"""One-off: copy a fork's edits from the old Mexicali baseline to a fork of the new one.

The new baseline's lot ids come from the cadastral key, so an edited lot of the old
baseline is matched to the new lot covering the same land: the largest overlap with
intersection over union of at least MIN_IOU. Elements with the same id in the new
baseline (blocks), and elements that exist only in the fork, keep their id.
Indicator values are not copied: run reimagina_scenario_write on the target fork.

    REIMAGINA_POSTGRES_DSN=... uv run python scripts/migrate_fork_edits.py \\
        SOURCE_FORK TARGET_FORK [--apply]

Without --apply it reports the mapping and rolls back.
"""

import argparse
import sys

import psycopg

from reimagina_test_port.scenario import resolve_dsn

MIN_IOU = 0.9

_FORKS = """
SELECT s.id::text, s.project_id::text, s.parent_scenario_id::text
FROM reimagina_urbano.scenarios s
WHERE s.id = ANY(%s::uuid[]) AND s.deleted_at IS NULL
"""

# Each edited element of the source fork and the id it takes in the target fork.
_MAPPING = """
WITH edited AS (
    SELECT es.element_id FROM reimagina_urbano.element_state es
    WHERE es.scenario_id = %(source)s::uuid
),
same AS (
    SELECT e.element_id, e.element_id AS target, 'same id' AS how, NULL::float AS iou
    FROM edited e
    WHERE EXISTS (SELECT 1 FROM reimagina_urbano.element_state n
                  WHERE n.scenario_id = %(new)s::uuid AND n.element_id = e.element_id)
       OR NOT EXISTS (SELECT 1 FROM reimagina_urbano.element_state o
                      WHERE o.scenario_id = %(old)s::uuid
                        AND o.element_id = e.element_id)
),
matched AS (
    SELECT DISTINCT ON (e.element_id) e.element_id, n.element_id AS target,
           'geometry' AS how,
           ST_Area(ST_Intersection(o.geometry, n.geometry)::geography)
             / NULLIF(ST_Area(ST_Union(o.geometry, n.geometry)::geography), 0) AS iou
    FROM edited e
    JOIN reimagina_urbano.element_state o
      ON o.scenario_id = %(old)s::uuid AND o.element_id = e.element_id
    JOIN reimagina_urbano.elements oe ON oe.id = e.element_id
    JOIN reimagina_urbano.element_state n
      ON n.scenario_id = %(new)s::uuid AND ST_Intersects(n.geometry, o.geometry)
    JOIN reimagina_urbano.elements ne ON ne.id = n.element_id AND ne.type = oe.type
    WHERE e.element_id NOT IN (SELECT element_id FROM same)
    ORDER BY e.element_id, iou DESC NULLS LAST
)
SELECT element_id::text, target::text, how, iou FROM same
UNION ALL
SELECT element_id::text, CASE WHEN iou >= %(min_iou)s THEN target::text END,
       how, iou FROM matched
UNION ALL
SELECT e.element_id::text, NULL, 'no overlap', NULL FROM edited e
WHERE e.element_id NOT IN (SELECT element_id FROM same)
  AND e.element_id NOT IN (SELECT element_id FROM matched)
"""

_COPY_STATE = """
INSERT INTO reimagina_urbano.element_state
    (scenario_id, element_id, parent_element_id, geometry, deleted_at)
SELECT %(target)s::uuid, m.target, COALESCE(p.target, s.parent_element_id),
       s.geometry, s.deleted_at
FROM reimagina_urbano.element_state s
JOIN fork_mapping m ON m.element_id = s.element_id
LEFT JOIN fork_mapping p ON p.element_id = s.parent_element_id
WHERE s.scenario_id = %(source)s::uuid AND m.target IS NOT NULL
"""

_COPY_ATTRIBUTES = """
INSERT INTO reimagina_urbano.element_state_attribute_value
    (scenario_id, element_id, attribute_id, value_num, value_text, value_json)
SELECT %(target)s::uuid, m.target, v.attribute_id, v.value_num, v.value_text,
       v.value_json
FROM reimagina_urbano.element_state_attribute_value v
JOIN fork_mapping m ON m.element_id = v.element_id
WHERE v.scenario_id = %(source)s::uuid AND m.target IS NOT NULL
"""


def migrate(dsn: str, source: str, target: str, *, apply: bool) -> list[tuple]:
    """Copy the edits; return ``(element_id, target_id, how, iou)`` per edit."""
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        forks = {row[0]: row for row in cur.execute(_FORKS, ([source, target],))}
        if set(forks) != {source, target}:
            msg = "Both forks must exist."
            raise ValueError(msg)
        if forks[source][1] != forks[target][1]:
            msg = "The forks belong to different projects."
            raise ValueError(msg)
        old, new = forks[source][2], forks[target][2]
        if old is None or new is None or old == new:
            msg = "Both must be forks, of two different baselines."
            raise ValueError(msg)
        cur.execute(
            "SELECT count(*) FROM reimagina_urbano.element_state "
            "WHERE scenario_id = %s::uuid",
            (target,),
        )
        if cur.fetchone()[0]:
            msg = f"Target fork {target} already has edits."
            raise ValueError(msg)
        params = {
            "source": source,
            "target": target,
            "old": old,
            "new": new,
            "min_iou": MIN_IOU,
        }
        mapping = cur.execute(_MAPPING, params).fetchall()
        cur.execute(
            "CREATE TEMP TABLE fork_mapping (element_id uuid, target uuid) "
            "ON COMMIT DROP"
        )
        with cur.copy("COPY fork_mapping (element_id, target) FROM STDIN") as copy:
            for element_id, target_id, _, _ in mapping:
                copy.write_row((element_id, target_id))
        cur.execute(_COPY_STATE, params)
        cur.execute(_COPY_ATTRIBUTES, params)
        if apply:
            conn.commit()
        else:
            conn.rollback()
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source_fork")
    parser.add_argument("target_fork")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    mapping = migrate(
        resolve_dsn(), args.source_fork, args.target_fork, apply=args.apply
    )
    for element_id, target_id, how, iou in mapping:
        overlap = "" if iou is None else f" (IoU {iou:.3f})"
        print(f"{element_id} -> {target_id or 'NOT COPIED'}: {how}{overlap}")
    skipped = sum(target_id is None for _, target_id, _, _ in mapping)
    print(
        f"{len(mapping)} edited elements, {skipped} not copied; "
        + ("applied." if args.apply else "rolled back (use --apply).")
    )
    sys.exit(1 if skipped else 0)


if __name__ == "__main__":
    main()
