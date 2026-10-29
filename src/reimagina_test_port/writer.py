"""Write a scenario's indicators to Reimagina's database (EAV schema).

Ported from reimagina-lyra ``store.write_indicators``. A fork stores only values
that differ from its baseline; rows whose stored value is unchanged are not
rewritten, so running it again for the same scenario changes nothing.
"""

import math
from collections.abc import Iterator, Mapping, Sequence
from typing import Any
from uuid import UUID

import pandas as pd
import psycopg
from psycopg import sql

from reimagina_test_port import contract
from reimagina_test_port.scenario import SCHEMA, resolve_dsn

# Decimals persisted, as reimagina-lyra ``base.INDICATOR_DECIMALS``; a fork's values
# compare exactly against its baseline's.
INDICATOR_DECIMALS = 4

_TABLES = {
    name: sql.Identifier(SCHEMA, name)
    for name in (
        "scenarios",
        "element_state_indicator_value",
        "job_run",
    )
}

# Same tolerance as the API's replace_indicators (math.isclose rel 1e-9, abs 1e-12).
_SAME_VALUE = sql.SQL(
    """
    ({a}.value_text IS NOT DISTINCT FROM {b}.value_text AND (
        ({a}.value_num IS NULL AND {b}.value_num IS NULL)
        OR abs({a}.value_num - {b}.value_num) <= greatest(
            1e-12, 1e-9 * greatest(abs({a}.value_num), abs({b}.value_num))
        )
    ) IS TRUE)
    """
)


def _same_value(a: str, b: str) -> sql.Composed:
    return _SAME_VALUE.format(a=sql.Identifier(a), b=sql.Identifier(b))


# Changes to the scenario's own rows:
# - remove: stored rows equal to the baseline again (or empty in a baseline
#   scenario), or of an element no longer present;
# - write: values that differ from the baseline and are new or changed.
_CHANGES_QUERY = sql.SQL(
    """
    CREATE TEMP TABLE indicator_change ON COMMIT DROP AS
    SELECT
        coalesce(p.element_id, s.element_id) AS element_id,
        coalesce(p.indicator_id, s.indicator_id) AS indicator_id,
        p.value_num,
        p.value_text,
        p.element_id IS NULL OR {same_as_baseline} AS remove
    FROM indicator_payload p
    LEFT JOIN {element_state_indicator_value} b
      ON b.scenario_id = %(baseline)s::uuid
     AND b.indicator_id = ANY(%(indicator_ids)s)
     AND b.element_id = p.element_id
     AND b.indicator_id = p.indicator_id
    FULL JOIN (
        SELECT element_id, indicator_id, value_num, value_text
        FROM {element_state_indicator_value}
        WHERE scenario_id = %(scenario)s::uuid
          AND indicator_id = ANY(%(indicator_ids)s)
    ) s ON s.element_id = p.element_id AND s.indicator_id = p.indicator_id
    WHERE CASE
        WHEN p.element_id IS NULL THEN TRUE
        WHEN {same_as_baseline} THEN s.element_id IS NOT NULL
        ELSE s.element_id IS NULL OR NOT {same_as_stored}
    END
    """
).format(
    same_as_baseline=_same_value("p", "b"),
    same_as_stored=_same_value("p", "s"),
    **_TABLES,
)

_DELETE_QUERY = sql.SQL(
    """
    DELETE FROM {element_state_indicator_value} v
    USING indicator_change c
    WHERE v.scenario_id = %(scenario)s::uuid
      AND v.element_id = c.element_id
      AND v.indicator_id = c.indicator_id
      AND c.remove
    """
).format(**_TABLES)

_UPSERT_QUERY = sql.SQL(
    """
    INSERT INTO {element_state_indicator_value}
        (scenario_id, element_id, indicator_id, value_num, value_text,
         job_run_id, computed_at)
    SELECT %(scenario)s::uuid, t.element_id, t.indicator_id, t.value_num, t.value_text,
           %(job_run)s::uuid, NOW()
    FROM indicator_change t
    WHERE NOT t.remove
    ON CONFLICT (scenario_id, element_id, indicator_id) DO UPDATE
    SET value_num = EXCLUDED.value_num,
        value_text = EXCLUDED.value_text,
        job_run_id = EXCLUDED.job_run_id,
        computed_at = EXCLUDED.computed_at,
        updated_at = NOW()
    """
).format(**_TABLES)


class ScenarioNotFoundError(LookupError):
    """The scenario does not exist or was deleted."""


def _plain(value: Any) -> Any:
    """Rounded float, text, or None for a missing value."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        return value
    number = float(value)
    return None if math.isinf(number) else round(number, INDICATOR_DECIMALS)


def element_values(
    tables: Sequence[tuple[str, pd.DataFrame]], keys: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """``(elements, values)`` from ``(element_type, table)`` pairs keyed by ``id``.

    ``elements`` has ``id`` and ``type_slug``; ``values`` maps id to its ``keys``
    found in the tables. Several tables of one type are merged by id.
    """
    values: dict[str, dict[str, Any]] = {}
    types: dict[str, str] = {}
    for element_type, table in tables:
        columns = [key for key in keys if key in table.columns]
        for row in table[["id", *columns]].to_dict("records"):
            identifier = str(row.pop("id"))
            types[identifier] = element_type
            entry = values.setdefault(identifier, {})
            entry.update({key: _plain(value) for key, value in row.items()})
    elements = pd.DataFrame({"id": list(types), "type_slug": list(types.values())})
    return elements, values


def payload_records(
    values: dict[str, dict[str, Any]],
    elements: pd.DataFrame,
    indicator_ids: dict[tuple[str, str], int],
) -> Iterator[tuple[UUID, int, float | None, str | None]]:
    """``(element_id, indicator_id, value_num, value_text)`` per owned indicator.

    Keys without a value are sent as NULL so stale values are cleared.
    """
    by_type: dict[str, list[tuple[str, int]]] = {}
    for (element_type, key), indicator_id in sorted(indicator_ids.items()):
        by_type.setdefault(element_type, []).append((key, indicator_id))
    for element_id, element_type in zip(
        elements["id"], elements["type_slug"], strict=True
    ):
        owned = by_type.get(element_type)
        if not owned:
            continue
        entry = values.get(str(element_id), {})
        element_uuid = UUID(str(element_id))
        for key, indicator_id in owned:
            value = entry.get(key)
            if isinstance(value, str):
                yield (element_uuid, indicator_id, None, value)
            else:
                yield (element_uuid, indicator_id, value, None)


def write_indicators(
    scenario_id: str,
    values: dict[str, dict[str, Any]],
    elements: pd.DataFrame,
    keys: Mapping[str, str],
    *,
    job_run_id: str | None = None,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Write the owned ``keys`` (``{key: kind}``) of ``elements`` in one transaction.

    Every key must be active in ``indicator_catalog`` for each element type
    written (``contract.check``). ``job_run_id`` is recorded only when that
    job_run exists. A dry run computes the changes and rolls them back.

    Returns:
        ``(values_written, values_cleared)``.
    """
    with psycopg.connect(resolve_dsn()) as conn, conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT parent_scenario_id::text FROM {scenarios} "
                "WHERE id = %s::uuid AND deleted_at IS NULL"
            ).format(**_TABLES),
            (scenario_id,),
        )
        row = cur.fetchone()
        if row is None:
            msg = f"Scenario {scenario_id!r} does not exist."
            raise ScenarioNotFoundError(msg)
        baseline_id = row[0]
        if job_run_id is not None:
            cur.execute(
                sql.SQL("SELECT 1 FROM {job_run} WHERE id = %s::uuid").format(
                    **_TABLES
                ),
                (job_run_id,),
            )
            if cur.fetchone() is None:
                job_run_id = None
        required = {
            (element_type, key): kind
            for element_type in sorted(set(elements["type_slug"]))
            for key, kind in keys.items()
        }
        indicator_ids = {
            pair: row[0]
            for pair, row in contract.check(cur, "indicator_catalog", required).items()
        }
        # JIT costs about a second per query and saves nothing here.
        cur.execute("SET LOCAL jit = off")
        cur.execute(
            "CREATE TEMP TABLE indicator_payload (element_id uuid, "
            "indicator_id integer, value_num double precision, value_text text) "
            "ON COMMIT DROP"
        )
        with cur.copy(
            "COPY indicator_payload (element_id, indicator_id, value_num, "
            "value_text) FROM STDIN (FORMAT BINARY)"
        ) as copy:
            copy.set_types(["uuid", "int4", "float8", "text"])
            for record in payload_records(values, elements, indicator_ids):
                copy.write_row(record)
        # Without stats the planner may pick a nested loop over the value table.
        cur.execute("ANALYZE indicator_payload")
        params = {
            "scenario": scenario_id,
            "baseline": baseline_id,
            "indicator_ids": sorted(indicator_ids.values()),
            "job_run": job_run_id,
        }
        cur.execute(_CHANGES_QUERY, params)
        cur.execute("CREATE INDEX ON indicator_change (element_id, indicator_id)")
        cur.execute("ANALYZE indicator_change")
        cur.execute(_DELETE_QUERY, params)
        cleared = cur.rowcount
        cur.execute(_UPSERT_QUERY, params)
        written = cur.rowcount
        if dry_run:
            conn.rollback()
        else:
            conn.commit()
    return written, cleared
