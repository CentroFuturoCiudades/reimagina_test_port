"""Write a city's baseline elements as a Reimagina baseline scenario.

The elements are blocks, lots, amenities, establishments and rights of way; each
element type
must exist in the database's ``element_type`` enum. Missing ``attribute_catalog``
and ``indicator_catalog`` rows are added first (``catalogs.upsert``), existing
ones left as they are.

Fills an existing baseline scenario (created in the API) with ``elements``,
``element_state`` and ``element_state_attribute_value`` rows, in one
transaction. Elements that already exist (blocks keep their CVEGEO id across
baselines) are reused unchanged, so other baselines are not touched. Running it
again replaces the scenario's own rows. Indicators are computed afterwards with
``reimagina_scenario_write``.

``append_baseline`` adds to a baseline instead, forks or not: it inserts the
elements it lacks and adds or updates the given attributes of those it has,
touching nothing else; forks see the new elements through the baseline.
"""

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import geopandas as gpd
import pandas as pd
import psycopg
import shapely
from psycopg import sql

from reimagina_test_port.ingestion import catalogs
from reimagina_test_port.scenario import contract
from reimagina_test_port.scenario.source import SCHEMA, resolve_dsn

_TABLES = {
    name: sql.Identifier(SCHEMA, name)
    for name in (
        "scenarios",
        "elements",
        "element_state",
        "element_state_attribute_value",
        "element_state_indicator_value",
        "attribute_catalog",
    )
}

_INSERT_ELEMENTS = sql.SQL(
    """
    INSERT INTO {elements} (id, type, parent_element_id, origin_scenario_id,
                            source_record_ref)
    SELECT id, type::{schema}.element_type, parent, %(scenario)s::uuid, ref
    FROM baseline_elements
    ON CONFLICT (id) DO NOTHING
    """
).format(schema=sql.Identifier(SCHEMA), **_TABLES)

_TYPE_CONFLICTS = sql.SQL(
    """
    SELECT count(*) FROM baseline_elements b JOIN {elements} e ON e.id = b.id
    WHERE e.type::text <> b.type
    """
).format(**_TABLES)

# Replace the scenario's rows: first values and state of absent elements, then
# all attribute values, which are written again below.
_DELETE_INDICATORS = sql.SQL(
    """
    DELETE FROM {element_state_indicator_value} v
    WHERE v.scenario_id = %(scenario)s::uuid
      AND NOT EXISTS (SELECT 1 FROM baseline_elements b WHERE b.id = v.element_id)
    """
).format(**_TABLES)

_DELETE_ATTRIBUTES = sql.SQL(
    "DELETE FROM {element_state_attribute_value} WHERE scenario_id = %(scenario)s::uuid"
).format(**_TABLES)

_DELETE_STATE = sql.SQL(
    """
    DELETE FROM {element_state} s
    WHERE s.scenario_id = %(scenario)s::uuid
      AND NOT EXISTS (SELECT 1 FROM baseline_elements b WHERE b.id = s.element_id)
    """
).format(**_TABLES)

_UPSERT_STATE = sql.SQL(
    """
    INSERT INTO {element_state} (scenario_id, element_id, parent_element_id, geometry)
    SELECT %(scenario)s::uuid, id, parent, ST_SetSRID(ST_GeomFromWKB(wkb), 4326)
    FROM baseline_elements
    ON CONFLICT (scenario_id, element_id) DO UPDATE
    SET parent_element_id = EXCLUDED.parent_element_id,
        geometry = EXCLUDED.geometry,
        deleted_at = NULL
    """
).format(**_TABLES)

_INSERT_ATTRIBUTES = sql.SQL(
    """
    INSERT INTO {element_state_attribute_value}
        (scenario_id, element_id, attribute_id, value_num, value_text, value_json)
    SELECT %(scenario)s::uuid, element_id, attribute_id, value_num, value_text,
           value_json
    FROM baseline_attributes
    """
).format(**_TABLES)


# What the scenario holds once written, read back from the database.
_COUNT_BY_TYPE = sql.SQL(
    """
    SELECT e.type::text, count(*) FROM {element_state} s
    JOIN {elements} e ON e.id = s.element_id
    WHERE s.scenario_id = %(scenario)s::uuid AND s.deleted_at IS NULL
    GROUP BY 1
    """
).format(**_TABLES)


class BaselineScenarioError(ValueError):
    """The target is not an existing baseline of the project (without forks, to
    replace it)."""


def _check_scenario(
    cur: psycopg.Cursor, project_id: str, scenario_id: str, *, forks: bool = False
) -> None:
    """Raise unless ``scenario_id`` is a baseline of ``project_id``; with forks
    too, unless ``forks``."""
    cur.execute(
        sql.SQL(
            "SELECT project_id::text, parent_scenario_id FROM {scenarios} "
            "WHERE id = %s::uuid AND deleted_at IS NULL"
        ).format(**_TABLES),
        (scenario_id,),
    )
    row = cur.fetchone()
    if row is None:
        msg = f"Scenario {scenario_id} does not exist; create it in the API first."
        raise BaselineScenarioError(msg)
    if row[0] != project_id:
        msg = f"Scenario {scenario_id} belongs to project {row[0]}, not {project_id}."
        raise BaselineScenarioError(msg)
    if row[1] is not None:
        msg = f"Scenario {scenario_id} is a fork; only a baseline can be written."
        raise BaselineScenarioError(msg)
    if forks:
        return
    cur.execute(
        sql.SQL(
            "SELECT count(*) FROM {scenarios} "
            "WHERE parent_scenario_id = %s::uuid AND deleted_at IS NULL"
        ).format(**_TABLES),
        (scenario_id,),
    )
    if cur.fetchone()[0]:
        msg = (
            f"Baseline {scenario_id} has forks; replacing it would leave their "
            "edits pointing at removed elements."
        )
        raise BaselineScenarioError(msg)


def _encode(value: Any, value_type: str) -> tuple[float | None, str | None, str | None]:
    """``(value_num, value_text, value_json)``; all None for a missing value. A json
    value may arrive already encoded, as a string."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None, None, None
    if value_type in ("number", "integer"):
        return float(value), None, None
    if value_type == "json":
        return None, None, value if isinstance(value, str) else json.dumps(value)
    return None, str(value), None


def _copy_elements(
    cur: psycopg.Cursor, name: str, tables: Mapping[str, gpd.GeoDataFrame]
) -> None:
    """A temporary table ``name`` with the elements of ``tables``."""
    cur.execute(
        sql.SQL(
            "CREATE TEMP TABLE {} (id uuid PRIMARY KEY, type text, parent uuid, "
            "ref text, wkb bytea, outcome text) ON COMMIT DROP"
        ).format(sql.Identifier(name))
    )
    with cur.copy(
        sql.SQL("COPY {} (id, type, parent, ref, wkb) FROM STDIN").format(
            sql.Identifier(name)
        )
    ) as copy:
        for element_type, table in tables.items():
            for identifier, parent, ref, geometry in zip(
                table["id"],
                table["parent_id"],
                table["source_record_ref"],
                shapely.to_wkb(table.geometry.to_numpy()),
                strict=True,
            ):
                copy.write_row((identifier, element_type, parent, ref, geometry))


def _copy_attributes(
    cur: psycopg.Cursor,
    name: str,
    tables: Mapping[str, pd.DataFrame],
    attributes: Mapping[str, Sequence[str]],
    ids: Mapping[tuple[str, str], tuple[int, str]],
) -> int:
    """A temporary table ``name`` with the ``attributes`` (``{element_type:
    keys}``) of ``tables``, missing values left out; returns its rows."""
    cur.execute(
        sql.SQL(
            "CREATE TEMP TABLE {} (element_id uuid, attribute_id integer, "
            "value_num double precision, value_text text, value_json jsonb, "
            "ref text, outcome text) ON COMMIT DROP"
        ).format(sql.Identifier(name))
    )
    values = 0
    with cur.copy(
        sql.SQL(
            "COPY {} (element_id, attribute_id, value_num, value_text, value_json, "
            "ref) FROM STDIN"
        ).format(sql.Identifier(name))
    ) as copy:
        for element_type, keys in attributes.items():
            table = tables[element_type]
            for key in keys:
                attribute_id, value_type = ids[(element_type, key)]
                for identifier, ref, value in zip(
                    table["id"], table["source_record_ref"], table[key], strict=True
                ):
                    encoded = _encode(value, value_type)
                    if any(part is not None for part in encoded):
                        copy.write_row((identifier, attribute_id, *encoded, ref))
                        values += 1
    return values


def write_baseline(
    project_id: str,
    scenario_id: str,
    tables: dict[str, gpd.GeoDataFrame],
    attributes: dict[str, dict[str, str]],
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """Write ``tables`` (``{element_type: table}``) into the baseline.

    ``attributes`` maps each element type to ``{column: kind}``, the columns
    written as attribute values; types not in ``tables`` are ignored. A dry run
    rolls everything back.

    Returns:
        Counts: ``elements``, ``elements_created``, ``attribute_values``,
        ``elements_removed`` (state rows of elements no longer present),
        ``attributes_added`` and ``indicators_added`` (catalog rows added) and
        ``by_type`` (``{element_type: elements}``, the scenario's state rows of
        each type of ``tables`` after the write, read back from the database).
    """
    attributes = {t: columns for t, columns in attributes.items() if t in tables}
    required = {
        (element_type, key): kind
        for element_type, columns in attributes.items()
        for key, kind in columns.items()
    }
    enums = {
        (element_type, key): {
            str(v) for v in tables[element_type][key].dropna().unique()
        }
        for (element_type, key), kind in required.items()
        if kind == "text"
    }
    with psycopg.connect(resolve_dsn()) as conn, conn.cursor() as cur:
        _check_scenario(cur, project_id, scenario_id)
        added = catalogs.upsert(cur, attributes)
        ids = contract.check(cur, "attribute_catalog", required, enums)
        _copy_elements(cur, "baseline_elements", tables)
        values = _copy_attributes(cur, "baseline_attributes", tables, attributes, ids)
        cur.execute("ANALYZE baseline_elements")
        params = {"scenario": scenario_id}
        cur.execute(_INSERT_ELEMENTS, params)
        created = cur.rowcount
        cur.execute(_TYPE_CONFLICTS)
        if conflicts := cur.fetchone()[0]:
            msg = f"{conflicts} element ids already exist with another type."
            raise BaselineScenarioError(msg)
        cur.execute(_DELETE_INDICATORS, params)
        cur.execute(_DELETE_ATTRIBUTES, params)
        cur.execute(_DELETE_STATE, params)
        removed = cur.rowcount
        cur.execute(_UPSERT_STATE, params)
        cur.execute(_INSERT_ATTRIBUTES, params)
        cur.execute(_COUNT_BY_TYPE, params)
        stored = dict(cur.fetchall())
        if dry_run:
            conn.rollback()
        else:
            conn.commit()
    return {
        "elements": sum(len(table) for table in tables.values()),
        "elements_created": created,
        "attribute_values": values,
        "elements_removed": removed,
        **added,
        "by_type": {
            element_type: stored.get(element_type, 0) for element_type in tables
        },
    }


# ---------------------------------------------------------------------- append

_APPEND_SQL = {
    name: sql.SQL(query).format(schema=sql.Identifier(SCHEMA), **_TABLES)
    for name, query in {
        # Elements to insert: those without a state row in the baseline (deleted
        # or not), whose parent, if any, is present in it.
        "classify": """
            UPDATE append_elements a SET outcome = CASE
                WHEN EXISTS (
                    SELECT 1 FROM {element_state} s
                    WHERE s.scenario_id = %(scenario)s::uuid AND s.element_id = a.id
                ) THEN 'already_present'
                WHEN a.parent IS NOT NULL AND NOT EXISTS (
                    SELECT 1 FROM {element_state} s
                    WHERE s.scenario_id = %(scenario)s::uuid
                      AND s.element_id = a.parent AND s.deleted_at IS NULL
                ) THEN 'missing_parent'
                ELSE 'inserted'
            END
        """,
        "type_conflicts": """
            SELECT count(*) FROM append_elements a JOIN {elements} e ON e.id = a.id
            WHERE a.outcome = 'inserted' AND e.type::text <> a.type
        """,
        "elements": """
            INSERT INTO {elements} (id, type, parent_element_id, origin_scenario_id,
                                    source_record_ref)
            SELECT id, type::{schema}.element_type, parent, %(scenario)s::uuid, ref
            FROM append_elements WHERE outcome = 'inserted'
            ON CONFLICT (id) DO NOTHING
        """,
        "state": """
            INSERT INTO {element_state}
                (scenario_id, element_id, parent_element_id, geometry)
            SELECT %(scenario)s::uuid, id, parent,
                   ST_SetSRID(ST_GeomFromWKB(wkb), 4326)
            FROM append_elements WHERE outcome = 'inserted'
        """,
        "attributes": """
            INSERT INTO {element_state_attribute_value}
                (scenario_id, element_id, attribute_id, value_num, value_text,
                 value_json)
            SELECT %(scenario)s::uuid, v.element_id, v.attribute_id, v.value_num,
                   v.value_text, v.value_json
            FROM append_attributes v
            JOIN append_elements a ON a.id = v.element_id AND a.outcome = 'inserted'
        """,
        # Inserted elements whose parent a fork deleted: present in that fork
        # still, on a lot it no longer has.
        "fork_orphans": """
            SELECT a.type, count(DISTINCT a.id),
                   (array_agg(DISTINCT a.ref))[1:5]
            FROM append_elements a
            JOIN {scenarios} f
              ON f.parent_scenario_id = %(scenario)s::uuid AND f.deleted_at IS NULL
            JOIN {element_state} s
              ON s.scenario_id = f.id AND s.element_id = a.parent
             AND s.deleted_at IS NOT NULL
            WHERE a.outcome = 'inserted'
            GROUP BY a.type
        """,
        # Attribute values of elements the baseline has (before any insert).
        "classify_updates": """
            UPDATE update_attributes u SET outcome = CASE
                WHEN NOT EXISTS (
                    SELECT 1 FROM {element_state} s
                    WHERE s.scenario_id = %(scenario)s::uuid
                      AND s.element_id = u.element_id
                ) THEN 'absent'
                WHEN NOT EXISTS (
                    SELECT 1 FROM {element_state_attribute_value} v
                    WHERE v.scenario_id = %(scenario)s::uuid
                      AND v.element_id = u.element_id
                      AND v.attribute_id = u.attribute_id
                ) THEN 'added'
                WHEN EXISTS (
                    SELECT 1 FROM {element_state_attribute_value} v
                    WHERE v.scenario_id = %(scenario)s::uuid
                      AND v.element_id = u.element_id
                      AND v.attribute_id = u.attribute_id
                      AND v.value_num IS NOT DISTINCT FROM u.value_num
                      AND v.value_text IS NOT DISTINCT FROM u.value_text
                      AND v.value_json IS NOT DISTINCT FROM u.value_json
                ) THEN 'unchanged'
                ELSE 'updated'
            END
        """,
        "delete_updated": """
            DELETE FROM {element_state_attribute_value} v
            USING update_attributes u
            WHERE u.outcome = 'updated' AND v.scenario_id = %(scenario)s::uuid
              AND v.element_id = u.element_id AND v.attribute_id = u.attribute_id
        """,
        "upsert_updates": """
            INSERT INTO {element_state_attribute_value}
                (scenario_id, element_id, attribute_id, value_num, value_text,
                 value_json)
            SELECT %(scenario)s::uuid, element_id, attribute_id, value_num,
                   value_text, value_json
            FROM update_attributes WHERE outcome IN ('added', 'updated')
        """,
        # One row per element type (and key, for updates) and outcome.
        "element_outcomes": """
            SELECT type, NULL, outcome, count(*), (array_agg(ref ORDER BY ref))[1:5]
            FROM append_elements GROUP BY type, outcome
        """,
        "update_outcomes": """
            SELECT ac.element_type::text, ac.key, u.outcome, count(*),
                   (array_agg(u.ref ORDER BY u.ref))[1:5]
            FROM update_attributes u
            JOIN {attribute_catalog} ac ON ac.id = u.attribute_id
            GROUP BY 1, 2, 3
        """,
    }.items()
}
APPEND_OUTCOMES = (
    "inserted", "already_present", "missing_parent", "parent_deleted_in_fork",
    "added", "updated", "unchanged", "absent",
)  # fmt: skip


def append_baseline(
    project_id: str,
    scenario_id: str,
    tables: Mapping[str, gpd.GeoDataFrame],
    attributes: Mapping[str, Mapping[str, str]],
    *,
    updates: Mapping[str, tuple[gpd.GeoDataFrame, Sequence[str]]] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Add to the baseline, forks or not, in one transaction; nothing else changes.

    - ``tables`` (``{element_type: table}``, as ``write_baseline``'s): the elements
      the baseline lacks are inserted, with their state and their ``attributes``
      (``{element_type: {column: kind}}``). Skipped: those it already has
      (``already_present``, deleted there or not) and those whose parent it lacks
      (``missing_parent``), as when another project has other lots.
    - ``updates`` (``{element_type: (table, keys)}``): of the elements the
      baseline has, the ``keys`` values are ``added`` or ``updated`` (or left
      ``unchanged``); a missing value is left as it is, and elements the baseline
      lacks are ``absent``.

    Forks read the baseline's rows, so they see what is added unless they have
    their own row; ``parent_deleted_in_fork`` counts inserted elements whose
    parent some fork deleted. A dry run rolls everything back.

    Returns:
        ``outcomes``, rows of ``element_type``, ``attribute`` (None for
        elements), ``outcome`` (``APPEND_OUTCOMES``), ``count`` and ``examples``
        (up to five ``source_record_ref``), and ``attributes_added`` and
        ``indicators_added`` (catalog rows added).

    Raises:
        BaselineScenarioError: the scenario is not a baseline of the project, or
            an element id exists with another type.
    """
    updates = {
        element_type: (table, attribute_kinds(attributes, element_type, keys))
        for element_type, (table, keys) in (updates or {}).items()
    }
    inserted = {t: columns for t, columns in attributes.items() if t in tables}
    catalog: dict[str, dict[str, str]] = {}
    enums: dict[tuple[str, str], set[str]] = {}
    for element_type, source, columns in [
        *((t, tables[t], columns) for t, columns in inserted.items()),
        *((t, table, columns) for t, (table, columns) in updates.items()),
    ]:
        catalog.setdefault(element_type, {}).update(columns)
        for key, kind in columns.items():
            if kind == "text":
                enums.setdefault((element_type, key), set()).update(
                    str(v) for v in source[key].dropna()
                )
    required = {
        (element_type, key): kind
        for element_type, columns in catalog.items()
        for key, kind in columns.items()
    }
    params = {"scenario": scenario_id}
    with psycopg.connect(resolve_dsn()) as conn, conn.cursor() as cur:
        _check_scenario(cur, project_id, scenario_id, forks=True)
        added = catalogs.upsert(cur, catalog)
        ids = contract.check(cur, "attribute_catalog", required, enums)
        _copy_elements(cur, "append_elements", tables)
        _copy_attributes(cur, "append_attributes", tables, inserted, ids)
        _copy_attributes(
            cur,
            "update_attributes",
            {t: table for t, (table, _) in updates.items()},
            {t: keys for t, (_, keys) in updates.items()},
            ids,
        )
        cur.execute("ANALYZE append_elements")
        cur.execute("ANALYZE update_attributes")
        # Updates first: they concern the elements the baseline had.
        cur.execute(_APPEND_SQL["classify_updates"], params)
        cur.execute(_APPEND_SQL["delete_updated"], params)
        cur.execute(_APPEND_SQL["upsert_updates"], params)
        cur.execute(_APPEND_SQL["classify"], params)
        cur.execute(_APPEND_SQL["type_conflicts"])
        if conflicts := cur.fetchone()[0]:
            msg = f"{conflicts} element ids already exist with another type."
            raise BaselineScenarioError(msg)
        for step in ("elements", "state", "attributes"):
            cur.execute(_APPEND_SQL[step], params)
        rows = []
        for step in ("element_outcomes", "update_outcomes"):
            cur.execute(_APPEND_SQL[step], params)
            rows += cur.fetchall()
        cur.execute(_APPEND_SQL["fork_orphans"], params)
        rows += [
            (element_type, None, "parent_deleted_in_fork", count, examples)
            for element_type, count, examples in cur.fetchall()
        ]
        if dry_run:
            conn.rollback()
        else:
            conn.commit()
    order = {outcome: i for i, outcome in enumerate(APPEND_OUTCOMES)}
    outcomes = [
        {
            "element_type": element_type,
            "attribute": attribute,
            "outcome": outcome,
            "count": count,
            "examples": list(examples),
        }
        for element_type, attribute, outcome, count, examples in sorted(
            rows, key=lambda row: (row[0], row[1] or "", order[row[2]])
        )
    ]
    return {"outcomes": outcomes, **added}


def attribute_kinds(
    attributes: Mapping[str, Mapping[str, str]],
    element_type: str,
    keys: Sequence[str],
) -> dict[str, str]:
    """``{key: kind}`` of ``keys``, attributes of ``element_type``.

    Raises:
        BaselineScenarioError: a key that is not one of its attributes.
    """
    known = attributes.get(element_type, {})
    if unknown := sorted(set(keys) - set(known)):
        msg = f"{element_type} has no attributes {unknown}; known: {sorted(known)}"
        raise BaselineScenarioError(msg)
    return {key: known[key] for key in keys}
