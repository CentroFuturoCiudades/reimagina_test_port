"""Write a city's baseline lots and blocks as a Reimagina baseline scenario.

Fills an existing baseline scenario (created in the API) with ``elements``,
``element_state`` and ``element_state_attribute_value`` rows, in one
transaction. Elements that already exist (blocks keep their CVEGEO id across
baselines) are reused unchanged, so other baselines are not touched. Running it
again replaces the scenario's own rows. Indicators are computed afterwards with
``reimagina_scenario_write``.
"""

import math
from typing import Any

import geopandas as gpd
import psycopg
import shapely
from psycopg import sql

from reimagina_test_port import contract
from reimagina_test_port.scenario import SCHEMA, resolve_dsn

_TABLES = {
    name: sql.Identifier(SCHEMA, name)
    for name in (
        "scenarios",
        "elements",
        "element_state",
        "element_state_attribute_value",
        "element_state_indicator_value",
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
        (scenario_id, element_id, attribute_id, value_num, value_text)
    SELECT %(scenario)s::uuid, element_id, attribute_id, value_num, value_text
    FROM baseline_attributes
    """
).format(**_TABLES)


class BaselineScenarioError(ValueError):
    """The target is not an existing, fork-free baseline of the project."""


def _check_scenario(cur: psycopg.Cursor, project_id: str, scenario_id: str) -> None:
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


def _encode(value: Any, value_type: str) -> tuple[float | None, str | None]:
    """``(value_num, value_text)``; both None for a missing value."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None, None
    if value_type in ("number", "integer"):
        return float(value), None
    return None, str(value)


def write_baseline(
    project_id: str,
    scenario_id: str,
    tables: dict[str, gpd.GeoDataFrame],
    attributes: dict[str, dict[str, str]],
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """Write ``tables`` (``{"lot": ..., "block": ...}``) into the baseline.

    ``attributes`` maps each element type to ``{column: kind}``, the columns
    written as attribute values. A dry run rolls everything back.

    Returns:
        Counts: ``elements``, ``elements_created``, ``attribute_values`` and
        ``elements_removed`` (state rows of elements no longer present).
    """
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
        ids = contract.check(cur, "attribute_catalog", required, enums)
        cur.execute(
            "CREATE TEMP TABLE baseline_elements (id uuid PRIMARY KEY, type text, "
            "parent uuid, ref text, wkb bytea) ON COMMIT DROP"
        )
        cur.execute(
            "CREATE TEMP TABLE baseline_attributes (element_id uuid, "
            "attribute_id integer, value_num double precision, value_text text) "
            "ON COMMIT DROP"
        )
        values = 0
        with cur.copy(
            "COPY baseline_elements (id, type, parent, ref, wkb) FROM STDIN"
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
        with cur.copy(
            "COPY baseline_attributes (element_id, attribute_id, value_num, "
            "value_text) FROM STDIN"
        ) as copy:
            for element_type, columns in attributes.items():
                table = tables[element_type]
                for key in columns:
                    attribute_id, value_type = ids[(element_type, key)]
                    for identifier, value in zip(table["id"], table[key], strict=True):
                        num, text = _encode(value, value_type)
                        if num is not None or text is not None:
                            copy.write_row((identifier, attribute_id, num, text))
                            values += 1
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
        if dry_run:
            conn.rollback()
        else:
            conn.commit()
    return {
        "elements": sum(len(table) for table in tables.values()),
        "elements_created": created,
        "attribute_values": values,
        "elements_removed": removed,
    }
