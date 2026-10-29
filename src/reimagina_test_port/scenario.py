"""Effective scenario lots and blocks from Reimagina's Postgres (EAV schema).

Ported from reimagina-lyra ``store.py``: a fork stores only what differs from its
baseline, so each element, geometry and attribute comes from the scenario's own row
when it has one (even an empty one) and from the baseline otherwise.
"""

import json
import os
from typing import Any
from urllib.parse import urlparse

import geopandas as gpd
import pandas as pd
import psycopg
from psycopg import sql

DSN_ENV_VAR = "REIMAGINA_POSTGRES_DSN"
SCHEMA = "reimagina_urbano"
CRS = "EPSG:4326"

_TABLES = {
    name: sql.Identifier(SCHEMA, name)
    for name in (
        "scenarios",
        "elements",
        "element_state",
        "attribute_catalog",
        "element_state_attribute_value",
    )
}

# Present elements: the scenario's own rows, else the baseline's.
_STATES = """
    WITH states AS (
        SELECT es.element_id, es.parent_element_id, es.geometry
        FROM {element_state} es
        WHERE es.scenario_id = %(scenario)s::uuid AND es.deleted_at IS NULL
        UNION ALL
        SELECT b.element_id, b.parent_element_id, b.geometry
        FROM {element_state} b
        WHERE b.scenario_id = %(baseline)s::uuid AND b.deleted_at IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM {element_state} o
              WHERE o.scenario_id = %(scenario)s::uuid AND o.element_id = b.element_id
          )
    )
"""

_ELEMENTS_QUERY = sql.SQL(
    _STATES
    + """
    SELECT
        st.element_id::text AS id,
        COALESCE(st.parent_element_id, e.parent_element_id)::text AS parent_id,
        e.source_record_ref,
        e.type::text AS type,
        ST_AsBinary(st.geometry) AS geometry
    FROM states st
    JOIN {elements} e ON e.id = st.element_id
    WHERE e.deleted_at IS NULL AND e.type::text IN ('block', 'lot')
    ORDER BY e.type::text, st.element_id::text
    """
).format(**_TABLES)

# Scenario row first, so an all-NULL scenario row hides the baseline value.
_ATTRIBUTES_QUERY = sql.SQL(
    """
    SELECT DISTINCT ON (v.element_id, v.attribute_id)
        v.element_id::text AS element_id,
        ac.key,
        ac.value_type::text AS value_type,
        v.value_num,
        v.value_text,
        v.value_json
    FROM {element_state_attribute_value} v
    JOIN {attribute_catalog} ac ON ac.id = v.attribute_id
    WHERE v.scenario_id IN (%(scenario)s::uuid, %(baseline)s::uuid)
      AND ac.key = ANY(%(keys)s)
    ORDER BY v.element_id, v.attribute_id, (v.scenario_id = %(scenario)s::uuid) DESC
    """
).format(**_TABLES)

_BASELINE_QUERY = sql.SQL(
    "SELECT parent_scenario_id::text FROM {scenarios} "
    "WHERE id = %s::uuid AND deleted_at IS NULL"
).format(**_TABLES)


class ScenarioNotFoundError(LookupError):
    """The scenario does not exist or was deleted."""


def resolve_dsn() -> str:
    """Read the DSN secret; TLS is required unless the DSN sets ``sslmode``."""
    dsn = os.environ.get(DSN_ENV_VAR)
    if not dsn:
        msg = f"{DSN_ENV_VAR} is not set on this worker."
        raise RuntimeError(msg)
    if "sslmode=" not in dsn:
        dsn += ("&" if urlparse(dsn).query else "?") + "sslmode=require"
    return dsn


def decode_value(value_type: str, num: Any, text: Any, js: Any) -> Any:
    """Python value of one EAV row (exe_reimagina ``eav.decode_value``)."""
    if value_type in ("number", "integer"):
        if num is None:
            return None
        return round(num) if value_type == "integer" else float(num)
    if value_type in ("text", "enum"):
        return None if text is None else str(text)
    if value_type == "json":
        return json.loads(js) if isinstance(js, str) else js
    msg = f"Unknown value_type: {value_type!r}"
    raise ValueError(msg)


def read_scenario(scenario_id: str, keys: list[str]) -> dict[str, gpd.GeoDataFrame]:
    """Return effective ``lot`` and ``block`` tables with attribute ``keys`` as columns."""
    with psycopg.connect(resolve_dsn()) as conn, conn.cursor() as cur:
        cur.execute(_BASELINE_QUERY, (scenario_id,))
        row = cur.fetchone()
        if row is None:
            msg = f"Scenario {scenario_id!r} does not exist."
            raise ScenarioNotFoundError(msg)
        params = {"scenario": scenario_id, "baseline": row[0], "keys": keys}
        cur.execute(_ELEMENTS_QUERY, params)
        elements = pd.DataFrame(
            cur.fetchall(),
            columns=["id", "parent_id", "source_record_ref", "type", "geometry"],
        )
        cur.execute(_ATTRIBUTES_QUERY, params)
        attribute_rows = cur.fetchall()

    attributes: dict[str, dict[str, Any]] = {}
    for element_id, key, value_type, num, text, js in attribute_rows:
        attributes.setdefault(element_id, {})[key] = decode_value(
            value_type, num, text, js
        )
    result = {}
    for kind in ("lot", "block"):
        frame = elements[elements["type"] == kind].reset_index(drop=True)
        values = [attributes.get(i, {}) for i in frame["id"]]
        table = frame[["id", "parent_id", "source_record_ref"]].copy()
        if kind == "lot":
            for key in keys:
                table[key] = pd.Series(
                    [value.get(key) for value in values], dtype=object
                )
        geometry = gpd.GeoSeries.from_wkb(
            frame["geometry"].map(bytes, na_action="ignore"), crs=CRS
        )
        result[kind] = gpd.GeoDataFrame(table, geometry=geometry, crs=CRS)
    return result
