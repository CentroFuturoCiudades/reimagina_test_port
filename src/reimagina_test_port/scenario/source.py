"""Effective scenario lots and blocks from Reimagina's Postgres (EAV schema).

Ported from reimagina-lyra ``store.py``: a fork stores only what differs from its
baseline, so each element, geometry and attribute comes from the scenario's own row
when it has one (even an empty one) and from the baseline otherwise.
"""

import json
import logging
import os
from typing import Any
from urllib.parse import urlparse

import geopandas as gpd
import pandas as pd
import psycopg
from psycopg import sql

from reimagina_test_port.common import conditions
from reimagina_test_port.common.vocabulary import DESTINATION_TYPES

logger = logging.getLogger(__name__)

DSN_ENV_VAR = "REIMAGINA_POSTGRES_DSN"
SCHEMA = "reimagina_urbano"
CRS = "EPSG:4326"

_TABLES = {
    name: sql.Identifier(SCHEMA, name)
    for name in (
        "projects",
        "scenarios",
        "elements",
        "element_state",
        "attribute_catalog",
        "element_state_attribute_value",
        "poi_catalog",
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

# Accessibility destinations: the poi_catalog rules of the active types. read_pois
# evaluates them (common.conditions) over the present amenity and establishment
# elements, with the effective values of the keys the rules read.
_POI_RULES_QUERY = sql.SQL(
    "SELECT poi_type, rule_criteria, params -> 'subcategories' FROM {poi_catalog} "
    "WHERE status = 'active' ORDER BY poi_type"
).format(**_TABLES)

_POI_COVERAGE_QUERY = sql.SQL(
    "SELECT poi_type, (params ->> 'coverage_m')::float FROM {poi_catalog} "
    "WHERE status = 'active' AND params ->> 'coverage_m' IS NOT NULL"
).format(**_TABLES)

_POI_DIVERSITY_QUERY = sql.SQL(
    "SELECT poi_type, (params ->> 'diversity_m')::float, "
    "jsonb_array_length(params -> 'subcategories') FROM {poi_catalog} "
    "WHERE status = 'active' AND params ->> 'diversity_m' IS NOT NULL "
    "AND jsonb_typeof(params -> 'subcategories') = 'array'"
).format(**_TABLES)

# Starts from those attribute values and only then looks up their state, instead
# of resolving every element of the scenario first. Capacity stays NULL, as
# before: the model counts each destination once.
_POI_VALUES = """
    WITH value AS (
        SELECT DISTINCT ON (v.element_id, v.attribute_id)
               v.element_id, ac.key, ac.value_type::text AS value_type,
               v.value_num, v.value_text, v.value_json
        FROM {element_state_attribute_value} v
        JOIN {attribute_catalog} ac ON ac.id = v.attribute_id
        WHERE v.scenario_id IN (%(scenario)s::uuid, %(baseline)s::uuid)
          AND ac.element_type IN ('amenity', 'establishment')
          AND ac.key = ANY(%(keys)s)
        ORDER BY v.element_id, v.attribute_id, (v.scenario_id = %(scenario)s::uuid) DESC
    )
"""
_POI_ELEMENTS_QUERY = sql.SQL(
    _POI_VALUES
    + """,
    state AS (
        SELECT DISTINCT ON (es.element_id) es.element_id, es.geometry, es.deleted_at
        FROM {element_state} es
        WHERE es.scenario_id IN (%(scenario)s::uuid, %(baseline)s::uuid)
          AND es.element_id IN (SELECT element_id FROM value)
        ORDER BY es.element_id, (es.scenario_id = %(scenario)s::uuid) DESC
    )
    SELECT st.element_id::text AS id, e.type::text AS type,
           ST_AsBinary(st.geometry) AS geometry
    FROM state st
    JOIN {elements} e ON e.id = st.element_id
    WHERE st.deleted_at IS NULL AND e.deleted_at IS NULL
    ORDER BY 1
    """
).format(**_TABLES)
_POI_ATTRIBUTES_QUERY = sql.SQL(
    _POI_VALUES
    + """
    SELECT element_id::text, key, value_type, value_num, value_text, value_json
    FROM value
    """
).format(**_TABLES)

_ROAD_NETWORK_QUERY = sql.SQL(
    "SELECT NULLIF(p.road_network_uri, '') FROM {scenarios} s "
    "JOIN {projects} p ON p.id = s.project_id "
    "WHERE s.id = %s::uuid AND s.deleted_at IS NULL"
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


def _baseline_id(cur: psycopg.Cursor, scenario_id: str) -> str | None:
    cur.execute(_BASELINE_QUERY, (scenario_id,))
    row = cur.fetchone()
    if row is None:
        msg = f"Scenario {scenario_id!r} does not exist."
        raise ScenarioNotFoundError(msg)
    return row[0]


def _geometry(values: pd.Series) -> gpd.GeoSeries:
    return gpd.GeoSeries.from_wkb(values.map(bytes, na_action="ignore"), crs=CRS)


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _destination_rules(
    cur: psycopg.Cursor,
) -> dict[str, tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Rule and subcategories of the active poi_catalog types the model has outputs
    for."""
    cur.execute(_POI_RULES_QUERY)
    rules = {}
    for poi_type, rule, subcategories in cur.fetchall():
        if poi_type not in DESTINATION_TYPES:
            logger.warning(
                "poi_catalog type without accessibility outputs: %s", poi_type
            )
            continue
        rules[poi_type] = (_json(rule), _json(subcategories) or [])
    return rules


def _subcategory(
    poi_type: str, subcategories: list[dict[str, Any]], element: dict[str, Any]
) -> str | None:
    """The first subcategory whose ``when`` the element meets. A malformed one only
    leaves the element without subcategory: its type, and accessibility, stay."""
    try:
        return next(
            (
                each["name"]
                for each in subcategories
                if conditions.evaluate(each["when"], element)
            ),
            None,
        )
    except (conditions.ConditionError, KeyError) as exc:
        logger.error("poi_catalog subcategories of %s are malformed: %s", poi_type, exc)
        return None


def _matches(
    rules: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
    element: dict[str, Any],
) -> list[tuple[str, str | None]]:
    """Types whose rule the element meets, each with its subcategory."""
    matched = []
    for poi_type, (rule, subcategories) in rules.items():
        try:
            if conditions.evaluate(rule, element):
                matched.append(
                    (poi_type, _subcategory(poi_type, subcategories, element))
                )
        except conditions.ConditionError as exc:
            logger.error("poi_catalog rule of %s is malformed: %s", poi_type, exc)
    return matched


def read_pois(scenario_id: str) -> gpd.GeoDataFrame:
    """Present amenities and establishments, one row per poi_catalog type whose
    rule they meet: id, poi_type, capacity, subcategory, geometry."""
    with psycopg.connect(resolve_dsn()) as conn, conn.cursor() as cur:
        params = {"scenario": scenario_id, "baseline": _baseline_id(cur, scenario_id)}
        rules = _destination_rules(cur)
        own: set[str] = set()
        for poi_type, (rule, subcategories) in rules.items():
            for condition in [rule, *(each.get("when", {}) for each in subcategories)]:
                rule_keys, parent_keys = conditions.keys(condition)
                own |= rule_keys - {"type"}
                if parent_keys:
                    # Only the element's own values are read; parent keys are missing.
                    logger.warning(
                        "%s reads its parent's %s", poi_type, sorted(parent_keys)
                    )
        params["keys"] = sorted(own)
        cur.execute(_POI_ELEMENTS_QUERY, params)
        elements = cur.fetchall()
        cur.execute(_POI_ATTRIBUTES_QUERY, params)
        attribute_rows = cur.fetchall()

    attributes: dict[str, dict[str, Any]] = {}
    for element_id, key, value_type, num, text, js in attribute_rows:
        attributes.setdefault(element_id, {})[key] = decode_value(
            value_type, num, text, js
        )
    rows = [
        (element_id, poi_type, subcategory, geometry)
        for element_id, element_type, geometry in elements
        for poi_type, subcategory in _matches(
            rules, {**attributes.get(element_id, {}), "type": element_type}
        )
    ]
    frame = pd.DataFrame(rows, columns=["id", "poi_type", "subcategory", "geometry"])
    frame.insert(2, "capacity", pd.Series(dtype="float64", index=frame.index))
    frame["subcategory"] = frame["subcategory"].astype("string")
    return gpd.GeoDataFrame(
        frame.drop(columns="geometry"), geometry=_geometry(frame["geometry"]), crs=CRS
    )


def read_coverage_m() -> dict[str, float]:
    """Coverage distance, in metres, of the active poi_catalog types that have one."""
    with psycopg.connect(resolve_dsn()) as conn, conn.cursor() as cur:
        cur.execute(_POI_COVERAGE_QUERY)
        return dict(cur.fetchall())


def read_diversity() -> dict[str, tuple[float, int]]:
    """Diversity radius, in metres, and number of subcategories of the active
    poi_catalog types that have both."""
    with psycopg.connect(resolve_dsn()) as conn, conn.cursor() as cur:
        cur.execute(_POI_DIVERSITY_QUERY)
        return {poi_type: (radius, count) for poi_type, radius, count in cur}


def read_road_network_uri(scenario_id: str) -> str | None:
    """The scenario's ``projects.road_network_uri``, or None when it is empty."""
    with psycopg.connect(resolve_dsn()) as conn, conn.cursor() as cur:
        _baseline_id(cur, scenario_id)
        cur.execute(_ROAD_NETWORK_QUERY, (scenario_id,))
        row = cur.fetchone()
    return None if row is None else row[0]


def read_scenario(scenario_id: str, keys: list[str]) -> dict[str, gpd.GeoDataFrame]:
    """Return effective ``lot`` and ``block`` tables with attribute ``keys`` as columns."""
    with psycopg.connect(resolve_dsn()) as conn, conn.cursor() as cur:
        params = {
            "scenario": scenario_id,
            "baseline": _baseline_id(cur, scenario_id),
            "keys": keys,
        }
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
        result[kind] = gpd.GeoDataFrame(
            table, geometry=_geometry(frame["geometry"]), crs=CRS
        )
    return result
