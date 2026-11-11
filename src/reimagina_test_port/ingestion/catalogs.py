"""The catalog rows a baseline needs, added to Reimagina's database when missing.

``attribute_catalog`` gets every attribute the baseline writes, and
``indicator_catalog`` every indicator the scenario writer computes on lots and
blocks, and each composite of composite_config.yaml on its elements (the scenario
writer also registers those, ``composites.catalog``). A row already there is left
as it is (``ON CONFLICT DO NOTHING``): the
API's migrations own existing rows, so an existing enum without a value the
baseline uses still fails the contract check. Types, units and names follow the
API's seed (alembic ``c8e1a4d92f37``) and ``urban_fabric.md``.

``element_type`` is the API's enum: a type the baseline writes and the enum lacks
fails, naming it, instead of being added here.
"""

from collections.abc import Mapping

import psycopg
from psycopg import sql

from reimagina_test_port.common.rules import LAND_USES
from reimagina_test_port.common.schema import UNITS
from reimagina_test_port.indicators.accessibility.nodes import ACCESSIBILITY_KEYS
from reimagina_test_port.indicators.composites import catalog as composite_catalog
from reimagina_test_port.indicators.population_access.nodes import (
    ACCESS_KEYS,
    DISTANCE_KEYS,
    DIVERSITY_KEY,
)
from reimagina_test_port.ingestion.crosswalk import (
    AMENITY_CATEGORIES,
    ESTABLISHMENT_CATEGORIES,
)
from reimagina_test_port.scenario.contract import ContractError
from reimagina_test_port.scenario.source import SCHEMA

# (element_type, key) -> (value_type, enum_values, unit), where the plugin's kind
# ("number" or "text") is not enough.
ATTRIBUTES = {
    ("lot", "land_use"): ("enum", LAND_USES, None),
    ("lot", "num_levels"): ("integer", None, None),
    ("lot", "area"): ("number", None, "m2"),
    ("lot", "unit_size_m2"): ("number", None, "m2"),
    ("lot", "housing.floor_share"): ("number", None, "%"),
    ("block", "lot.size_m2"): ("number", None, "m2"),
    ("block", "lot.frontage_m"): ("number", None, "m"),
    ("amenity", "category"): ("enum", AMENITY_CATEGORIES, None),
    ("amenity", "num_workers"): ("integer", None, None),
    ("amenity", "area_m2"): ("number", None, "m2"),
    ("establishment", "category"): ("enum", ESTABLISHMENT_CATEGORIES, None),
    ("establishment", "num_workers"): ("integer", None, None),
}

# key -> (name, unit, value_type, job_code), on lots and blocks.
INDICATORS = {
    "cos": ("COS (coeficiente de ocupacion del suelo)", None, "number", "indicators"),
    "num_levels": ("Numero de niveles", None, "number", "indicators"),
    "cus": ("CUS (coeficiente de utilizacion del suelo)", None, "number", "indicators"),
    "area": ("Area", "m2", "number", "indicators"),
    "unit_size_m2": ("Tamano promedio de unidad", "m2", "number", "indicators"),
    "num_housing_units": ("Unidades de vivienda", None, "number", "indicators"),
    "population": ("Poblacion", None, "number", "indicators"),
    "density": ("Densidad", "viv/ha", "number", "indicators"),
    "minutes": ("Minutos a la amenidad mas lejana", "min", "number", "accessibility"),
    "farthest_amenity": ("Amenidad mas lejana", None, "text", "accessibility"),
    **{
        key: (f"Tiempo de viaje a {key[12:]}", "min", "number", "accessibility")
        for key in ACCESSIBILITY_KEYS
        if key.startswith("distance_to_")
    },
    **{
        key: (f"Accesibilidad a {key[14:]}", "score", "number", "accessibility")
        for key in ACCESSIBILITY_KEYS
        if key.startswith("accessibility_")
    },
    **{
        key: (f"Distancia en metros a {key[11:]}", "m", "number", "population_access")
        for key in DISTANCE_KEYS
    },
    **{
        key: (f"Poblacion con acceso a {key[23:]}", None, "number", "population_access")
        for key in ACCESS_KEYS
    },
    DIVERSITY_KEY: ("Diversidad de destinos a 500 m", None, "number", "poi_diversity"),
}
INDICATOR_ELEMENT_TYPES = ("lot", "block")
if missing := sorted(set(UNITS) - set(INDICATORS)):
    msg = f"Built-form keys without an indicator_catalog row: {missing}"
    raise RuntimeError(msg)

_ENUM_QUERY = sql.SQL("SELECT unnest(enum_range(NULL::{}))::text").format(
    sql.Identifier(SCHEMA, "element_type")
)
_INSERT_ATTRIBUTE = sql.SQL(
    """
    INSERT INTO {table} (element_type, key, value_type, enum_values, unit)
    VALUES (%s::{element_type}, %s, %s::{value_type}, %s, %s)
    ON CONFLICT (element_type, key) DO NOTHING
    RETURNING id
    """
).format(
    table=sql.Identifier(SCHEMA, "attribute_catalog"),
    element_type=sql.Identifier(SCHEMA, "element_type"),
    value_type=sql.Identifier(SCHEMA, "attribute_value_type"),
)
_INSERT_INDICATOR = sql.SQL(
    """
    INSERT INTO {table} (element_type, key, name, unit, value_type, scope, job_code)
    VALUES (%s::{element_type}, %s, %s, %s, %s::{value_type}, 'scenario', %s)
    ON CONFLICT (element_type, key) DO NOTHING
    RETURNING id
    """
).format(
    table=sql.Identifier(SCHEMA, "indicator_catalog"),
    element_type=sql.Identifier(SCHEMA, "element_type"),
    value_type=sql.Identifier(SCHEMA, "attribute_value_type"),
)


def attribute_rows(
    attributes: Mapping[str, Mapping[str, str]],
) -> list[tuple[str, str, str, list[str] | None, str | None]]:
    """``attribute_catalog`` rows of ``{element_type: {key: kind}}``."""
    rows = []
    for element_type, columns in attributes.items():
        for key, kind in columns.items():
            value_type, enum_values, unit = ATTRIBUTES.get(
                (element_type, key), (kind, None, None)
            )
            enum = None if enum_values is None else list(enum_values)
            rows.append((element_type, key, value_type, enum, unit))
    return rows


def indicator_rows() -> list[tuple[str, str, str, str | None, str, str]]:
    """``indicator_catalog`` rows of every indicator, on lots and blocks, and of
    every composite (its index, around 100) on the composites' elements."""
    return [
        (element_type, key, name, unit, value_type, job_code)
        for element_type in INDICATOR_ELEMENT_TYPES
        for key, (name, unit, value_type, job_code) in INDICATORS.items()
    ] + composite_catalog.catalog_rows()


def upsert(
    cur: psycopg.Cursor, attributes: Mapping[str, Mapping[str, str]]
) -> dict[str, int]:
    """Add the missing catalog rows; returns ``attributes_added``, ``indicators_added``.

    Raises:
        ContractError: when ``element_type`` lacks a type of ``attributes``.
    """
    cur.execute(_ENUM_QUERY)
    types = {row[0] for row in cur.fetchall()}
    needed = set(attributes) | set(INDICATOR_ELEMENT_TYPES)
    if missing := sorted(needed - types):
        msg = (
            f"{SCHEMA}.element_type lacks {missing}: add them in the API "
            "(alembic) before writing a baseline."
        )
        raise ContractError(msg)
    added = {"attributes_added": 0, "indicators_added": 0}
    for row in attribute_rows(attributes):
        cur.execute(_INSERT_ATTRIBUTE, row)
        added["attributes_added"] += cur.fetchone() is not None
    for row in indicator_rows():
        cur.execute(_INSERT_INDICATOR, row)
        added["indicators_added"] += cur.fetchone() is not None
    return added
