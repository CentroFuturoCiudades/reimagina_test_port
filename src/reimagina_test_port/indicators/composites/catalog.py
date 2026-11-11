"""The composites' rows in Reimagina's catalog, derived from composite_config.yaml.

The API owns ``indicator_catalog`` and ``indicator_input`` (alembic) except for the
composites: their keys and inputs are the tree, so the plugin registers them before
writing (``register``, in the write's own transaction). Adding a composite to the
tree is then enough; no migration follows it.

- ``indicator_catalog``: one ``composite_<name>`` row per composite, on the tree's
  element, named by its ``label``. A composite no longer in the tree is marked
  ``deprecated``, not deleted, so its stored values stay.
- ``indicator_input``: each composite's children, a child composite's key or the
  indicator its leaf reads. A share of residents with access
  (``share_with_access_<type>``) is not stored: its inputs are
  ``population_with_access_<type>`` and ``population``.
"""

import psycopg
from psycopg import sql

from reimagina_test_port.indicators.composites.composite_metric import (
    ELEMENT_TYPES,
    check_tree,
    child_name,
    composite_key,
    load_config,
    reads_composite,
)
from reimagina_test_port.scenario.source import SCHEMA

JOB_CODE = "composites"
UNIT = "score"
SHARE_PREFIX = "share_with_access_"


def composite_name(node: dict) -> str:
    """Its ``label``, or a name made of its key."""
    return node.get("label") or f"Indice compuesto {node['name'].replace('_', ' ')}"


def catalog_rows(
    config: dict | None = None,
) -> list[tuple[str, str, str, str, str, str]]:
    """``(element_type, key, name, unit, value_type, job_code)`` per composite."""
    element, composites, _ = check_tree(load_config() if config is None else config)
    return [
        (
            ELEMENT_TYPES[element],
            composite_key(node["name"]),
            composite_name(node),
            UNIT,
            "number",
            JOB_CODE,
        )
        for node in composites
    ]


def child_keys(child: dict) -> list[str]:
    """The catalog keys a child reads."""
    if reads_composite(child):
        return [composite_key(child_name(child))]
    variable = child["variable"]
    if variable.startswith(SHARE_PREFIX):
        poi_type = variable.removeprefix(SHARE_PREFIX)
        return [f"population_with_access_{poi_type}", "population"]
    return [variable]


def input_pairs(config: dict | None = None) -> list[tuple[str, str, str]]:
    """``(element_type, composite key, input key)``, each pair once."""
    element, composites, _ = check_tree(load_config() if config is None else config)
    pairs = {
        (ELEMENT_TYPES[element], composite_key(node["name"]), key): None
        for node in composites
        for child in node["children"]
        for key in child_keys(child)
    }
    return list(pairs)


_T = {
    "catalog": sql.Identifier(SCHEMA, "indicator_catalog"),
    "inputs": sql.Identifier(SCHEMA, "indicator_input"),
    "element_type": sql.Identifier(SCHEMA, "element_type"),
    "value_type": sql.Identifier(SCHEMA, "attribute_value_type"),
}
_UPSERT = sql.SQL(
    """
    INSERT INTO {catalog} (element_type, key, name, unit, value_type, scope, job_code)
    VALUES (%s::{element_type}, %s, %s, %s, %s::{value_type}, 'scenario', %s)
    ON CONFLICT (element_type, key) DO UPDATE
    SET name = EXCLUDED.name, status = 'active'
    WHERE {catalog}.name IS DISTINCT FROM EXCLUDED.name
       OR {catalog}.status <> 'active'
    RETURNING (xmax = 0)
    """
).format(**_T)
_DEPRECATE = sql.SQL(
    """
    UPDATE {catalog} SET status = 'deprecated'
    WHERE job_code = %s AND key LIKE 'composite\\_%%' AND status = 'active'
      AND NOT (key = ANY(%s))
    """
).format(**_T)
_INPUTS_OF = sql.SQL(
    """
    SELECT ic.element_type::text, ic.key, inp.key
    FROM {inputs} ii
    JOIN {catalog} ic ON ic.id = ii.indicator_id
    JOIN {catalog} inp ON inp.id = ii.input_indicator_id
    WHERE ic.job_code = %s AND ic.key LIKE 'composite\\_%%'
    """
).format(**_T)
_ADD_INPUT = sql.SQL(
    """
    INSERT INTO {inputs} (indicator_id, input_indicator_id)
    SELECT ic.id, inp.id
    FROM {catalog} ic JOIN {catalog} inp ON inp.element_type = ic.element_type
    WHERE ic.element_type = %s::{element_type} AND ic.key = %s AND inp.key = %s
    """
).format(**_T)
_DROP_INPUT = sql.SQL(
    """
    DELETE FROM {inputs} ii
    USING {catalog} ic, {catalog} inp
    WHERE ii.indicator_id = ic.id AND ii.input_indicator_id = inp.id
      AND ic.element_type = %s::{element_type} AND ic.key = %s AND inp.key = %s
    """
).format(**_T)


def register(cur: psycopg.Cursor, config: dict | None = None) -> dict[str, int]:
    """Bring the catalog in line with the tree, on ``cur``'s transaction.

    Returns:
        Counts: ``added`` or ``updated`` composites, ``deprecated`` ones, and
        ``inputs_added`` / ``inputs_removed``.
    """
    rows = catalog_rows(config)
    counts = dict.fromkeys(
        ("added", "updated", "deprecated", "inputs_added", "inputs_removed"), 0
    )
    for row in rows:
        cur.execute(_UPSERT, row)
        result = cur.fetchone()
        if result is not None:
            counts["added" if result[0] else "updated"] += 1
    cur.execute(_DEPRECATE, (JOB_CODE, [row[1] for row in rows]))
    counts["deprecated"] = cur.rowcount
    wanted = set(input_pairs(config))
    cur.execute(_INPUTS_OF, (JOB_CODE,))
    stored = set(cur.fetchall())
    keys = {row[1] for row in rows}
    for pair in sorted(wanted - stored):
        cur.execute(_ADD_INPUT, pair)
        counts["inputs_added"] += cur.rowcount
    for pair in sorted(stored - wanted):
        if pair[1] in keys:  # a deprecated composite keeps its inputs
            cur.execute(_DROP_INPUT, pair)
            counts["inputs_removed"] += cur.rowcount
    return counts
