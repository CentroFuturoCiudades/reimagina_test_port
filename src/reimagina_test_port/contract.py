"""The catalog keys this plugin writes, checked against Reimagina's database.

The API owns ``attribute_catalog`` and ``indicator_catalog`` (alembic). Before
writing, the plugin checks that every key it writes is active there, with a
compatible value type and, for enums, the values it uses; otherwise it fails
and names what to add, instead of silently skipping keys.
"""

from collections.abc import Iterable, Mapping

import psycopg
from psycopg import sql

from reimagina_test_port.scenario import SCHEMA

# Plugin kind -> catalog value types it can be stored as.
KINDS = {"number": ("number", "integer"), "text": ("text", "enum")}


class ContractError(RuntimeError):
    """The database catalog lacks or contradicts keys the plugin writes."""


def catalog(
    cur: psycopg.Cursor, table: str
) -> dict[tuple[str, str], tuple[int, str, list[str] | None]]:
    """``{(element_type, key): (id, value_type, enum_values)}`` of active rows."""
    enum = "enum_values" if table == "attribute_catalog" else "NULL::text[]"
    cur.execute(
        sql.SQL(
            "SELECT element_type::text, key, id, value_type::text, {enum} "
            "FROM {table} WHERE status = 'active'"
        ).format(enum=sql.SQL(enum), table=sql.Identifier(SCHEMA, table))
    )
    return {(t, k): (i, v, e) for t, k, i, v, e in cur.fetchall()}


def check(
    cur: psycopg.Cursor,
    table: str,
    keys: Mapping[tuple[str, str], str],
    enum_values: Mapping[tuple[str, str], Iterable[str]] | None = None,
) -> dict[tuple[str, str], tuple[int, str]]:
    """Check ``keys`` (``{(element_type, key): kind}``) and return ``(id, value_type)``.

    ``enum_values`` are the values the plugin writes for enum keys.

    Raises:
        ContractError: listing missing keys, type conflicts and unknown enum values.
    """
    rows = catalog(cur, table)
    problems = []
    for (element_type, key), kind in sorted(keys.items()):
        row = rows.get((element_type, key))
        if row is None:
            problems.append(f"missing {element_type}.{key} ({kind})")
        elif row[1] not in KINDS[kind]:
            problems.append(f"{element_type}.{key} is {row[1]}, expected {kind}")
    for (element_type, key), values in (enum_values or {}).items():
        row = rows.get((element_type, key))
        # An enum without enum_values does not restrict its values.
        if row is None or row[1] != "enum" or row[2] is None:
            continue
        unknown = sorted(set(values) - set(row[2] or []))
        if unknown:
            problems.append(f"{element_type}.{key} enum lacks {unknown}")
    if problems:
        msg = f"{SCHEMA}.{table} does not match the plugin: " + "; ".join(problems)
        raise ContractError(msg)
    return {key: (rows[key][0], rows[key][1]) for key in keys}
