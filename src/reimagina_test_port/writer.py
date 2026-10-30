"""Write a scenario's indicators to Reimagina's database (EAV schema).

Ported from reimagina-lyra ``store.write_indicators``. A fork stores only values
that differ from its baseline; rows whose stored value is unchanged are not
rewritten, so running it again for the same scenario changes nothing.

For a fork, lot values that cannot have changed are not sent at all: a lot the
fork did not edit and has no stored value for has its baseline value, which for
built-form is what it computes and, for values inherited from its block, is its
block's baseline value. ``full`` sends everything.
"""

import logging
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd
import psycopg
from psycopg import sql

from reimagina_test_port import contract
from reimagina_test_port.scenario import SCHEMA, resolve_dsn

logger = logging.getLogger(__name__)


class _Timer:
    """Logs each stage's duration (s) and a final summary."""

    def __init__(self, label: str, log: Any) -> None:
        self.label, self.log, self.stages = label, log, {}
        self.start = self.last = time.perf_counter()

    def __call__(self, stage: str) -> None:
        now = time.perf_counter()
        self.stages[stage] = round(now - self.last, 2)
        self.last = now

    def done(self, **counts: int) -> None:
        total = round(time.perf_counter() - self.start, 2)
        self.log.info("%s: %ss %s %s", self.label, total, self.stages, counts)


# Decimals persisted, as reimagina-lyra ``base.INDICATOR_DECIMALS``; a fork's values
# compare exactly against its baseline's.
INDICATOR_DECIMALS = 4

_TABLES = {
    name: sql.Identifier(SCHEMA, name)
    for name in (
        "scenarios",
        "element_state",
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


def long_values(
    table: pd.DataFrame, element_type: str, keys: Mapping[str, str]
) -> pd.DataFrame:
    """One row per element and key: ``id``, ``element_type``, ``parent_id``, ``key``,
    ``value_num`` (rounded) and ``value_text``. Missing values are kept, as None,
    so stale stored values are cleared."""
    parent = (
        table["parent_id"]
        if "parent_id" in table
        else pd.Series(None, index=table.index)
    )
    frames = []
    for key, kind in keys.items():
        if key not in table:
            continue
        column = table[key]
        if kind == "text":
            num = pd.Series(np.nan, index=table.index)
            text = (
                column.astype(object)
                .where(column.notna(), None)
                .map(lambda v: None if v is None else str(v))
            )
        else:
            num = pd.to_numeric(column, errors="coerce").astype("float64")
            num = num.where(np.isfinite(num)).round(INDICATOR_DECIMALS)
            text = pd.Series(None, index=table.index, dtype=object)
        frames.append(
            pd.DataFrame(
                {
                    "id": table["id"].astype(str).to_numpy(),
                    "element_type": element_type,
                    "parent_id": parent.to_numpy(),
                    "key": key,
                    "value_num": num.to_numpy(),
                    "value_text": text.to_numpy(),
                }
            )
        )
    columns = ["id", "element_type", "parent_id", "key", "value_num", "value_text"]
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=columns)
    )


def same_value(a_num, a_text, b_num, b_text) -> np.ndarray:
    """Vectorized ``_SAME_VALUE``: equal text and numbers within the API tolerance."""
    a_num, b_num = np.asarray(a_num, float), np.asarray(b_num, float)
    a_text = pd.Series(a_text, dtype=object).where(
        pd.notna(pd.Series(a_text, dtype=object)), None
    )
    b_text = pd.Series(b_text, dtype=object).where(
        pd.notna(pd.Series(b_text, dtype=object)), None
    )
    text_equal = (a_text.to_numpy() == b_text.to_numpy()) | (
        a_text.isna().to_numpy() & b_text.isna().to_numpy()
    )
    both_nan = np.isnan(a_num) & np.isnan(b_num)
    with np.errstate(invalid="ignore"):
        close = np.abs(a_num - b_num) <= np.maximum(
            1e-12, 1e-9 * np.maximum(np.abs(a_num), np.abs(b_num))
        )
    return text_equal & (both_nan | np.nan_to_num(close, nan=False).astype(bool))


def _rows(
    cur: psycopg.Cursor, query: str, params: Sequence[Any], columns: list[str]
) -> pd.DataFrame:
    cur.execute(sql.SQL(query).format(**_TABLES), params)
    return pd.DataFrame(cur.fetchall(), columns=columns)


def _to_send(
    cur: psycopg.Cursor,
    values: pd.DataFrame,
    scenario_id: str,
    baseline_id: str,
    indicator_ids: dict[tuple[str, str], int],
    *,
    inherited: bool,
) -> np.ndarray:
    """Mask of the fork's rows that may differ from what it shows today."""
    lots = (values["element_type"] == "lot").to_numpy()
    edited = _rows(
        cur,
        "SELECT element_id::text FROM {element_state} WHERE scenario_id = %s::uuid",
        (scenario_id,),
        ["id"],
    )
    lot_ids = sorted({i for (t, _), i in indicator_ids.items() if t == "lot"})
    stored = _rows(
        cur,
        "SELECT element_id::text, indicator_id FROM {element_state_indicator_value} "
        "WHERE scenario_id = %s::uuid AND indicator_id = ANY(%s)",
        (scenario_id, lot_ids),
        ["id", "indicator_id"],
    )
    stored_pairs = pd.MultiIndex.from_frame(stored)
    pairs = pd.MultiIndex.from_frame(values[["id", "indicator_id"]])
    send = ~lots | values["id"].isin(edited["id"]).to_numpy() | pairs.isin(stored_pairs)
    if inherited:
        # A lot shows its block's baseline value; send it when the new one differs.
        block_ids = {k: i for (t, k), i in indicator_ids.items() if t == "block"}
        base = _rows(
            cur,
            "SELECT element_id::text, indicator_id, value_num, value_text "
            "FROM {element_state_indicator_value} "
            "WHERE scenario_id = %s::uuid AND indicator_id = ANY(%s)",
            (baseline_id, sorted(block_ids.values())),
            ["parent_id", "block_indicator_id", "base_num", "base_text"],
        )
        candidates = values.loc[
            lots & ~send, ["parent_id", "key", "value_num", "value_text"]
        ]
        candidates = candidates.assign(
            block_indicator_id=candidates["key"].map(block_ids)
        )
        merged = candidates.merge(
            base, on=["parent_id", "block_indicator_id"], how="left"
        )
        changed = ~same_value(
            merged["value_num"],
            merged["value_text"],
            merged["base_num"],
            merged["base_text"],
        )
        send[candidates.index.to_numpy()[changed]] = True
    return send


def write_indicators(
    scenario_id: str,
    values: pd.DataFrame,
    keys: Mapping[str, str],
    *,
    inherited: bool = False,
    full: bool = False,
    job_run_id: str | None = None,
    dry_run: bool = False,
    log: Any = logger,
) -> tuple[int, int]:
    """Write the owned ``keys`` (``{key: kind}``) of ``values`` in one transaction.

    ``values`` comes from ``long_values``. ``inherited`` says lots repeat their
    block's values (accessibility). Every key must be active in
    ``indicator_catalog`` for each element type written (``contract.check``).
    ``job_run_id`` is recorded only when that job_run exists. A dry run computes
    the changes and rolls them back.

    Returns:
        ``(values_written, values_cleared)``.
    """
    timer = _Timer(f"write_indicators {scenario_id} {sorted(keys)[:2]}…", log)
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
            for element_type in sorted(pd.unique(values["element_type"]))
            for key, kind in keys.items()
        }
        indicator_ids = {
            pair: row[0]
            for pair, row in contract.check(cur, "indicator_catalog", required).items()
        }
        # Vectorized lookup: a Python loop over millions of rows took ~30 s.
        lookup = pd.Series(
            list(indicator_ids.values()),
            index=pd.MultiIndex.from_tuples(list(indicator_ids)),
        )
        values = values.assign(
            indicator_id=lookup.reindex(
                pd.MultiIndex.from_arrays([values["element_type"], values["key"]])
            ).to_numpy()
        )
        timer("catalog")
        if baseline_id is not None and not full:
            values = values[
                _to_send(
                    cur,
                    values,
                    scenario_id,
                    baseline_id,
                    indicator_ids,
                    inherited=inherited,
                )
            ]
        timer("select")
        # JIT costs about a second per query and saves nothing here.
        cur.execute("SET LOCAL jit = off")
        cur.execute(
            "CREATE TEMP TABLE indicator_payload (element_id uuid, "
            "indicator_id integer, value_num double precision, value_text text) "
            "ON COMMIT DROP"
        )
        # Plain Python values: the binary COPY rejects numpy scalars.
        uuids = {i: UUID(i) for i in pd.unique(values["id"])}
        nums = [
            None if math.isnan(v) else v
            for v in values["value_num"].astype(float).tolist()
        ]
        texts = [
            v if isinstance(v, str) else None for v in values["value_text"].tolist()
        ]
        with cur.copy(
            "COPY indicator_payload (element_id, indicator_id, value_num, "
            "value_text) FROM STDIN (FORMAT BINARY)"
        ) as copy:
            copy.set_types(["uuid", "int4", "float8", "text"])
            for record in zip(
                [uuids[i] for i in values["id"].tolist()],
                values["indicator_id"].astype(int).tolist(),
                nums,
                texts,
                strict=True,
            ):
                copy.write_row(record)
        payload = len(values)
        timer("copy")
        # Without stats the planner may pick a nested loop over the value table.
        cur.execute("ANALYZE indicator_payload")
        params = {
            "scenario": scenario_id,
            "baseline": baseline_id,
            "indicator_ids": sorted(indicator_ids.values()),
            "job_run": job_run_id,
        }
        timer("analyze")
        cur.execute(_CHANGES_QUERY, params)
        cur.execute("CREATE INDEX ON indicator_change (element_id, indicator_id)")
        cur.execute("ANALYZE indicator_change")
        timer("compare")
        cur.execute(_DELETE_QUERY, params)
        cleared = cur.rowcount
        timer("delete")
        cur.execute(_UPSERT_QUERY, params)
        written = cur.rowcount
        timer("upsert")
        if dry_run:
            conn.rollback()
        else:
            conn.commit()
        timer("commit")
    timer.done(payload=payload, written=written, cleared=cleared)
    return written, cleared
