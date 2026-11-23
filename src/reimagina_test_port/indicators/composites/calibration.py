"""A project's calibration of the composite tree's leaves.

``composite_config.yaml`` fixes each leaf's AMPI reference, compress stats and
goalposts from a Mexicali baseline. A project's baseline replaces them with its
own, computed over its elements after its indicators:

- reference: the average of the transformed leaf (ampi leaves);
- compress: the transformed leaf's stats (leaves that compress);
- goalposts: for a count (no transform, goalposts other than 0-1), its max
  becomes the 99th percentile of the (compressed) transformed leaf, rounded up
  to 1, 2 or 5 × 10^k; its min stays.

A leaf without a value on the baseline (no police station, say) has none, and the
composites leave it missing, even where a fork gives it a value: it is never
scored against the YAML's. The calibration is saved as JSON at ``calibration_uri(project)``, overwritten by
each run of the baseline, and its forks read it: B and C score against the same
values. A fork of a project without one fails (scenario/nodes.py). The tree and
the weights stay the YAML's.
"""

import copy
import math
import os
from typing import Any

import geopandas as gpd
import pandas as pd
from lyra.sdk import TableColumn, TableOutput

from reimagina_test_port.common.nodes import column
from reimagina_test_port.indicators.composites.composite import (
    COMPRESSION_KEYS,
    compress_outliers,
    compression_parameters,
    fixed_compression,
    transform_indicator,
)

# scenario_file.kind of a baseline's calibration.
KIND = "composite_calibration"
PREFIX_ENV = "REIMAGINA_CALIBRATION_PREFIX"
DEFAULT_PREFIX = "az://sandbox/projects/EXE/cleaned/composite_calibration"
GOALPOST_QUANTILE = 0.99
UNIT_GOALPOSTS = (0.0, 1.0)


def calibration_uri(project_id: str) -> str:
    """Where a project's calibration is: one file per project, overwritten."""
    prefix = os.environ.get(PREFIX_ENV) or DEFAULT_PREFIX
    return f"{prefix.rstrip('/')}/{project_id}.json"


def is_count(leaf: dict) -> bool:
    """A leaf whose goalposts the calibration sets: no transform, not 0-1."""
    goalposts = leaf.get("goalposts")
    if leaf.get("transform") or not goalposts:
        return False
    return (float(goalposts["min"]), float(goalposts["max"])) != UNIT_GOALPOSTS


def nice_ceiling(value: float) -> float | None:
    """The smallest 1, 2 or 5 × 10^k at or above ``value``; None unless positive."""
    if not value > 0:
        return None
    power = 10.0 ** math.floor(math.log10(value))
    for step in (1, 2, 5, 10):
        if step * power >= value * (1 - 1e-12):
            return float(step * power)
    return None  # pragma: no cover: 10 × power is always above


def leaf_calibration(leaf: dict, values: pd.Series) -> dict[str, Any]:
    """The leaf's reference, compress stats and goalposts over ``values``; empty
    when none has a value."""
    x = transform_indicator(values, leaf.get("transform")).dropna()
    if x.empty:
        return {}
    result: dict[str, Any] = {}
    ampi = leaf.get("method", "ampi") == "ampi"
    if ampi:
        result["reference"] = float(x.mean())
    if leaf.get("compress"):
        result["compress"] = compression_parameters(x)
        x = compress_outliers(x, **result["compress"])
    if ampi and is_count(leaf):
        low = float(leaf["goalposts"]["min"])
        high = nice_ceiling(float(x.quantile(GOALPOST_QUANTILE)))
        if high is not None and high > low:
            result["goalposts"] = {"min": low, "max": high}
    return result


def inside(grain: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame | None):
    """The elements whose representative point lies inside ``boundary``."""
    if boundary is None:
        return grain
    area = boundary.to_crs(grain.crs).geometry.union_all()
    return grain[grain.geometry.representative_point().within(area)]


def calibrate(
    leaves: list[dict],
    tables: dict[str, gpd.GeoDataFrame],
    boundary: gpd.GeoDataFrame | None = None,
) -> dict[str, dict]:
    """Each leaf's calibration over its source's elements (``tables`` by source),
    inside ``boundary`` when there is one: {leaf name: values}."""
    result = {}
    for leaf in leaves:
        table = inside(tables[leaf["source"]], boundary)
        if values := leaf_calibration(leaf, table[leaf["variable"]]):
            result[leaf["name"]] = values
    return result


def calibrated(leaf: dict, calibration: dict[str, dict]) -> dict:
    """``leaf`` with its project's values; the YAML's where there are none. Only
    what the calibration may set: an ampi reference, the stats of a leaf that
    compresses, a count's goalposts."""
    values = calibration.get(leaf.get("name"))
    if not values:
        return leaf
    leaf = copy.deepcopy(leaf)
    if "reference" in values and leaf.get("method", "ampi") == "ampi":
        leaf["reference"] = values["reference"]
    if "compress" in values and leaf.get("compress"):
        leaf["compress"] = fixed_compression(values["compress"])
    if "goalposts" in values and is_count(leaf):
        leaf["goalposts"] = dict(values["goalposts"])
    return leaf


# The calibration as a table, one row per leaf, as Lyra passes it between nodes.
COLUMNS = ["reference", *[f"compress_{k}" for k in COMPRESSION_KEYS],
           "goalposts_min", "goalposts_max"]  # fmt: skip
DECLARATION = TableOutput(
    columns=[
        TableColumn(name="leaf", type="string", unit=None, description="Leaf name."),
        column("reference", "number", None, "AMPI reference; missing: the YAML's."),
        *[
            column(
                f"compress_{key}",
                "number",
                None,
                f"compress's {key}; missing: the YAML's.",
            )
            for key in COMPRESSION_KEYS
        ],
        column("goalposts_min", "number", None, "A count's goalposts: min."),
        column("goalposts_max", "number", None, "A count's goalposts: max."),
        TableColumn(
            name="origin",
            type="string",
            unit=None,
            description=(
                "computed (a baseline, now) or the URI it was read from (a fork)."
            ),
        ),
    ]
)
COMPUTED = "computed"


def to_table(calibration: dict[str, dict], origin: str) -> pd.DataFrame:
    rows = []
    for leaf, values in calibration.items():
        compress = values.get("compress", {})
        goalposts = values.get("goalposts", {})
        rows.append(
            {
                "leaf": leaf,
                "reference": values.get("reference"),
                **{f"compress_{k}": compress.get(k) for k in COMPRESSION_KEYS},
                "goalposts_min": goalposts.get("min"),
                "goalposts_max": goalposts.get("max"),
                "origin": origin,
            }
        )
    table = pd.DataFrame(rows, columns=["leaf", *COLUMNS, "origin"])
    table[COLUMNS] = table[COLUMNS].astype("float64")
    return table


def from_table(table: pd.DataFrame) -> dict[str, dict]:
    def number(row: dict, key: str) -> float | None:
        value = row.get(key)
        return None if value is None or pd.isna(value) else float(value)

    result = {}
    for row in table.to_dict("records"):
        values: dict[str, Any] = {}
        if (reference := number(row, "reference")) is not None:
            values["reference"] = reference
        compress = {k: number(row, f"compress_{k}") for k in COMPRESSION_KEYS}
        if all(v is not None for v in compress.values()):
            values["compress"] = compress
        goalposts = {k: number(row, f"goalposts_{k}") for k in ("min", "max")}
        if all(v is not None for v in goalposts.values()):
            values["goalposts"] = goalposts
        result[row["leaf"]] = values
    return result


def document(
    calibration: dict[str, dict], project_id: str, scenario_id: str, element: str
) -> dict:
    """The JSON saved in the lake."""
    return {
        "project_id": project_id,
        "baseline_scenario_id": scenario_id,
        "element": element,
        "leaves": calibration,
    }


def from_document(document: dict) -> dict[str, dict]:
    leaves = document.get("leaves")
    if not isinstance(leaves, dict):
        msg = "a composite calibration needs 'leaves'"
        raise TypeError(msg)
    return leaves
