"""Snapshot acquisition and the public built-form dependency graph."""

import os
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from lyra.sdk import (
    Dependency,
    MetricParameters,
    PluginDefinition,
    TableColumn,
    TableOutput,
    Unit,
    dataset,
    metric,
)
from lyra.sdk.errors import MetricInputError
from pydantic import Field

from reimagina_test_port.calculation import block_indicators, lot_indicators

IDENTITY = ["id", "parent_id", "source_record_ref"]
ATTRIBUTES = [
    "cos",
    "num_levels",
    "area",
    "land_use",
    "unit_size_m2",
    "people_per_household",
    "occupancy_rate",
]
UNITS = {
    "cos": Unit.RATIO,
    "num_levels": Unit.COUNT,
    "cus": Unit.RATIO,
    "area": Unit.SQUARE_METRE,
    "unit_size_m2": Unit.SQUARE_METRE,
    "num_housing_units": Unit.COUNT,
    "population": Unit.COUNT,
    "density": Unit.COUNT_PER_SQUARE_KILOMETRE,
}


def declaration(*, source: bool = False, blocks: bool = False) -> TableOutput:
    """Describe identity, numerical meaning, and spatial output."""
    columns = [
        TableColumn(
            name=k,
            type="string",
            unit=None,
            nullable=k != "id",
            description=k.replace("_", " "),
        )
        for k in IDENTITY
    ]
    if source:
        for key in [] if blocks else ATTRIBUTES:
            columns.append(
                TableColumn(
                    name=key,
                    type="string" if key == "land_use" else "number",
                    unit=None
                    if key == "land_use"
                    else (
                        {
                            "people_per_household": Unit.COUNT,
                            "occupancy_rate": Unit.RATIO,
                        }
                        | UNITS
                    )[key],
                    nullable=True,
                    description=key.replace("_", " "),
                )
            )
    else:
        for key, unit in UNITS.items():
            description = (
                "Housing density in homes per square kilometre."
                if key == "density"
                else key.replace("_", " ")
            )
            columns.append(
                TableColumn(
                    name=key,
                    type="number",
                    unit=unit,
                    nullable=True,
                    description=description,
                )
            )
    return TableOutput(columns=columns, geometry="geometry")


class SnapshotParameters(MetricParameters):
    """Select an explicit prepared snapshot beneath the configured root."""

    snapshot: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
        description="Prepared snapshot identifier; synthetic selects the bundled example.",
    )


def snapshot_error(message: str) -> MetricInputError:
    """Attach source-node and parameter context to acquisition failures."""
    return MetricInputError(
        "reimagina_prepared_snapshot", "parameters.snapshot", message
    )


def read_snapshot(name: str) -> dict[str, gpd.GeoDataFrame]:
    """Read prepared tables without changing source identity or area values."""
    root = Path(
        os.environ.get("REIMAGINA_SNAPSHOT_ROOT", str(Path(__file__).parent / "data"))
    ).resolve()
    folder = (root / name).resolve()
    if folder.parent != root:
        raise snapshot_error("Snapshot must be directly beneath the configured root")
    result = {}
    for kind in ("lot", "block"):
        path = folder / f"{kind}.parquet"
        try:
            frame = gpd.read_parquet(path)
        except (OSError, ValueError) as exc:
            raise snapshot_error(
                f"Cannot read {kind} table for snapshot {name!r}: {exc}"
            ) from exc
        if frame.crs is None or frame.geometry.name != "geometry":
            raise snapshot_error(f"{kind}: geometry column with declared CRS required")
        if "id" not in frame or "parent_id" not in frame:
            raise snapshot_error(f"{kind}: id and parent_id columns required")
        if (
            frame.id.isna().any()
            or frame.id.duplicated().any()
            or not frame.id.map(
                lambda value: isinstance(value, str) and bool(value.strip())
            ).all()
        ):
            raise snapshot_error(f"{kind}: IDs must be unique nonempty strings")
        for key in IDENTITY + ([] if kind == "block" else ATTRIBUTES):
            if key not in frame:
                frame[key] = None
        result[f"reimagina_prepared_{kind}s"] = frame[
            IDENTITY + ([] if kind == "block" else ATTRIBUTES) + ["geometry"]
        ].copy()
    return result


def plain_record(row: dict[str, Any]) -> dict[str, Any]:
    """Translate tabular missing values to the source algorithm's nulls."""
    return {key: None if pd.isna(value) else value for key, value in row.items()}


def result_frame(source: gpd.GeoDataFrame, values: list[dict]) -> gpd.GeoDataFrame:
    """Attach indicators positionally, preserving IDs, order, and geometry."""
    result = source[IDENTITY + ["geometry"]].copy()
    for key in UNITS:
        result[key] = [value[key] for value in values]
    return result


@dataset(
    name="reimagina_prepared_snapshot",
    description="Acquire prepared lot and block GeoParquet tables.",
    outputs={
        "reimagina_prepared_lots": declaration(source=True),
        "reimagina_prepared_blocks": declaration(source=True, blocks=True),
    },
)
def prepared_snapshot(parameters: SnapshotParameters) -> dict[str, gpd.GeoDataFrame]:
    """Acquire both files once per graph invocation."""
    return read_snapshot(parameters.snapshot)


@metric(
    name="reimagina_lot_built_form",
    description="Calculate built-form indicators for every prepared lot.",
    inputs={"lots": Dependency("reimagina_prepared_lots")},
    outputs={"reimagina_lot_indicators": declaration()},
)
def lot_built_form(
    parameters: SnapshotParameters, lots: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Preserve source arithmetic and publish housing density in homes/km²."""
    values = [
        lot_indicators(plain_record(row)) for row in lots[ATTRIBUTES].to_dict("records")
    ]
    for value in values:
        if value["density"] is not None:
            value["density"] *= 100
    return result_frame(lots, values)


@metric(
    name="reimagina_built_form",
    description="Return lot indicators and their block aggregates.",
    inputs={
        "lots": Dependency("reimagina_lot_indicators"),
        "blocks": Dependency("reimagina_prepared_blocks"),
    },
    outputs={
        "reimagina_built_form_lots": declaration(),
        "reimagina_built_form_blocks": declaration(),
    },
)
def built_form(
    parameters: SnapshotParameters, lots: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame
) -> dict[str, gpd.GeoDataFrame]:
    """Aggregate upstream values without recalculating lots or converting density."""
    children: dict[str, list[dict]] = {}
    for row in lots[["parent_id", *UNITS]].to_dict("records"):
        record = plain_record(row)
        children.setdefault(record.pop("parent_id"), []).append(record)
    values = [
        block_indicators(children.get(identifier, [])) for identifier in blocks.id
    ]
    return {
        "reimagina_built_form_lots": lots.copy(),
        "reimagina_built_form_blocks": result_frame(blocks, values),
    }


def create_plugin() -> PluginDefinition:
    """Register declarations without reading snapshots or contacting services."""
    return PluginDefinition(nodes=[prepared_snapshot, lot_built_form, built_form])
