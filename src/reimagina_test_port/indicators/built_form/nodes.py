"""Lyra nodes for built-form indicators: reimagina_lot_built_form (lot formulas)
and reimagina_built_form (lot values summed into their blocks)."""

from typing import Any

import geopandas as gpd
import pandas as pd
from lyra.sdk import (
    Dependency,
    MetricParameters,
    RunContext,
    TableColumn,
    TableOutput,
    metric,
)

from reimagina_test_port.common.nodes import timed
from reimagina_test_port.common.schema import (
    ATTRIBUTES,
    IDENTITY,
    UNITS,
    identity_columns,
)
from reimagina_test_port.indicators.built_form.calculation import (
    block_indicators,
    lot_indicators,
)


def built_form_declaration() -> TableOutput:
    """Identity, built-form keys (as indicator_catalog) and geometry."""
    columns = identity_columns()
    for key, unit in UNITS.items():
        description = (
            "Housing density in homes per hectare (viv/ha)."
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


def plain_record(row: dict[str, Any]) -> dict[str, Any]:
    """Translate tabular missing values to the source algorithm's nulls."""
    return {key: None if pd.isna(value) else value for key, value in row.items()}


def result_frame(source: gpd.GeoDataFrame, values: list[dict]) -> gpd.GeoDataFrame:
    """Attach indicators positionally, preserving IDs, order, and geometry."""
    result = source[IDENTITY + ["geometry"]].copy()
    for key in UNITS:
        result[key] = [value[key] for value in values]
    return result


def lot_built_form(lots: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Lot indicators; density in viv/ha as indicator_catalog."""
    values = [
        lot_indicators(plain_record(row)) for row in lots[ATTRIBUTES].to_dict("records")
    ]
    return result_frame(lots, values)


def built_form(
    lots: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame
) -> dict[str, gpd.GeoDataFrame]:
    """Aggregate upstream lot values into their blocks."""
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


def built_form_metrics(
    parameters_type: type[MetricParameters], lots_output: str, blocks_output: str
) -> list:
    """Built-form metrics over one source's lot and block outputs."""

    def lot_handler(
        parameters, lots: gpd.GeoDataFrame, context: RunContext
    ) -> gpd.GeoDataFrame:
        with timed("lot built-form", context.logger):
            return lot_built_form(lots)

    def block_handler(
        parameters,
        lots: gpd.GeoDataFrame,
        blocks: gpd.GeoDataFrame,
        context: RunContext,
    ) -> dict[str, gpd.GeoDataFrame]:
        with timed("block built-form", context.logger):
            return built_form(lots, blocks)

    lot_handler.__annotations__["parameters"] = parameters_type
    block_handler.__annotations__["parameters"] = parameters_type
    return [
        metric(
            name="reimagina_lot_built_form",
            description="Calculate built-form indicators for every lot.",
            inputs={"lots": Dependency(lots_output)},
            outputs={"reimagina_lot_indicators": built_form_declaration()},
        )(lot_handler),
        metric(
            name="reimagina_built_form",
            description="Return lot indicators and their block aggregates.",
            inputs={
                "lots": Dependency("reimagina_lot_indicators"),
                "blocks": Dependency(blocks_output),
            },
            outputs={
                "reimagina_built_form_lots": built_form_declaration(),
                "reimagina_built_form_blocks": built_form_declaration(),
            },
        )(block_handler),
    ]
