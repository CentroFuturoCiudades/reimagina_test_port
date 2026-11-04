"""Lyra node reimagina_accessibility and the declarations of accessibility's
outputs and inputs (POIs, road network nodes and edges)."""

import logging
from typing import Any

import geopandas as gpd
import pandas as pd
from lyra.sdk import (
    Dependency,
    MetricParameters,
    RunContext,
    TableColumn,
    TableOutput,
    Unit,
    metric,
)

from reimagina_test_port.common.nodes import column, timed
from reimagina_test_port.common.schema import IDENTITY, identity_columns
from reimagina_test_port.common.vocabulary import AMENITY_TYPES

logger = logging.getLogger(__name__)

SCORE_KEYS = [f"accessibility_{amenity}" for amenity in AMENITY_TYPES]
DISTANCE_KEYS = [f"distance_to_{amenity}" for amenity in AMENITY_TYPES]
ACCESSIBILITY_KEYS = ["minutes", "farthest_amenity", *SCORE_KEYS, *DISTANCE_KEYS]


def accessibility_declaration() -> TableOutput:
    """Identity, accessibility keys (as indicator_catalog) and geometry."""
    identity = identity_columns()
    # Minutes as indicator_catalog ("min"); Lyra has no minute unit.
    columns = [
        *identity,
        column(
            "minutes",
            "number",
            None,
            "Walking minutes to the nearest amenity of the slowest reachable type.",
        ),
        column("farthest_amenity", "string", None, "Slowest reachable amenity type."),
        *(
            column(
                key,
                "number",
                Unit.SCORE,
                f"Gravity score for {amenity}, as a 0-1 percentile rank among the "
                "run's blocks.",
            )
            for key, amenity in zip(SCORE_KEYS, AMENITY_TYPES, strict=True)
        ),
        *(
            column(key, "number", None, f"Walking minutes to the nearest {amenity}.")
            for key, amenity in zip(DISTANCE_KEYS, AMENITY_TYPES, strict=True)
        ),
    ]
    return TableOutput(columns=columns, geometry="geometry")


POIS_DECLARATION = TableOutput(
    columns=[
        TableColumn(name="id", type="string", unit=None, description="Element ID."),
        column("amenity_type", "string", None, "Amenity type of the element."),
        column("capacity", "number", Unit.COUNT, "Capacity; missing counts as one."),
    ],
    geometry="geometry",
)
NODES_DECLARATION = TableOutput(
    columns=[
        TableColumn(name="id", type="integer", unit=None, description="Node ID."),
        TableColumn(
            name="x", type="number", unit=None, description="X in the network's CRS."
        ),
        TableColumn(
            name="y", type="number", unit=None, description="Y in the network's CRS."
        ),
    ]
)
EDGES_DECLARATION = TableOutput(
    columns=[
        TableColumn(name="from", type="integer", unit=None, description="Node ID."),
        TableColumn(name="to", type="integer", unit=None, description="Node ID."),
        TableColumn(
            name="distance",
            type="number",
            unit=Unit.METRE,
            description="Edge length (the network's first impedance).",
        ),
        TableColumn(
            name="two_way",
            type="boolean",
            unit=None,
            description="Whether edges are traversable both ways.",
        ),
    ]
)


def accessibility_frame(
    source: gpd.GeoDataFrame, values: dict[str, dict[str, Any]]
) -> gpd.GeoDataFrame:
    """Accessibility keys by id; elements reaching nothing stay null."""
    result = source[IDENTITY + ["geometry"]].copy()
    rows = [values.get(str(identifier), {}) for identifier in result["id"]]
    for key in ACCESSIBILITY_KEYS:
        result[key] = pd.Series(
            [row.get(key) for row in rows],
            index=result.index,
            dtype=object if key == "farthest_amenity" else "float64",
        )
    return result


def accessibility_tables(
    lots: gpd.GeoDataFrame,
    blocks: gpd.GeoDataFrame,
    pois: gpd.GeoDataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    log: Any = logger,
) -> dict[str, gpd.GeoDataFrame]:
    """Per block, inherited by each lot of that block."""
    # pandana only where accessibility runs.
    from reimagina_test_port.indicators.accessibility import model as accessibility

    with timed("accessibility", log):
        graph = accessibility.build_network(nodes, edges)
        by_block = accessibility.accessibility_by_block(blocks, pois, graph)
    return {
        "reimagina_accessibility_lots": accessibility_frame(
            lots, accessibility.inherit_to_lots(lots, by_block)
        ),
        "reimagina_accessibility_blocks": accessibility_frame(blocks, by_block),
    }


def accessibility_metric(
    parameters_type: type[MetricParameters],
    pois_output: str,
    nodes_output: str,
    edges_output: str,
):
    """Accessibility over built-form results and one source's POIs and network."""

    def handler(
        parameters,
        lots: gpd.GeoDataFrame,
        blocks: gpd.GeoDataFrame,
        pois: gpd.GeoDataFrame,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        context: RunContext,
    ) -> dict[str, gpd.GeoDataFrame]:
        return accessibility_tables(lots, blocks, pois, nodes, edges, context.logger)

    handler.__annotations__["parameters"] = parameters_type
    return metric(
        name="reimagina_accessibility",
        description=(
            "Walking accessibility per block, inherited by its lots (exe_reimagina "
            "model job). Blocks are origins weighted by the scenario's modelled "
            "population; destinations are amenities and establishments, by the "
            "amenity_type their category and SCIAN code give. Up to five "
            "nearest destinations per amenity within 10 km on the road network at "
            "4.8 km/h. Null means none is reachable."
        ),
        inputs={
            "lots": Dependency("reimagina_built_form_lots"),
            "blocks": Dependency("reimagina_built_form_blocks"),
            "pois": Dependency(pois_output),
            "nodes": Dependency(nodes_output),
            "edges": Dependency(edges_output),
        },
        outputs={
            "reimagina_accessibility_lots": accessibility_declaration(),
            "reimagina_accessibility_blocks": accessibility_declaration(),
        },
    )(handler)
