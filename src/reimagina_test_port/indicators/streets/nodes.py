"""Lyra metric reimagina_streets: bike lane and street lighting coverage per block,
from the scenario's right_of_way streets, and street connectivity, from its road
network."""

import geopandas as gpd
import pandas as pd
from lyra.sdk import Dependency, RunContext, TableOutput, Unit, metric

from reimagina_test_port.common.nodes import column, timed
from reimagina_test_port.common.schema import IDENTITY, identity_columns
from reimagina_test_port.indicators.accessibility.model import resolve_access_crs
from reimagina_test_port.indicators.streets.calculation import (
    COVERAGE_KEYS,
    coverage_by_block,
    intersection_density,
)
from reimagina_test_port.scenario.nodes import ScenarioParameters

OUTPUT = "reimagina_streets_blocks"


@metric(
    name="reimagina_streets",
    description=(
        "What the streets around each block are like (ULI 'Bicycle lanes "
        "coverage', 'Protected bicycle lanes coverage' and 'Presence of street "
        "lighting'): of the length of the streets (right_of_way) within 500 m "
        "of the block, the share whose bands include a bike lane, of any kind "
        "and protected only, and the share that is lit; and how connected they "
        "are ('Street connectivity'): intersections per km² within 1 km over the "
        "road network, from the node nearest the block."
    ),
    inputs={
        "blocks": Dependency("reimagina_built_form_blocks"),
        "streets": Dependency("reimagina_scenario_rights_of_way"),
        "nodes": Dependency("reimagina_road_network_nodes"),
        "edges": Dependency("reimagina_road_network_edges"),
    },
    outputs={
        OUTPUT: TableOutput(
            columns=[
                *identity_columns(),
                column(
                    "bike_lane_coverage",
                    "number",
                    Unit.RATIO,
                    "Share (0-1) of the street length within 500 m of the block with "
                    "a bike lane band. Null without streets.",
                ),
                column(
                    "protected_bike_lane_coverage",
                    "number",
                    Unit.RATIO,
                    "As bike_lane_coverage, counting only protected bike lanes.",
                ),
                column(
                    "street_lighting_coverage",
                    "number",
                    Unit.RATIO,
                    "Share (0-1) of the street length within 500 m of the block that "
                    "is lit (streetlights weighted by length). Null without streets "
                    "with a value.",
                ),
                column(
                    "intersection_density",
                    "number",
                    Unit.COUNT_PER_SQUARE_KILOMETRE,
                    "Intersections (nodes of three or more streets) within 1 km over "
                    "the road network, per km² of a 1 km circle.",
                ),
            ],
            geometry="geometry",
        )
    },
)
def streets(
    parameters: ScenarioParameters,
    blocks: gpd.GeoDataFrame,
    streets: gpd.GeoDataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    context: RunContext,
) -> gpd.GeoDataFrame:
    # pandana only where the streets metric runs.
    from reimagina_test_port.indicators.accessibility.model import build_network

    with timed("street coverage", context.logger):
        values = coverage_by_block(blocks, streets, resolve_access_crs(blocks))
    result = blocks[IDENTITY + ["geometry"]].copy()
    for key in COVERAGE_KEYS:
        result[key] = values[key].to_numpy()
    with timed("street connectivity", context.logger):
        network = build_network(nodes, edges)
        result["intersection_density"] = intersection_density(blocks, network, edges)
    return result
