"""Lyra metric reimagina_land_use: net residential density, commercial land share and
land use mix per block, from its lots' attributes."""

import geopandas as gpd
from lyra.sdk import Dependency, RunContext, TableOutput, Unit, metric

from reimagina_test_port.common.nodes import column, timed
from reimagina_test_port.common.schema import IDENTITY, identity_columns
from reimagina_test_port.indicators.land_use.calculation import (
    KEYS,
    land_use_by_block,
)
from reimagina_test_port.scenario.nodes import ScenarioParameters

OUTPUT = "reimagina_land_use_blocks"


@metric(
    name="reimagina_land_use",
    description=(
        "Walkability components per block from what its lots are (ULI 'Net "
        "residential density', 'Commercial land use proportion' and 'Land use "
        "mix'): homes per hectare of the lots that hold homes (homes from "
        "housing.floor_share and housing.unit_size_m2, as urban_fabric.md), the "
        "share of lot area in commerce or office, and the normalised entropy of "
        "land uses by area."
    ),
    inputs={
        "lots": Dependency("reimagina_scenario_lots"),
        "blocks": Dependency("reimagina_built_form_blocks"),
    },
    outputs={
        OUTPUT: TableOutput(
            columns=[
                *identity_columns(),
                # viv/ha, as density; Lyra has no count/ha unit.
                column(
                    "net_residential_density",
                    "number",
                    None,
                    "Homes per hectare of the lots that hold homes (viv/ha); a lot's "
                    "homes are its built space x housing.floor_share / "
                    "housing.unit_size_m2. Null without residential land.",
                ),
                column(
                    "commercial_land_share",
                    "number",
                    Unit.RATIO,
                    "Share (0-1) of the block's lot area in commerce or office lots.",
                ),
                column(
                    "land_use_mix",
                    "number",
                    Unit.RATIO,
                    "Entropy of the lots' land uses by area over the log of the uses "
                    "present (0-1). Null without lots with a land use.",
                ),
            ],
            geometry="geometry",
        )
    },
)
def land_use(
    parameters: ScenarioParameters,
    lots: gpd.GeoDataFrame,
    blocks: gpd.GeoDataFrame,
    context: RunContext,
) -> gpd.GeoDataFrame:
    with timed("land use", context.logger):
        values = land_use_by_block(lots, blocks["id"])
    result = blocks[IDENTITY + ["geometry"]].copy()
    for key in KEYS:
        result[key] = values[key].to_numpy()
    return result
