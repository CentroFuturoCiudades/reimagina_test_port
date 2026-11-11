"""Lyra nodes of population access to the poi_catalog types over the road network:

- reimagina_nearest_poi_distance: metres from each lot to its nearest POI of each type.
- reimagina_population_access: residents within each type's coverage distance, their
  share of the residents, and the average distance to its nearest POI, per lot and
  per block.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from lyra.sdk import (
    Dependency,
    MetricParameters,
    RunContext,
    TableOutput,
    Unit,
    metric,
)
from pydantic import Field

from reimagina_test_port.common.nodes import column, timed
from reimagina_test_port.common.schema import IDENTITY, identity_columns
from reimagina_test_port.common.vocabulary import (
    DESTINATION_COVERAGE_M,
    DESTINATION_TYPES,
)
from reimagina_test_port.indicators.accessibility.model import resolve_access_crs
from reimagina_test_port.indicators.population_access.calculation import (
    nearest_distances,
    population_access,
    street_network,
)

POI_TYPE_COLUMN = "poi_type"
MAX_DISTANCE_M = 10_000
DISTANCE_OUTPUT = "reimagina_nearest_poi_distance_lots"
PoiType = Literal[DESTINATION_TYPES]  # ty: ignore[invalid-type-form]
DISTANCE_KEYS = [f"distance_m_{poi_type}" for poi_type in DESTINATION_TYPES]
ACCESS_KEYS = [f"population_with_access_{poi_type}" for poi_type in DESTINATION_TYPES]
KEYS = [*DISTANCE_KEYS, *ACCESS_KEYS]
# Read by the composites, not written: population_with_access_<type> / population.
SHARE_KEYS = [f"share_with_access_{poi_type}" for poi_type in DESTINATION_TYPES]

DISTANCE_COLUMNS = [
    column(
        key,
        "number",
        Unit.METRE,
        "Walk over the road network to the nearest POI of the type; for a block, its "
        "lots' average weighted by population. Null beyond 10 km or if not measured.",
    )
    for key in DISTANCE_KEYS
]
ACCESS_COLUMNS = [
    column(
        key,
        "number",
        Unit.COUNT,
        "Residents of lots within the type's coverage distance of one of its POIs.",
    )
    for key in ACCESS_KEYS
]
SHARE_COLUMNS = [
    column(
        key,
        "number",
        None,
        "Share (0-1) of the residents within the type's coverage distance of one of "
        "its POIs ('Percentage of population with access'). Null without residents.",
    )
    for key in SHARE_KEYS
]


def result_frame(source: gpd.GeoDataFrame, values: pd.DataFrame) -> gpd.GeoDataFrame:
    """Identity and geometry of ``source`` with every key, in its order, and each
    type's share of ``source``'s population; the keys of types not measured stay
    null."""
    result = source[IDENTITY + ["geometry"]].copy()
    values = values.reindex(columns=KEYS)
    for key in KEYS:
        result[key] = values[key].to_numpy()
    population = pd.to_numeric(source["population"]).astype("float64").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        for access, share in zip(ACCESS_KEYS, SHARE_KEYS, strict=True):
            counted = result[access].to_numpy(dtype="float64")
            result[share] = np.where(population > 0, counted / population, np.nan)
    return result


def population_access_metrics(
    parameters_type: type[MetricParameters],
    lots_output: str,
    pois_output: str,
    nodes_output: str,
    edges_output: str,
    coverage_m: Callable[[], Mapping[str, float]],
) -> list:
    """Both nodes over one source's lots, POIs, road network and coverage distances."""

    class PopulationAccessParameters(parameters_type):  # ty: ignore[unsupported-base]
        """The source's parameter, plus the poi_catalog types to measure."""

        poi_types: list[PoiType] | None = Field(
            default=None, description="Types to measure; all when omitted."
        )

    def requested(parameters) -> Sequence[str]:
        return (
            DESTINATION_TYPES if parameters.poi_types is None else parameters.poi_types
        )

    def distance_handler(
        parameters,
        lots: gpd.GeoDataFrame,
        pois: gpd.GeoDataFrame,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        context: RunContext,
    ) -> pd.DataFrame:
        crs = resolve_access_crs(lots)
        with timed("road network", context.logger):
            network = street_network(nodes, edges)
        selected = {
            poi_type: pois.geometry[pois[POI_TYPE_COLUMN] == poi_type]
            for poi_type in requested(parameters)
        }
        with timed("nearest POI distances", context.logger):
            distances = nearest_distances(lots, selected, network, crs, MAX_DISTANCE_M)
        values = pd.DataFrame(
            {f"distance_m_{key}": value for key, value in distances.items()},
            index=lots.index,
        ).reindex(columns=DISTANCE_KEYS)
        return pd.concat([pd.DataFrame(lots[IDENTITY]), values], axis=1)

    def access_handler(
        parameters,
        lots: gpd.GeoDataFrame,
        blocks: gpd.GeoDataFrame,
        distances: pd.DataFrame,
        context: RunContext,
    ) -> dict[str, gpd.GeoDataFrame]:
        catalog = coverage_m()
        missing = [key for key in requested(parameters) if key not in catalog]
        if missing:
            context.logger.warning(
                "poi_catalog types without coverage_m, using the default: %s", missing
            )
        coverage = {**DESTINATION_COVERAGE_M, **catalog}
        by_lot = distances.set_index("id").reindex(lots["id"])
        measured = {
            poi_type: by_lot[f"distance_m_{poi_type}"].to_numpy()
            for poi_type in requested(parameters)
        }
        with timed("population access", context.logger):
            values = population_access(lots, blocks, measured, coverage)
        return {
            "reimagina_population_access_lots": result_frame(lots, values["lots"]),
            "reimagina_population_access_blocks": result_frame(
                blocks, values["blocks"]
            ),
        }

    distance_handler.__annotations__["parameters"] = PopulationAccessParameters
    access_handler.__annotations__["parameters"] = PopulationAccessParameters
    population_access_output = TableOutput(
        columns=[
            *identity_columns(),
            *DISTANCE_COLUMNS,
            *ACCESS_COLUMNS,
            *SHARE_COLUMNS,
        ],
        geometry="geometry",
    )
    return [
        metric(
            name="reimagina_nearest_poi_distance",
            description=(
                "Walking distance over the road network from each lot to the "
                "nearest POI of each poi_catalog type, as GHSCI: points are their "
                "own entry and polygons are entered anywhere along their outline (a "
                "point every 20 m within 30 m of a street); entries join the street "
                "at their projection on the nearest segment, every street two-way."
            ),
            inputs={
                "lots": Dependency(lots_output),
                "pois": Dependency(pois_output),
                "nodes": Dependency(nodes_output),
                "edges": Dependency(edges_output),
            },
            outputs={
                DISTANCE_OUTPUT: TableOutput(
                    columns=[*identity_columns(), *DISTANCE_COLUMNS]
                )
            },
        )(distance_handler),
        metric(
            name="reimagina_population_access",
            description=(
                "Residents with a POI of each poi_catalog type within its coverage "
                "distance over the road network, and the average distance to the "
                "nearest one (ULI 'Percentage of population with access to <poi> "
                "within X meters' and 'Average distance to closest <poi>'). Lots "
                "within the distance count their built-form population; blocks add "
                "up their lots and average their distances weighted by population."
            ),
            inputs={
                "lots": Dependency("reimagina_built_form_lots"),
                "blocks": Dependency("reimagina_built_form_blocks"),
                "distances": Dependency(DISTANCE_OUTPUT),
            },
            outputs={
                "reimagina_population_access_lots": population_access_output,
                "reimagina_population_access_blocks": population_access_output,
            },
        )(access_handler),
    ]
