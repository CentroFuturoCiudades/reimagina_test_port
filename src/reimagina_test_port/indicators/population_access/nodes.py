"""Lyra nodes of population access to one type of POI over the road network:

- reimagina_nearest_poi_distance: metres from each lot to its nearest POI of the type.
- reimagina_population_access: residents within a distance of one, and the average
  distance, per lot and per block.
"""

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
from pydantic import Field, model_validator

from reimagina_test_port.common.nodes import column, timed
from reimagina_test_port.common.schema import IDENTITY, identity_columns
from reimagina_test_port.indicators.accessibility.model import resolve_access_crs
from reimagina_test_port.indicators.population_access.calculation import (
    nearest_distance,
    population_access,
    street_network,
)

# Today POIs carry their type as amenity_type; the node that selects and labels
# them will be the one to say which column.
POI_TYPE_COLUMN = "amenity_type"
MAX_DISTANCE_M = 10_000
DISTANCE_OUTPUT = "reimagina_nearest_poi_distance_lots"
KEYS = [
    "distance_m",
    "population",
    "population_with_access",
    "access_share",
]


DISTANCE_DECLARATION = TableOutput(
    columns=[
        *identity_columns(),
        column(
            "distance_m",
            "number",
            Unit.METRE,
            "Walking distance over the road network from the lot's nearest entry "
            "to the nearest POI entry of the type; null beyond max_distance_m.",
        ),
    ]
)


def population_access_declaration() -> TableOutput:
    """Identity, distance, residents, residents with access, their share, geometry."""
    return TableOutput(
        columns=[
            *identity_columns(),
            column(
                "distance_m",
                "number",
                Unit.METRE,
                "Network distance to the nearest POI of the type; for a block, its "
                "lots' average weighted by population (null without residents "
                "within max_distance_m).",
            ),
            column("population", "number", Unit.COUNT, "Residents (built-form)."),
            column(
                "population_with_access",
                "number",
                Unit.COUNT,
                "Residents of lots within within_m of a POI of the type.",
            ),
            column(
                "access_share",
                "number",
                Unit.RATIO,
                "population_with_access / population (0-1); null without residents.",
            ),
        ],
        geometry="geometry",
    )


def result_frame(source: gpd.GeoDataFrame, values: pd.DataFrame) -> gpd.GeoDataFrame:
    """Identity and geometry of ``source`` with the values, in its order."""
    result = source[IDENTITY + ["geometry"]].copy()
    for key in KEYS:
        result[key] = values[key].to_numpy()
    return result


def population_access_metrics(
    parameters_type: type[MetricParameters],
    lots_output: str,
    pois_output: str,
    nodes_output: str,
    edges_output: str,
) -> list:
    """Both nodes over one source's lots, POIs and road network."""

    class DistanceParameters(parameters_type):  # ty: ignore[unsupported-base]
        """The source's parameter, plus the POIs to measure and the search limit."""

        poi_type: str = Field(
            min_length=1,
            description=f"Type of the POIs to measure, as their {POI_TYPE_COLUMN} "
            "(e.g. pharmacy).",
        )
        max_distance_m: float = Field(
            default=MAX_DISTANCE_M,
            gt=0,
            description="Search limit over the network, in metres; lots farther "
            "from every POI have no distance.",
        )

    class PopulationAccessParameters(DistanceParameters):
        """Plus the distance within which residents have access."""

        within_m: float = Field(
            gt=0, description="Access distance over the network, in metres (e.g. 300)."
        )

        @model_validator(mode="after")
        def _within_search(self):
            if self.within_m > self.max_distance_m:
                msg = "within_m cannot exceed max_distance_m"
                raise ValueError(msg)
            return self

    def distance_handler(
        parameters,
        lots: gpd.GeoDataFrame,
        pois: gpd.GeoDataFrame,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        context: RunContext,
    ) -> pd.DataFrame:
        selected = pois[pois[POI_TYPE_COLUMN] == parameters.poi_type]
        if selected.empty:
            context.logger.warning(
                "No POI of type %r; types present: %s",
                parameters.poi_type,
                sorted(pois[POI_TYPE_COLUMN].dropna().unique()),
            )
        crs = resolve_access_crs(lots)
        with timed("road network", context.logger):
            network = street_network(nodes, edges)
        with timed("nearest POI distance", context.logger):
            distance = nearest_distance(
                lots, selected, network, crs, parameters.max_distance_m
            )
        context.logger.info(
            "%s: %d POIs; %d of %d lots within %s m of one",
            parameters.poi_type,
            len(selected),
            np.isfinite(distance).sum(),
            len(lots),
            parameters.max_distance_m,
        )
        result = pd.DataFrame(lots[IDENTITY].copy())
        result["distance_m"] = distance
        return result

    def access_handler(
        parameters,
        lots: gpd.GeoDataFrame,
        blocks: gpd.GeoDataFrame,
        distances: pd.DataFrame,
        context: RunContext,
    ) -> dict[str, gpd.GeoDataFrame]:
        distance = (
            distances.set_index("id")["distance_m"].reindex(lots["id"]).to_numpy()
        )
        with timed("population access", context.logger):
            values = population_access(lots, blocks, distance, parameters.within_m)
        by_lot = values["lots"]
        total = by_lot[["population", "population_with_access"]].sum()
        reached = by_lot["population"].where(by_lot["distance_m"].notna())
        context.logger.info(
            "%s within %s m: %.0f of %.0f residents (%.1f%%); average distance %.0f m; "
            "%.0f residents without one within %s m",
            parameters.poi_type,
            parameters.within_m,
            total["population_with_access"],
            total["population"],
            100 * total["population_with_access"] / total["population"]
            if total["population"]
            else float("nan"),
            (reached * by_lot["distance_m"]).sum() / reached.sum()
            if reached.sum()
            else float("nan"),
            total["population"] - reached.sum(),
            parameters.max_distance_m,
        )
        return {
            "reimagina_population_access_lots": result_frame(lots, by_lot),
            "reimagina_population_access_blocks": result_frame(
                blocks, values["blocks"]
            ),
        }

    distance_handler.__annotations__["parameters"] = DistanceParameters
    access_handler.__annotations__["parameters"] = PopulationAccessParameters
    return [
        metric(
            name="reimagina_nearest_poi_distance",
            description=(
                "Walking distance over the road network from each lot to the "
                "nearest POI of one type (poi_type), as GHSCI: points are their own "
                "entry and polygons are entered anywhere along their outline (a "
                "point every 20 m within 30 m of a street); entries join the street "
                "at their projection on the nearest segment, every street two-way. "
                "Null beyond max_distance_m."
            ),
            inputs={
                "lots": Dependency(lots_output),
                "pois": Dependency(pois_output),
                "nodes": Dependency(nodes_output),
                "edges": Dependency(edges_output),
            },
            outputs={DISTANCE_OUTPUT: DISTANCE_DECLARATION},
        )(distance_handler),
        metric(
            name="reimagina_population_access",
            description=(
                "Residents with a POI of one type (poi_type) within within_m over "
                "the road network, and the average distance to the nearest one "
                "(ULI 'Percentage of population with access to <poi> within X "
                "meters' and 'Average distance to closest <poi>'). Lots within "
                "within_m count their built-form population; blocks add up their "
                "lots and average their distances weighted by population."
            ),
            inputs={
                "lots": Dependency("reimagina_built_form_lots"),
                "blocks": Dependency("reimagina_built_form_blocks"),
                "distances": Dependency(DISTANCE_OUTPUT),
            },
            outputs={
                "reimagina_population_access_lots": population_access_declaration(),
                "reimagina_population_access_blocks": population_access_declaration(),
            },
        )(access_handler),
    ]
