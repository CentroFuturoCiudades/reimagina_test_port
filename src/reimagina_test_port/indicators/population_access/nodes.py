"""Lyra nodes of population access to the poi_catalog types over the road network:

- reimagina_nearest_poi_distance: metres from each lot to its nearest POI of each type.
- reimagina_population_access: residents within each type's coverage distance, their
  share of the residents, and the average distance to its nearest POI, per lot and
  per block.
- reimagina_poi_diversity: entropy of the subcategories of each diversity set among
  the POIs within walking distance, and the jobs (workers of the establishments)
  within it, per lot and per block.
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
from reimagina_test_port.indicators.population_access.access_calculation import (
    lot_entries,
    nearest_distances,
    owned_by,
    population_access,
    street_network,
)
from reimagina_test_port.indicators.population_access.diversity_calculation import (
    diversity,
    entropy,
    reached,
    within_lots,
)
from reimagina_test_port.ingestion.crosswalk import ESTABLISHMENT_CATEGORIES

POI_TYPE_COLUMN = "poi_type"
MAX_DISTANCE_M = 10_000
DISTANCE_OUTPUT = "reimagina_nearest_poi_distance_lots"
# The lots' entries, shared by the distances and the diversity: ~5 s in Mexicali.
ENTRIES_OUTPUT = "reimagina_lot_entries"
ENTRIES_COLUMNS = [
    column("id", "string", None, "The lot's id."),
    column("segment", "integer", None, "Street segment the entry joins."),
    column("along_start", "number", Unit.METRE, "Along the segment to its start."),
    column("along_end", "number", Unit.METRE, "Along the segment to its end."),
    column("offset", "number", Unit.METRE, "Straight from the entry to the segment."),
]
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

    def entries_handler(
        parameters,
        lots: gpd.GeoDataFrame,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        context: RunContext,
    ) -> pd.DataFrame:
        crs = resolve_access_crs(lots)
        with timed("road network", context.logger):
            network = street_network(nodes, edges)
        with timed("lot entries", context.logger):
            origins = lot_entries(lots, network, crs)
        return origins.assign(
            id=lots["id"].astype(str).to_numpy()[origins["owner"].to_numpy()]
        ).reindex(columns=[c.name for c in ENTRIES_COLUMNS])

    def distance_handler(
        parameters,
        lots: gpd.GeoDataFrame,
        pois: gpd.GeoDataFrame,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        entries: pd.DataFrame,
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
            distances = nearest_distances(
                lots,
                selected,
                network,
                crs,
                MAX_DISTANCE_M,
                owned_by(entries, lots),
            )
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

    # The source's parameters alone: one run for the distances and the diversity.
    entries_handler.__annotations__["parameters"] = parameters_type
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
            name="reimagina_lot_entries",
            description=(
                "Where each lot joins the road network, as GHSCI: points are their "
                "own entry and polygons are entered anywhere along their outline (a "
                "point every 20 m within 30 m of a street), each joining the street "
                "at its projection on the nearest segment. Shared by the nearest "
                "POI distances and the diversity."
            ),
            inputs={
                "lots": Dependency(lots_output),
                "nodes": Dependency(nodes_output),
                "edges": Dependency(edges_output),
            },
            outputs={ENTRIES_OUTPUT: TableOutput(columns=ENTRIES_COLUMNS)},
        )(entries_handler),
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
                "entries": Dependency(ENTRIES_OUTPUT),
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


CATEGORY_COLUMN = "category"
DIVERSITY_M = 500
# Each diversity set: its subcategories, each the element categories it groups
# (crosswalk.CATEGORIES); k is how many subcategories it has. A category may sit in
# several subcategories and counts in each (a school of several levels). As the
# ULI's diversity groups (MX_Mexicali_2025_ULI.yml); the city's own layers' places
# count in the group of their category (a cultural centre in culture). The sets are
# not poi_catalog types: that table holds destinations only.
DIVERSITY_SETS = {
    "fresh_food": {
        "meat": ("butcher", "poultry", "fishmonger"),
        "greengrocer": ("greengrocer",),
        "grains_spices": ("grains_spices",),
        "dairy": ("dairy",),
        "supermarket": ("supermarket",),
        "convenience_store": ("convenience_store",),
        "bakery_tortilleria": ("bakery", "tortilleria"),
    },
    "education": {
        "preschool": ("preschool", "multilevel_school"),
        "primary": ("primary_education", "multilevel_school"),
        "secondary": (
            "secondary_education",
            "upper_secondary_education",
            "multilevel_school",
        ),
        "tertiary": ("tertiary_education",),
    },
    "health": {
        "primary_care": ("health_center", "clinic"),
        "specialist_clinic": ("specialist_clinic",),
        "hospital": ("hospital",),
        "pharmacy": ("pharmacy",),
        "care": (
            "residence",
            "daycare",
            "social_assistance",
            "community_food",
            "community_center",
        ),
    },
    "care": {
        "daycare": ("daycare",),
        "residence": ("residence",),
        "social_assistance": (
            "social_assistance",
            "community_food",
            "community_center",
        ),
    },
    "community_culture_recreation": {
        "community": ("religious", "civic_association", "social_welfare"),
        "culture": ("cultural_center", "museum", "library", "cinema"),
        "sport": ("sports", "sports_school"),
        "recreation": ("recreation_venue",),
    },
    # Job diversity: the establishments' categories (not the SCIAN sectors of the
    # ULI's), establishments only (JOBS_ELEMENT); as the ULI's, among the places
    # in the lot (for a block, in its lots), with k the categories there.
    "jobs": {category: (category,) for category in ESTABLISHMENT_CATEGORIES},
}
DIVERSITY_KEYS = [f"diversity_{name}" for name in DIVERSITY_SETS]
ELEMENT_TYPE_COLUMN = "element_type"
WORKERS_COLUMN = "num_workers"
# The set whose places are establishments only; its walk also adds up their workers.
JOBS_SET = "jobs"
JOBS_KEY = f"diversity_{JOBS_SET}"
JOBS_ELEMENT = "establishment"
PARENT_COLUMN = "parent_id"
REACHABLE_JOBS_KEY = "reachable_jobs"
REACHABLE_JOBS_COLUMN = column(
    REACHABLE_JOBS_KEY,
    "number",
    Unit.COUNT,
    "Workers (DENUE's num_workers) of the establishments within 500 m over the road "
    "network; for a block, its lots' average weighted by population. Null without "
    "geometry or residents.",
)
DIVERSITY_COLUMNS = [
    column(
        key,
        "number",
        Unit.RATIO,
        "Shannon entropy of the set's subcategories (by category) among the POIs "
        "within 500 m over the road network, over ln k, k being how many "
        "subcategories the set has (0 = one or none, 1 = all k equally); for a "
        "block, its lots' average weighted by population. Null without geometry "
        "or residents."
        if key != JOBS_KEY
        else "Shannon entropy of the categories of the establishments of the lot "
        "(for a block, of its lots) over ln k, k being how many categories are "
        "there (0 = one or none), as the ULI's job diversity. Null for a block "
        "without lots.",
    )
    for key in DIVERSITY_KEYS
]


def poi_diversity_metric(
    parameters_type: type[MetricParameters],
    pois_output: str,
    nodes_output: str,
    edges_output: str,
):
    """The diversity node over one source's POIs and road network."""

    def diversity_handler(
        parameters,
        lots: gpd.GeoDataFrame,
        blocks: gpd.GeoDataFrame,
        pois: gpd.GeoDataFrame,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        entries: pd.DataFrame,
        context: RunContext,
    ) -> dict[str, gpd.GeoDataFrame]:
        crs = resolve_access_crs(lots)
        with timed("road network", context.logger):
            network = street_network(nodes, edges)
        origins = owned_by(entries, lots)
        # One row per POI, whatever its poi_types: its category gives its
        # subcategories, one row each.
        places = pois.drop_duplicates("id").reset_index(drop=True)
        categories = places.get(CATEGORY_COLUMN, pd.Series(None, index=places.index))
        missing = pd.Series(None, index=places.index)
        element_types = places.get(ELEMENT_TYPE_COLUMN, missing)
        workers = places.get(WORKERS_COLUMN, missing)
        parents = places.get(PARENT_COLUMN, missing)
        values = {}
        for name, subcategories in DIVERSITY_SETS.items():
            groups = {}
            for sub, group in subcategories.items():
                for cat in group:
                    groups.setdefault(cat, []).append(sub)
            subcategory = categories.map(groups).explode().dropna()
            if name == JOBS_SET:
                is_job = element_types.loc[subcategory.index] == JOBS_ELEMENT
                subcategory = subcategory[is_job.to_numpy()]
            rows = subcategory.index
            with timed(f"{name} diversity", context.logger):
                counts, totals, measured = reached(
                    lots,
                    places.geometry.loc[rows],
                    subcategory,
                    network,
                    crs,
                    DIVERSITY_M,
                    origins,
                    workers.loc[rows] if name == JOBS_SET else None,
                )
                values[f"diversity_{name}"] = diversity(
                    lots, blocks, entropy(counts, measured, len(subcategories))
                )
            if name == JOBS_SET:
                values[REACHABLE_JOBS_KEY] = diversity(
                    lots, blocks, np.where(measured, totals, np.nan)
                )
                # Its diversity is the ULI's: within the lots, not the walk's.
                jobs = (element_types == JOBS_ELEMENT) & categories.isin(
                    list(subcategories)
                )
                values[JOBS_KEY] = within_lots(
                    lots, blocks, parents[jobs], categories[jobs]
                )
        return {
            f"reimagina_poi_diversity_{name}": source[IDENTITY + ["geometry"]].assign(
                **{key: value[name] for key, value in values.items()}
            )
            for name, source in (("lots", lots), ("blocks", blocks))
        }

    diversity_handler.__annotations__["parameters"] = parameters_type
    output = TableOutput(
        columns=[*identity_columns(), *DIVERSITY_COLUMNS, REACHABLE_JOBS_COLUMN],
        geometry="geometry",
    )
    return metric(
        name="reimagina_poi_diversity",
        description=(
            "Diversity of each set of destinations within walking distance: the "
            "Shannon entropy of its subcategories, by category, among the POIs "
            "within 500 m over the road network, over ln k, k being how many "
            "subcategories the set has; and the workers of the establishments "
            "within 500 m (reachable jobs). Lots count the POIs their entries "
            "reach, walked as in reimagina_nearest_poi_distance; blocks average "
            "their lots weighted by population. Job diversity is the ULI's "
            "instead: the entropy of the categories of the establishments of each "
            "lot, and of each block's lots, over ln of the categories there."
        ),
        inputs={
            "lots": Dependency("reimagina_built_form_lots"),
            "blocks": Dependency("reimagina_built_form_blocks"),
            "pois": Dependency(pois_output),
            "nodes": Dependency(nodes_output),
            "edges": Dependency(edges_output),
            "entries": Dependency(ENTRIES_OUTPUT),
        },
        outputs={
            "reimagina_poi_diversity_lots": output,
            "reimagina_poi_diversity_blocks": output,
        },
    )(diversity_handler)
