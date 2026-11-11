"""Lyra nodes that build a city's baseline from the data lake and write it into a
Reimagina baseline scenario: reimagina_baseline, reimagina_baseline_elements and
reimagina_baseline_write. The elements are blocks, lots, amenities and
establishments. Its street network: reimagina_street_network,
reimagina_street_network_tables and reimagina_street_network_write
(ingestion/network.py)."""

from typing import Literal

import geopandas as gpd
import pandas as pd
import shapely
from lyra.sdk import (
    DatasetContext,
    Dependency,
    MetricParameters,
    RunContext,
    TableColumn,
    TableOutput,
    Unit,
    dataset,
    metric,
)
from lyra.sdk.errors import MetricInputError
from pydantic import Field

from reimagina_test_port.common.nodes import column, timed
from reimagina_test_port.common.schema import UNITS, identity_columns
from reimagina_test_port.indicators.accessibility.nodes import (
    EDGES_DECLARATION,
    NODES_DECLARATION,
)
from reimagina_test_port.ingestion.cities import CITIES
from reimagina_test_port.ingestion.points import (
    AMENITY_ATTRIBUTES,
    ESTABLISHMENT_ATTRIBUTES,
)

# One per YAML in ingestion/cities.
CityName = Literal[tuple(CITIES)]  # ty: ignore[invalid-type-form]


class BaselineParameters(MetricParameters):
    """City whose baseline is built from the data lake."""

    city: CityName = Field(description="City preset (ingestion/cities/<city>.yaml).")


def point_declaration(attributes: dict[str, str]) -> TableOutput:
    """Identity, amenity or establishment attributes and EPSG:4326 points."""
    units = {
        "capacity": Unit.COUNT,
        "num_workers": Unit.COUNT,
        "area_m2": Unit.SQUARE_METRE,
    }
    return TableOutput(
        columns=[
            *identity_columns(),
            *(
                column(
                    key,
                    "string" if kind == "text" else "number",
                    None if kind == "text" else units[key],
                    key.replace("_", " "),
                )
                for key, kind in attributes.items()
            ),
        ],
        geometry="geometry",
    )


AMENITY_DECLARATION = point_declaration(AMENITY_ATTRIBUTES)
ESTABLISHMENT_DECLARATION = point_declaration(ESTABLISHMENT_ATTRIBUTES)


def baseline_declaration(*, blocks: bool) -> TableOutput:
    """Identity, element attributes and EPSG:4326 geometry."""
    identity = identity_columns()
    if blocks:
        attributes = [
            column("vivtot", "number", Unit.COUNT, "Census dwellings of the block."),
            column(
                "people_per_household",
                "number",
                Unit.COUNT,
                "Census average occupants per inhabited dwelling.",
            ),
            column(
                "lot.size_m2",
                "number",
                Unit.SQUARE_METRE,
                "Target area of each lot it is subdivided into (unset when loaded).",
            ),
            column(
                "lot.frontage_m",
                "number",
                Unit.METRE,
                "Target lot frontage (unset when loaded).",
            ),
        ]
    else:
        units = {
            "people_per_household": Unit.COUNT,
            "occupancy_rate": Unit.RATIO,
            "housing.floor_share": Unit.PERCENT,
        }
        attributes = [
            column(
                key,
                "string" if key in TEXT_ATTRIBUTES else "number",
                None if key in TEXT_ATTRIBUTES else (units | UNITS)[key],
                key.replace("_", " "),
            )
            for key in BASELINE_LOT_ATTRIBUTES
        ]
    return TableOutput(columns=[*identity, *attributes], geometry="geometry")


TEXT_ATTRIBUTES = {"land_use", "commerce_type"}
BASELINE_LOT_ATTRIBUTES = [
    "land_use",
    "cos",
    "num_levels",
    "area",
    "unit_size_m2",
    "people_per_household",
    "occupancy_rate",
    "commerce_type",
    "housing.floor_share",
]


@dataset(
    name="reimagina_baseline",
    description=(
        "Build a city's baseline elements from the CFC data lake with "
        "exe_reimagina's rules: INEGI blocks and census, cadastral lots, "
        "normativa, Open Buildings heights, the city's amenity layers (normalized "
        "into the amenities file) and the DENUE. "
        "Blocks are keyed by CVEGEO; lots by their cadastral key. Each DENUE "
        "record is an amenity or an establishment by the SCIAN crosswalk; with "
        "the amenities file's, they go to the nearest lot within 50 m as a point "
        "at its center, and each lot's land use follows from what it holds."
    ),
    outputs={
        "reimagina_baseline_lots": baseline_declaration(blocks=False),
        "reimagina_baseline_blocks": baseline_declaration(blocks=True),
        "reimagina_baseline_amenities": AMENITY_DECLARATION,
        "reimagina_baseline_establishments": ESTABLISHMENT_DECLARATION,
    },
)
def baseline_dataset(
    parameters: BaselineParameters, context: DatasetContext
) -> dict[str, gpd.GeoDataFrame]:
    """Download the sources into the job's scratch folder and build the tables."""
    # rasterio only where the baseline is built.
    from reimagina_test_port.ingestion import baseline

    def progress(stage: str, current: int, total: int) -> None:
        context.report_progress(stage=stage, current=current, total=total)

    result = baseline.build(CITIES[parameters.city], context.temp_dir, progress)
    context.logger.info("Baseline: %s", result["report"])
    return {
        "reimagina_baseline_lots": result["lots"],
        "reimagina_baseline_blocks": result["blocks"],
        "reimagina_baseline_amenities": result["amenities"],
        "reimagina_baseline_establishments": result["establishments"],
    }


BASELINE_INPUTS = {
    "lots": Dependency("reimagina_baseline_lots"),
    "blocks": Dependency("reimagina_baseline_blocks"),
    "amenities": Dependency("reimagina_baseline_amenities"),
    "establishments": Dependency("reimagina_baseline_establishments"),
}


@metric(
    name="reimagina_baseline_elements",
    description=(
        "Return a city's baseline lots, blocks, amenities and establishments as "
        "built from the data lake (reimagina_baseline). Nothing is written to "
        "Reimagina's database."
    ),
    inputs=BASELINE_INPUTS,
    outputs={
        "reimagina_baseline_elements_lots": baseline_declaration(blocks=False),
        "reimagina_baseline_elements_blocks": baseline_declaration(blocks=True),
        "reimagina_baseline_elements_amenities": AMENITY_DECLARATION,
        "reimagina_baseline_elements_establishments": ESTABLISHMENT_DECLARATION,
    },
)
def baseline_elements(
    parameters: BaselineParameters,
    lots: gpd.GeoDataFrame,
    blocks: gpd.GeoDataFrame,
    amenities: gpd.GeoDataFrame,
    establishments: gpd.GeoDataFrame,
) -> dict[str, gpd.GeoDataFrame]:
    """The dataset's tables."""
    return {
        "reimagina_baseline_elements_lots": lots,
        "reimagina_baseline_elements_blocks": blocks,
        "reimagina_baseline_elements_amenities": amenities,
        "reimagina_baseline_elements_establishments": establishments,
    }


UUID_PATTERN = (
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


class BaselineWriteParameters(BaselineParameters):
    """Baseline scenario to fill, created beforehand in the API."""

    project_id: str = Field(pattern=UUID_PATTERN, description="Project UUID.")
    scenario_id: str = Field(
        pattern=UUID_PATTERN,
        description="Baseline scenario UUID of the project, without forks.",
    )
    dry_run: bool = Field(
        default=False, description="Compute and count the rows, then roll back."
    )


BASELINE_ATTRIBUTES = {
    "lot": {
        key: "text" if key in TEXT_ATTRIBUTES else "number"
        for key in BASELINE_LOT_ATTRIBUTES
    },
    "block": {
        "vivtot": "number",
        "people_per_household": "number",
        "lot.size_m2": "number",
        "lot.frontage_m": "number",
    },
    "amenity": AMENITY_ATTRIBUTES,
    "establishment": ESTABLISHMENT_ATTRIBUTES,
}
# Element type -> column of the write summary with its elements in the scenario.
WRITTEN_COLUMNS = {
    "block": "blocks",
    "lot": "lots",
    "amenity": "amenities",
    "establishment": "establishments",
}
BASELINE_WRITE_DECLARATION = TableOutput(
    columns=[
        TableColumn(name=name, type="integer", unit=Unit.COUNT, description=description)
        for name, description in (
            ("elements", "Blocks, lots, amenities and establishments written."),
            ("elements_created", "Elements new to the database."),
            ("attribute_values", "Attribute values written."),
            ("elements_removed", "Scenario elements no longer in the baseline."),
            ("attributes_added", "attribute_catalog rows added (missing before)."),
            ("indicators_added", "indicator_catalog rows added (missing before)."),
            *(
                (name, f"{name.capitalize()} in the scenario after the write.")
                for name in WRITTEN_COLUMNS.values()
            ),
        )
    ]
    + [
        TableColumn(
            name="dry_run",
            type="boolean",
            unit=None,
            description="Whether the changes were rolled back.",
        )
    ]
)


@metric(
    name="reimagina_baseline_write",
    description=(
        "Build a city's baseline from the data lake (reimagina_baseline) and write "
        "it into an existing baseline scenario of a project: elements, state and "
        "attribute values of blocks, lots, amenities and establishments, in one "
        "transaction. Existing elements are reused "
        "unchanged; running it again replaces the scenario's rows. Missing "
        "attribute_catalog and indicator_catalog rows are added first; existing "
        "ones are left as they are and must be active. Compute indicators "
        "afterwards with reimagina_scenario_write."
    ),
    inputs=BASELINE_INPUTS,
    outputs={"reimagina_baseline_write": BASELINE_WRITE_DECLARATION},
)
def baseline_write(
    parameters: BaselineWriteParameters,
    lots: gpd.GeoDataFrame,
    blocks: gpd.GeoDataFrame,
    amenities: gpd.GeoDataFrame,
    establishments: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Final node: one summary row."""
    from reimagina_test_port.ingestion import baseline_writer

    try:
        counts = baseline_writer.write_baseline(
            parameters.project_id,
            parameters.scenario_id,
            {
                "block": blocks,
                "lot": lots,
                "amenity": amenities,
                "establishment": establishments,
            },
            BASELINE_ATTRIBUTES,
            dry_run=parameters.dry_run,
        )
    except baseline_writer.BaselineScenarioError as exc:
        raise MetricInputError(
            "reimagina_baseline_write", "parameters.scenario_id", str(exc)
        ) from exc
    by_type = counts.pop("by_type")
    written = {name: by_type[t] for t, name in WRITTEN_COLUMNS.items()}
    return pd.DataFrame([{**counts, **written, "dry_run": parameters.dry_run}])


# ------------------------------------------------------------------ street network


class NetworkParameters(BaselineParameters):
    """City whose street network is built, and an optional file of streets."""

    streets_uri: str | None = Field(
        default=None,
        description=(
            "Optional (Geo)Parquet or vector file of street lines (az:// or a path) "
            "that replaces the OpenStreetMap network inside its zone."
        ),
    )
    zone_uri: str | None = Field(
        default=None,
        description=(
            "Optional polygon file of the zone streets_uri replaces; without it, "
            "the streets' convex hull."
        ),
    )


def _network_input_error(field: str, message: str) -> MetricInputError:
    return MetricInputError("reimagina_street_network", f"parameters.{field}", message)


@dataset(
    name="reimagina_street_network",
    description=(
        "Build a city's street network from scratch: OpenStreetMap's ways over "
        "the convex hull of the city's blocks plus a margin, within its "
        "municipality (sources.streets.boundary_uri), with the city's filter (Mexicali: the "
        "ULI's pedestrian network), in its metric CRS. With streets_uri, those "
        "streets replace the network inside their zone (zone_uri, or their "
        "convex hull) and are sewn to it, as exe_reimagina's Condesa network."
    ),
    outputs={
        "reimagina_street_network_nodes": NODES_DECLARATION,
        "reimagina_street_network_edges": EDGES_DECLARATION,
    },
)
def street_network(
    parameters: NetworkParameters, context: DatasetContext
) -> dict[str, pd.DataFrame]:
    """Nodes and edges in the tables reimagina_road_network returns."""
    from reimagina_test_port.common import lake
    from reimagina_test_port.ingestion import network

    city = CITIES[parameters.city]
    if city.streets is None:
        raise _network_input_error(
            "city", f"{city.name} has no sources.streets in its YAML."
        )
    with timed("blocks", context.logger):
        raw = lake.read_parquet(city.blocks_uri, ["geometry"])
        blocks = gpd.GeoSeries.from_wkb(raw["geometry"], crs="EPSG:4326")
    with timed("OpenStreetMap", context.logger):
        boundary = None
        if city.streets.boundary_uri:
            raw = lake.read_parquet(city.streets.boundary_uri, ["geometry"])
            boundary = gpd.GeoSeries.from_wkb(raw["geometry"], crs="EPSG:4326")
        nodes, edges = network.osm_network(
            blocks, city.target_crs, city.streets, boundary
        )
    if parameters.streets_uri:
        folder = context.temp_dir / "streets"
        lines = network.read_geometries(
            lake.download(parameters.streets_uri, folder),
            city.target_crs,
            ("LineString", "MultiLineString"),
        )
        zone = None
        if parameters.zone_uri:
            polygons = network.read_geometries(
                lake.download(parameters.zone_uri, folder),
                city.target_crs,
                ("Polygon", "MultiPolygon"),
            )
            zone = shapely.union_all(polygons.to_numpy())
        with timed("streets merge", context.logger):
            nodes, edges, report = network.merge_streets(
                nodes, edges, lines, city.streets, zone
            )
        context.logger.info("Streets merged: %s", report)
    elif parameters.zone_uri:
        raise _network_input_error("zone_uri", "zone_uri needs streets_uri.")
    return {
        "reimagina_street_network_nodes": nodes.reset_index(),
        "reimagina_street_network_edges": edges.rename(
            columns={"length": "distance"}
        ).assign(two_way=True),
    }


NETWORK_INPUTS = {
    "nodes": Dependency("reimagina_street_network_nodes"),
    "edges": Dependency("reimagina_street_network_edges"),
}


@metric(
    name="reimagina_street_network_tables",
    description=(
        "Return a city's street network as built (reimagina_street_network), to "
        "inspect it. Nothing is uploaded or written to Reimagina's database."
    ),
    inputs=NETWORK_INPUTS,
    outputs={
        "reimagina_street_network_tables_nodes": NODES_DECLARATION,
        "reimagina_street_network_tables_edges": EDGES_DECLARATION,
    },
)
def street_network_tables(
    parameters: NetworkParameters, nodes: pd.DataFrame, edges: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """The dataset's tables."""
    return {
        "reimagina_street_network_tables_nodes": nodes,
        "reimagina_street_network_tables_edges": edges,
    }


class NetworkWriteParameters(NetworkParameters):
    """Project whose road_network_uri gets the new network."""

    project_id: str = Field(pattern=UUID_PATTERN, description="Project UUID.")
    dry_run: bool = Field(
        default=False,
        description=(
            "Build the H5 and report, without uploading it or changing the project."
        ),
    )


NETWORK_WRITE_DECLARATION = TableOutput(
    columns=[
        TableColumn(
            name="nodes", type="integer", unit=Unit.COUNT, description="Network nodes."
        ),
        TableColumn(
            name="edges", type="integer", unit=Unit.COUNT, description="Network edges."
        ),
        TableColumn(
            name="road_network_uri",
            type="string",
            unit=None,
            description="Where the H5 is (or would be, on a dry run).",
        ),
        TableColumn(
            name="previous_road_network_uri",
            type="string",
            unit=None,
            description="The project's road_network_uri before this run.",
        ),
        TableColumn(
            name="dry_run",
            type="boolean",
            unit=None,
            description="Whether nothing was uploaded or changed.",
        ),
    ]
)


@metric(
    name="reimagina_street_network_write",
    description=(
        "Save a city's street network (reimagina_street_network) as the pandana "
        "H5 the indicators read, upload it to the city's output_prefix in the "
        "lake, named by its contents (an existing file is never overwritten), and "
        "set the project's road_network_uri to it. Recompute indicators "
        "afterwards with reimagina_scenario_write."
    ),
    inputs=NETWORK_INPUTS,
    outputs={"reimagina_street_network_write": NETWORK_WRITE_DECLARATION},
)
def street_network_write(
    parameters: NetworkWriteParameters,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    context: RunContext,
) -> pd.DataFrame:
    """Final node: one summary row."""
    from reimagina_test_port.common import lake
    from reimagina_test_port.ingestion import network

    city = CITIES[parameters.city]
    nodes = nodes.set_index("id")[["x", "y"]]
    edges = edges.rename(columns={"distance": "length"})[["from", "to", "length"]]
    name = network.h5_name(city.cve_code, nodes, edges)
    path = network.save_h5(nodes, edges, context.temp_dir / name)
    uri = f"{city.streets.output_prefix}/{name}"
    if not parameters.dry_run:
        try:
            lake.upload(path, uri)
        except FileExistsError:
            context.logger.info("%s is already in the lake: the same network", uri)
    try:
        previous = network.set_road_network_uri(
            parameters.project_id, uri, dry_run=parameters.dry_run
        )
    except network.ProjectNotFoundError as exc:
        raise MetricInputError(
            "reimagina_street_network_write", "parameters.project_id", str(exc)
        ) from exc
    context.logger.info("road_network_uri: %s -> %s", previous, uri)
    return pd.DataFrame(
        [
            {
                "nodes": len(nodes),
                "edges": len(edges),
                "road_network_uri": uri,
                "previous_road_network_uri": previous,
                "dry_run": parameters.dry_run,
            }
        ]
    )
