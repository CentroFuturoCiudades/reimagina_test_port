"""Lyra nodes that build a city's baseline from the data lake and write it into a
Reimagina baseline scenario: reimagina_baseline, reimagina_baseline_elements and
reimagina_baseline_write; reimagina_baseline_append adds to an existing one. The
elements are blocks, lots, amenities, establishments, rights of way (reimagina_baseline_streets, over the street
network) and public transport stops (reimagina_baseline_transit_stops). Its street network: reimagina_street_network,
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
        "tree_coverage": Unit.PERCENT,
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


RIGHT_OF_WAY_DECLARATION = TableOutput(
    columns=[
        *identity_columns(),
        column(
            "bands.use",
            "string",
            None,
            "The street's bands, left to right, as a JSON list: sidewalk, lanes "
            "(all traffic lanes), shared_lanes (the lanes, shared with bikes), "
            "protected_bike_lane (its own band), median.",
        ),
        column(
            "bands.share",
            "string",
            None,
            "Each band's share (0-100) of the width, as a JSON list.",
        ),
        column(
            "streetlights",
            "number",
            Unit.PERCENT,
            "Share of its block faces with street lighting.",
        ),
        column(
            "tree_coverage",
            "number",
            Unit.PERCENT,
            "Share of its area under tree canopy (Meta's canopy height); only a "
            "street drawn as a polygon has one.",
        ),
    ],
    geometry="geometry",
)
TRANSIT_DECLARATION = TableOutput(
    columns=[
        *identity_columns(),
        column("element_type", "string", None, "bus_stop or station."),
        column("source", "string", None, "Where the stop comes from: IMIP, OSM."),
        column("name", "string", None, "The stop's name, when its source has one."),
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
                "housing.people_per_household",
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
            "housing.unit_size_m2": Unit.SQUARE_METRE,
            "housing.people_per_household": Unit.COUNT,
            "occupancy_rate": Unit.RATIO,
            "housing.floor_share": Unit.PERCENT,
            "tree_coverage": Unit.PERCENT,
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
    "housing.unit_size_m2",
    "housing.people_per_household",
    "occupancy_rate",
    "commerce_type",
    "housing.floor_share",
    "tree_coverage",
]


@dataset(
    name="reimagina_baseline",
    description=(
        "Build a city's baseline elements from the CFC data lake with "
        "exe_reimagina's rules: INEGI blocks and census, cadastral lots, "
        "normativa, Open Buildings heights, Meta's canopy height (each lot's "
        "tree_coverage), the city's amenity layers (normalized "
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


@metric(
    name="reimagina_baseline_streets",
    description=(
        "A city's streets as right_of_way elements, one per street of its street "
        "network (reimagina_street_network), with streetlights from the block faces "
        "near each street and a bike lane band where the street's midpoint lies "
        "near one of the city's bike lanes (sources.street_lighting, "
        "sources.bike_lanes). With sources.streets.params.refine, the vehicular "
        "streets are fitted to the cadastre's lots instead: one right of way per "
        "stretch from intersection to intersection, centred between the lot fronts, "
        "with bands of the measured width and the lighting and bike lane matches "
        "scaled to it; pedestrian streets stay as above."
    ),
    inputs={
        "nodes": Dependency("reimagina_street_network_nodes"),
        "edges": Dependency("reimagina_street_network_edges"),
    },
    outputs={"reimagina_baseline_right_of_way": RIGHT_OF_WAY_DECLARATION},
)
def baseline_streets(
    parameters: BaselineParameters,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    context: RunContext,
) -> gpd.GeoDataFrame:
    """Download the lighting and bike lane files into the job's scratch folder and
    build the table."""
    from reimagina_test_port.ingestion import streets

    table = streets.build(CITIES[parameters.city], nodes, edges, context.temp_dir)
    fit = table.attrs.get("fit")
    if fit and fit["fitted"]:
        context.logger.info(
            "Streets fitted to the lots: %d (both sides measured: %d), as ever: %d",
            fit["fitted"],
            fit["both_sides"],
            fit["unfitted"],
        )
    context.logger.info(
        "Streets: %d, with streetlights %d, with a bike lane %d",
        len(table),
        int(table["streetlights"].notna().sum()),
        int(table["bands.use"].str.contains("shared_lanes|protected_bike_lane").sum()),
    )
    return table


@dataset(
    name="reimagina_baseline_transit_stops",
    description=(
        "A city's public transport stops as bus_stop and station elements, as the "
        "ULI's pt_any: its own layer (sources.transit_stops; Mexicali: IMIP's "
        "equipment category transp without the freight terminals) pooled with "
        "OpenStreetMap's stops by GHSCI's tags (sources.osm_transit_stops, "
        "extracted by scripts/osm_transit.py), each at its own point. Empty "
        "without either source."
    ),
    outputs={"reimagina_baseline_transit_stops": TRANSIT_DECLARATION},
)
def baseline_transit_stops(
    parameters: BaselineParameters, context: DatasetContext
) -> gpd.GeoDataFrame:
    """Download the sources into the job's scratch folder and build the table."""
    from reimagina_test_port.ingestion import transit

    table = transit.build(CITIES[parameters.city], context.temp_dir / "transit")
    context.logger.info(
        "Transit stops: %s",
        table.groupby(["source", "element_type"]).size().to_dict(),
    )
    return table


BASELINE_INPUTS = {
    "lots": Dependency("reimagina_baseline_lots"),
    "blocks": Dependency("reimagina_baseline_blocks"),
    "amenities": Dependency("reimagina_baseline_amenities"),
    "establishments": Dependency("reimagina_baseline_establishments"),
    "right_of_way": Dependency("reimagina_baseline_right_of_way"),
    "transit_stops": Dependency("reimagina_baseline_transit_stops"),
}


@metric(
    name="reimagina_baseline_elements",
    description=(
        "Return a city's baseline lots, blocks, amenities, establishments, "
        "streets and transit stops as built from the data lake "
        "(reimagina_baseline, reimagina_baseline_streets, "
        "reimagina_baseline_transit_stops). Nothing is written to Reimagina's database."
    ),
    inputs=BASELINE_INPUTS,
    outputs={
        "reimagina_baseline_elements_lots": baseline_declaration(blocks=False),
        "reimagina_baseline_elements_blocks": baseline_declaration(blocks=True),
        "reimagina_baseline_elements_amenities": AMENITY_DECLARATION,
        "reimagina_baseline_elements_establishments": ESTABLISHMENT_DECLARATION,
        "reimagina_baseline_elements_right_of_way": RIGHT_OF_WAY_DECLARATION,
        "reimagina_baseline_elements_transit_stops": TRANSIT_DECLARATION,
    },
)
def baseline_elements(
    parameters: BaselineParameters,
    lots: gpd.GeoDataFrame,
    blocks: gpd.GeoDataFrame,
    amenities: gpd.GeoDataFrame,
    establishments: gpd.GeoDataFrame,
    right_of_way: gpd.GeoDataFrame,
    transit_stops: gpd.GeoDataFrame,
) -> dict[str, gpd.GeoDataFrame]:
    """The datasets' tables."""
    return {
        "reimagina_baseline_elements_lots": lots,
        "reimagina_baseline_elements_blocks": blocks,
        "reimagina_baseline_elements_amenities": amenities,
        "reimagina_baseline_elements_establishments": establishments,
        "reimagina_baseline_elements_right_of_way": right_of_way,
        "reimagina_baseline_elements_transit_stops": transit_stops,
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
        "housing.people_per_household": "number",
        "lot.size_m2": "number",
        "lot.frontage_m": "number",
    },
    "amenity": AMENITY_ATTRIBUTES,
    "establishment": ESTABLISHMENT_ATTRIBUTES,
    "right_of_way": {
        "bands.use": "json",
        "bands.share": "json",
        "streetlights": "number",
        "tree_coverage": "number",
    },
    # Public transport stops (ingestion/transit.py).
    "bus_stop": {"source": "text", "name": "text"},
    "station": {"source": "text", "name": "text"},
}
TRANSIT_TYPES = ("bus_stop", "station")
# Element type -> column of the write summary with its elements in the scenario.
WRITTEN_COLUMNS = {
    "block": "blocks",
    "lot": "lots",
    "amenity": "amenities",
    "establishment": "establishments",
    "right_of_way": "rights_of_way",
    "bus_stop": "bus_stops",
    "station": "stations",
}
BASELINE_WRITE_DECLARATION = TableOutput(
    columns=[
        TableColumn(name=name, type="integer", unit=Unit.COUNT, description=description)
        for name, description in (
            (
                "elements",
                "Blocks, lots, amenities, establishments, streets and stops written.",
            ),
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
        "attribute values of blocks, lots, amenities, establishments, streets "
        "(right_of_way, from reimagina_baseline_streets; left out when the city "
        "has none) and transit stops (bus_stop and station, from "
        "reimagina_baseline_transit_stops; left out when the city has none), in "
        "one transaction. Existing elements are reused "
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
    right_of_way: gpd.GeoDataFrame,
    transit_stops: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Final node: one summary row."""
    from reimagina_test_port.ingestion import baseline_writer

    tables = {
        "block": blocks,
        "lot": lots,
        "amenity": amenities,
        "establishment": establishments,
        "right_of_way": right_of_way,
    }
    # A city without streets writes none, so its database needs no right_of_way type.
    if right_of_way.empty:
        del tables["right_of_way"]
    # Likewise a city without stops needs no bus_stop or station type.
    for element_type in TRANSIT_TYPES:
        stops = transit_stops[transit_stops["element_type"] == element_type]
        if not stops.empty:
            tables[element_type] = stops
    try:
        counts = baseline_writer.write_baseline(
            parameters.project_id,
            parameters.scenario_id,
            tables,
            BASELINE_ATTRIBUTES,
            dry_run=parameters.dry_run,
        )
    except baseline_writer.BaselineScenarioError as exc:
        raise MetricInputError(
            "reimagina_baseline_write", "parameters.scenario_id", str(exc)
        ) from exc
    by_type = counts.pop("by_type")
    written = {name: by_type.get(t, 0) for t, name in WRITTEN_COLUMNS.items()}
    return pd.DataFrame([{**counts, **written, "dry_run": parameters.dry_run}])


# ------------------------------------------------------------------------ append

APPEND_TYPES = ("amenity", "establishment", *TRANSIT_TYPES)
AppendType = Literal[APPEND_TYPES]  # ty: ignore[invalid-type-form]
# Its inputs without the streets: appending needs no street fit.
APPEND_INPUTS = {
    name: dependency
    for name, dependency in BASELINE_INPUTS.items()
    if name != "right_of_way"
}


class BaselineAppendParameters(BaselineParameters):
    """What to add to an existing baseline scenario, forks or not."""

    project_id: str = Field(pattern=UUID_PATTERN, description="Project UUID.")
    scenario_id: str = Field(
        pattern=UUID_PATTERN,
        description="Baseline scenario UUID of the project; it may have forks.",
    )
    element_types: list[AppendType] = Field(
        default=[],
        description=(
            "Element types whose missing elements are inserted: amenity, "
            "establishment, bus_stop, station."
        ),
    )
    categories: list[str] | None = Field(
        default=None,
        description=(
            "Only amenities and establishments of these categories (e.g. "
            "police_station, fire_station); all when unset."
        ),
    )
    update_attributes: list[str] = Field(
        default=[],
        description=(
            "Attributes added or updated on the elements the baseline already has, "
            "as <element_type>.<key> (e.g. amenity.tree_coverage, a park's canopy)."
        ),
    )
    dry_run: bool = Field(
        default=False, description="Compute and count the rows, then roll back."
    )


APPEND_DECLARATION = TableOutput(
    columns=[
        TableColumn(
            name="element_type", type="string", unit=None, description="Element type."
        ),
        TableColumn(
            name="attribute",
            type="string",
            unit=None,
            description="The attribute updated; empty for inserted elements.",
        ),
        TableColumn(
            name="outcome",
            type="string",
            unit=None,
            description=(
                "Elements: inserted, already_present (the baseline has it), "
                "missing_parent (the baseline lacks its lot: left out), "
                "parent_deleted_in_fork (inserted, on a lot a fork deleted). "
                "Attributes: added, updated, unchanged, absent (the baseline lacks "
                "the element)."
            ),
        ),
        TableColumn(name="count", type="integer", unit=Unit.COUNT, description="Rows."),
        TableColumn(
            name="examples",
            type="string",
            unit=None,
            description="Up to five of their source_record_ref, comma-separated.",
        ),
        TableColumn(
            name="dry_run",
            type="boolean",
            unit=None,
            description="Whether the changes were rolled back.",
        ),
    ]
)


def _append_error(field: str, message: str) -> MetricInputError:
    return MetricInputError("reimagina_baseline_append", f"parameters.{field}", message)


def _of_categories(
    table: gpd.GeoDataFrame, categories: list[str] | None
) -> gpd.GeoDataFrame:
    if categories is None:
        return table
    return table[table["category"].isin(categories)]


@metric(
    name="reimagina_baseline_append",
    description=(
        "Add to an existing baseline scenario, with forks or not, from the same "
        "build as reimagina_baseline_write (reimagina_baseline and "
        "reimagina_baseline_transit_stops; not the streets): insert the elements "
        "of element_types (amenities and establishments of categories, if given) "
        "it lacks, with their state and attributes, and add or update "
        "update_attributes on the elements it has. Nothing else is changed; forks "
        "see the new elements through the baseline. Elements whose lot the "
        "baseline lacks are left out. One row per element type, attribute and "
        "outcome. Compute indicators afterwards with reimagina_scenario_write, "
        "the baseline first."
    ),
    inputs=APPEND_INPUTS,
    outputs={"reimagina_baseline_append": APPEND_DECLARATION},
)
def baseline_append(
    parameters: BaselineAppendParameters,
    lots: gpd.GeoDataFrame,
    blocks: gpd.GeoDataFrame,
    amenities: gpd.GeoDataFrame,
    establishments: gpd.GeoDataFrame,
    transit_stops: gpd.GeoDataFrame,
    context: RunContext,
) -> pd.DataFrame:
    """Final node: the outcomes."""
    from reimagina_test_port.ingestion import baseline_writer
    from reimagina_test_port.ingestion.crosswalk import CATEGORIES

    known = {c for categories in CATEGORIES.values() for c in categories}
    if unknown := sorted(set(parameters.categories or []) - known):
        raise _append_error("categories", f"Unknown categories {unknown}.")
    available = {
        "block": blocks,
        "lot": lots,
        "amenity": amenities,
        "establishment": establishments,
        **{
            element_type: transit_stops[transit_stops["element_type"] == element_type]
            for element_type in TRANSIT_TYPES
        },
    }
    tables = {}
    for element_type in parameters.element_types:
        table = available[element_type]
        if element_type in ("amenity", "establishment"):
            table = _of_categories(table, parameters.categories)
        if not table.empty:
            tables[element_type] = table
    updates: dict[str, tuple[gpd.GeoDataFrame, list[str]]] = {}
    for name in parameters.update_attributes:
        element_type, _, key = name.partition(".")
        if element_type not in available or not key:
            raise _append_error(
                "update_attributes",
                f"{name!r} is not <element_type>.<key> of {sorted(available)}.",
            )
        updates.setdefault(element_type, (available[element_type], []))[1].append(key)
    if not tables and not updates:
        raise _append_error(
            "element_types", "Nothing to add: give element_types or update_attributes."
        )
    try:
        result = baseline_writer.append_baseline(
            parameters.project_id,
            parameters.scenario_id,
            tables,
            BASELINE_ATTRIBUTES,
            updates=updates,
            dry_run=parameters.dry_run,
        )
    except baseline_writer.BaselineScenarioError as exc:
        raise _append_error("scenario_id", str(exc)) from exc
    context.logger.info(
        "Catalog rows added: %d attributes, %d indicators",
        result["attributes_added"],
        result["indicators_added"],
    )
    rows = pd.DataFrame(
        result["outcomes"],
        columns=["element_type", "attribute", "outcome", "count", "examples"],
    )
    for row in rows.itertuples():
        context.logger.info(
            "%s %s %s: %d (%s)",
            row.element_type,
            row.attribute or "",
            row.outcome,
            row.count,
            ", ".join(row.examples),
        )
    return rows.assign(
        examples=rows["examples"].map(", ".join), dry_run=parameters.dry_run
    )


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
    refresh_osm: bool = Field(
        default=False,
        description=(
            "Download OpenStreetMap again instead of reading the copy saved in the "
            "lake by an earlier run, and replace that copy."
        ),
    )


# The road network's edges with each street's shape, in the city's metric CRS.
# plus OpenStreetMap's highway, name and lanes (text), which refine fits the streets
# with (ingestion/street_fit.py); none where the network added the edge.
STREET_EDGES_DECLARATION = TableOutput(
    columns=[
        *EDGES_DECLARATION.columns,
        TableColumn(
            name="highway",
            type="string",
            unit=None,
            description="OpenStreetMap's highway tag (its first, if several).",
        ),
        TableColumn(
            name="name",
            type="string",
            unit=None,
            description="OpenStreetMap's street name.",
        ),
        TableColumn(
            name="lanes",
            type="string",
            unit=None,
            description="OpenStreetMap's lanes tag, as text (both directions).",
        ),
    ],
    geometry="geometry",
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
        "convex hull) and are sewn to it, as exe_reimagina's Condesa network. "
        "The OpenStreetMap download is saved in the lake beside the H5s and read "
        "from there by later runs (refresh_osm downloads it again)."
    ),
    outputs={
        "reimagina_street_network_nodes": NODES_DECLARATION,
        "reimagina_street_network_edges": STREET_EDGES_DECLARATION,
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
            blocks,
            city.target_crs,
            city.streets,
            boundary,
            cve_code=city.cve_code,
            workdir=context.temp_dir / "osm",
            refresh=parameters.refresh_osm,
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
    lines = network.shapes(nodes, edges)
    return {
        "reimagina_street_network_nodes": nodes.reset_index(),
        "reimagina_street_network_edges": gpd.GeoDataFrame(
            edges[["from", "to"]]
            .assign(distance=edges["length"], two_way=True)
            .join(edges.reindex(columns=list(network.OSM_TAGS))),
            geometry=lines,
            crs=city.target_crs,
        ),
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
        "reimagina_street_network_tables_edges": STREET_EDGES_DECLARATION,
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
