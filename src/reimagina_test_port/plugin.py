"""Scenario and snapshot acquisition and the public built-form and accessibility graph.

Production reads a Reimagina scenario from Postgres and its road network from the
data lake. The bundled snapshot datasets feed the same metrics offline
(``create_snapshot_plugin``).
"""

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import pandas as pd
from lyra.sdk import (
    DatasetContext,
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

from reimagina_test_port import network, scenario, writer
from reimagina_test_port.calculation import block_indicators, lot_indicators
from reimagina_test_port.vocabulary import AMENITY_TYPES

logger = logging.getLogger(__name__)


@contextmanager
def timed(stage: str) -> Iterator[None]:
    """Log how long a stage of a job took."""
    start = time.perf_counter()
    yield
    logger.info("%s: %.2fs", stage, time.perf_counter() - start)


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
    # viv/ha, as indicator_catalog; Lyra has no count/ha unit.
    "density": None,
}
NUMERIC_ATTRIBUTES = [key for key in ATTRIBUTES if key != "land_use"]
SCORE_KEYS = [f"accessibility_{amenity}" for amenity in AMENITY_TYPES]
DISTANCE_KEYS = [f"distance_to_{amenity}" for amenity in AMENITY_TYPES]
ACCESSIBILITY_KEYS = ["minutes", "farthest_amenity", *SCORE_KEYS, *DISTANCE_KEYS]


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


def column(name: str, kind: str, unit: Unit | None, description: str) -> TableColumn:
    """Nullable column."""
    return TableColumn(
        name=name, type=kind, unit=unit, nullable=True, description=description
    )


def accessibility_declaration() -> TableOutput:
    """Identity, accessibility keys (as indicator_catalog) and geometry."""
    identity = declaration(source=True, blocks=True).columns
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


def snapshot_folder(name: str) -> Path:
    """The snapshot's folder, directly beneath the configured root."""
    root = Path(
        os.environ.get("REIMAGINA_SNAPSHOT_ROOT", str(Path(__file__).parent / "data"))
    ).resolve()
    folder = (root / name).resolve()
    if folder.parent != root:
        raise snapshot_error("Snapshot must be directly beneath the configured root")
    return folder


def read_snapshot(name: str) -> dict[str, gpd.GeoDataFrame]:
    """Read prepared tables without changing source identity or area values."""
    folder = snapshot_folder(name)
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


class ScenarioParameters(MetricParameters):
    """Select a Reimagina scenario; forks fall back to their baseline."""

    scenario_id: str = Field(
        pattern=r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
        description="Reimagina scenario UUID.",
    )


def scenario_error(message: str) -> MetricInputError:
    """Attach source-node and parameter context to scenario failures."""
    return MetricInputError("reimagina_scenario", "parameters.scenario_id", message)


def read_scenario(scenario_id: str) -> dict[str, gpd.GeoDataFrame]:
    """Read effective lots and blocks in the snapshot table contract."""
    try:
        tables = scenario.read_scenario(scenario_id, ATTRIBUTES)
    except scenario.ScenarioNotFoundError as exc:
        raise scenario_error(str(exc)) from exc
    if tables["lot"].empty and tables["block"].empty:
        raise scenario_error(f"No block or lot elements for scenario {scenario_id!r}")
    lots = tables["lot"]
    lots[NUMERIC_ATTRIBUTES] = lots[NUMERIC_ATTRIBUTES].astype("float64")
    return {
        "reimagina_scenario_lots": lots,
        "reimagina_scenario_blocks": tables["block"],
    }


@dataset(
    name="reimagina_scenario",
    description="Acquire a scenario's effective lots and blocks from Reimagina.",
    outputs={
        "reimagina_scenario_lots": declaration(source=True),
        "reimagina_scenario_blocks": declaration(source=True, blocks=True),
    },
)
def scenario_dataset(parameters: ScenarioParameters) -> dict[str, gpd.GeoDataFrame]:
    """Acquire both tables once per graph invocation."""
    with timed("read scenario"):
        return read_scenario(parameters.scenario_id)


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

    def lot_handler(parameters, lots: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        with timed("lot built-form"):
            return lot_built_form(lots)

    def block_handler(
        parameters, lots: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame
    ) -> dict[str, gpd.GeoDataFrame]:
        with timed("block built-form"):
            return built_form(lots, blocks)

    lot_handler.__annotations__["parameters"] = parameters_type
    block_handler.__annotations__["parameters"] = parameters_type
    return [
        metric(
            name="reimagina_lot_built_form",
            description="Calculate built-form indicators for every lot.",
            inputs={"lots": Dependency(lots_output)},
            outputs={"reimagina_lot_indicators": declaration()},
        )(lot_handler),
        metric(
            name="reimagina_built_form",
            description="Return lot indicators and their block aggregates.",
            inputs={
                "lots": Dependency("reimagina_lot_indicators"),
                "blocks": Dependency(blocks_output),
            },
            outputs={
                "reimagina_built_form_lots": declaration(),
                "reimagina_built_form_blocks": declaration(),
            },
        )(block_handler),
    ]


def read_snapshot_accessibility(name: str) -> dict[str, pd.DataFrame]:
    """POIs and road network tables of a prepared snapshot."""
    folder = snapshot_folder(name)
    try:
        pois = gpd.read_parquet(folder / "poi.parquet")
        nodes = pd.read_parquet(folder / "network_nodes.parquet")
        edges = pd.read_parquet(folder / "network_edges.parquet")
    except (OSError, ValueError) as exc:
        raise MetricInputError(
            "reimagina_prepared_accessibility",
            "parameters.snapshot",
            f"Cannot read accessibility inputs for snapshot {name!r}: {exc}",
        ) from exc
    return {
        "reimagina_prepared_pois": pois,
        "reimagina_prepared_network_nodes": nodes,
        "reimagina_prepared_network_edges": edges,
    }


@dataset(
    name="reimagina_prepared_accessibility",
    description="Acquire a prepared snapshot's POIs and road network tables.",
    outputs={
        "reimagina_prepared_pois": POIS_DECLARATION,
        "reimagina_prepared_network_nodes": NODES_DECLARATION,
        "reimagina_prepared_network_edges": EDGES_DECLARATION,
    },
)
def prepared_accessibility(parameters: SnapshotParameters) -> dict[str, pd.DataFrame]:
    """Acquire the three files once per graph invocation."""
    return read_snapshot_accessibility(parameters.snapshot)


@dataset(
    name="reimagina_scenario_pois",
    description="Acquire a scenario's present elements with an amenity_type.",
    outputs={"reimagina_scenario_pois": POIS_DECLARATION},
)
def scenario_pois(parameters: ScenarioParameters) -> gpd.GeoDataFrame:
    """Destinations of the accessibility model, with baseline fallback."""
    try:
        with timed("read pois"):
            return scenario.read_pois(parameters.scenario_id)
    except scenario.ScenarioNotFoundError as exc:
        raise MetricInputError(
            "reimagina_scenario_pois", "parameters.scenario_id", str(exc)
        ) from exc


@dataset(
    name="reimagina_road_network",
    description=(
        "Acquire the road network H5 of the scenario's project "
        "(projects.road_network_uri) as node and edge tables."
    ),
    outputs={
        "reimagina_road_network_nodes": NODES_DECLARATION,
        "reimagina_road_network_edges": EDGES_DECLARATION,
    },
)
def road_network(
    parameters: ScenarioParameters, context: DatasetContext
) -> dict[str, pd.DataFrame]:
    """Download into the job's scratch folder and read the H5."""

    def error(message: str) -> MetricInputError:
        return MetricInputError(
            "reimagina_road_network", "parameters.scenario_id", message
        )

    try:
        uri = scenario.read_road_network_uri(parameters.scenario_id)
    except scenario.ScenarioNotFoundError as exc:
        raise error(str(exc)) from exc
    if uri is None:
        raise error(
            f"The project of scenario {parameters.scenario_id} has no road_network_uri."
        )
    with timed("road network"):
        tables = network.read_network(network.download(uri, context.temp_dir))
    return {
        "reimagina_road_network_nodes": tables["nodes"],
        "reimagina_road_network_edges": tables["edges"],
    }


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
) -> dict[str, gpd.GeoDataFrame]:
    """Per block, inherited by each lot of that block."""
    # pandana only where accessibility runs.
    from reimagina_test_port import accessibility

    with timed("accessibility"):
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
    ) -> dict[str, gpd.GeoDataFrame]:
        return accessibility_tables(lots, blocks, pois, nodes, edges)

    handler.__annotations__["parameters"] = parameters_type
    return metric(
        name="reimagina_accessibility",
        description=(
            "Walking accessibility per block, inherited by its lots (exe_reimagina "
            "model job). Blocks are origins weighted by the scenario's modelled "
            "population; destinations are elements with an amenity_type. Up to five "
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


class WriteParameters(ScenarioParameters):
    """Scenario to write, with exe_reimagina's job options."""

    dry_run: bool = Field(
        default=False,
        description="Compute and count the changes, then roll them back.",
    )
    job_run_id: str | None = Field(
        default=None,
        description=(
            "UUID of the Reimagina job_run that requested this run; recorded on "
            "written values when that job_run exists."
        ),
    )


WRITE_DECLARATION = TableOutput(
    columns=[
        TableColumn(
            name="elements",
            type="integer",
            unit=Unit.COUNT,
            description="Blocks and lots of the scenario.",
        ),
        TableColumn(
            name="values_written",
            type="integer",
            unit=Unit.COUNT,
            description=(
                "Values inserted or changed; unchanged values are not rewritten."
            ),
        ),
        TableColumn(
            name="values_cleared",
            type="integer",
            unit=Unit.COUNT,
            description=(
                "Scenario values removed because they match the baseline, are "
                "empty in a baseline scenario, or belong to an absent element."
            ),
        ),
        TableColumn(
            name="dry_run",
            type="boolean",
            unit=None,
            description="Whether the changes were rolled back.",
        ),
    ]
)


def write_scenario(
    parameters: WriteParameters,
    tables: dict[str, gpd.GeoDataFrame],
) -> pd.DataFrame:
    """Write built-form and accessibility values in one transaction."""
    keys = {
        key: "text" if key == "farthest_amenity" else "number"
        for key in [*UNITS, *ACCESSIBILITY_KEYS]
    }
    with timed("prepare values"):
        elements, values = writer.element_values(
            [
                ("lot", tables["built_form_lots"]),
                ("lot", tables["accessibility_lots"]),
                ("block", tables["built_form_blocks"]),
                ("block", tables["accessibility_blocks"]),
            ],
            keys,
        )
    try:
        written, cleared = writer.write_indicators(
            parameters.scenario_id,
            values,
            elements,
            keys,
            job_run_id=parameters.job_run_id,
            dry_run=parameters.dry_run,
        )
    except writer.ScenarioNotFoundError as exc:
        raise MetricInputError(
            "reimagina_scenario_write", "parameters.scenario_id", str(exc)
        ) from exc
    return pd.DataFrame(
        [
            {
                "elements": len(elements),
                "values_written": written,
                "values_cleared": cleared,
                "dry_run": parameters.dry_run,
            }
        ]
    )


@metric(
    name="reimagina_scenario_write",
    description=(
        "Compute a scenario's built-form and accessibility indicators and write "
        "them to Reimagina's element_state_indicator_value in one transaction, as "
        "exe_reimagina's model job does. Only keys active in indicator_catalog are "
        "written; a fork stores only values that differ from its baseline; "
        "unchanged values are not rewritten, so repeating it changes nothing."
    ),
    inputs={
        "built_form_lots": Dependency("reimagina_built_form_lots"),
        "built_form_blocks": Dependency("reimagina_built_form_blocks"),
        "accessibility_lots": Dependency("reimagina_accessibility_lots"),
        "accessibility_blocks": Dependency("reimagina_accessibility_blocks"),
    },
    outputs={"reimagina_scenario_write": WRITE_DECLARATION},
)
def scenario_write(
    parameters: WriteParameters,
    built_form_lots: gpd.GeoDataFrame,
    built_form_blocks: gpd.GeoDataFrame,
    accessibility_lots: gpd.GeoDataFrame,
    accessibility_blocks: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Final node: the scenario's summary row."""
    return write_scenario(
        parameters,
        {
            "built_form_lots": built_form_lots,
            "built_form_blocks": built_form_blocks,
            "accessibility_lots": accessibility_lots,
            "accessibility_blocks": accessibility_blocks,
        },
    )


class BaselineParameters(MetricParameters):
    """City whose baseline is built from the data lake."""

    city: Literal["mexicali"] = Field(description="City preset.")


def baseline_declaration(*, blocks: bool) -> TableOutput:
    """Identity, element attributes and EPSG:4326 geometry."""
    identity = declaration(source=True, blocks=True).columns
    if blocks:
        attributes = [
            column("vivtot", "number", Unit.COUNT, "Census dwellings of the block."),
            column(
                "people_per_household",
                "number",
                Unit.COUNT,
                "Census average occupants per inhabited dwelling.",
            ),
        ]
    else:
        units = {"people_per_household": Unit.COUNT, "occupancy_rate": Unit.RATIO}
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


TEXT_ATTRIBUTES = {"land_use", "amenity_type", "commerce_type"}
BASELINE_LOT_ATTRIBUTES = [
    "land_use",
    "cos",
    "num_levels",
    "area",
    "unit_size_m2",
    "people_per_household",
    "occupancy_rate",
    "amenity_type",
    "commerce_type",
]


@dataset(
    name="reimagina_baseline",
    description=(
        "Build a city's baseline lots and blocks from the CFC data lake with "
        "exe_reimagina's rules: INEGI blocks and census, cadastral lots, "
        "normativa, Open Buildings heights and amenity POIs. Blocks are keyed by "
        "CVEGEO; lots by their cadastral key."
    ),
    outputs={
        "reimagina_baseline_lots": baseline_declaration(blocks=False),
        "reimagina_baseline_blocks": baseline_declaration(blocks=True),
    },
)
def baseline_dataset(
    parameters: BaselineParameters, context: DatasetContext
) -> dict[str, gpd.GeoDataFrame]:
    """Download the sources into the job's scratch folder and build the tables."""
    # rasterio only where the baseline is built.
    from reimagina_test_port import baseline
    from reimagina_test_port.cities import CITIES

    def progress(stage: str, current: int, total: int) -> None:
        context.report_progress(stage=stage, current=current, total=total)

    result = baseline.build(CITIES[parameters.city], context.temp_dir, progress)
    context.logger.info("Cadastral keys: %s", result["report"])
    return {
        "reimagina_baseline_lots": result["lots"],
        "reimagina_baseline_blocks": result["blocks"],
    }


@metric(
    name="reimagina_baseline_elements",
    description=(
        "Return a city's baseline lots and blocks as built from the data lake "
        "(reimagina_baseline). Nothing is written to Reimagina's database."
    ),
    inputs={
        "lots": Dependency("reimagina_baseline_lots"),
        "blocks": Dependency("reimagina_baseline_blocks"),
    },
    outputs={
        "reimagina_baseline_elements_lots": baseline_declaration(blocks=False),
        "reimagina_baseline_elements_blocks": baseline_declaration(blocks=True),
    },
)
def baseline_elements(
    parameters: BaselineParameters,
    lots: gpd.GeoDataFrame,
    blocks: gpd.GeoDataFrame,
) -> dict[str, gpd.GeoDataFrame]:
    """The dataset's tables."""
    return {
        "reimagina_baseline_elements_lots": lots,
        "reimagina_baseline_elements_blocks": blocks,
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
    "block": {"vivtot": "number", "people_per_household": "number"},
}
BASELINE_WRITE_DECLARATION = TableOutput(
    columns=[
        TableColumn(name=name, type="integer", unit=Unit.COUNT, description=description)
        for name, description in (
            ("elements", "Blocks and lots written."),
            ("elements_created", "Elements new to the database."),
            ("attribute_values", "Attribute values written."),
            ("elements_removed", "Scenario elements no longer in the baseline."),
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
        "attribute values, in one transaction. Existing elements are reused "
        "unchanged; running it again replaces the scenario's rows. Every attribute "
        "must be active in attribute_catalog. Compute indicators afterwards with "
        "reimagina_scenario_write."
    ),
    inputs={
        "lots": Dependency("reimagina_baseline_lots"),
        "blocks": Dependency("reimagina_baseline_blocks"),
    },
    outputs={"reimagina_baseline_write": BASELINE_WRITE_DECLARATION},
)
def baseline_write(
    parameters: BaselineWriteParameters,
    lots: gpd.GeoDataFrame,
    blocks: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Final node: one summary row."""
    from reimagina_test_port import baseline_writer

    try:
        counts = baseline_writer.write_baseline(
            parameters.project_id,
            parameters.scenario_id,
            {"lot": lots, "block": blocks},
            BASELINE_ATTRIBUTES,
            dry_run=parameters.dry_run,
        )
    except baseline_writer.BaselineScenarioError as exc:
        raise MetricInputError(
            "reimagina_baseline_write", "parameters.scenario_id", str(exc)
        ) from exc
    return pd.DataFrame([{**counts, "dry_run": parameters.dry_run}])


def create_plugin() -> PluginDefinition:
    """Production graph over Reimagina scenarios; registers without connecting."""
    return PluginDefinition(
        nodes=[
            scenario_dataset,
            scenario_pois,
            road_network,
            *built_form_metrics(
                ScenarioParameters,
                "reimagina_scenario_lots",
                "reimagina_scenario_blocks",
            ),
            accessibility_metric(
                ScenarioParameters,
                "reimagina_scenario_pois",
                "reimagina_road_network_nodes",
                "reimagina_road_network_edges",
            ),
            scenario_write,
            baseline_dataset,
            baseline_elements,
            baseline_write,
        ]
    )


def create_snapshot_plugin() -> PluginDefinition:
    """The same metrics over bundled or prepared snapshots, for offline runs."""
    return PluginDefinition(
        nodes=[
            prepared_snapshot,
            prepared_accessibility,
            *built_form_metrics(
                SnapshotParameters,
                "reimagina_prepared_lots",
                "reimagina_prepared_blocks",
            ),
            accessibility_metric(
                SnapshotParameters,
                "reimagina_prepared_pois",
                "reimagina_prepared_network_nodes",
                "reimagina_prepared_network_edges",
            ),
        ]
    )
