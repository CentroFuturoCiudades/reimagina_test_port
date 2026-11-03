"""Scenario and snapshot acquisition and the public built-form and accessibility graph.

Production reads a Reimagina scenario from Postgres and its road network from the
data lake. The bundled snapshot datasets feed the same metrics offline
(``create_snapshot_plugin``).
"""

import logging
import os
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from lyra.sdk import (
    DatasetContext,
    Dependency,
    MetricParameters,
    PluginDefinition,
    RunContext,
    TableColumn,
    TableOutput,
    Unit,
    dataset,
    metric,
)
from lyra.sdk.errors import MetricInputError
from pydantic import Field

from reimagina_test_port import scenario, writer
from reimagina_test_port.common.nodes import timed
from reimagina_test_port.common.schema import (
    ATTRIBUTES,
    IDENTITY,
    NUMERIC_ATTRIBUTES,
    UNITS,
    source_declaration,
)
from reimagina_test_port.indicators.accessibility import network
from reimagina_test_port.indicators.accessibility.nodes import (
    ACCESSIBILITY_KEYS,
    EDGES_DECLARATION,
    NODES_DECLARATION,
    POIS_DECLARATION,
    accessibility_metric,
)
from reimagina_test_port.indicators.built_form import nodes as built_form_nodes
from reimagina_test_port.ingestion import nodes as ingestion_nodes

logger = logging.getLogger(__name__)


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
        "reimagina_scenario_lots": source_declaration(),
        "reimagina_scenario_blocks": source_declaration(blocks=True),
    },
)
def scenario_dataset(
    parameters: ScenarioParameters, context: DatasetContext
) -> dict[str, gpd.GeoDataFrame]:
    """Acquire both tables once per graph invocation."""
    with timed("read scenario", context.logger):
        return read_scenario(parameters.scenario_id)


@dataset(
    name="reimagina_prepared_snapshot",
    description="Acquire prepared lot and block GeoParquet tables.",
    outputs={
        "reimagina_prepared_lots": source_declaration(),
        "reimagina_prepared_blocks": source_declaration(blocks=True),
    },
)
def prepared_snapshot(parameters: SnapshotParameters) -> dict[str, gpd.GeoDataFrame]:
    """Acquire both files once per graph invocation."""
    return read_snapshot(parameters.snapshot)


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
def scenario_pois(
    parameters: ScenarioParameters, context: DatasetContext
) -> gpd.GeoDataFrame:
    """Destinations of the accessibility model, with baseline fallback."""
    try:
        with timed("read pois", context.logger):
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
    with timed("road network", context.logger):
        tables = network.read_network(network.download(uri, context.temp_dir))
    return {
        "reimagina_road_network_nodes": tables["nodes"],
        "reimagina_road_network_edges": tables["edges"],
    }


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
    full: bool = Field(
        default=False,
        description=(
            "Send every value instead of only those a fork may have changed. "
            "Slower; for after the baseline's values were computed another way."
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
BUILT_FORM_KEYS = dict.fromkeys(UNITS, "number")
ACCESSIBILITY_WRITE_KEYS = {
    key: "text" if key == "farthest_amenity" else "number" for key in ACCESSIBILITY_KEYS
}


def write_group(
    parameters: WriteParameters,
    tables: dict[str, gpd.GeoDataFrame],
    keys: dict[str, str],
    *,
    node: str,
    log: Any = logger,
) -> pd.DataFrame:
    """Write one group of keys (built-form or accessibility) in its own transaction.

    ``tables`` maps each element type written (``lot``, ``block``) to its values.
    """
    with timed("prepare values", log):
        values = pd.concat(
            [
                writer.long_values(table, element_type, keys)
                for element_type, table in tables.items()
            ],
            ignore_index=True,
        )
    try:
        written, cleared = writer.write_indicators(
            parameters.scenario_id,
            values,
            keys,
            full=parameters.full,
            job_run_id=parameters.job_run_id,
            dry_run=parameters.dry_run,
            log=log,
        )
    except writer.ScenarioNotFoundError as exc:
        raise MetricInputError(node, "parameters.scenario_id", str(exc)) from exc
    return pd.DataFrame(
        [
            {
                "elements": sum(len(table) for table in tables.values()),
                "values_written": written,
                "values_cleared": cleared,
                "dry_run": parameters.dry_run,
            }
        ]
    )


@metric(
    name="reimagina_scenario_indicators",
    description=(
        "Compute a scenario's built-form indicators and write them to Reimagina's "
        "element_state_indicator_value. Only keys active in indicator_catalog are "
        "written; a fork stores only values that differ from its baseline; "
        "unchanged values are not rewritten, so repeating it changes nothing. "
        "Accessibility values are not touched."
    ),
    inputs={
        "lots": Dependency("reimagina_built_form_lots"),
        "blocks": Dependency("reimagina_built_form_blocks"),
    },
    outputs={"reimagina_scenario_indicators": WRITE_DECLARATION},
)
def scenario_indicators(
    parameters: WriteParameters,
    lots: gpd.GeoDataFrame,
    blocks: gpd.GeoDataFrame,
    context: RunContext,
) -> pd.DataFrame:
    """Built-form values, in their own transaction."""
    return write_group(
        parameters,
        {"lot": lots, "block": blocks},
        BUILT_FORM_KEYS,
        node="reimagina_scenario_indicators",
        log=context.logger,
    )


@metric(
    name="reimagina_scenario_accessibility",
    description=(
        "Compute a scenario's accessibility and write it to Reimagina's "
        "element_state_indicator_value, under the same rules as "
        "reimagina_scenario_indicators. Only blocks are written: a lot's "
        "accessibility is its block's. Built-form values are not touched."
    ),
    inputs={"blocks": Dependency("reimagina_accessibility_blocks")},
    outputs={"reimagina_scenario_accessibility": WRITE_DECLARATION},
)
def scenario_accessibility(
    parameters: WriteParameters,
    blocks: gpd.GeoDataFrame,
    context: RunContext,
) -> pd.DataFrame:
    """Block accessibility values, in their own transaction."""
    return write_group(
        parameters,
        {"block": blocks},
        ACCESSIBILITY_WRITE_KEYS,
        node="reimagina_scenario_accessibility",
        log=context.logger,
    )


@metric(
    name="reimagina_scenario_write",
    description=(
        "Compute and write a scenario's built-form indicators and accessibility, "
        "as exe_reimagina's model job does: reimagina_scenario_indicators and "
        "reimagina_scenario_accessibility, each in its own transaction. The result "
        "adds up both."
    ),
    inputs={
        "indicators": Dependency("reimagina_scenario_indicators"),
        "accessibility": Dependency("reimagina_scenario_accessibility"),
    },
    outputs={"reimagina_scenario_write": WRITE_DECLARATION},
)
def scenario_write(
    parameters: WriteParameters,
    indicators: pd.DataFrame,
    accessibility: pd.DataFrame,
) -> pd.DataFrame:
    """Final node: both writes' summary in one row."""
    first, second = indicators.iloc[0], accessibility.iloc[0]
    return pd.DataFrame(
        [
            {
                "elements": int(first["elements"]),
                "values_written": int(
                    first["values_written"] + second["values_written"]
                ),
                "values_cleared": int(
                    first["values_cleared"] + second["values_cleared"]
                ),
                "dry_run": parameters.dry_run,
            }
        ]
    )


def create_plugin() -> PluginDefinition:
    """Production graph over Reimagina scenarios; registers without connecting."""
    return PluginDefinition(
        nodes=[
            scenario_dataset,
            scenario_pois,
            road_network,
            *built_form_nodes.built_form_metrics(
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
            scenario_indicators,
            scenario_accessibility,
            scenario_write,
            ingestion_nodes.baseline_dataset,
            ingestion_nodes.baseline_elements,
            ingestion_nodes.baseline_write,
        ]
    )


def create_snapshot_plugin() -> PluginDefinition:
    """The same metrics over bundled or prepared snapshots, for offline runs."""
    return PluginDefinition(
        nodes=[
            prepared_snapshot,
            prepared_accessibility,
            *built_form_nodes.built_form_metrics(
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
