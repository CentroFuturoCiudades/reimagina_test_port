"""Lyra nodes over a Reimagina scenario: reimagina_scenario (its lots and blocks),
reimagina_scenario_pois and reimagina_road_network (accessibility inputs), and the
writes reimagina_scenario_indicators, reimagina_scenario_accessibility,
reimagina_scenario_population_access and reimagina_scenario_write."""

import logging
from typing import Any

import geopandas as gpd
import pandas as pd
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

from reimagina_test_port.common.nodes import timed
from reimagina_test_port.common.schema import (
    ATTRIBUTES,
    NUMERIC_ATTRIBUTES,
    UNITS,
    source_declaration,
)
from reimagina_test_port.common.vocabulary import DESTINATION_TYPES
from reimagina_test_port.indicators.accessibility import network
from reimagina_test_port.indicators.accessibility.nodes import (
    ACCESSIBILITY_KEYS,
    EDGES_DECLARATION,
    NODES_DECLARATION,
    POIS_DECLARATION,
)
from reimagina_test_port.indicators.population_access.nodes import PoiType
from reimagina_test_port.scenario import source, writer

logger = logging.getLogger(__name__)


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
        tables = source.read_scenario(scenario_id, ATTRIBUTES)
    except source.ScenarioNotFoundError as exc:
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
    name="reimagina_scenario_pois",
    description=(
        "Acquire a scenario's present amenities and establishments, one row per "
        "active poi_catalog type whose rule they meet."
    ),
    outputs={"reimagina_scenario_pois": POIS_DECLARATION},
)
def scenario_pois(
    parameters: ScenarioParameters, context: DatasetContext
) -> gpd.GeoDataFrame:
    """Destinations of the accessibility model, with baseline fallback."""
    try:
        with timed("read pois", context.logger):
            return source.read_pois(parameters.scenario_id)
    except source.ScenarioNotFoundError as exc:
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
        uri = source.read_road_network_uri(parameters.scenario_id)
    except source.ScenarioNotFoundError as exc:
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


class PopulationAccessWriteParameters(WriteParameters):
    """Scenario to write, and the poi_catalog types to write."""

    poi_types: list[PoiType] | None = Field(
        default=None, description="Types to write; all when omitted."
    )


@metric(
    name="reimagina_scenario_population_access",
    description=(
        "Compute a scenario's population access and write it to Reimagina's "
        "element_state_indicator_value, under the same rules as "
        "reimagina_scenario_indicators. Only blocks are written, and only the keys "
        "of poi_types, so a partial run keeps the stored values of the other types."
    ),
    inputs={"blocks": Dependency("reimagina_population_access_blocks")},
    outputs={"reimagina_scenario_population_access": WRITE_DECLARATION},
)
def scenario_population_access(
    parameters: PopulationAccessWriteParameters,
    blocks: gpd.GeoDataFrame,
    context: RunContext,
) -> pd.DataFrame:
    """Block values of the requested types, in their own transaction."""
    poi_types = (
        DESTINATION_TYPES if parameters.poi_types is None else parameters.poi_types
    )
    keys = {
        f"{prefix}{poi_type}": "number"
        for poi_type in poi_types
        for prefix in ("distance_m_", "population_with_access_")
    }
    return write_group(
        parameters,
        {"block": blocks},
        keys,
        node="reimagina_scenario_population_access",
        log=context.logger,
    )


@metric(
    name="reimagina_scenario_write",
    description=(
        "Compute and write a scenario's built-form indicators, accessibility and "
        "population access: reimagina_scenario_indicators, "
        "reimagina_scenario_accessibility and "
        "reimagina_scenario_population_access, each in its own transaction. The "
        "result adds up all three."
    ),
    inputs={
        "indicators": Dependency("reimagina_scenario_indicators"),
        "accessibility": Dependency("reimagina_scenario_accessibility"),
        "population_access": Dependency("reimagina_scenario_population_access"),
    },
    outputs={"reimagina_scenario_write": WRITE_DECLARATION},
)
def scenario_write(
    parameters: WriteParameters,
    indicators: pd.DataFrame,
    accessibility: pd.DataFrame,
    population_access: pd.DataFrame,
) -> pd.DataFrame:
    """Final node: the writes' summary in one row."""
    writes = pd.concat([indicators, accessibility, population_access])
    return pd.DataFrame(
        [
            {
                "elements": int(indicators["elements"].iloc[0]),
                "values_written": int(writes["values_written"].sum()),
                "values_cleared": int(writes["values_cleared"].sum()),
                "dry_run": parameters.dry_run,
            }
        ]
    )
