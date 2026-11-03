"""Lyra nodes that build a city's baseline from the data lake and write it into a
Reimagina baseline scenario: reimagina_baseline, reimagina_baseline_elements and
reimagina_baseline_write."""

from typing import Literal

import geopandas as gpd
import pandas as pd
from lyra.sdk import (
    DatasetContext,
    Dependency,
    MetricParameters,
    TableColumn,
    TableOutput,
    Unit,
    dataset,
    metric,
)
from lyra.sdk.errors import MetricInputError
from pydantic import Field

from reimagina_test_port.common.nodes import column
from reimagina_test_port.common.schema import UNITS, identity_columns


class BaselineParameters(MetricParameters):
    """City whose baseline is built from the data lake."""

    city: Literal["mexicali"] = Field(description="City preset.")


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
    from reimagina_test_port.ingestion import baseline
    from reimagina_test_port.ingestion.cities import CITIES

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
    from reimagina_test_port.ingestion import baseline_writer

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
