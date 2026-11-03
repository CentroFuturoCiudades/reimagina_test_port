"""Lyra nodes that build a city's baseline from the data lake and write it into a
Reimagina baseline scenario: reimagina_baseline, reimagina_baseline_elements and
reimagina_baseline_write. The elements are blocks, lots, amenities and
establishments."""

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
    units = {"capacity": Unit.COUNT, "employees": Unit.COUNT}
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
        "Build a city's baseline elements from the CFC data lake with "
        "exe_reimagina's rules: INEGI blocks and census, cadastral lots, "
        "normativa, Open Buildings heights, the equipment layer and the DENUE. "
        "Blocks are keyed by CVEGEO; lots by their cadastral key; amenities "
        "(equipment) and establishments (DENUE) hang from the lot holding them."
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
    "block": {"vivtot": "number", "people_per_household": "number"},
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
        "unchanged; running it again replaces the scenario's rows. Every attribute "
        "must be active in attribute_catalog. Compute indicators afterwards with "
        "reimagina_scenario_write."
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
