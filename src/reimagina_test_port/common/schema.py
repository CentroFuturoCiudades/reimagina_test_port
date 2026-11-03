"""Element schema shared by ingestion and indicators: identity columns, the lot
attributes indicators read, and the units of attributes and built-form keys."""

from lyra.sdk import TableColumn, Unit

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


def identity_columns() -> list[TableColumn]:
    """``id`` (required), ``parent_id`` and ``source_record_ref``."""
    return [
        TableColumn(
            name=k,
            type="string",
            unit=None,
            nullable=k != "id",
            description=k.replace("_", " "),
        )
        for k in IDENTITY
    ]
