"""Domain thresholds that ingestion and indicators must agree on."""

MIN_BUILT_M2 = 20  # a single-family lot with less built area counts as empty

# A lot's land_use values (reimagina_urbano_api urban_fabric.md).
LAND_USES = (
    "single_family", "multifamily", "mixed_use", "commerce", "office", "industrial",
    "amenity",
)  # fmt: skip
