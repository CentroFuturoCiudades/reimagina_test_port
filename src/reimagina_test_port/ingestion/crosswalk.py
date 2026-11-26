"""SCIAN code -> element type (amenity or establishment) and category.

The platform's crosswalk (reimagina_urbano_api ``urban_fabric.md``, *Platform
configuration*), from the codes the team's DENUE crosswalk (``denue_crosswalk.xlsx``,
November 2026) lists under *What does this include specifically?*:
``crosswalk.csv``, one SCIAN prefix per row, the longest prefix winning (``461121``
before ``4611``). The rest of each SCIAN sector (its two digits) is a category of
its own (``wholesale``, ``professional_services``…), so that no record loses its
sector; a code no prefix matches, not a SCIAN code, is an establishment of
category ``other``. Each DENUE record is one or the other, never both.

A category is the finest kind of place the platform tells apart (a bakery, a
daycare, a library), the one attribute that says what a place is: ``poi_catalog``
rules and the diversity sets group categories, and a place counts in every group
that lists its category (a school of several levels counts in each level).
``ECONOMIC_SECTORS`` gives each category its SCIAN sector, so that a place whose
category changes changes sector.
"""

import csv
from functools import cache
from pathlib import Path

import pandas as pd

PATH = Path(__file__).with_name("crosswalk.csv")

AMENITY_CATEGORIES = (
    "preschool", "primary_education", "secondary_education", "upper_secondary_education",
    "tertiary_education", "multilevel_school", "special_education", "health_center",
    "clinic", "specialist_clinic", "hospital", "residence", "daycare",
    "social_assistance", "community_food", "community_center", "religious",
    "civic_association", "social_welfare", "cultural_center", "museum", "library",
    "sports", "sports_school", "recreation_venue", "park", "police_station",
    "fire_station",
)  # fmt: skip
ESTABLISHMENT_CATEGORIES = (
    "supermarket", "convenience_store", "grocery", "butcher", "poultry", "fishmonger",
    "greengrocer", "grains_spices", "dairy", "bakery", "tortilleria", "other_food",
    "basic_household_goods", "pharmacy", "financial", "petrol_station", "cinema", "sports",
    "sports_school", "manufacturing_food_textile", "manufacturing_materials",
    "manufacturing_metal_machinery", "agriculture", "mining", "utilities",
    "construction", "wholesale", "other_retail", "transport", "postal_storage", "media", "other_financial", "real_estate",
    "professional_services", "corporate", "business_support", "other_education",
    "other_health", "other_recreation", "food_accommodation", "other_services",
    "government", "other",
)  # fmt: skip
CATEGORIES = {"amenity": AMENITY_CATEGORIES, "establishment": ESTABLISHMENT_CATEGORIES}
# Manufacturing (SCIAN 31-33), by subsector: a lot mostly of it is industrial.
MANUFACTURING_CATEGORIES = (
    "manufacturing_food_textile", "manufacturing_materials",
    "manufacturing_metal_machinery",
)  # fmt: skip
# The SCIAN sector (its two digits) of each category, as the ULI's job diversity
# counts them (uli_mexicali_calculations_notebooks, add_economic_sector_diversity:
# 31, 32 and 33, and 48 and 49, apart). A category from the DENUE is in the sector
# of its codes; the cities' own layers' as the DENUE would place them (a cultural
# centre 71, a police station 93). A park and ``other`` have none.
ECONOMIC_SECTORS = {
    **dict.fromkeys(
        ("preschool", "primary_education", "secondary_education",
         "upper_secondary_education", "tertiary_education", "multilevel_school",
         "special_education", "sports_school", "other_education"),
        "61",
    ),
    **dict.fromkeys(
        ("health_center", "clinic", "specialist_clinic", "hospital", "residence",
         "daycare", "social_assistance", "community_food", "community_center",
         "other_health"),
        "62",
    ),
    **dict.fromkeys(
        ("cultural_center", "museum", "sports", "recreation_venue", "other_recreation"),
        "71",
    ),
    **dict.fromkeys(("library", "cinema", "media"), "51"),
    **dict.fromkeys(("religious", "civic_association", "other_services"), "81"),
    **dict.fromkeys(
        ("police_station", "fire_station", "social_welfare", "government"), "93"
    ),
    **dict.fromkeys(
        ("supermarket", "convenience_store", "grocery", "butcher", "poultry",
         "fishmonger", "greengrocer", "grains_spices", "dairy", "other_food",
         "basic_household_goods", "pharmacy", "petrol_station", "other_retail"),
        "46",
    ),
    **dict.fromkeys(("bakery", "tortilleria", "manufacturing_food_textile"), "31"),
    "manufacturing_materials": "32",
    "manufacturing_metal_machinery": "33",
    **dict.fromkeys(("financial", "other_financial"), "52"),
    "agriculture": "11",
    "mining": "21",
    "utilities": "22",
    "construction": "23",
    "wholesale": "43",
    "transport": "48",
    "postal_storage": "49",
    "real_estate": "53",
    "professional_services": "54",
    "corporate": "55",
    "business_support": "56",
    "food_accommodation": "72",
    "park": None,
    "other": None,
}  # fmt: skip


@cache
def table(path: Path = PATH) -> tuple[tuple[str, str, str], ...]:
    """(prefix, element, category) rows, longest prefix first."""
    with path.open(encoding="utf-8", newline="") as file:
        rows = [
            (row["scian_prefix"], row["element"], row["category"])
            for row in csv.DictReader(file)
        ]
    for prefix, element, category in rows:
        if category not in CATEGORIES.get(element, ()):
            msg = f"{path.name}: {prefix} -> {element} {category!r} is not a category"
            raise ValueError(msg)
    if repeated := sorted({p for p, *_ in rows if [r[0] for r in rows].count(p) > 1}):
        msg = f"{path.name}: prefixes repeated: {repeated}"
        raise ValueError(msg)
    return tuple(sorted(rows, key=lambda row: -len(row[0])))


def classify(scian_codes: pd.Series) -> pd.DataFrame:
    """``element`` and ``category`` of each SCIAN code (``other`` establishments
    when no prefix, not even its sector's, matches)."""
    codes = scian_codes.astype(str).str.strip()
    element = pd.Series(None, index=codes.index, dtype=object)
    category = pd.Series(None, index=codes.index, dtype=object)
    for prefix, kind, name in table():
        hit = element.isna() & codes.str.startswith(prefix)
        element[hit], category[hit] = kind, name
    rest = element.isna()
    element[rest], category[rest] = "establishment", "other"
    return pd.DataFrame({"element": element, "category": category})
