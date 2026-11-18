"""SCIAN code -> element type (amenity or establishment) and category.

The platform's crosswalk (reimagina_urbano_api ``urban_fabric.md``, *Platform
configuration*), from the codes the team's DENUE crosswalk (``denue_crosswalk.xlsx``,
November 2026) lists under *What does this include specifically?*:
``crosswalk.csv``, one SCIAN prefix per row, the longest prefix winning (``461121``
before ``4611``). A code no prefix matches is an establishment
of category ``other``: it counts only as jobs. Each DENUE record is one or the
other, never both.

A category is the finest kind of place the platform tells apart (a bakery, a
daycare, a library), the one attribute that says what a place is: ``poi_catalog``
rules and the diversity sets group categories, and a place counts in every group
that lists its category (a preschool is both education and care).
"""

import csv
from functools import cache
from pathlib import Path

import pandas as pd

PATH = Path(__file__).with_name("crosswalk.csv")

AMENITY_CATEGORIES = (
    "preschool", "primary_school", "secondary_school", "high_school", "university",
    "special_education", "health_center", "clinic", "hospital", "residence",
    "daycare", "community_food", "community_center", "religious", "cultural_center",
    "museum", "library", "sports", "park", "police_station", "fire_station",
)  # fmt: skip
ESTABLISHMENT_CATEGORIES = (
    "supermarket", "mini_supermarket", "grocery", "butcher", "poultry", "fishmonger",
    "greengrocer", "grains_spices", "dairy", "bakery", "tortilleria", "other_food",
    "household_goods", "pharmacy", "bank", "petrol_station", "cinema", "sports",
    "manufacturing", "other",
)  # fmt: skip
CATEGORIES = {"amenity": AMENITY_CATEGORIES, "establishment": ESTABLISHMENT_CATEGORIES}


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
    when no prefix matches)."""
    codes = scian_codes.astype(str).str.strip()
    element = pd.Series(None, index=codes.index, dtype=object)
    category = pd.Series(None, index=codes.index, dtype=object)
    for prefix, kind, name in table():
        hit = element.isna() & codes.str.startswith(prefix)
        element[hit], category[hit] = kind, name
    rest = element.isna()
    element[rest], category[rest] = "establishment", "other"
    return pd.DataFrame({"element": element, "category": category})
