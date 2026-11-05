"""SCIAN code -> element type (amenity or establishment) and category.

The platform's crosswalk (reimagina_urbano_api ``urban_fabric.md``, *Platform
configuration*), the codes the team's DENUE crosswalk (``denue_crosswalk.xlsx``,
November 2026) lists under *What does this include specifically?*:
``crosswalk.csv``, one SCIAN prefix per row, the longest prefix winning (``46112``
before ``4611``). A code no prefix matches is an establishment
of category ``other``: it counts only as jobs. Each DENUE record is one or the
other, never both.
"""

import csv
from functools import cache
from pathlib import Path

import pandas as pd

PATH = Path(__file__).with_name("crosswalk.csv")

AMENITY_CATEGORIES = (
    "primary_education", "secondary_education", "tertiary_education", "university",
    "primary_healthcare", "medical_clinic", "hospital", "care", "community_facility",
    "cultural_facility", "sports_recreation", "public_space", "police",
    "emergency_services",
)  # fmt: skip
ESTABLISHMENT_CATEGORIES = (
    "food_retail", "fresh_food", "convenience_store", "basic_household_goods",
    "medicines", "financial", "petrol_station", "cultural_facility",
    "sports_recreation", "manufacturing", "other",
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
