"""reimagina/built-form: the formulas in 2-what/data/indicators.md, lot by lot, summed into blocks.

Ported from exe_reimagina `indicators_core.base.compute_lot_indicators`. Pure arithmetic, no network.
"""

from __future__ import annotations

from typing import Any

MIN_BUILT_M2 = 20  # a single-family lot with less built area counts as empty (same threshold as ingestion)
NON_RESIDENTIAL = {"amenity", "commerce", "office"}
BLOCK_SUM = ["num_housing_units", "population", "area"]
BLOCK_AVERAGE = ["num_levels", "unit_size_m2"]
BLOCK_AREA_WEIGHTED = ["cos", "cus", "density"]


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def lot_indicators(attributes: dict[str, Any]) -> dict[str, float | None]:
    """Every key is always present: None clears a value that no longer applies (a house turned shop)."""
    area, cos, levels = (
        _number(attributes.get(k)) for k in ("area", "cos", "num_levels")
    )
    cus = cos * levels if cos is not None and levels is not None else None
    built = cus * area if cus is not None and area is not None else None
    out: dict[str, float | None] = {
        "cos": cos,
        "num_levels": levels,
        "cus": cus,
        "area": area,
        "unit_size_m2": None,
        "num_housing_units": None,
        "population": None,
        "density": None,
    }
    land_use = str(attributes.get("land_use") or "").lower()
    units: int | None = None
    if land_use == "single_family":
        if built is not None and built > MIN_BUILT_M2:
            out["unit_size_m2"], units = (
                built,
                1,
            )  # its own built area: exactly one home
        else:
            units = 0
    elif land_use in ("multifamily", "mixed_use"):
        out["unit_size_m2"] = size = _number(attributes.get("unit_size_m2"))
        if built is not None and size:
            units = round(built / size)
    elif land_use in NON_RESIDENTIAL:
        units = 0
    if units is not None:
        people = _number(attributes.get("people_per_household"))
        occupancy = _number(
            attributes.get("occupancy_rate")
        )  # vacant homes out; absent = 1
        out["num_housing_units"] = units
        out["population"] = (
            0
            if units == 0
            else round(units * people * (1.0 if occupancy is None else occupancy))
            if people is not None
            else None
        )
        out["density"] = units / (area / 10_000) if area else None
    return out


def block_indicators(lots: list[dict[str, float | None]]) -> dict[str, float | None]:
    """Sums for counts, averages for ratios, weighted by lot area where indicators.md says so."""
    out: dict[str, float | None] = {}
    for key in BLOCK_SUM:
        values = [value for lot in lots if (value := lot[key]) is not None]
        out[key] = sum(values) if values else None
    for key in BLOCK_AVERAGE:
        values = [value for lot in lots if (value := lot[key]) is not None]
        out[key] = sum(values) / len(values) if values else None
    for key in BLOCK_AREA_WEIGHTED:
        pairs = [
            (value, area)
            for lot in lots
            if (value := lot[key]) is not None and (area := lot["area"])
        ]
        weight = sum(area for _, area in pairs)
        out[key] = sum(v * area for v, area in pairs) / weight if weight else None
    return out
