"""Zoning for lots outside the zoning plan's area, from a cadastral column.

A city's zoning plan may cover only part of it (Mexicali's *Normativa síntesis*:
Condesa, 17 % of the lots). Configured in the city YAML
(``sources.parcels.params.zoning_fallback``), lots whose representative point is
outside the plan's area (the convex hull of its polygons) take their zoning from
a cadastral column (Mexicali: ``Giro``), mapped by a CSV next to the YAML
(``giro,zoning``) to the plan's own vocabulary ("habitacional", "comercio y
servicios"...), so the rest of the pipeline reads them like any zone. Lots inside
the area keep the plan's zoning, or none. Values the CSV does not list give no
zoning. A city without the section is unchanged.
"""

import logging

import geopandas as gpd
import pandas as pd

from reimagina_test_port.ingestion.cities import ZoningFallback

logger = logging.getLogger(__name__)


def normalize(value: pd.Series) -> pd.Series:
    """Cadastral values as the CSV lists them: stripped, upper case, ``_`` for spaces."""
    return value.fillna("").astype(str).str.strip().str.upper().str.replace(" ", "_")


def apply(
    lots: gpd.GeoDataFrame,
    raw: pd.Series,
    zoning_plan: gpd.GeoSeries,
    fallback: ZoningFallback,
    *,
    metric_crs: str,
) -> tuple[pd.Series, dict[str, int]]:
    """``lots["zoning"]`` with the fallback for lots outside the plan's area.

    ``raw`` is the cadastral column, aligned with ``lots``; ``zoning_plan`` the
    plan's polygons. Returns the zoning and a report: lots zoned by the plan,
    lots outside its area, of these those zoned by the fallback and those whose
    value the CSV does not list (by value, the most frequent).
    """
    area = zoning_plan.to_crs(metric_crs).union_all().convex_hull
    points = lots.geometry.to_crs(metric_crs).representative_point()
    outside = ~points.within(area)
    value = normalize(raw)
    mapped = value.map(fallback.values)
    zoning = lots["zoning"].where(lots["zoning"].notna() | ~outside, mapped)
    unlisted = value[outside & mapped.isna() & value.ne("")]
    report = {
        "from_zoning_plan": int(lots["zoning"].notna().sum()),
        "outside_plan_area": int(outside.sum()),
        "from_fallback": int((outside & mapped.notna() & lots["zoning"].isna()).sum()),
        "fallback_value_empty": int((outside & value.eq("")).sum()),
        "fallback_value_unlisted": len(unlisted),
    }
    logger.info(
        "Zoning fallback (%s): %s; unlisted values: %s",
        fallback.column,
        report,
        unlisted.value_counts().head(10).to_dict(),
    )
    return zoning, report
