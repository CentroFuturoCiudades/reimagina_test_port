"""Amenity and establishment elements, each hanging from the lot that holds it.

From reimagina_pipelines_lab ``reimagina_ingestion``: the ``equipment_layer`` and
``denue`` normalizers, the ``amenities.catalog`` and ``establishments.registry``
builders and the lot parent of ``elements._parents``. Amenities are the cleaned
equipment layer (DENUE + IMIP, ``pois_accesibilidad``), the same points that give
lots their ``amenity_type``; establishments are every business of the raw INEGI
DENUE of the municipality.
"""

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from reimagina_test_port.common.vocabulary import normalize_amenity_type
from reimagina_test_port.ingestion.cadastral import stable_element_uuid
from reimagina_test_port.ingestion.cities import CityPreset

logger = logging.getLogger(__name__)

AMENITY_ATTRIBUTES = {"amenity_type": "text", "capacity": "number"}
ESTABLISHMENT_ATTRIBUTES = {
    "scian_class": "text",
    "scian_sector": "text",
    "employees": "number",
}
DENUE_COLUMNS = ["id", "codigo_act", "per_ocu", "municipio", "latitud", "longitud"]
# DENUE ``per_ocu`` band -> employees (the band's midpoint; the top band, its floor).
DENUE_EMPLOYEES = {
    "0 a 5 personas": 3,
    "6 a 10 personas": 8,
    "11 a 30 personas": 20,
    "31 a 50 personas": 40,
    "51 a 100 personas": 75,
    "101 a 250 personas": 175,
    "251 y más personas": 251,
}


def read_denue(path: Path, municipio: str, encoding: str) -> gpd.GeoDataFrame:
    """The DENUE CSV's businesses of ``municipio``, as EPSG:4326 points."""
    rows = pd.read_csv(path, encoding=encoding, usecols=DENUE_COLUMNS, dtype=str)
    rows = rows[rows["municipio"].str.strip() == municipio]
    x = pd.to_numeric(rows["longitud"], errors="coerce")
    y = pd.to_numeric(rows["latitud"], errors="coerce")
    located = x.notna() & y.notna()
    if not located.all():
        logger.info("DENUE: %s businesses without coordinates", int((~located).sum()))
    rows = rows[located]
    return gpd.GeoDataFrame(
        rows.reset_index(drop=True),
        geometry=gpd.points_from_xy(x[located], y[located]),
        crs="EPSG:4326",
    )


def lot_holders(
    points: gpd.GeoDataFrame,
    lots: gpd.GeoDataFrame,
    *,
    metric_crs: str,
    max_distance_m: float,
) -> pd.Series:
    """Id of the lot holding each point, else the nearest within ``max_distance_m``.

    Points farther from every lot are NaN. Ties go to the first lot in ``lots``.
    """
    shapes = lots[["id", "geometry"]].to_crs(metric_crs).reset_index(drop=True)
    located = gpd.GeoDataFrame(geometry=points.geometry.to_crs(metric_crs))
    inside = gpd.sjoin(located, shapes, how="inner", predicate="within")
    inside = inside.sort_values("index_right", kind="mergesort")
    inside = inside[~inside.index.duplicated()]
    near = located.drop(index=inside.index).sjoin_nearest(
        shapes, how="inner", max_distance=max_distance_m, distance_col="distance"
    )
    near = near.sort_values(["distance", "index_right"], kind="mergesort")
    near = near[~near.index.duplicated()]
    holder = pd.concat([inside["id"], near["id"]])
    return points.index.to_series().map(holder)


def _point_table(
    points: gpd.GeoDataFrame,
    element_type: str,
    keys: pd.Series,
    attributes: pd.DataFrame,
    lots: gpd.GeoDataFrame,
    city: CityPreset,
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    parents = lot_holders(
        points, lots, metric_crs=city.target_crs, max_distance_m=city.lot_max_distance_m
    )
    kept = parents.notna()
    frame = pd.DataFrame(
        {
            "id": [stable_element_uuid(city.key, element_type, k) for k in keys[kept]],
            "parent_id": parents[kept].to_numpy(),
            "source_record_ref": keys[kept].to_numpy(),
        }
    )
    for column in attributes:
        frame[column] = attributes.loc[kept, column].to_numpy()
    report = {"elements": int(kept.sum()), "outside_every_lot": int((~kept).sum())}
    logger.info("%s: %s", element_type, report)
    table = gpd.GeoDataFrame(
        frame, geometry=points.geometry[kept].to_crs(4326).to_numpy(), crs=4326
    )
    return table, report


def amenity_table(
    pois: gpd.GeoDataFrame, lots: gpd.GeoDataFrame, city: CityPreset
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """One amenity per equipment point inside or near a lot, and a report.

    The key is ``fuente:category:source_id`` (the lab's ``poi_key``); a missing
    capacity is ``default_capacity``.
    """
    pois = gpd.GeoDataFrame(pois.reset_index(drop=True))
    category = pois["category"].astype(str)
    # Raises on a category with no amenity type.
    amenity_type = category.map(normalize_amenity_type)
    source_id = pois["source_id"] if "source_id" in pois else pois["id"]
    keys = pois["fuente"].astype(str) + ":" + category + ":" + source_id.astype(str)
    if repeated := sorted(set(keys[keys.duplicated()])):
        msg = f"Equipment keys repeated: {repeated[:5]}"
        raise ValueError(msg)
    capacity = (
        pd.to_numeric(pois["capacity"], errors="coerce")
        if "capacity" in pois
        else pd.Series(float("nan"), index=pois.index)
    )
    attributes = pd.DataFrame(
        {
            "amenity_type": amenity_type,
            "capacity": capacity.fillna(city.default_capacity).astype(float),
        }
    )
    return _point_table(pois, "amenity", keys, attributes, lots, city)


def establishment_table(
    businesses: gpd.GeoDataFrame, lots: gpd.GeoDataFrame, city: CityPreset
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """One establishment per DENUE business inside or near a lot, and a report.

    The key is ``denue:<id>``; SCIAN sector = the class's first two digits;
    employees from the ``per_ocu`` band (an unknown band is left out).
    """
    businesses = gpd.GeoDataFrame(businesses.reset_index(drop=True))
    keys = "denue:" + businesses["id"].astype(str).str.strip()
    if repeated := sorted(set(keys[keys.duplicated()])):
        msg = f"DENUE ids repeated: {repeated[:5]}"
        raise ValueError(msg)
    scian_class = businesses["codigo_act"].astype(str).str.strip()
    attributes = pd.DataFrame(
        {
            "scian_class": scian_class,
            "scian_sector": scian_class.str[:2],
            "employees": businesses["per_ocu"].str.strip().map(DENUE_EMPLOYEES),
        }
    )
    return _point_table(businesses, "establishment", keys, attributes, lots, city)
