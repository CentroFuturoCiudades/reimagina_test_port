"""Amenity and establishment elements, each a point at the center of its lot.

From reimagina_pipelines_lab ``reimagina_ingestion`` (the ``equipment_layer`` and
``denue`` normalizers, ``amenities.catalog``, ``establishments.registry``), placed
as reimagina_urbano_api ``urban_fabric.md`` says: every DENUE record of the
municipality is an amenity or an establishment by the SCIAN crosswalk
(``crosswalk``), never both; the amenities file (the IMIP rows of the equipment
layer) adds amenities by its own categories. Each goes to the lot holding it, else
the nearest within ``lot_max_distance_m`` (50 m), as a point at that lot's center;
farther, it is not loaded.
"""

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from reimagina_test_port.ingestion import crosswalk
from reimagina_test_port.ingestion.cadastral import stable_element_uuid
from reimagina_test_port.ingestion.cities import CityPreset

logger = logging.getLogger(__name__)

AMENITY_ATTRIBUTES = {
    "category": "text",
    "scian_code": "text",
    "num_workers": "number",
    "capacity": "number",
    "name": "text",
}
ESTABLISHMENT_ATTRIBUTES = {
    "category": "text",
    "scian_code": "text",
    "scian_sector": "text",
    "num_workers": "number",
    "name": "text",
}
DENUE_COLUMNS = [
    "id", "nom_estab", "codigo_act", "per_ocu", "municipio", "latitud", "longitud",
]  # fmt: skip
# DENUE ``per_ocu`` band -> workers (the band's midpoint; the top band, its floor).
DENUE_WORKERS = {
    "0 a 5 personas": 3,
    "6 a 10 personas": 8,
    "11 a 30 personas": 20,
    "31 a 50 personas": 40,
    "51 a 100 personas": 75,
    "101 a 250 personas": 175,
    "251 y más personas": 251,
}
# The equipment layer's own rows; its DENUE rows come from the DENUE itself.
AMENITY_FILE_SOURCES = {"IMIP"}


def read_denue(
    path: Path, municipio: str, encoding: str
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """The DENUE CSV's records of ``municipio``, as EPSG:4326 points, and a report.

    Each record gets its ``element`` and ``category`` from the crosswalk. The
    report counts the CSV's rows (the whole state), those of ``municipio`` and, of
    these, those without coordinates (left out).
    """
    rows = pd.read_csv(path, encoding=encoding, usecols=DENUE_COLUMNS, dtype=str)
    state = len(rows)
    rows = rows[rows["municipio"].str.strip() == municipio]
    x = pd.to_numeric(rows["longitud"], errors="coerce")
    y = pd.to_numeric(rows["latitud"], errors="coerce")
    located = x.notna() & y.notna()
    report = {
        "denue_state": state,
        "denue_municipio": len(rows),
        "without_coordinates": int((~located).sum()),
    }
    rows = rows[located].reset_index(drop=True)
    records = gpd.GeoDataFrame(
        pd.concat([rows, crosswalk.classify(rows["codigo_act"])], axis=1),
        geometry=gpd.points_from_xy(x[located], y[located]),
        crs="EPSG:4326",
    )
    report["denue_amenities"] = int((records["element"] == "amenity").sum())
    report["denue_establishments"] = int((records["element"] == "establishment").sum())
    logger.info("DENUE: %s", report)
    return records, report


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


def _workers(per_ocu: pd.Series) -> pd.Series:
    bands = per_ocu.astype(str).str.strip()
    workers = bands.map(DENUE_WORKERS)
    if unknown := sorted(set(bands[workers.isna()])):
        msg = f"DENUE staff ranges without a number of workers: {unknown}"
        raise ValueError(msg)
    return workers.astype(float)


def _denue_attributes(records: gpd.GeoDataFrame) -> pd.DataFrame:
    scian_code = records["codigo_act"].astype(str).str.strip()
    name = records["nom_estab"].astype(str).str.strip()
    return pd.DataFrame(
        {
            "category": records["category"],
            "scian_code": scian_code,
            "scian_sector": scian_code.str[:2],
            "num_workers": _workers(records["per_ocu"]),
            "name": name.where(name != ""),
        },
        index=records.index,
    )


def _point_table(
    points: gpd.GeoDataFrame,
    element_type: str,
    keys: pd.Series,
    attributes: pd.DataFrame,
    lots: gpd.GeoDataFrame,
    city: CityPreset,
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    if repeated := sorted(set(keys[keys.duplicated()])):
        msg = f"{element_type} keys repeated: {repeated[:5]}"
        raise ValueError(msg)
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
    report = {
        "source": len(points),
        "elements": int(kept.sum()),
        "outside_every_lot": int((~kept).sum()),
    }
    logger.info("%s: %s", element_type, report)
    # A point at its lot's center, as urban_fabric.md places them.
    centers = lots.set_index("id").geometry.to_crs(4326).representative_point()
    geometry = centers.loc[frame["parent_id"]].to_numpy()
    table = gpd.GeoDataFrame(frame, geometry=geometry, crs=4326)
    return table, report


def amenity_table(
    equipment: gpd.GeoDataFrame,
    denue: gpd.GeoDataFrame,
    lots: gpd.GeoDataFrame,
    city: CityPreset,
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """One amenity per amenities-file point and DENUE amenity near a lot, and a report.

    Amenities-file keys are ``fuente:category:source_id`` (the lab's ``poi_key``),
    with the file's category mapped and no SCIAN code, workers or name; DENUE ones
    are ``denue:<id>``. A missing capacity is ``default_capacity``.
    """
    equipment = equipment[equipment["fuente"].isin(AMENITY_FILE_SOURCES)]
    raw = equipment["category"].astype(str)
    if unknown := sorted(set(raw) - set(crosswalk.AMENITY_FILE_CATEGORIES)):
        msg = f"Amenities file categories without a category: {unknown}"
        raise ValueError(msg)
    source_id = equipment["source_id"] if "source_id" in equipment else equipment["id"]
    capacity = (
        pd.to_numeric(equipment["capacity"], errors="coerce")
        if "capacity" in equipment
        else pd.Series(float("nan"), index=equipment.index)
    )
    from_file = pd.DataFrame(
        {
            "key": equipment["fuente"].astype(str) + ":" + raw + ":"
            + source_id.astype(str),
            "category": raw.map(crosswalk.AMENITY_FILE_CATEGORIES),
            "capacity": capacity,
        }
    )  # fmt: skip
    denue = denue[denue["element"] == "amenity"]
    from_denue = _denue_attributes(denue).drop(columns="scian_sector")
    from_denue.insert(0, "key", "denue:" + denue["id"].astype(str).str.strip())
    attributes = pd.concat([from_file, from_denue], ignore_index=True)
    attributes["capacity"] = attributes["capacity"].fillna(city.default_capacity)
    points = gpd.GeoDataFrame(
        geometry=pd.concat(
            [equipment.geometry.to_crs(4326), denue.geometry], ignore_index=True
        ),
        crs=4326,
    )
    keys = attributes.pop("key")
    return _point_table(points, "amenity", keys, attributes, lots, city)


def establishment_table(
    denue: gpd.GeoDataFrame, lots: gpd.GeoDataFrame, city: CityPreset
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """One establishment per DENUE establishment near a lot, and a report.

    The key is ``denue:<id>``; SCIAN sector = the code's first two digits; workers
    from the staff range's midpoint (required: an unknown range raises).
    """
    denue = gpd.GeoDataFrame(
        denue[denue["element"] == "establishment"].reset_index(drop=True)
    )
    keys = "denue:" + denue["id"].astype(str).str.strip()
    attributes = _denue_attributes(denue)
    return _point_table(denue, "establishment", keys, attributes, lots, city)
