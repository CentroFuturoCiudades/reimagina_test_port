"""A redesign of part of the city, such as a master plan, replacing the baseline
inside its zone.

The redesign is a folder (``overlay_uri``, in the lake or local) of Reimagina
elements, one GeoParquet per type in EPSG:4326: ``blocks``, ``lots``,
``amenities``, ``establishments`` and ``road_segments``. Each table has ``id``,
``parent_id`` and its attributes as columns. The zone (``zone_uri``) is a polygon
file: Mexicali's master plan is Condesa's zone of interest.

Inside the zone, the city's own elements give way to the redesign's. A block or a
lot goes when its representative point is in the zone, and so does an amenity or
establishment, or one whose lot went. A lot of a block that went goes too. The
redesign's elements are added as they are: they keep their ids, take the columns
the baseline's tables have (the rest are left out, missing ones are empty), and
their ``source_record_ref`` is ``overlay:<id>``. An establishment's
``scian_sector`` comes from its ``scian_code``.

The streets are not handled here: ``road_segments`` replaces the street network
inside the same zone in ``reimagina_street_network`` (``ingestion/network.py``,
step 2).
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from reimagina_test_port.common import lake
from reimagina_test_port.ingestion import network

# Baseline table -> the redesign's file.
FILES = {
    "blocks": "blocks",
    "lots": "lots",
    "amenities": "amenities",
    "establishments": "establishments",
}
STREETS_FILE = "road_segments"


class OverlayError(ValueError):
    """The redesign's elements do not fit together or clash with the city's."""


def file_uri(overlay_uri: str, name: str) -> str:
    """The redesign's ``name`` GeoParquet within ``overlay_uri``."""
    return f"{overlay_uri.rstrip('/')}/{name}.parquet"


def read(overlay_uri: str, workdir: Path) -> dict[str, gpd.GeoDataFrame]:
    """The redesign's tables, keyed as the baseline's, in EPSG:4326."""
    tables = {}
    for table, name in FILES.items():
        path = lake.download(file_uri(overlay_uri, name), workdir)
        tables[table] = gpd.read_parquet(path).to_crs(4326)
    return tables


def read_zone(zone_uri: str, workdir: Path, crs: str) -> shapely.Geometry:
    """The zone's polygons, as one geometry in ``crs``."""
    path = lake.download(zone_uri, workdir)
    polygons = network.read_geometries(path, crs, ("Polygon", "MultiPolygon"))
    return shapely.union_all(polygons.to_numpy())


def _inside(table: gpd.GeoDataFrame, zone: shapely.Geometry, crs: str) -> np.ndarray:
    points = table.geometry.to_crs(crs).representative_point().to_numpy()
    shapely.prepare(zone)
    return shapely.contains(zone, points)


def _conform(redesign: gpd.GeoDataFrame, like: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """``redesign`` with ``like``'s columns, in its order."""
    frame = pd.DataFrame(
        {
            column: (
                redesign[column].to_numpy()
                if column in redesign
                else np.full(len(redesign), None, dtype=object)
            )
            for column in like.columns
            if column != like.geometry.name
        }
    )
    frame["id"] = redesign["id"].astype(str).to_numpy()
    frame["parent_id"] = redesign["parent_id"].astype(object).to_numpy()
    frame["source_record_ref"] = ("overlay:" + frame["id"]).to_numpy()
    if "scian_sector" in frame and "scian_code" in redesign:
        frame["scian_sector"] = redesign["scian_code"].astype("string").str[:2]
    return gpd.GeoDataFrame(
        frame, geometry=redesign.geometry.to_numpy(), crs=redesign.crs
    )


def _check(redesign: dict[str, gpd.GeoDataFrame]) -> None:
    """Every lot in one of the redesign's blocks, every point in one of its lots,
    and no id twice."""
    blocks = set(redesign["blocks"]["id"].astype(str))
    lots = set(redesign["lots"]["id"].astype(str))
    problems = []
    if not redesign["lots"]["parent_id"].astype(str).isin(blocks).all():
        problems.append("lots outside its blocks")
    for table in ("amenities", "establishments"):
        if not redesign[table]["parent_id"].astype(str).isin(lots).all():
            problems.append(f"{table} outside its lots")
    ids = pd.concat([t["id"].astype(str) for t in redesign.values()])
    if ids.duplicated().any():
        problems.append("repeated ids")
    if problems:
        msg = f"Overlay: {', '.join(problems)}"
        raise OverlayError(msg)


def replace(
    tables: dict[str, gpd.GeoDataFrame],
    redesign: dict[str, gpd.GeoDataFrame],
    zone: shapely.Geometry,
    crs: str,
) -> tuple[dict[str, gpd.GeoDataFrame], dict[str, dict[str, int]]]:
    """``tables`` (the baseline's blocks, lots, amenities and establishments) with
    their elements in ``zone`` (in ``crs``) replaced by ``redesign``'s. Returns the
    tables and, per table, how many were ``removed`` and ``added``."""
    _check(redesign)
    blocks = tables["blocks"]
    gone_blocks = set(blocks["id"][_inside(blocks, zone, crs)])
    lots = tables["lots"]
    lots_out = _inside(lots, zone, crs) | lots["parent_id"].isin(gone_blocks).to_numpy()
    gone_lots = set(lots["id"][lots_out])
    keep = {
        "blocks": ~blocks["id"].isin(gone_blocks).to_numpy(),
        "lots": ~lots_out,
    }
    for table in ("amenities", "establishments"):
        points = tables[table]
        keep[table] = ~(
            _inside(points, zone, crs) | points["parent_id"].isin(gone_lots).to_numpy()
        )
    result, report = {}, {}
    for table, kept in keep.items():
        own = tables[table][kept]
        added = _conform(redesign[table], tables[table])
        clash = sorted(set(added["id"]) & set(own["id"].astype(str)))
        if clash:
            msg = f"Overlay: {table} ids already in the city, e.g. {clash[:3]}"
            raise OverlayError(msg)
        result[table] = gpd.GeoDataFrame(
            pd.concat([own, added], ignore_index=True), crs=tables[table].crs
        )
        report[table] = {"removed": int((~kept).sum()), "added": len(added)}
    return result, report
