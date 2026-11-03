"""Building footprint, built-up area and floors per lot from Open Buildings 2.5D.

Moved from exe_reimagina ``src/temporal_footprints`` (``raster.py``, ``metrics.py``)
and the tile selection of ``src/data_prep/vrt.py``. The VRT mosaics are written
directly instead of with ``gdal.BuildVRT``, so only rasterio is needed.
"""

import json
import logging
import math
import os
import re
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.features import geometry_mask
from rasterio.windows import Window
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

# Open Buildings 2.5D bands: fractional count, height (m), building presence (0-1).
HEIGHT_BAND = 2
PRESENCE_BAND = 3
NODATA = -99.0
HEIGHT_STAT_COLUMNS = ["max", "p95", "building_area", "builtup_area"]
_GDAL_TYPES = {
    "uint8": "Byte", "int8": "Int8", "uint16": "UInt16", "int16": "Int16",
    "uint32": "UInt32", "int32": "Int32", "float32": "Float32", "float64": "Float64",
}  # fmt: skip


@dataclass(frozen=True)
class FootprintParams:
    presence_threshold: float = 0.5
    height_percentile: float = 95.0
    average_level_height: float = 2.7
    gsd_pixel: float = 0.5

    @property
    def pixel_area(self) -> float:
        return self.gsd_pixel**2


def tiles_for_lots(manifest_paths: Sequence[Path], lots: gpd.GeoDataFrame) -> list[str]:
    """File names of the tiles in ``lots.crs`` whose footprint touches a lot."""
    epsg = lots.crs.to_epsg() if lots.crs is not None else None
    if epsg is None:
        msg = "Lots need an EPSG CRS to pick height tiles."
        raise ValueError(msg)
    footprints: list[tuple[str, BaseGeometry]] = []
    for path in manifest_paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        match = re.search(r"EPSG_(\d+)", manifest["name"])
        if match is None or int(match.group(1)) != epsg:
            continue
        for tileset in manifest.get("tilesets", []):
            for source in tileset.get("sources", []):
                affine, size = source["affineTransform"], source["dimensions"]
                x1, y1 = float(affine["translateX"]), float(affine["translateY"])
                x2 = x1 + int(size["width"]) * float(affine["scaleX"])
                y2 = y1 + int(size["height"]) * float(affine["scaleY"])
                tile = box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                footprints.extend(
                    (Path(uri).name, tile) for uri in source.get("uris", [])
                )
    if not footprints:
        msg = f"No Open Buildings manifest covers EPSG:{epsg}."
        raise ValueError(msg)
    tree = shapely.STRtree([geometry for _, geometry in footprints])
    hits = tree.query(lots.geometry.to_numpy(), predicate="intersects")[1]
    return sorted({footprints[i][0] for i in hits})


def write_vrt(tile_paths: Sequence[Path], band: int, path: Path) -> Path:
    """Mosaic one band of same-grid tiles as a single-band VRT."""
    sources = []
    for tile_path in tile_paths:
        with rasterio.open(tile_path) as tile:
            sources.append(
                (tile_path, tile.transform, tile.width, tile.height, tile.crs,
                 tile.dtypes[band - 1], tile.nodatavals[band - 1])
            )  # fmt: skip
    first = sources[0]
    resolution_x, resolution_y = first[1].a, first[1].e
    if any(
        s[1].a != resolution_x or s[1].e != resolution_y or s[4] != first[4]
        for s in sources
    ):
        msg = "Tiles must share CRS and resolution to be mosaicked."
        raise ValueError(msg)
    west = min(s[1].c for s in sources)
    north = max(s[1].f for s in sources)
    east = max(s[1].c + s[2] * resolution_x for s in sources)
    south = min(s[1].f + s[3] * resolution_y for s in sources)
    width = round((east - west) / resolution_x)
    height = round((south - north) / resolution_y)
    nodata = "" if first[6] is None else f"<NoDataValue>{first[6]}</NoDataValue>"
    simple_sources = "".join(
        "<SimpleSource>"
        f'<SourceFilename relativeToVRT="0">{escape(str(tile_path))}</SourceFilename>'
        f"<SourceBand>{band}</SourceBand>"
        f'<SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>'
        f'<DstRect xOff="{round((t.c - west) / resolution_x)}" '
        f'yOff="{round((t.f - north) / resolution_y)}" xSize="{w}" ySize="{h}"/>'
        "</SimpleSource>"
        for tile_path, t, w, h, *_ in sources
    )
    path.write_text(
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">'
        f"<SRS>{escape(first[4].to_wkt())}</SRS>"
        f"<GeoTransform>{west}, {resolution_x}, 0, {north}, 0, {resolution_y}"
        "</GeoTransform>"
        f'<VRTRasterBand dataType="{_GDAL_TYPES[first[5]]}" band="1">'
        f"{nodata}{simple_sources}</VRTRasterBand></VRTDataset>"
    )
    return path


def _bounds_window(
    bounds: tuple[float, float, float, float], transform: rasterio.Affine
) -> Window:
    """Window covering the whole bbox, rounded like rasterstats."""
    west, south, east, north = bounds
    row_start = math.floor((north - transform.f) / transform.e)
    col_start = math.floor((west - transform.c) / transform.a)
    row_stop = math.ceil((south - transform.f) / transform.e)
    col_stop = math.ceil((east - transform.c) / transform.a)
    width, height = max(col_stop - col_start, 0), max(row_stop - row_start, 0)
    return Window(col_start, row_start, width, height)  # ty: ignore[too-many-positional-arguments]


def lot_stats(
    geometry: BaseGeometry,
    height: rasterio.DatasetReader,
    presence: rasterio.DatasetReader | None,
    params: FootprintParams,
) -> dict[str, float]:
    """Stats over the pixels whose centre falls inside the lot.

    A pixel is built when its height is valid and positive and, with the presence
    mask, its building confidence reaches ``presence_threshold``. ``max`` and ``p95``
    are 0 for a lot on the raster without buildings and NaN off the raster.
    """
    empty = {"max": np.nan, "p95": np.nan, "building_area": 0.0, "builtup_area": 0.0}
    window = _bounds_window(geometry.bounds, height.transform)
    if window.width == 0 or window.height == 0:
        return empty
    heights = height.read(1, window=window, boundless=True, fill_value=NODATA)
    inside = geometry_mask(
        [geometry],
        out_shape=heights.shape,
        transform=height.window_transform(window),
        invert=True,
        all_touched=False,
    )
    valid = inside & (heights != NODATA) & np.isfinite(heights)
    if not valid.any():
        return empty
    built = valid & (heights > 0)
    if presence is not None:
        confidence = presence.read(1, window=window, boundless=True, fill_value=NODATA)
        built &= (confidence != NODATA) & np.isfinite(confidence)
        built &= confidence >= params.presence_threshold
    built_heights = heights[built].astype(float, copy=False)
    if built_heights.size == 0:
        return {"max": 0.0, "p95": 0.0, "building_area": 0.0, "builtup_area": 0.0}
    levels = np.maximum(1.0, np.ceil(built_heights / params.average_level_height))
    return {
        "max": float(built_heights.max()),
        "p95": float(np.percentile(built_heights, params.height_percentile)),
        "building_area": float(built_heights.size * params.pixel_area),
        "builtup_area": float(levels.sum() * params.pixel_area),
    }


def _stats_chunk(
    args: tuple[list[bytes], str, str | None, FootprintParams],
) -> list[dict[str, float]]:
    wkbs, height_path, presence_path, params = args
    with rasterio.open(height_path) as height:
        presence = rasterio.open(presence_path) if presence_path else None
        try:
            return [
                lot_stats(shapely.from_wkb(wkb), height, presence, params)
                for wkb in wkbs
            ]
        finally:
            if presence is not None:
                presence.close()


def compute_height_stats(  # noqa: PLR0913
    lots: gpd.GeoDataFrame,
    height_vrt: Path,
    presence_vrt: Path | None,
    params: FootprintParams,
    *,
    workers: int | None = None,
    chunk_size: int = 5_000,
) -> pd.DataFrame:
    """``max``, ``p95``, ``building_area`` and ``builtup_area`` per lot, in parallel."""
    with rasterio.open(height_vrt) as height:
        if height.crs != lots.crs:
            msg = f"CRS differ: lots={lots.crs}, raster={height.crs}."
            raise ValueError(msg)
    wkbs = shapely.to_wkb(lots.geometry.to_numpy())
    chunks = [
        (list(wkbs[start : start + chunk_size]), str(height_vrt),
         None if presence_vrt is None else str(presence_vrt), params)
        for start in range(0, len(wkbs), chunk_size)
    ]  # fmt: skip
    workers = workers or min(8, os.cpu_count() or 1)
    rows: list[dict[str, float]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for chunk_rows in pool.map(_stats_chunk, chunks):
            rows.extend(chunk_rows)
            logger.info("Heights: %s/%s lots", f"{len(rows):,}", f"{len(wkbs):,}")
    return pd.DataFrame(rows, index=lots.index, columns=HEIGHT_STAT_COLUMNS).astype(
        float
    )


def add_urban_metrics(
    lots: gpd.GeoDataFrame, stats: pd.DataFrame, *, average_level_height: float
) -> gpd.GeoDataFrame:
    """Coverage, floor-area ratio, floors and normative capacity per lot."""
    if not stats.index.equals(lots.index):
        msg = "Height stats are not aligned with the lots."
        raise ValueError(msg)
    lots = lots.copy()
    lot_area = pd.to_numeric(lots["lot_area"], errors="coerce").to_numpy(float)
    building_area = stats["building_area"].to_numpy(float)
    builtup_area = stats["builtup_area"].to_numpy(float)
    height_p95 = stats["p95"].to_numpy(float)
    max_cos = pd.to_numeric(lots["max_cos"], errors="coerce").to_numpy(float)
    max_cus = pd.to_numeric(lots["max_cus"], errors="coerce").to_numpy(float)
    max_density = pd.to_numeric(lots["max_density"], errors="coerce").to_numpy(float)
    zeros = np.zeros_like(building_area)
    lots["cos"] = np.divide(
        building_area, lot_area, out=zeros.copy(), where=lot_area > 0
    )
    lots["cus"] = np.divide(
        builtup_area, lot_area, out=zeros.copy(), where=lot_area > 0
    )
    lots["max_height"] = stats["max"].to_numpy(float)
    lots["height_p95"] = height_p95
    # Average floors implied by the raster heights.
    lots["num_levels"] = np.divide(
        builtup_area, building_area, out=zeros.copy(), where=building_area > 0
    )
    lots["max_num_levels"] = np.where(
        np.isfinite(height_p95),
        np.ceil(np.clip(height_p95, 0, None) / average_level_height),
        0.0,
    )
    lots["building_area"] = building_area
    lots["builtup_area"] = builtup_area
    lots["max_building_area"] = max_cos * lot_area
    lots["max_builtup_area"] = max_cus * lot_area
    lots["max_home_units"] = np.where(
        np.isfinite(max_density) & np.isfinite(lot_area),
        np.round(max_density * lot_area / 10_000),
        np.nan,
    )
    return lots
