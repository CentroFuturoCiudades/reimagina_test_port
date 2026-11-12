"""``tree_coverage`` of lots and rights of way from a city's canopy height raster.

The raster (``sources.tree_canopy``) is prepared once per city, outside the
pipeline, by ``scripts/canopy_raster.py``. That script cuts Meta & WRI's *High
Resolution Canopy Height Maps* (Tolan et al., 2024) to the city. Heights are whole
metres (uint8), with pixels of about 1.19 m in EPSG:3857. The pipeline only reads
that file, which lives in the lake or at a local path.

A polygon's ``tree_coverage`` (0-100) is the share of the pixels whose centre lies
inside it that are canopy: at least ``min_height_m`` tall. The default is 2 m, the
lab's canopy threshold in ``meta_canopy_trees.py``. Every pixel of one polygon is
the same size, so the share of pixels is the share of area. Lines and points
(today's rights of way) have no area and get no value. Neither does a polygon
smaller than a pixel.
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.features import geometry_mask
from shapely.geometry.base import BaseGeometry

from reimagina_test_port.common import lake
from reimagina_test_port.ingestion.cities import TreeCanopy
from reimagina_test_port.ingestion.footprints import _bounds_window

logger = logging.getLogger(__name__)

# Sentinel for pixels read past the raster's edge. It is not a real height:
# canopy heights stay well below 255 m.
OFF_RASTER = 255


def coverage(
    geometry: BaseGeometry, raster: rasterio.DatasetReader, min_height_m: float
) -> float:
    """Percent (0-100) of the pixels centred inside ``geometry`` that are at least
    ``min_height_m`` tall. ``geometry`` must be in the raster's CRS. NaN when no
    pixel centre inside it is on the raster."""
    window = _bounds_window(geometry.bounds, raster.transform)
    if window.width == 0 or window.height == 0:
        return np.nan
    heights = raster.read(1, window=window, boundless=True, fill_value=OFF_RASTER)
    inside = geometry_mask(
        [geometry],
        out_shape=heights.shape,
        transform=raster.window_transform(window),
        invert=True,
        all_touched=False,
    )
    valid = inside & (heights != OFF_RASTER)
    pixels = int(valid.sum())
    if pixels == 0:
        return np.nan
    return 100.0 * int((valid & (heights >= min_height_m)).sum()) / pixels


def _coverage_chunk(args: tuple[list[bytes], str, float]) -> list[float]:
    wkbs, path, min_height_m = args
    with rasterio.open(path) as raster:
        return [coverage(shapely.from_wkb(w), raster, min_height_m) for w in wkbs]


def coverage_from_raster(
    geometries: gpd.GeoSeries,
    raster_path: Path,
    min_height_m: float,
    *,
    workers: int | None = None,
    chunk_size: int = 5_000,
) -> pd.Series:
    """``coverage`` of each polygon of ``geometries``, computed in parallel.
    ``geometries`` can be in any CRS. Non-polygons get NaN."""
    result = pd.Series(np.nan, index=geometries.index, dtype="float64")
    polygonal = geometries.geom_type.isin(["Polygon", "MultiPolygon"]).to_numpy()
    if not polygonal.any():
        return result
    with rasterio.open(raster_path) as raster:
        crs = raster.crs
    shapes = geometries[polygonal].to_crs(crs)
    # Sort north to south so each chunk reads nearby rows of the raster.
    order = np.argsort(-shapes.centroid.y.to_numpy(), kind="stable")
    wkbs = shapely.to_wkb(shapes.to_numpy()[order])
    chunks = [
        (list(wkbs[start : start + chunk_size]), str(raster_path), min_height_m)
        for start in range(0, len(wkbs), chunk_size)
    ]
    values: list[float] = []
    workers = workers or min(8, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for chunk in pool.map(_coverage_chunk, chunks):
            values.extend(chunk)
            logger.info("Tree coverage: %s/%s", f"{len(values):,}", f"{len(wkbs):,}")
    covered = np.empty(len(wkbs))
    covered[order] = values
    result.iloc[np.flatnonzero(polygonal)] = covered
    return result


def tree_coverage(
    geometries: gpd.GeoSeries, canopy: TreeCanopy, workdir: Path
) -> pd.Series:
    """``tree_coverage`` (0-100) of each polygon of ``geometries``, from the city's
    canopy raster, copied into ``workdir``. Lines, points and polygons off the
    raster get NaN. The raster is copied only when there is a polygon to measure."""
    if not geometries.geom_type.isin(["Polygon", "MultiPolygon"]).any():
        return pd.Series(np.nan, index=geometries.index, dtype="float64")
    raster = lake.download(canopy.uri, workdir / "canopy")
    return coverage_from_raster(geometries, raster, canopy.min_height_m)
