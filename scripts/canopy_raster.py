"""Cut Meta's canopy height map to a city: the raster ``sources.tree_canopy`` reads.

Source: Meta & WRI, *High Resolution Canopy Height Maps* (Tolan et al., 2024),
public on AWS Open Data (no credentials). The map is one GeoTIFF per zoom-9
quadkey, with heights in whole metres (uint8) and pixels of about 1.19 m in
EPSG:3857. Only the rows over the city are read, over HTTP range requests.

The cut covers the city's blocks (``sources.blocks``, from the lake) plus
``--margin`` metres. It keeps Meta's own grid, so nothing is resampled, and is
written as a tiled, compressed GeoTIFF:

    uv run python scripts/canopy_raster.py mexicali \\
        --out src/reimagina_test_port/data/mexicali/meta_canopy_height.tif

Upload it to the lake (Mexicali:
``az://sandbox/projects/EXE/cleaned/canopy_height/meta_canopy_height_02002.tif``)
and point the city's ``sources.tree_canopy.uri`` at it.
"""

import argparse
import logging
import math
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.merge import merge

from reimagina_test_port.common import lake
from reimagina_test_port.ingestion.canopy import OFF_RASTER
from reimagina_test_port.ingestion.cities import CITIES

logger = logging.getLogger("canopy_raster")

TILES = (
    "/vsicurl/https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/"
    "alsgedi_global_v6_float/chm/{}.tif"
)
ZOOM = 9


def quadkey(x: int, y: int, zoom: int = ZOOM) -> str:
    """Bing quadkey of web mercator tile ``x``, ``y``."""
    return "".join(
        str(((x >> bit) & 1) + 2 * ((y >> bit) & 1)) for bit in range(zoom - 1, -1, -1)
    )


def quadkeys(bounds: tuple[float, float, float, float], zoom: int = ZOOM) -> list[str]:
    """Quadkeys of the tiles covering lon/lat ``bounds`` (west, south, east, north)."""

    def tile(lon: float, lat: float) -> tuple[int, int]:
        n = 2**zoom
        s = math.sin(math.radians(lat))
        y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n
        return int((lon + 180) / 360 * n), int(y)

    west, south, east, north = bounds
    (x0, y0), (x1, y1) = tile(west, north), tile(east, south)
    return [quadkey(x, y, zoom) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


def cut(bounds_4326: tuple[float, float, float, float], out: Path) -> Path:
    """Write the canopy heights over ``bounds_4326`` to ``out``, on Meta's grid.
    Where no tile has data: 255, which canopy.py reads as off the raster."""
    keys = quadkeys(bounds_4326)
    logger.info("Tiles: %s", keys)
    bounds = (
        gpd.GeoSeries.from_xy(bounds_4326[::2], bounds_4326[1::2], crs=4326)
        .to_crs(3857)
        .total_bounds
    )
    sources = [rasterio.open(TILES.format(key)) for key in keys]
    try:
        partial = out.with_name(f".{out.name}.part")
        # Tiles' pixel sizes differ only by float rounding: take the first one's.
        merge(
            sources,
            bounds=tuple(bounds),
            res=sources[0].res,
            nodata=OFF_RASTER,
            mem_limit=1024,
            dst_path=partial,
            dst_kwds={
                "driver": "GTiff",
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "compress": "deflate",
                "predictor": 2,
                "BIGTIFF": "IF_SAFER",
            },
        )
    finally:
        for source in sources:
            source.close()
    partial.replace(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("city", choices=sorted(CITIES))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=500.0, help="Metres.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    city = CITIES[args.city]
    raw = lake.read_parquet(city.blocks_uri, ["geometry"])
    blocks = gpd.GeoSeries.from_wkb(raw["geometry"], crs=4326).to_crs(city.target_crs)
    west, south, east, north = blocks.total_bounds
    m = args.margin
    area = gpd.GeoSeries.from_xy(
        [west - m, east + m], [south - m, north + m], crs=city.target_crs
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Open only the tile, without listing the bucket; read its strips in big ranges.
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", VSI_CACHE=True):
        cut(tuple(area.to_crs(4326).total_bounds), args.out)
    logger.info("Written %s", args.out)


if __name__ == "__main__":
    main()
