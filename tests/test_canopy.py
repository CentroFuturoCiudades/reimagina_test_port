"""tree_coverage from a canopy height raster: the share of a polygon's pixels that
are canopy, for lots and for a right of way drawn as a polygon."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, box

from reimagina_test_port.ingestion import canopy
from reimagina_test_port.ingestion.cities import TreeCanopy

CRS = "EPSG:3857"


@pytest.fixture
def raster(tmp_path) -> Path:
    """100 x 100 m at 1 m, origin (0, 100). Trees 12 m tall west of x = 50, and a
    1 m shrub (not canopy) east of it."""
    heights = np.where(np.arange(100) < 50, 12, 1).astype("uint8")[None, :]
    path = tmp_path / "canopy.tif"
    with rasterio.open(
        path, "w", driver="GTiff", width=100, height=100, count=1, dtype="uint8",
        crs=CRS, transform=from_origin(0, 100, 1, 1),
    ) as target:  # fmt: skip
        target.write(heights.repeat(100, 0), 1)
    return path


def test_polygon_right_of_way(raster, tmp_path):
    # A hypothetical right of way as a polygon: a 20 m wide street from x = 0 to
    # 80 (its line buffered 10 m, flat ends). 50 of its 80 m are under trees.
    street = LineString([(0, 50), (80, 50)]).buffer(10, cap_style="flat")
    shapes = gpd.GeoSeries(
        {
            "right_of_way": street,
            "lot_under_trees": box(10, 10, 30, 30),
            "lot_half": box(40, 60, 60, 80),
            "line": LineString([(0, 50), (80, 50)]),
            "off_raster": box(500, 500, 520, 520),
            "smaller_than_a_pixel": Point(10, 10).buffer(0.1),  # no pixel centre
        },
        crs=CRS,
    )
    # Any CRS: the polygons are reprojected to the raster's.
    result = canopy.tree_coverage(
        shapes.to_crs(4326), TreeCanopy(str(raster)), tmp_path / "work"
    )
    assert result["right_of_way"] == pytest.approx(62.5)
    assert result["lot_under_trees"] == pytest.approx(100.0, abs=1e-6)
    assert result["lot_half"] == pytest.approx(50.0, abs=1e-6)
    assert result[["line", "off_raster", "smaller_than_a_pixel"]].isna().all()


def test_lines_do_not_read_the_raster(tmp_path):
    lines = gpd.GeoSeries([LineString([(0, 0), (1, 1)])], crs=CRS)
    missing = TreeCanopy(str(tmp_path / "missing.tif"))
    assert canopy.tree_coverage(lines, missing, tmp_path).isna().all()
