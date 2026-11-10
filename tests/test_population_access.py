"""Population access: hand-checked distances on the synthetic snapshot and on small
hand-made street networks."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from lyra.sdk import LocalRunner
from shapely.geometry import Point, box

from reimagina_test_port.common import vocabulary
from reimagina_test_port.indicators.population_access.access_calculation import (
    nearest_distances,
    street_network,
)
from reimagina_test_port.indicators.population_access.nodes import KEYS
from reimagina_test_port.plugin import create_snapshot_plugin

CRS = "EPSG:32611"
# The synthetic pharmacy: a 10 m box (x 640450-640460) on the street y=3610000.
DISTANCE = "distance_m_health_services"
ACCESS = "population_with_access_health_services"


def run(**parameters) -> dict[str, gpd.GeoDataFrame]:
    runner = LocalRunner.from_definitions(
        {"reimagina-test-port": create_snapshot_plugin()}
    )
    outputs = runner.run(
        "reimagina_population_access",
        {"parameters": {"snapshot": "synthetic", **parameters}},
    )
    return {
        "lots": outputs["reimagina_population_access_lots"].set_index("id"),
        "blocks": outputs["reimagina_population_access_blocks"].set_index("id"),
    }


def test_walks_from_frontage_to_frontage():
    # 20 m lots on the same street as the pharmacy: the walk is the gap between them.
    lots = run()["lots"]
    np.testing.assert_allclose(lots[DISTANCE], 640460 - lots.geometry.bounds["maxx"])


def test_blocks_add_up_their_lots(monkeypatch):
    monkeypatch.setitem(vocabulary.DESTINATION_COVERAGE_M, "health_services", 300)
    blocks = run()["blocks"]
    # Only lot-mixed (90 residents, 165 m) is within 300 m.
    assert blocks.loc["block-z", ACCESS] == 90
    # Weighted by population: lot-half_empty (45, 365 m) and lot-mixed (90, 165 m).
    assert blocks.loc["block-z", DISTANCE] == pytest.approx((45 * 365 + 90 * 165) / 135)
    assert blocks.loc["block-empty", [DISTANCE, ACCESS]].isna().all()


def test_each_type_uses_its_coverage():
    # health_services reaches 500 m: every resident of block-z.
    assert run()["blocks"].loc["block-z", ACCESS] == 135


def test_lots_without_residents():
    lots = run()["lots"]
    assert lots.loc["lot-shop", ACCESS] == 0
    assert pd.isna(lots.loc["lot-unknown", ACCESS])


def test_type_without_pois_has_no_access():
    lots = run()["lots"]
    assert lots["distance_m_fresh_food"].isna().all()
    assert lots["population_with_access_fresh_food"].fillna(0).eq(0).all()


def test_only_the_requested_types_are_measured():
    lots = run(poi_types=["health"])["lots"]
    assert set(KEYS) <= set(lots.columns)
    assert lots["distance_m_health"].notna().all()
    assert lots[[DISTANCE, ACCESS]].isna().all(axis=None)


def street() -> pd.DataFrame:
    """A street 0-400 m along y=0 (nodes every 200 m) and a side street up x=400."""
    nodes = pd.DataFrame(
        {"id": [1, 2, 3, 4], "x": [0.0, 200, 400, 400], "y": [0.0, 0, 0, 400]}
    )
    edges = pd.DataFrame(
        {"from": [1, 2, 3], "to": [2, 3, 4], "distance": [200.0, 200, 400]}
    )
    return street_network(nodes, edges)


LOTS = gpd.GeoDataFrame(
    geometry=[box(40, 5, 60, 25), box(150, 5, 170, 25), box(300, 200, 320, 220)],
    crs=CRS,
)
POINT = gpd.GeoSeries([Point(120, 8)], crs=CRS)


def test_same_segment_walks_along_it():
    distance = nearest_distances(LOTS, {"x": POINT}, street(), CRS, 10_000)["x"]
    # 5 m to the street, along it, 8 m to the point; the third lot has no entry
    # within 30 m of a street, so its closest one (80 m from x=400) is used.
    np.testing.assert_allclose(distance, [73, 43, 80 + 200 + 280 + 8])


def test_beyond_the_search_limit_is_null():
    distance = nearest_distances(LOTS, {"x": POINT}, street(), CRS, 100)["x"]
    np.testing.assert_allclose(distance, [73, 43, np.nan])


def test_polygons_are_entered_by_their_nearest_side():
    park = gpd.GeoSeries([box(405, 50, 605, 350)], crs=CRS)
    distance = nearest_distances(LOTS, {"x": park}, street(), CRS, 10_000)["x"]
    # To x=400, up the side street to the park's corner (y=50), 5 m in; the third
    # lot reaches the west side at y=210, its nearest outline point every 20 m.
    np.testing.assert_allclose(distance, [5 + 340 + 50 + 5, 5 + 230 + 50 + 5, 95])


def test_network_in_another_crs_fails():
    far = gpd.GeoDataFrame(geometry=LOTS.translate(xoff=500_000), crs=CRS)
    with pytest.raises(ValueError, match="ACCESS_CRS"):
        nearest_distances(far, {"x": POINT}, street(), CRS, 10_000)
