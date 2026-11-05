"""Population access: hand-checked distances on the synthetic snapshot and on small
hand-made street networks."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from lyra.sdk import LocalRunner
from lyra.sdk.errors import MetricInputError
from shapely.geometry import Point, box

from reimagina_test_port.indicators.population_access.calculation import (
    nearest_distance,
    street_network,
)
from reimagina_test_port.plugin import create_snapshot_plugin

CRS = "EPSG:32611"


def run(**parameters) -> dict[str, gpd.GeoDataFrame]:
    runner = LocalRunner.from_definitions(
        {"reimagina-test-port": create_snapshot_plugin()}
    )
    defaults = {"snapshot": "synthetic", "poi_type": "pharmacy", "within_m": 300}
    outputs = runner.run(
        "reimagina_population_access", {"parameters": {**defaults, **parameters}}
    )
    return {
        "lots": outputs["reimagina_population_access_lots"].set_index("id"),
        "blocks": outputs["reimagina_population_access_blocks"].set_index("id"),
    }


def test_walks_from_frontage_to_frontage():
    # 20 m lots and the 10 m pharmacy box (x 640450-640460) front the street
    # y=3610000 of the 50 m grid: the walk is the gap between them.
    lots = run()["lots"]
    expected = 640460 - lots.geometry.bounds["maxx"]
    np.testing.assert_allclose(lots["distance_m"], expected)


def test_blocks_add_up_their_lots():
    blocks = run()["blocks"]
    z = blocks.loc["block-z"]
    assert z["population"] == 135
    # Only lot-mixed (90 residents, 165 m) is within 300 m.
    assert z["population_with_access"] == 90
    assert z["access_share"] == pytest.approx(90 / 135)
    # Weighted by population: lot-half_empty (45, 365 m) and lot-mixed (90, 165 m).
    assert z["distance_m"] == pytest.approx((45 * 365 + 90 * 165) / 135)
    assert blocks.loc["block-empty", ["distance_m", "population"]].isna().all()


def test_lots_without_residents():
    lots = run()["lots"]
    assert lots.loc["lot-shop", "population"] == 0
    assert pd.isna(lots.loc["lot-shop", "access_share"])
    assert pd.isna(lots.loc["lot-unknown", "population_with_access"])


def test_beyond_the_search_limit_is_null():
    lots = run(max_distance_m=300)["lots"]
    assert lots["distance_m"].notna().sum() == 8
    assert lots["distance_m"].max() <= 300


def test_type_without_pois_has_no_access():
    lots = run(poi_type="fresh_food")["lots"]
    assert lots["distance_m"].isna().all()
    assert lots["population_with_access"].fillna(0).eq(0).all()


def test_access_distance_within_the_search_limit():
    with pytest.raises(MetricInputError, match="within_m cannot exceed"):
        run(within_m=500, max_distance_m=400)


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


def test_same_segment_walks_along_it():
    pois = gpd.GeoDataFrame(geometry=[Point(120, 8)], crs=CRS)
    distance = nearest_distance(LOTS, pois, street(), CRS, 10_000)
    # 5 m to the street, along it, 8 m to the point; the third lot has no entry
    # within 30 m of a street, so its closest one (80 m from x=400) is used.
    np.testing.assert_allclose(distance, [73, 43, 80 + 200 + 280 + 8])


def test_polygons_are_entered_by_their_nearest_side():
    park = gpd.GeoDataFrame(geometry=[box(405, 50, 605, 350)], crs=CRS)
    distance = nearest_distance(LOTS, park, street(), CRS, 10_000)
    # To x=400, up the side street to the park's corner (y=50), 5 m in; the third
    # lot reaches the west side at y=210, its nearest outline point every 20 m.
    np.testing.assert_allclose(distance, [5 + 340 + 50 + 5, 5 + 230 + 50 + 5, 95])
    assert (distance >= LOTS.distance(park.geometry.iloc[0])).all()


def test_network_in_another_crs_fails():
    far = gpd.GeoDataFrame(geometry=LOTS.translate(xoff=500_000), crs=CRS)
    pois = gpd.GeoDataFrame(geometry=[Point(120, 8)], crs=CRS)
    with pytest.raises(ValueError, match="ACCESS_CRS"):
        nearest_distance(far, pois, street(), CRS, 10_000)
