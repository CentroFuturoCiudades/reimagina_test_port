"""POI diversity: hand-checked entropies on a hand-made street and on the synthetic
snapshot."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from lyra.sdk import LocalRunner
from shapely.geometry import Point, box

from reimagina_test_port.indicators.population_access.access_calculation import (
    entries,
    street_network,
)
from reimagina_test_port.indicators.population_access.diversity_calculation import (
    diversity,
    poi_diversity,
    reachable,
)
from reimagina_test_port.plugin import create_snapshot_plugin

CRS = "EPSG:32611"


def street():
    """A straight street along y=0, with nodes every 1000 m up to 3000 m."""
    nodes = pd.DataFrame({"id": [1, 2, 3, 4], "x": [0.0, 1000, 2000, 3000], "y": 0.0})
    edges = pd.DataFrame({"from": [1, 2, 3], "to": [2, 3, 4], "distance": 1000.0})
    return street_network(nodes, edges)


def points(*xy) -> gpd.GeoSeries:
    return gpd.GeoSeries([Point(x, y) for x, y in xy], crs=CRS)


def normalised(*counts: int, k: int) -> float:
    share = np.array(counts) / sum(counts)
    return float(-(share * np.log(share)).sum() / np.log(k))


def test_reachable_walks_like_the_nearest_distance():
    network = street()
    lots = entries(points((500, 10), (1500, 10)), network)
    pois = entries(points((550, 5), (1100, 5)), network)
    # Along the street: 50 + 10 + 5 = 65 and 400 + 10 + 5 = 415 m; across the
    # node at 1000 m: 500 + 10 + 100 + 5 = 615 and 950 + 10 + 5 = 965 m.
    pairs = pd.concat(reachable(lots, pois, network, 500))
    assert set(map(tuple, pairs.to_numpy())) == {(0, 0), (1, 1)}
    pairs = pd.concat(reachable(lots, pois, network, 700))
    assert set(map(tuple, pairs.to_numpy())) == {(0, 0), (1, 1), (0, 1)}


def test_entropy_of_the_subcategories_reached():
    lots = gpd.GeoDataFrame(
        geometry=[Point(500, 10), Point(1500, 10), Point(2500, 10), None], crs=CRS
    )
    pois = gpd.GeoSeries(
        [
            # Around the first lot, one of a and one of b.
            Point(480, 5),
            Point(520, 5),
            # Around the second, three of a and one of b: two POIs share the place
            # at 1470 m, and the box is one POI however many entries it has.
            Point(1450, 5),
            Point(1470, 5),
            Point(1470, 5),
            box(1520, 2, 1540, 8),
            # Around the third, only one of b.
            Point(2520, 5),
            # Far from every lot.
            Point(2900, 5),
        ],
        crs=CRS,
    )
    types = ["a", "b", "a", "a", "b", "a", "b", "a"]
    # k = 3 is the set's, though only two subcategories are there.
    values = poi_diversity(lots, pois, types, street(), CRS, 100, 3)
    np.testing.assert_allclose(
        values, [normalised(1, 1, k=3), normalised(3, 1, k=3), 0, np.nan]
    )
    # One subcategory reached is 0, not -0.0.
    assert not np.signbit(values[2])
    assert values[0] == pytest.approx(np.log(2) / np.log(3))


def test_blocks_average_their_lots_weighted_by_population():
    lots = pd.DataFrame(
        {
            "parent_id": ["a", "a", "a", "b"],
            "population": [10.0, 30.0, 20.0, 0.0],
        }
    )
    blocks = pd.DataFrame({"id": ["a", "b", "c"]})
    block = diversity(lots, blocks, np.array([0.5, 1.0, np.nan, 0.8]))["blocks"]
    # The lot without a value does not count; without residents, no average.
    assert block[0] == pytest.approx((10 * 0.5 + 30 * 1.0) / 40)
    assert np.isnan(block[1:]).all()


def test_snapshot_without_scian_code_scores_zero():
    runner = LocalRunner.from_definitions(
        {"reimagina-test-port": create_snapshot_plugin()}
    )
    outputs = runner.run(
        "reimagina_poi_diversity", {"parameters": {"snapshot": "synthetic"}}
    )
    # The synthetic POIs have no scian_code, so no subcategory: 0 where measured.
    lots = outputs["reimagina_poi_diversity_lots"]["diversity_fresh_food"]
    assert lots.notna().any()
    assert (lots.dropna() == 0).all()
    blocks = outputs["reimagina_poi_diversity_blocks"].set_index("id")
    assert np.isnan(blocks.loc["block-empty", "diversity_fresh_food"])
