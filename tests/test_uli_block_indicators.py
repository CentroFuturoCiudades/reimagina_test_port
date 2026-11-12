"""Land use and streets: hand-checked values on hand-made lots and
streets, and on the synthetic snapshot's road network."""

import logging
import math
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, box

from reimagina_test_port.indicators.land_use.calculation import land_use_by_block
from reimagina_test_port.indicators.streets.calculation import (
    coverage_by_block,
    intersections,
    lane_flags,
)
from reimagina_test_port.indicators.streets.nodes import streets

CRS = "EPSG:32611"
SYNTHETIC = Path(__file__).parents[1] / "src/reimagina_test_port/data/synthetic"
CONTEXT = SimpleNamespace(logger=logging.getLogger(__name__))


def test_land_use_by_block():
    lots = pd.DataFrame(
        {
            "parent_id": ["a", "a", "a", "a", "b", "b"],
            "land_use": [
                "single_family",
                "multifamily",
                "commerce",
                None,
                "office",
                None,
            ],
            "area": [1_000.0, 1_000, 2_000, 1_000, 500, 0],
            # Built form's homes: a house, a building of 50, a shop.
            "num_housing_units": [1.0, 50, 0, None, None, None],
        }
    )
    result = land_use_by_block(lots, pd.Series(["a", "b", "c"])).set_index(
        pd.Index(["a", "b", "c"])
    )
    # 1 + 50 homes on the 0.2 ha of the two lots with homes: the single-family house
    # counts. The shop has none and the fourth lot's homes are unknown.
    assert result.loc["a", "net_residential_density"] == pytest.approx(255)
    # 2,000 of 5,000 m² (the lot without land use counts as land).
    assert result.loc["a", "commercial_land_share"] == pytest.approx(0.4)
    # Three uses at 1/4, 1/4, 1/2.
    p = np.array([0.25, 0.25, 0.5])
    assert result.loc["a", "land_use_mix"] == pytest.approx(
        -(p * np.log(p)).sum() / math.log(3)
    )
    # One use: no mix; no residential land: no density; no lots: all null.
    assert result.loc["b"].tolist()[1:] == [1.0, 0.0]
    assert pd.isna(result.loc["b", "net_residential_density"])
    assert result.loc["c"].isna().all()


def test_streets_metric_on_the_synthetic_network():
    # One lit street with a protected lane between block-z and block-a.
    street = gpd.GeoDataFrame(
        {"bike_lane": [True], "protected_bike_lane": [True], "streetlights": [100.0]},
        geometry=[LineString([(640_000, 3_610_095), (640_400, 3_610_095)])],
        crs=CRS,
    )
    blocks = streets(
        None,
        gpd.read_parquet(SYNTHETIC / "block.parquet"),
        street,
        pd.read_parquet(SYNTHETIC / "network_nodes.parquet"),
        pd.read_parquet(SYNTHETIC / "network_edges.parquet"),
        CONTEXT,
    ).set_index("id")
    # The only street is within 500 m of every block, all of it with a lane and lit.
    assert blocks["protected_bike_lane_coverage"].tolist() == [1.0] * 3
    assert blocks["street_lighting_coverage"].tolist() == [1.0] * 3
    # Every node of the 13 x 9 grid but its 4 corners is within 1 km.
    assert blocks.loc["block-z", "intersection_density"] == pytest.approx(113 / math.pi)


def test_intersections_count_distinct_neighbours():
    edges = pd.DataFrame({"from": [1, 1, 1, 2, 2, 3], "to": [2, 3, 4, 1, 3, 3]})
    # 1 joins 2, 3 and 4; 2 and 3 only two others (repeats and loops aside).
    assert intersections(edges).tolist() == [1]


def test_lane_flags_read_the_bands():
    bands = pd.Series(
        [
            ["sidewalk", "protected_bike_lane", "lanes", "sidewalk"],
            ["sidewalk", "shared_lanes", "sidewalk"],
            ["sidewalk", "lanes", "sidewalk"],
            None,
        ]
    )
    flags = lane_flags(bands)
    assert flags["bike_lane"].tolist() == [True, True, False, False]
    assert flags["protected_bike_lane"].tolist() == [True, False, False, False]


def test_street_coverage():
    # Streets of 400 m (protected), 300 m (painted) and 300 m (none) near the first
    # block; only part of the last one within 500 m of it.
    streets = gpd.GeoDataFrame(
        {
            "bike_lane": [True, True, False, None],
            "protected_bike_lane": [True, False, False, None],
            "streetlights": [100.0, 0.0, 50.0, None],
        },
        geometry=[
            LineString([(0, 0), (400, 0)]),
            LineString([(0, 100), (300, 100)]),
            LineString([(0, 200), (300, 200)]),
            LineString([(0, 35), (1_000, 35)]),
        ],
        crs=CRS,
    )
    blocks = gpd.GeoDataFrame(
        geometry=[box(0, 10, 50, 60), box(5_000, 5_000, 5_050, 5_050)], crs=CRS
    )
    result = coverage_by_block(blocks, streets, CRS)
    # The last street, level with the block, counts its 550 m within 500 m of it.
    total = 400 + 300 + 300 + 550
    assert result.loc[0, "bike_lane_coverage"] == pytest.approx(700 / total)
    assert result.loc[0, "protected_bike_lane_coverage"] == pytest.approx(400 / total)
    # Lit: all of the first, none of the second, half of the third; the last has no
    # value, so it is left out.
    assert result.loc[0, "street_lighting_coverage"] == pytest.approx(
        (400 + 150) / 1_000
    )
    # No street within 500 m.
    assert result.loc[1].isna().all()
