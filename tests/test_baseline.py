"""Cadastral lot ids and the baseline build from local copies of the lake sources."""

import json
import uuid
from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from lyra.sdk import LocalRunner
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from reimagina_test_port import baseline, cadastral, lake, plugin, prep
from reimagina_test_port.cities import CITIES

CRS = "EPSG:32611"
KEYS = ("Numdel", "Colonia", "Manzana", "Lote", "Fraccion")
NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://exe-reimagina.local/elements")


def parcels(rows: list[tuple], geometries: list) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        pd.DataFrame(rows, columns=list(KEYS)), geometry=geometries, crs=CRS
    )


def test_key_joins_strips_and_marks_empty():
    lots = parcels(
        [
            (" 1", "A ", "12", "3", "0"),
            ("1", "A", "12", None, None),
            ("1", None, "12", "3", None),
            (None, None, None, None, None),
            ("", " ", "", "", ""),
        ],
        [box(i * 10, 0, i * 10 + 5, 5) for i in range(5)],
    )
    keyed, report = cadastral.lot_keys(lots, KEYS)
    assert sorted(keyed.index) == [
        "1--12-3",
        "1-A-12",
        "1-A-12-3-0",
        "sin_clave",
        "sin_clave#2",
    ]
    assert report == {
        "exact_copies_dropped": 0,
        "lots_sharing_a_key": 2,
        "lots_without_key": 2,
    }


def test_copies_dropped_and_repeats_ordered_by_position():
    shape = box(0, 0, 5, 5)
    lots = parcels(
        [("1", "A", "1", "1", "")] * 4,
        # Two exact copies; the other two sorted by representative point x, then y.
        [box(50, 0, 55, 5), shape, shape, box(20, 0, 25, 5)],
    )
    keyed, report = cadastral.lot_keys(lots, KEYS)
    assert report["exact_copies_dropped"] == 1
    assert keyed.index.tolist() == ["1-A-1-1", "1-A-1-1#2", "1-A-1-1#3"]
    assert [g.bounds[0] for g in keyed.geometry] == [0, 20, 50]
    # Source order does not matter.
    shuffled, _ = cadastral.lot_keys(lots.iloc[::-1], KEYS)
    assert shuffled.index.tolist() == keyed.index.tolist()
    assert shuffled.geometry.equals(keyed.geometry)


def test_missing_key_column():
    with pytest.raises(ValueError, match="Fraccion"):
        cadastral.lot_keys(parcels([], []).drop(columns="Fraccion"), KEYS)


def test_ids():
    assert cadastral.lot_id("mexicali", "1-A-12") == str(
        uuid.uuid5(NAMESPACE, "mexicali:lot:1-A-12")
    )
    # Blocks keep exe_reimagina's id.
    assert cadastral.block_id("020020001001a007") == str(
        uuid.uuid5(NAMESPACE, "block:020020001001a007")
    )


# ------------------------------------------------------------- build from sources

CVEGEO = ["020020001001a007", "020020001001a008"]


def write_sources(root: Path) -> dict[str, str]:
    """A tiny Mexicali: two blocks, five cadastral rows, one height tile."""
    blocks = gpd.GeoDataFrame(
        {"CVEGEO": [c.upper() for c in CVEGEO]},
        geometry=[box(0, 0, 40, 40), box(60, 0, 100, 40)],
        crs=CRS,
    ).to_crs(4326)
    pd.DataFrame(
        {"CVEGEO": blocks["CVEGEO"], "geometry": blocks.geometry.to_wkb()}
    ).to_parquet(root / "blocks.parquet")
    census = {
        column: [0, 0]
        for column in prep.CENSUS_SOURCE_COLUMNS
        if column not in {"loc", "ageb", "mza"}
    }
    census |= {
        "loc": ["0001", "0001"],
        "ageb": ["001A", "001A"],
        "mza": ["7", "8"],
        "vivtot": [4, 2],
        "tvivparhab": [3, 2],
        "prom_ocup": [3.25, 2.0],
        "pobtot": [10, 4],
    }
    pd.DataFrame(census).to_parquet(root / "census.parquet")
    lots = gpd.GeoDataFrame(
        {
            "Numdel": ["1", "1", "1", "1", None],
            "Colonia": ["A", "A", "A", "B", None],
            "Manzana": ["7", "7", "7", "8", None],
            "Lote": ["1", "2", "2", "1", None],
            "Fraccion": ["", "", "", "", None],
            "Giro": [None, None, None, "TIENDA COMERCIAL", None],
        },
        # Lot "1-A-7-2" appears twice with different shapes, then an exact copy.
        geometry=[
            box(0, 0, 20, 20),
            box(20, 20, 40, 40),
            box(0, 20, 20, 40),
            box(60, 0, 80, 20),
            box(80, 0, 100, 20),
        ],
        crs=CRS,
    )
    lots = pd.concat([lots, lots.iloc[[1]]], ignore_index=True)
    (root / "lots").mkdir()
    lots.to_file(root / "lots" / "PREDIOS.shp", engine="pyogrio")
    gpd.GeoDataFrame(
        {
            "USOSUELO24": ["Habitacional", "Comercio y servicios"],
            "Max_cos": [0.8, 1.0],
            "Max_cus": [1.6, 3.0],
            "Dens_U_max": [60.0, None],
        },
        geometry=[box(-5, -5, 45, 45), box(55, -5, 105, 45)],
        crs=CRS,
    ).to_file(root / "normativa.geojson", engine="pyogrio")
    (root / "pois").mkdir()
    gpd.GeoDataFrame(
        {"category": ["Farmacia"]}, geometry=[Point(10, 30)], crs=CRS
    ).to_parquet(root / "pois" / "pois_02002.geoparquet")
    # One 120 x 60 m tile at 0.5 m: 5 m (2 floors, exact in float32), built.
    tiles = root / "tiles" / "2023"
    tiles.mkdir(parents=True)
    heights = np.full((120, 240), 5.0, dtype="float32")
    with rasterio.open(
        tiles / "t.tif", "w", driver="GTiff", width=240, height=120, count=3,
        dtype="float32", crs=CRS, nodata=-99.0, transform=from_origin(-10, 50, 0.5, 0.5),
    ) as tile:  # fmt: skip
        tile.write(np.zeros_like(heights), 1)
        tile.write(heights, 2)
        tile.write(np.full_like(heights, 0.9), 3)
    manifests = root / "manifests" / "2023" / "manifests"
    manifests.mkdir(parents=True)
    source = {
        "affineTransform": {
            "translateX": -10,
            "translateY": 50,
            "scaleX": 0.5,
            "scaleY": -0.5,
        },
        "dimensions": {"width": 240, "height": 120},
        "uris": ["x/t.tif"],
    }
    (manifests / "m.json").write_text(
        json.dumps({"name": "81_EPSG_32611_2023", "tilesets": [{"sources": [source]}]})
    )
    return {
        "lots_uri": str(root / "lots" / "PREDIOS.shp"),
        "normativa_uri": str(root / "normativa.geojson"),
        "pois_uri_prefix": str(root / "pois"),
        "tiles_uri": str(root / "tiles"),
        "manifests_uri": str(root / "manifests"),
    }


@pytest.fixture
def city(tmp_path, monkeypatch):
    sources = tmp_path / "sources"
    sources.mkdir()
    preset = replace(CITIES["mexicali"], **write_sources(sources))
    local = {
        preset.census_uri: sources / "census.parquet",
        preset.blocks_uri: sources / "blocks.parquet",
    }
    original = lake.read_parquet
    monkeypatch.setattr(
        lake, "read_parquet", lambda uri, columns: original(str(local[uri]), columns)
    )
    return preset


def test_build(city, tmp_path):
    stages = []
    result = baseline.build(
        city, tmp_path / "job", lambda stage, *_: stages.append(stage)
    )
    assert stages == list(baseline.STAGES)
    assert result["report"] == {
        "exact_copies_dropped": 1,
        "lots_sharing_a_key": 2,
        "lots_without_key": 1,
    }
    lots = result["lots"].set_index("source_record_ref")
    blocks = result["blocks"].set_index("source_record_ref")
    assert sorted(lots.index) == [
        "1-A-7-1",
        "1-A-7-2",
        "1-A-7-2#2",
        "1-B-8-1",
        "sin_clave",
    ]
    assert sorted(blocks.index) == CVEGEO
    assert lots.crs == blocks.crs == "EPSG:4326"
    for key, row in lots.iterrows():
        assert row["id"] == cadastral.lot_id("mexicali", key)
    assert blocks.loc[CVEGEO[0], "id"] == cadastral.block_id(CVEGEO[0])
    assert (
        lots.loc[["1-A-7-1", "1-A-7-2"], "parent_id"] == blocks.loc[CVEGEO[0], "id"]
    ).all()
    # The pharmacy makes its residential lot mixed use; the shop zone is commerce.
    # It is in box(0, 20, 20, 40): the first "1-A-7-2" by x.
    assert lots.loc["1-A-7-2", "amenity_type"] == "pharmacy"
    assert lots.loc["1-A-7-2", "land_use"] == "mixed_use"
    assert lots.loc["1-B-8-1", "land_use"] == "commerce"
    assert lots.loc["1-B-8-1", "commerce_type"] == "shop"
    # Heights: 5 m everywhere is 2 floors over the whole lot.
    assert lots.loc["1-A-7-1", "cos"] == pytest.approx(1.0)
    # num_levels is an integer attribute: whole floors.
    assert all(float(v).is_integer() for v in lots["num_levels"].dropna())
    assert lots.loc["1-A-7-1", "num_levels"] == 2
    assert lots.loc["1-A-7-1", "area"] == pytest.approx(400.0)
    assert lots.loc["1-A-7-1", "occupancy_rate"] == pytest.approx(0.75)
    assert blocks.loc[CVEGEO[0], "people_per_household"] == 3.25
    # Same sources, same ids.
    again = baseline.build(city, tmp_path / "again", lambda *_: None)
    assert again["lots"]["id"].tolist() == result["lots"]["id"].tolist()


def test_baseline_metric(city, monkeypatch):
    monkeypatch.setitem(CITIES, "mexicali", city)
    runner = LocalRunner.from_definitions(
        {"reimagina-test-port": plugin.create_plugin()}
    )
    tables = runner.run(
        "reimagina_baseline_elements", {"parameters": {"city": "mexicali"}}
    )
    assert len(tables["reimagina_baseline_elements_lots"]) == 5
    assert len(tables["reimagina_baseline_elements_blocks"]) == 2
