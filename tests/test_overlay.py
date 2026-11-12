"""A redesign (master plan) replacing the city's elements inside its zone."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from reimagina_test_port.ingestion import overlay

CRS = "EPSG:32611"


def table(rows: list[dict], geometries: list) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(rows, geometry=geometries, crs=CRS).to_crs(4326)


def city() -> dict[str, gpd.GeoDataFrame]:
    """Two blocks: ``in`` inside the zone (x 0-100), ``out`` outside (x 200-300),
    each with a lot, an amenity and an establishment."""
    return {
        "blocks": table(
            [
                {"id": "in", "parent_id": None, "source_record_ref": "1", "vivtot": 5},
                {"id": "out", "parent_id": None, "source_record_ref": "2", "vivtot": 9},
            ],
            [box(0, 0, 100, 100), box(200, 0, 300, 100)],
        ),
        "lots": table(
            [
                {
                    "id": "lot-in",
                    "parent_id": "in",
                    "source_record_ref": "a",
                    "land_use": "single_family",
                    "tree_coverage": 10.0,
                },
                {
                    "id": "lot-out",
                    "parent_id": "out",
                    "source_record_ref": "b",
                    "land_use": "commerce",
                    "tree_coverage": 20.0,
                },
                # Outside the zone, but its block is inside: it goes too.
                {
                    "id": "lot-edge",
                    "parent_id": "in",
                    "source_record_ref": "c",
                    "land_use": "office",
                    "tree_coverage": None,
                },
            ],
            [box(10, 10, 50, 50), box(210, 10, 250, 50), box(110, 10, 150, 50)],
        ),
        "amenities": table(
            [
                {
                    "id": "park-in",
                    "parent_id": "lot-in",
                    "source_record_ref": "p",
                    "category": "public_space",
                    "capacity": 1,
                },
                {
                    "id": "park-out",
                    "parent_id": "lot-out",
                    "source_record_ref": "q",
                    "category": "public_space",
                    "capacity": 1,
                },
            ],
            [Point(30, 30), Point(230, 30)],
        ),
        "establishments": table(
            [
                # Outside the zone, but its lot went.
                {
                    "id": "shop-edge",
                    "parent_id": "lot-edge",
                    "source_record_ref": "s",
                    "category": "other",
                    "scian_code": "811111",
                    "scian_sector": "81",
                },
                {
                    "id": "shop-out",
                    "parent_id": "lot-out",
                    "source_record_ref": "t",
                    "category": "other",
                    "scian_code": "461110",
                    "scian_sector": "46",
                },
            ],
            [Point(130, 30), Point(230, 30)],
        ),
    }


def plan() -> dict[str, gpd.GeoDataFrame]:
    """The redesign: one block with one lot, a clinic and a pharmacy, in Reimagina's
    element format (type and name columns the baseline does not have)."""
    return {
        "blocks": table(
            [{"id": "pm-b", "type": "block", "parent_id": None, "name": None,
              "lot.size_m2": None}],
            [box(5, 5, 95, 95)],
        ),
        "lots": table(
            [{"id": "pm-l", "type": "lot", "parent_id": "pm-b", "name": None,
              "land_use": "mixed_use", "housing.unit_size_m2": 90.0}],
            [box(10, 10, 90, 90)],
        ),
        "amenities": table(
            [{"id": "pm-a", "type": "amenity", "parent_id": "pm-l", "name": "Clinica",
              "category": "primary_healthcare", "tree_coverage": 15.0}],
            [Point(50, 50)],
        ),
        "establishments": table(
            [{"id": "pm-e", "type": "establishment", "parent_id": "pm-l",
              "name": None, "category": "medicines", "scian_code": "464111"}],
            [Point(60, 60)],
        ),
    }  # fmt: skip


ZONE = box(-1, -1, 101, 101)


def test_the_zone_is_replaced():
    tables, report = overlay.replace(city(), plan(), ZONE, CRS)
    assert report == {
        "blocks": {"removed": 1, "added": 1},
        "lots": {"removed": 2, "added": 1},
        "amenities": {"removed": 1, "added": 1},
        "establishments": {"removed": 1, "added": 1},
    }
    assert sorted(tables["blocks"]["id"]) == ["out", "pm-b"]
    assert sorted(tables["lots"]["id"]) == ["lot-out", "pm-l"]
    assert sorted(tables["amenities"]["id"]) == ["park-out", "pm-a"]
    assert sorted(tables["establishments"]["id"]) == ["pm-e", "shop-out"]
    # The baseline's columns, in its order; the redesign's others are left out.
    for name, frame in tables.items():
        assert list(frame.columns) == list(city()[name].columns)
        assert frame.crs.to_epsg() == 4326
    lots = tables["lots"].set_index("id")
    assert lots.loc["pm-l", "source_record_ref"] == "overlay:pm-l"
    assert lots.loc["pm-l", "parent_id"] == "pm-b"
    assert lots.loc["pm-l", "land_use"] == "mixed_use"
    assert pd.isna(lots.loc["pm-l", "tree_coverage"])
    assert lots.loc["lot-out", "tree_coverage"] == 20
    shops = tables["establishments"].set_index("id")
    assert shops.loc["pm-e", "scian_sector"] == "46"
    assert "tree_coverage" not in tables["amenities"]


def test_a_redesign_that_does_not_fit_is_refused():
    broken = plan()
    broken["lots"]["parent_id"] = "nowhere"
    with pytest.raises(overlay.OverlayError, match="lots outside its blocks"):
        overlay.replace(city(), broken, ZONE, CRS)
    clash = plan()
    clash["blocks"]["id"] = "out"
    clash["lots"]["parent_id"] = "out"
    with pytest.raises(overlay.OverlayError, match="blocks ids already in the city"):
        overlay.replace(city(), clash, ZONE, CRS)


def test_read_from_a_folder(tmp_path):
    for name, frame in plan().items():
        frame.to_parquet(tmp_path / f"{name}.parquet")
    gpd.GeoDataFrame(geometry=[ZONE], crs=CRS).to_file(tmp_path / "zone.fgb")
    tables = overlay.read(str(tmp_path), tmp_path / "work")
    assert {name: len(frame) for name, frame in tables.items()} == {
        "blocks": 1, "lots": 1, "amenities": 1, "establishments": 1,
    }  # fmt: skip
    zone = overlay.read_zone(str(tmp_path / "zone.fgb"), tmp_path / "work", CRS)
    assert zone.equals_exact(ZONE, 1e-6)
    assert overlay.file_uri("az://x/plan/", "road_segments") == (
        "az://x/plan/road_segments.parquet"
    )
