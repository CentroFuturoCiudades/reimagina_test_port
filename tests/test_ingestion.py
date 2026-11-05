"""reimagina-lyra's ingestion tests for the copied prep, footprints and elements."""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from reimagina_test_port.common import vocabulary
from reimagina_test_port.ingestion import crosswalk, elements, footprints, prep

CRS = "EPSG:32611"
PARAMS = footprints.FootprintParams()


def expect_equal(actual: object, expected: object) -> None:
    if actual != expected:
        raise AssertionError((actual, expected))


# --------------------------------------------------------------------------- prep


def test_prepare_census_builds_cvegeo_and_adults() -> None:
    census = pd.DataFrame(
        {"loc": [1], "ageb": ["12A"], "mza": [7], "p_18ymas": [100], "p_18a24": [20],
         "p_60ymas": [15], "graproes": [9.6]}
    )  # fmt: skip
    out = prep.prepare_census(census, "02", "002")
    expect_equal(out.loc[0, "cvegeo"], "020020001012a007")
    expect_equal(out.loc[0, "p_25a59"], 65)
    expect_equal(out.loc[0, "graproes"], 10)


def test_assign_normativa_containment_nearest_and_none() -> None:
    lots = gpd.GeoDataFrame(
        {"lot_id": ["inside", "near", "far"]},
        geometry=[box(0, 0, 10, 10), box(100.5, 0, 103, 10), box(500, 0, 510, 10)],
        crs=CRS,
    )
    normativa = gpd.GeoDataFrame(
        {"USOSUELO24": ["Habitacional", "Mixto", "Comercio y servicios"],
         "Max_cos": [0.7, None, 0.9]},
        geometry=[box(-5, -5, 20, 20), box(-5, -5, 20, 20), box(20, 0, 100, 10)],
        crs=CRS,
    )  # fmt: skip
    prepared = prep.prepare_normativa(
        normativa,
        target_crs=CRS,
        columns={"USOSUELO24": "zoning", "Max_cos": "max_cos"},
    )
    out = prep.assign_normativa(lots, prepared, max_distance=3.0).set_index("lot_id")
    # Two polygons contain the lot: the more complete one (with Max_cos) wins.
    expect_equal(out.loc["inside", "zoning"], "Habitacional")
    # Representative point 1.75 m from the commerce polygon, within 3 m.
    expect_equal(out.loc["near", "zoning"], "Comercio y servicios")
    expect_equal(pd.isna(out.loc["far", "zoning"]), np.True_)


# --------------------------------------------------------------------- footprints


def write_tile(
    path: Path, west: float, heights: np.ndarray, presence: np.ndarray
) -> Path:
    """A 3-band Open Buildings-like tile with 0.5 m pixels, top at y=100."""
    with rasterio.open(
        path, "w", driver="GTiff", width=heights.shape[1], height=heights.shape[0],
        count=3, dtype="float32", crs=CRS, nodata=-99.0,
        transform=from_origin(west, 100, 0.5, 0.5),
    ) as tile:  # fmt: skip
        tile.write(np.zeros_like(heights, dtype="float32"), 1)
        tile.write(heights.astype("float32"), 2)
        tile.write(presence.astype("float32"), 3)
    return path


@pytest.fixture
def mosaic(tmp_path: Path) -> tuple[Path, Path]:
    """Two 20x20-pixel tiles side by side: x 0-10 m and 10-20 m, y 90-100 m."""
    left = np.full((20, 20), 5.0)  # 2 floors at 2.7 m, exact in float32
    right = np.full((20, 20), 8.0)  # 3 floors
    right[:, 10:] = -99.0  # nodata half
    confidence = np.full((20, 20), 0.9)
    confidence[:10, :] = 0.2  # top half below the presence threshold
    tiles = [
        write_tile(tmp_path / "a.tif", 0, left, confidence),
        write_tile(tmp_path / "b.tif", 10, right, np.full((20, 20), 0.9)),
    ]
    return (
        footprints.write_vrt(tiles, footprints.HEIGHT_BAND, tmp_path / "h.vrt"),
        footprints.write_vrt(tiles, footprints.PRESENCE_BAND, tmp_path / "p.vrt"),
    )


def test_write_vrt_mosaics_tiles(mosaic: tuple[Path, Path]) -> None:
    with rasterio.open(mosaic[0]) as vrt:
        expect_equal(
            (vrt.width, vrt.height, vrt.crs.to_epsg(), vrt.nodata), (40, 20, 32611, -99)
        )
        band = vrt.read(1)
    expect_equal(
        (float(band[0, 0]), float(band[0, 25]), float(band[0, 35])), (5.0, 8.0, -99.0)
    )


def test_lot_stats_masks_presence_and_spans_tiles(mosaic: tuple[Path, Path]) -> None:
    with rasterio.open(mosaic[0]) as height, rasterio.open(mosaic[1]) as presence:
        # Left tile, rows 90-95 m only: presence 0.9, 5.0 m -> 2 floors.
        lower_left = footprints.lot_stats(box(0, 90, 10, 95), height, presence, PARAMS)
        # Upper half of the left tile: presence 0.2, no building.
        upper_left = footprints.lot_stats(box(0, 95, 10, 100), height, presence, PARAMS)
        # Across both tiles: 100 px at 5.0 m and 100 px at 8.0 m (x 15-20 m is nodata).
        across = footprints.lot_stats(box(5, 90, 20, 95), height, presence, PARAMS)
        off_raster = footprints.lot_stats(
            box(500, 0, 510, 10), height, presence, PARAMS
        )
    expect_equal(lower_left, {"max": 5.0, "p95": 5.0, "building_area": 50.0,
                              "builtup_area": 100.0})  # fmt: skip
    expect_equal(
        upper_left, {"max": 0.0, "p95": 0.0, "building_area": 0.0, "builtup_area": 0.0}
    )
    expect_equal(across["building_area"], 50.0)
    expect_equal(across["builtup_area"], 25.0 * 2 + 25.0 * 3)
    expect_equal(across["max"], 8.0)
    expect_equal(np.isnan(off_raster["max"]), True)


def test_compute_height_stats_matches_lot_stats(mosaic: tuple[Path, Path]) -> None:
    lots = gpd.GeoDataFrame(
        {"lot_area": [50.0, 75.0], "max_cos": [0.8, 0.8], "max_cus": [2.0, 2.0],
         "max_density": [100.0, np.nan]},
        geometry=[box(0, 90, 10, 95), box(5, 90, 20, 95)],
        crs=CRS,
    )  # fmt: skip
    stats = footprints.compute_height_stats(
        lots, mosaic[0], mosaic[1], PARAMS, workers=1
    )
    expect_equal(stats["building_area"].tolist(), [50.0, 50.0])
    urban = footprints.add_urban_metrics(lots, stats, average_level_height=2.7)
    expect_equal(urban["cos"].tolist(), [1.0, 50.0 / 75.0])
    expect_equal(urban["num_levels"].tolist(), [2.0, 2.5])
    expect_equal(urban["max_home_units"].iloc[0], 0.0)


def test_tiles_for_lots_reads_matching_manifest(tmp_path: Path) -> None:
    source = {
        "affineTransform": {
            "translateX": 0,
            "translateY": 100,
            "scaleX": 0.5,
            "scaleY": -0.5,
        },
        "dimensions": {"width": 20, "height": 20},
    }
    far = {
        **source,
        "affineTransform": {**source["affineTransform"], "translateX": 9000},
    }
    manifest = {
        "name": "81_EPSG_32611_2023",
        "tilesets": [{"sources": [{**source, "uris": ["x/near.tif"]},
                                  {**far, "uris": ["x/far.tif"]}]}],
    }  # fmt: skip
    other_zone = {**manifest, "name": "81_EPSG_32612_2023"}
    (tmp_path / "a.json").write_text(json.dumps(manifest))
    (tmp_path / "b.json").write_text(json.dumps(other_zone))
    lots = gpd.GeoDataFrame(geometry=[box(1, 91, 3, 93)], crs=CRS)
    names = footprints.tiles_for_lots([tmp_path / "a.json", tmp_path / "b.json"], lots)
    expect_equal(names, ["near.tif"])


# ----------------------------------------------------------------------- elements


def lots_frame(**columns: list) -> gpd.GeoDataFrame:
    n = len(columns["lot_id"])
    geometry = columns.pop(
        "geometry", [box(i * 20, 0, i * 20 + 10, 10) for i in range(n)]
    )
    return gpd.GeoDataFrame(columns, geometry=geometry, crs=CRS)


def contents_frame(*rows: tuple) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["lot_id", "element_type", "category", "scian_code", "num_workers"],
    )


def test_classify_land_use_from_contents() -> None:
    lots = lots_frame(
        lot_id=["sf", "mf", "mixed", "school", "shop", "office", "plant", "big",
                "empty", "vacant"],
        cvegeo=["b1", "b1", "b2", "b3", "b3", "b3", "b3", "b4", "b5", "b6"],
        vivtot=[10, 10, 2, 0, 0, 0, 0, 1, 5, 0],
        builtup_area=[100.0, 900.0, 300.0, 300.0, 50.0, 80.0, 500.0, 900.0, 10.0,
                      200.0],
        num_levels=[1.0, 3.0, 2.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 1.0],
    )  # fmt: skip
    zoning = pd.Series(
        ["habitacional"] * 3 + ["comercio y servicios"] * 4 + ["habitacional"] * 2
        + ["industrial"]
    )  # fmt: skip
    contents = contents_frame(
        # A clinic doesn't change the class of a lot with homes. Offices are not
        # classified: the design does not say which establishments are offices.
        ("sf", "amenity", "primary_healthcare", "621112", 3),
        ("mixed", "establishment", "medicines", "464111", 3),
        ("school", "amenity", "primary_education", "611122", 40),
        ("shop", "establishment", "convenience_store", "461110", 3),
        ("shop", "amenity", "care", "624411", 8),
        ("office", "establishment", "financial", "522110", 8),
        ("office", "establishment", "other", "541110", 3),
        ("plant", "establishment", "manufacturing", "311830", 20),
        ("plant", "establishment", "food_retail", "461150", 3),
    )
    land_use, unit_size = elements.classify_land_use(
        lots, zoning, contents, mf_min_units=1.5
    )
    # b1: 10 homes over 2 built lots; the big one gets 9 -> multifamily. "big" has
    # one home in 900 m² over 3 floors -> multifamily. "empty" is unbuilt with
    # nothing: its zone's use; "vacant" is built with nothing: industrial zone.
    expect_equal(
        land_use.tolist(),
        ["single_family", "multifamily", "mixed_use", "amenity", "commerce", "commerce",
         "industrial", "multifamily", "single_family", "industrial"],
    )  # fmt: skip
    # Remaining homes = 10 - 1 reserved for the single-family lot -> 900 / 9 m².
    expect_equal(unit_size.iloc[1], 100.0)
    expect_equal(pd.isna(unit_size.iloc[0]), True)


def test_lot_attributes_keep_conditional_keys() -> None:
    residential = elements.lot_attributes(
        "multifamily", 0.6, 3.2, 200.0, 75.0, 3.456, 0.9, "CASA HABITACION"
    )
    expect_equal(
        residential,
        {"land_use": "multifamily", "cos": 0.6, "num_levels": 3.2, "area": 200.0,
         "unit_size_m2": 75.0, "people_per_household": 3.46, "occupancy_rate": 0.9},
    )  # fmt: skip
    commerce = elements.lot_attributes(
        "commerce", 0.5, 1.0, 100.0, 80.0, 3.0, 0.9, "tienda comercial"
    )
    expect_equal(commerce["commerce_type"], "shop")
    expect_equal(set(commerce) & {"unit_size_m2", "people_per_household"}, set())


# ---------------------------------------------------------------------- crosswalk


def test_crosswalk_longest_prefix_and_other() -> None:
    codes = pd.Series(["461110", "461150", "462112", "611122", "311830", "811111",
                       " 464111", "461122", "461160"])  # fmt: skip
    out = crosswalk.classify(codes)
    expect_equal(
        list(zip(out["element"], out["category"], strict=True)),
        [("establishment", "food_retail"), ("establishment", "fresh_food"),
         ("establishment", "basic_household_goods"),
         ("amenity", "primary_education"),
         ("establishment", "manufacturing"), ("establishment", "other"),
         ("establishment", "medicines"), ("establishment", "fresh_food"),
         ("establishment", "food_retail")],
    )  # fmt: skip


def test_crosswalk_education_and_care() -> None:
    # Preschool and special education are care; upper secondary tertiary;
    # universities are not in the workbook. Community food services (62421) are
    # care, the rest of 6242 community facilities.
    codes = pd.Series(["611112", "611182", "611132", "611162", "611312", "624212",
                       "624232", "813210"])  # fmt: skip
    expect_equal(
        crosswalk.classify(codes)["category"].tolist(),
        ["care", "care", "secondary_education", "tertiary_education", "other",
         "care", "community_facility", "community_facility"],
    )  # fmt: skip


def test_crosswalk_rejects_unknown_category(tmp_path: Path) -> None:
    path = tmp_path / "crosswalk.csv"
    path.write_text("scian_prefix,element,category,note\n61,amenity,school,x\n")
    with pytest.raises(ValueError, match="school"):
        crosswalk.table(path)


def test_amenity_type_of_scian_then_category() -> None:
    expect_equal(vocabulary.amenity_type_of("primary_education", "611112"), "preschool")
    expect_equal(vocabulary.amenity_type_of("medicines", "464111"), "pharmacy")
    expect_equal(vocabulary.amenity_type_of("public_space", None), "park")
    expect_equal(vocabulary.amenity_type_of("sports_recreation", "713943"), "gym")
    # Schools of the amenities file: the DENUE already has them.
    expect_equal(vocabulary.amenity_type_of("primary_education", None), None)
    expect_equal(vocabulary.amenity_type_of("other", "811111"), None)
