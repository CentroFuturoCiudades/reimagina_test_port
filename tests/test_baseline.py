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
import yaml
from lyra.sdk import LocalRunner
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, box

from reimagina_test_port import plugin
from reimagina_test_port.common import lake
from reimagina_test_port.ingestion import (
    baseline,
    cadastral,
    cities,
    network,
    prep,
    raw_amenities,
    street_fit,
    streets,
)
from reimagina_test_port.ingestion.cities import (
    CITIES,
    BikeLanes,
    Source,
    StreetLighting,
    TreeCanopy,
)

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
# IMIP layer -> its name column (None: esp_pub has none).
IMIP_NAMES = {
    "edu_basica": "NOM_ESC", "edu_sup": "NOM_ESC", "med_sup": "NOM_ESC",
    "deporte": "NOMBRE", "cult_rec": "NOMBRE", "esp_pub": None,
}  # fmt: skip


def write_imip(folder: Path, **layers: tuple[list, list, list]) -> None:
    """IMIP's six layers, empty unless given as (OBJECTID, TIPO, points)."""
    folder.mkdir()
    for name, name_column in IMIP_NAMES.items():
        ids, tipos, geometry = layers.get(name, ([], [], []))
        columns = {"OBJECTID": ids, "TIPO": tipos}
        if name_column:
            columns[name_column] = [f"{name} {i}" for i in ids]
        if name == "esp_pub":
            columns["Sup_M2"] = [1500.0 + 100 * i for i in range(len(ids))]
        # With Z, as the lake's layers.
        points = [Point(p.x, p.y, 1) for p in geometry]
        gpd.GeoDataFrame(columns, geometry=points, crs=CRS).to_parquet(
            folder / f"{name}.geoparquet"
        )


def test_existing_land_use_then_zoning_plan():
    # The existing use covers lots at x 0 and 50; the plan, lots at x 0 and 10.
    lots = gpd.GeoDataFrame(
        {
            "lot_id": ["a", "b", "c", "d"],
            "zoning": ["Mixto", "Habitacional", None, None],
        },
        geometry=[box(x, 0, x + 5, 5) for x in (0, 10, 50, 70)],
        crs=CRS,
    )
    existing = gpd.GeoDataFrame(
        {"USOSUELO24": ["Habitacional", "Baldíos"]},
        geometry=[box(-1, -1, 6, 6), box(49, -1, 56, 6)],
        crs=CRS,
    )
    zoning, report = prep.assign_existing_land_use(
        lots, existing, column="USOSUELO24", target_crs=CRS, max_distance=3.0
    )
    # The existing use first; the plan where it has none; else none.
    assert zoning.fillna("-").tolist() == [
        "Habitacional",
        "Habitacional",
        "Baldíos",
        "-",
    ]
    assert report == {
        "from_existing_land_use": 2,
        "from_zoning_plan": 1,
        "without_zoning": 1,
    }


def test_imip_normalizer(tmp_path):
    write_imip(
        tmp_path / "imip",
        edu_basica=([1, 2, 3, 4, 5],
                    ["PRIMARIA", "PREESCOLAR (JARDIN DE NIÑOS)",
                     "ESCUELA SECUNDARIA TECNICA", "TELESECUNDARIA",
                     "CENTRO DE DESARROLLO INFANTIL (CENDI)"],
                    [Point(i, 0) for i in range(5)]),
        esp_pub=([1, 2, 3], ["PARQUE URBANO", "GLORIETAS", "ESPACIO DEPORTIVO"],
                 [Point(0, 0), Point(1, 0), Point(2, 0)]),
        deporte=([9], ["UNIDAD DEPORTIVA"], [Point(0, 0)]),
    )  # fmt: skip
    raw = raw_amenities.imip(tmp_path / "imip")
    raw_amenities.check(raw)
    table = raw.set_index("source_id")
    assert table["category"].to_dict() == {
        "edu_basica:1": "primary_education",
        "edu_basica:3": "secondary_education",
        "edu_basica:2": "care",
        "esp_pub:1": "public_space",
        "esp_pub:3": "sports_recreation",
        "deporte:9": "sports_recreation",
    }
    assert table.loc["edu_basica:1", "name"] == "edu_basica 1"
    assert table.loc["esp_pub:1", "area_m2"] == 1500
    assert pd.isna(table.loc["esp_pub:1", "name"])
    assert pd.isna(table.loc["deporte:9", "area_m2"])
    # area_m2 is for public spaces only.
    assert pd.isna(table.loc["esp_pub:3", "area_m2"])
    assert (table["source"] == "IMIP").all()


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
    # The zoning plan covers the first block; the existing land use, one lot of it
    # ("1-A-7-1") and the second block, for commerce.
    gpd.GeoDataFrame(
        {
            "USOSUELO24": ["Habitacional"],
            "Max_cos": [0.8],
            "Max_cus": [1.6],
            "Dens_U_max": [60.0],
        },
        geometry=[box(-5, -5, 45, 45)],
        crs=CRS,
    ).to_file(root / "normativa.geojson", engine="pyogrio")
    gpd.GeoDataFrame(
        {"USOSUELO24": ["Habitacional", "Comercio y servicios"]},
        geometry=[box(-1, -1, 21, 21), box(55, -5, 105, 45)],
        crs=CRS,
    ).to_file(root / "existing_land_use.geojson", engine="pyogrio")
    # IMIP's layers: a garden 5 m from a lot and a school far from every lot; a
    # median and a CENDI, which no rule keeps.
    write_imip(
        root / "imip",
        esp_pub=([12, 13], ["JARDIN VECINAL", "CAMELLONES"], [Point(105, 10), Point(10, 10)]),
        edu_basica=([7, 8], ["PRIMARIA", "CENTRO DE DESARROLLO INFANTIL (CENDI)"],
                    [Point(500, 500), Point(10, 10)]),
    )  # fmt: skip
    # Five Mexicali records (one far from every lot) and one elsewhere. The
    # crosswalk makes the school an amenity; it is 30 m from the nearest lot.
    businesses = gpd.GeoDataFrame(
        {
            "id": ["101", "102", "103", "104", "105", "106", "107"],
            "nom_estab": ["Tienda", "Taller", "Lejos", "Tijuana", "Farmacia",
                          "Escuela", "Unidad deportiva"],
            "codigo_act": ["461110", "811111", "461110", "461110", "464111",
                           "611122", "713942"],
            "per_ocu": ["0 a 5 personas", "11 a 30 personas", "0 a 5 personas",
                        "6 a 10 personas", "6 a 10 personas", "31 a 50 personas",
                        "6 a 10 personas"],
            "municipio": ["Mexicali", "Mexicali", "Mexicali", "Tijuana", "Mexicali",
                          "Mexicali", "Mexicali"],
        },
        geometry=[Point(70, 10), Point(15, 5), Point(500, 500), Point(70, 10),
                  Point(10, 30), Point(70, 50), Point(10, 10)],
        crs=CRS,
    ).to_crs(4326)  # fmt: skip
    businesses.assign(
        latitud=businesses.geometry.y, longitud=businesses.geometry.x
    ).drop(columns="geometry").to_csv(
        root / "denue.csv", index=False, encoding="latin-1"
    )
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
    # Canopy height at 1 m: 10 m trees west of x = 10, none east of it.
    canopy = np.where(np.arange(120) < 20, 10, 0).astype("uint8")[None, :].repeat(60, 0)
    with rasterio.open(
        root / "canopy.tif", "w", driver="GTiff", width=120, height=60, count=1,
        dtype="uint8", crs=CRS, transform=from_origin(-10, 50, 1, 1),
    ) as raster:  # fmt: skip
        raster.write(canopy, 1)
    return {
        "tree_canopy": TreeCanopy(str(root / "canopy.tif")),
        "lots_uri": str(root / "lots" / "PREDIOS.shp"),
        "normativa_uri": str(root / "normativa.geojson"),
        "existing_land_use_uri": str(root / "existing_land_use.geojson"),
        "amenities_uri": str(root / "imip"),
        "denue_uri": str(root / "denue.csv"),
        "tiles_uri": str(root / "tiles"),
        "manifests_uri": str(root / "manifests"),
        **write_streets(root),
    }


# The street network (reimagina_street_network): three streets, in the city's CRS;
# the first both ways, the third from its higher node.
NETWORK_NODES = pd.DataFrame(
    {"x": [0.0, 40, 60, 100, 0, 40], "y": [-5.0, -5, -5, -5, 50, 50]},
    index=pd.Index([1, 2, 3, 4, 5, 6], name="id"),
)
NETWORK_EDGES = pd.DataFrame(
    {"from": [1, 2, 3, 6], "to": [2, 1, 4, 5], "length": [40.0, 40, 40, 40]}
)
STREET_KEYS = ["1-2", "3-4", "5-6"]


def network_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The network as reimagina_street_network returns it."""
    edges = NETWORK_EDGES.rename(columns={"length": "distance"}).assign(two_way=True)
    return NETWORK_NODES.reset_index(), edges


def write_streets(root: Path) -> dict:
    """The block faces (lit, not lit, unanswered, another municipality's) and two
    bike lanes along the network's streets, as local files."""
    # Each face 5-7 m from its street and 40 m or more from the others.
    gpd.GeoDataFrame(
        {"CVE_MUN": ["002", "002", "002", "002", "003", "002"],
         "ALUMPUB": ["1", "3", "1", "9", "3", "8"]},
        geometry=[LineString([(5, -12), (35, -12)]), LineString([(5, 0), (35, 0)]),
                  LineString([(65, -12), (95, -12)]), LineString([(65, 0), (95, 0)]),
                  LineString([(65, -8), (95, -8)]), LineString([(5, 45), (35, 45)])],
        crs=CRS,
    ).to_file(root / "faces.geojson")  # fmt: skip
    gpd.GeoDataFrame(
        {"TIPO": ["SEGREGADO INDEPENDIENTE", "COMPARTIDO"]},
        geometry=[LineString([(0, 0), (40, 0)]), LineString([(0, 60), (40, 60)])],
        crs=CRS,
    ).to_file(root / "lanes.geojson")
    return {
        "street_lighting": StreetLighting(
            Source(str(root / "faces.geojson")),
            column="ALUMPUB",
            lit=("1",),
            unlit=("3",),
            where={"CVE_MUN": "002"},
            max_distance_m=25,
        ),
        "bike_lanes": BikeLanes(
            Source(str(root / "lanes.geojson")),
            type_column="TIPO",
            protected=("SEGREGADO INDEPENDIENTE",),
            max_distance_m=15,
        ),
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
    # OpenStreetMap is the network above; no municipality boundary to read.
    monkeypatch.setattr(
        network,
        "osm_network",
        lambda blocks, crs, params, boundary, **_: (NETWORK_NODES, NETWORK_EDGES),
    )
    return replace(preset, streets=replace(preset.streets, boundary_uri=None))


def test_build(city, tmp_path):
    stages = []
    result = baseline.build(
        city, tmp_path / "job", lambda stage, *_: stages.append(stage)
    )
    assert stages == list(baseline.STAGES)
    assert result["report"] == {
        "lots": {
            "exact_copies_dropped": 1,
            "lots_sharing_a_key": 2,
            "lots_without_key": 1,
        },
        "heights": "computed",
        "zoning": {
            "from_existing_land_use": 3,
            "from_zoning_plan": 2,
            "without_zoning": 0,
        },
        "amenities_file": {"public_space": 1, "primary_education": 1},
        # One source per kind: the IMIP park, not its school; the DENUE school,
        # not its sports club (sports come from the file).
        "amenities": {
            "from_file": 1,
            "file_skipped": 1,
            "from_denue": 1,
            "denue_skipped": 1,
            "source": 2,
            "elements": 2,
            "outside_every_lot": 0,
        },
        # Seven DENUE rows in the state, six of Mexicali: two amenities.
        "establishments": {
            "denue_state": 7,
            "denue_municipio": 6,
            "without_coordinates": 0,
            "denue_amenities": 2,
            "denue_establishments": 4,
            "source": 4,
            "elements": 3,
            "outside_every_lot": 1,
        },
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
    # Land use from contents. Homes and a business: mixed use ("1-A-7-2" is
    # box(0, 20, 20, 40), the first by x, with the pharmacy). Homes only: single
    # family. Only a business: commerce, whatever the school next to it. Only the
    # park: amenity, though zoned for commerce.
    assert "amenity_type" not in lots
    assert lots.loc["1-A-7-1", "land_use"] == "mixed_use"
    assert lots.loc["1-A-7-2", "land_use"] == "mixed_use"
    assert lots.loc["1-A-7-2#2", "land_use"] == "single_family"
    assert lots.loc["1-B-8-1", "land_use"] == "commerce"
    assert lots.loc["1-B-8-1", "commerce_type"] == "shop"
    assert lots.loc["sin_clave", "land_use"] == "amenity"
    # Heights: 5 m everywhere is 2 floors over the whole lot.
    assert lots.loc["1-A-7-1", "cos"] == pytest.approx(1.0)
    # num_levels is an integer attribute: whole floors.
    assert all(float(v).is_integer() for v in lots["num_levels"].dropna())
    assert lots.loc["1-A-7-1", "num_levels"] == 2
    assert lots.loc["1-A-7-1", "area"] == pytest.approx(400.0)
    assert lots.loc["1-A-7-1", "occupancy_rate"] == pytest.approx(0.75)
    # Space allocation by land use: residential 100, mixed use 80, the rest 0.
    assert lots.loc["1-A-7-1", "housing.floor_share"] == 80
    assert lots.loc["1-A-7-2#2", "housing.floor_share"] == 100
    assert lots.loc["1-B-8-1", "housing.floor_share"] == 0
    assert lots.loc["sin_clave", "housing.floor_share"] == 0
    # Canopy west of x = 10: half of the lots starting at x = 0, none of the rest.
    west = lots.geometry.to_crs(CRS).bounds["minx"].round(6) == 0
    assert lots.loc[west, "tree_coverage"].tolist() == [50.0, 50.0]
    assert (lots.loc[~west, "tree_coverage"] == 0).all()
    assert blocks.loc[CVEGEO[0], "housing.people_per_household"] == 3.25
    # Targets: declared, unset when loaded.
    assert blocks[["lot.size_m2", "lot.frontage_m"]].isna().all().all()
    # Amenities and establishments go to the lot holding them, or the nearest
    # within 50 m, as a point at its center; farther ones are not loaded.
    centers = result["lots"].set_index("id").geometry.representative_point()
    amenities = result["amenities"].set_index("source_record_ref")
    assert sorted(amenities.index) == ["IMIP:esp_pub:12", "denue:106"]
    assert amenities.crs == "EPSG:4326"
    assert not amenities.has_z.any()
    park = amenities.loc["IMIP:esp_pub:12"]
    assert park["id"] == cadastral.stable_element_uuid(
        "mexicali", "amenity", "IMIP:esp_pub:12"
    )
    # 5 m east of the last lot, box(80, 0, 100, 20) ("sin_clave").
    assert park["parent_id"] == lots.loc["sin_clave", "id"]
    assert park.geometry.equals_exact(centers[park["parent_id"]], 1e-12)
    assert park["category"] == "public_space"
    assert park["area_m2"] == 1500
    assert pd.isna(park["name"])
    assert park["capacity"] == 1
    assert pd.isna(park["scian_code"])
    school = amenities.loc["denue:106"]
    assert school["parent_id"] == lots.loc["1-B-8-1", "id"]
    assert school["category"] == "primary_education"
    assert school["scian_code"] == "611122"
    assert school["num_workers"] == 40
    assert school["name"] == "Escuela"
    assert school["capacity"] == 1
    establishments = result["establishments"].set_index("source_record_ref")
    assert sorted(establishments.index) == ["denue:101", "denue:102", "denue:105"]
    shop = establishments.loc["denue:101"]
    assert shop["parent_id"] == lots.loc["1-B-8-1", "id"]
    assert shop.geometry.equals_exact(centers[shop["parent_id"]], 1e-12)
    assert shop["category"] == "food_retail"
    assert shop["scian_code"] == "461110"
    assert shop["scian_sector"] == "46"
    assert shop["num_workers"] == 3
    assert shop["name"] == "Tienda"
    assert establishments.loc["denue:102", "category"] == "other"
    assert establishments.loc["denue:102", "num_workers"] == 20
    assert establishments.loc["denue:102", "parent_id"] == lots.loc["1-A-7-1", "id"]
    assert establishments.loc["denue:105", "category"] == "medicines"
    # Same sources, same ids.
    again = baseline.build(city, tmp_path / "again", lambda *_: None)
    # The same workdir reuses the height statistics, with the same result.
    reused = baseline.build(city, tmp_path / "job", lambda *_: None)
    assert reused["report"]["heights"] == "cached"
    pd.testing.assert_frame_equal(
        reused["lots"].drop(columns="geometry"), result["lots"].drop(columns="geometry")
    )
    assert again["lots"]["id"].tolist() == result["lots"]["id"].tolist()
    assert again["amenities"]["id"].tolist() == result["amenities"]["id"].tolist()


def test_streets(city, tmp_path):
    table = streets.build(city, *network_tables(), tmp_path / "work")
    table = table.set_index("source_record_ref")
    # One element per street, whatever the direction of its edges.
    assert table.index.tolist() == STREET_KEYS
    first, second, third = STREET_KEYS
    # Faces 1 and 3 next to it: half lit; 1 and 9 (unanswered): lit, the other
    # municipality's face aside; only an 8: no value.
    assert table.loc[first, "streetlights"] == 50
    assert table.loc[second, "streetlights"] == 100
    assert pd.isna(table.loc[third, "streetlights"])
    # tree_coverage over each street's polygon (x 0-40 or 60-100, 10 m wide): trees
    # west of x = 10. The third's polygon is half off the raster: its on-raster half.
    assert table.loc[first, "tree_coverage"] == 25
    assert table.loc[second, "tree_coverage"] == 0
    assert table.loc[third, "tree_coverage"] == 25
    # The protected lane runs 5 m from the first street: a band of its own. The
    # shared one, 10 m from the third, makes its lanes shared_lanes. The second's
    # midpoint is 40 m from the protected lane's end: lanes only.
    bands = {key: json.loads(use) for key, use in table["bands.use"].items()}
    assert bands[first] == ["sidewalk", "protected_bike_lane", "lanes", "sidewalk"]
    assert bands[third] == ["sidewalk", "shared_lanes", "sidewalk"]
    assert bands[second] == ["sidewalk", "lanes", "sidewalk"]
    for use, share in zip(table["bands.use"], table["bands.share"], strict=True):
        assert len(json.loads(use)) == len(json.loads(share))
        assert sum(json.loads(share)) == 100
    assert table.loc[first, "id"] == str(
        uuid.uuid5(NAMESPACE, f"mexicali:right_of_way:{first}")
    )
    assert table.crs.to_epsg() == 4326


def test_streets_take_the_network_shapes(city, tmp_path):
    nodes, edges = network_tables()
    # The second street bends 30 m south; the others have no shape (straight).
    bend = LineString([(60, -5), (80, -35), (100, -5)])
    shaped = gpd.GeoDataFrame(edges, geometry=[None, None, bend, None], crs=CRS).to_crs(
        4326
    )
    bare = replace(city, street_lighting=None, bike_lanes=None)
    table = streets.build(bare, nodes, shaped, tmp_path).set_index("source_record_ref")
    shapes = table.geometry.to_crs(CRS)
    # A strip of the class's default width (residential: 10 m) around each line.
    width = 2 * street_fit.DEFAULTS["default_half_width_m"]["residential"]
    assert shapes["3-4"].buffer(0.2).contains(bend.interpolate(0.5, normalized=True))
    assert street_fit.strip_width([shapes["3-4"]])[0] == pytest.approx(width, rel=0.1)
    plain = LineString([(0, -5), (40, -5)]).buffer(width / 2, cap_style="flat")
    # Less what its junction with the next street takes.
    assert shapes["1-2"].difference(plain).area < 1
    assert shapes["1-2"].area > 0.7 * plain.area


def test_streets_without_lighting_or_lanes(city, tmp_path):
    bare = replace(city, street_lighting=None, bike_lanes=None)
    table = streets.build(bare, *network_tables(), tmp_path)
    assert table["streetlights"].isna().all()
    assert set(table["bands.use"]) == {'["sidewalk", "lanes", "sidewalk"]'}


# A network to refine, south of the fixture's blocks (lots at y 0-40): one residential
# street in two edges meeting at node 2 (each both ways; 2-3's shape is drawn from 3),
# a trunk far away, and a footway.
REFINED_NODES = pd.DataFrame(
    {
        "id": [1, 2, 3, 5, 6, 7, 8],
        "x": [-10.0, 50, 110, 0, 100, 0, 50],
        "y": [-5.0, -5, -5, -300, -300, 50, 50],
    }
)
REFINED_EDGES = [  # from, to, highway, name, lanes, shape
    (1, 2, "residential", "Calle Uno", "2;3", None),
    (2, 1, "residential", "Calle Uno", "2;3", None),
    (2, 3, "residential", "Calle Uno", None, LineString([(110, -5), (50, -5)])),
    (3, 2, "residential", "Calle Uno", None, LineString([(110, -5), (50, -5)])),
    (5, 6, "trunk", "Autopista", "4", None),
    (6, 5, "trunk", "Autopista", "4", None),
    (7, 8, "footway", None, None, None),
    (8, 7, "footway", None, None, None),
]


def refined_network(rows=REFINED_EDGES) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Nodes and edges as reimagina_street_network returns them: EPSG:4326 shapes,
    OSM's highway, name and lanes."""
    xy = REFINED_NODES.set_index("id")[["x", "y"]]
    lines = [
        shape
        if shape is not None
        else LineString([xy.loc[a].to_numpy(), xy.loc[b].to_numpy()])
        for a, b, *_, shape in rows
    ]
    edges = gpd.GeoDataFrame(
        {
            "from": [r[0] for r in rows],
            "to": [r[1] for r in rows],
            "distance": [line.length for line in lines],
            "two_way": True,
            "highway": [r[2] for r in rows],
            "name": [r[3] for r in rows],
            "lanes": [r[4] for r in rows],
        },
        geometry=lines,
        crs=CRS,
    ).to_crs(4326)
    return REFINED_NODES, edges


@pytest.fixture
def refined_city(city):
    config = replace(city.streets, refine=True)
    return replace(city, streets=config, bike_lanes=None)


def test_roads_from_network_dedupes_directions():
    nodes, edges = refined_network()
    roads = streets.roads_from_network(nodes, edges, CRS).set_index("road_key")
    # Each way both directions is one road; the footway stays out.
    assert sorted(roads.index) == ["1-2", "2-3", "5-6"]
    assert roads.loc[["5-6"], "road_type"].tolist() == ["trunk"]
    # 2-3's shape is drawn from 3: it starts there.
    assert roads.loc["2-3", ["from_node", "to_node"]].tolist() == [3, 2]
    assert roads.loc["1-2", ["from_node", "to_node"]].tolist() == [1, 2]
    # "2;3" isn't a number: the class's default, not tagged.
    assert roads.loc[["1-2", "2-3", "5-6"], "lanes"].tolist() == [2, 2, 4]
    assert roads["lanes_tagged"].tolist() == [False, False, True]
    assert roads.loc["1-2", "name"] == "Calle Uno"
    assert roads.crs == CRS


def test_roads_from_network_classes():
    nodes, edges = refined_network(
        [
            (1, 2, None, None, None, None),  # a merged street: residential
            (2, 3, "motorway_link", None, None, None),
            (5, 6, "track", None, None, None),  # unknown: not vehicular
            (7, 8, "pedestrian", None, None, None),
        ]
    )
    roads = streets.roads_from_network(nodes, edges, CRS).set_index("road_key")
    assert roads["road_type"].to_dict() == {"1-2": "residential", "2-3": "motorway"}


def test_refined_streets(refined_city, tmp_path):
    table = streets.build(refined_city, *refined_network(), tmp_path / "work")
    # The two edges are one right of way, keyed by the first; the highway another;
    # the footway as ever, by its nodes.
    assert sorted(table.source_record_ref) == ["1-2", "5-6", "7-8"]
    assert table.attrs["fit"] == {"fitted": 2, "both_sides": 0, "unfitted": 1}
    table = table.set_index("source_record_ref", drop=False)
    row = table.loc["1-2"]
    assert row.id == str(uuid.uuid5(NAMESPACE, "mexicali:right_of_way:1-2"))
    shape = gpd.GeoSeries([row.geometry], crs=4326).to_crs(CRS).iloc[0]
    # A polygon, dead end to dead end: as long as its axis.
    assert shape.geom_type == "Polygon" and shape.is_valid
    assert street_fit.strip_length([shape])[0] == pytest.approx(120, abs=1)
    # Bands: a measured width's, one value per band, summing 100.
    for use, share in zip(table["bands.use"], table["bands.share"], strict=True):
        assert len(json.loads(use)) == len(json.loads(share))
        assert sum(json.loads(share)) == pytest.approx(100)
    assert {"sidewalk", "lanes"} <= set(json.loads(row["bands.use"]))
    assert table.crs.to_epsg() == 4326
    # Only the four attributes: nothing of width, lanes or name is written.
    assert list(table.columns) == [
        "id", "parent_id", "source_record_ref", "bands.use", "bands.share",
        "streetlights", "tree_coverage", "geometry",
    ]  # fmt: skip


def test_streets_are_polygons_with_their_attributes(city, refined_city, tmp_path):
    for preset in (city, refined_city):
        table = streets.build(preset, *refined_network(), tmp_path / preset.key)
        assert set(table.geom_type) == {"Polygon"}
        assert table.geometry.is_valid.all() and not table.geometry.is_empty.any()
        assert list(table.columns) == [
            "id", "parent_id", "source_record_ref", "bands.use", "bands.share",
            "streetlights", "tree_coverage", "geometry",
        ]  # fmt: skip
        assert set(table.attrs["footprints"]) == {"inside", "simplified"}


def test_refined_keeps_pedestrian(refined_city, tmp_path):
    table = streets.build(refined_city, *refined_network(), tmp_path)
    row = table.set_index("source_record_ref").loc["7-8"]
    # As today: the fixed profile, on the edge's own line, keyed by its nodes.
    assert json.loads(row["bands.use"]) == ["sidewalk", "lanes", "sidewalk"]
    assert json.loads(row["bands.share"]) == [20, 60, 20]
    assert row.id == str(uuid.uuid5(NAMESPACE, "mexicali:right_of_way:7-8"))


def test_unrefined_streets_unchanged(city, tmp_path):
    # Refine off: the network's tags change nothing.
    nodes, edges = refined_network()
    plain = streets.build(city, nodes, edges, tmp_path / "a")
    untagged = streets.build(
        city, nodes, edges.drop(columns=["highway", "name", "lanes"]), tmp_path / "b"
    )
    assert sorted(plain.source_record_ref) == ["1-2", "2-3", "5-6", "7-8"]
    assert "fit" in plain.attrs and plain.attrs["fit"]["fitted"] == 0
    # Only the polygons' widths follow the class (untagged: residential).
    pd.testing.assert_frame_equal(
        plain.drop(columns="geometry"), untagged.drop(columns="geometry")
    )
    assert set(plain.geom_type) == {"Polygon"} == set(untagged.geom_type)


def bike_lanes_of(rows: list[tuple], path: Path) -> BikeLanes:
    """A bike lanes file of rows (TIPO, line) in CRS."""
    gpd.GeoDataFrame(
        {"TIPO": [r[0] for r in rows]},
        geometry=[LineString(r[1]) for r in rows],
        crs=CRS,
    ).to_file(path)
    return BikeLanes(Source(str(path)), type_column="TIPO", protected=("SEGREGADO",))


def test_refined_bike_lanes(refined_city, tmp_path):
    # 7 m south of the street, along all of it: its right side (the axis runs west to
    # east); the footway's lane would be the fixed profile's.
    bike = bike_lanes_of(
        [("SEGREGADO", [(-10, -12), (110, -12)])], tmp_path / "lanes.geojson"
    )
    table = streets.build(
        replace(refined_city, bike_lanes=bike), *refined_network(), tmp_path / "w"
    ).set_index("source_record_ref")
    row = table.loc["1-2"]
    use, share = json.loads(row["bands.use"]), json.loads(row["bands.share"])
    assert use == ["sidewalk", "lanes", "protected_bike_lane", "sidewalk"]
    assert sum(share) == pytest.approx(100)
    assert "protected_bike_lane" not in table.loc["5-6", "bands.use"]


def test_refined_lane_bands(tmp_path):
    # An avenue 40 m wide, west to east: its lanes reach 20 m, past the 15 m.
    axes = gpd.GeoSeries(
        [LineString([(0, 0), (200, 0)]), LineString([(0, 500), (200, 500)]),
         LineString([(0, 1000), (200, 1000)])],
        index=["shared", "partial", "both"],
        crs=CRS,
    )  # fmt: skip
    bike = bike_lanes_of(
        [
            ("COMPARTIDO", [(0, 18), (200, 18)]),
            # Along 30% of it: not enough.
            ("SEGREGADO", [(0, 482), (60, 482)]),
            ("COMPARTIDO", [(0, 1018), (200, 1018)]),
            ("SEGREGADO", [(0, 982), (200, 982)]),
        ],
        tmp_path / "lanes.geojson",
    )
    lines = gpd.read_file(bike.source.uri)
    found = streets.refined_lane_bands(axes, np.array([40.0] * 3), lines, bike)
    assert found.loc["shared"].tolist() == ["shared_lanes", "left"]
    assert found.loc["partial"].isna().all()
    # Protected and shared: protected wins.
    assert found.loc["both"].tolist() == ["protected_bike_lane", "right"]
    # At 15 m, the avenue's lanes are too far.
    narrow = streets.refined_lane_bands(axes, np.array([10.0] * 3), lines, bike)
    assert narrow["kind"].isna().all()


def test_with_bike_lane():
    use = ["sidewalk", "lanes", "median", "lanes", "sidewalk"]
    share = [15.0, 30.0, 10.0, 30.0, 15.0]
    assert streets.with_bike_lane(use, share, "shared_lanes", "right", 40, 2) == (
        ["sidewalk", "lanes", "median", "shared_lanes", "sidewalk"],
        share,
    )
    # 2 m of 40: 5% from the left lanes, next to the left sidewalk.
    assert streets.with_bike_lane(use, share, "protected_bike_lane", "left", 40, 2) == (
        ["sidewalk", "protected_bike_lane", "lanes", "median", "lanes", "sidewalk"],
        [15.0, 5.0, 25.0, 10.0, 30.0, 15.0],
    )
    # Never more than half the lanes.
    one = streets.with_bike_lane(
        ["sidewalk", "lanes", "sidewalk"], [40, 20, 40], "protected_bike_lane",
        "right", 5, 2,
    )  # fmt: skip
    assert one == (
        ["sidewalk", "lanes", "protected_bike_lane", "sidewalk"],
        [40, 10.0, 10.0, 40],
    )
    assert streets.with_bike_lane(use, share, None, None, 40, 2) == (use, share)


def test_refined_lighting():
    # A face 32 m from the axis of an avenue 60 m wide: max(25, 30 + 5) = 35 m.
    street = gpd.GeoSeries([LineString([(0, 0), (100, 0)])], crs=CRS)
    faces = gpd.GeoDataFrame(
        {"ALUMPUB": ["1"]}, geometry=[LineString([(20, 32), (80, 32)])], crs=CRS
    )
    lighting = StreetLighting(
        Source("unused"), column="ALUMPUB", lit=("1",), unlit=("3",)
    )
    lit = streets.streetlights(street, faces, lighting, np.array([60.0]))
    assert lit.iloc[0] == 100
    # Unrefined, 25 m; 40 m wide, max(25, 20 + 5) = 25 m: out of reach.
    assert pd.isna(streets.streetlights(street, faces, lighting).iloc[0])
    narrow = streets.streetlights(street, faces, lighting, np.array([40.0]))
    assert pd.isna(narrow.iloc[0])
    # An unfitted street (NaN width) keeps max_distance_m.
    assert pd.isna(
        streets.streetlights(street, faces, lighting, np.array([np.nan])).iloc[0]
    )


def test_refined_streets_in_the_yaml(tmp_path):
    def refine(config):
        config["sources"]["streets"]["params"].update(
            refine=True, fit={"max_half_width_m": 30}
        )
        config["sources"]["bike_lanes"]["params"].update(
            min_share=0.6, protected_width_m=2.5
        )
        config["sources"]["street_lighting"]["params"].update(front_tolerance_m=3)

    city = cities.from_yaml(write_city(tmp_path, "mexicali", refine))
    assert city.streets.refine and city.streets.fit == {"max_half_width_m": 30}
    assert (city.bike_lanes.min_share, city.bike_lanes.protected_width_m) == (0.6, 2.5)
    assert city.street_lighting.front_tolerance_m == 3
    defaults = CITIES["mexicali"]
    assert (defaults.bike_lanes.min_share, defaults.bike_lanes.protected_width_m) == (
        0.5,
        2.0,
    )
    assert defaults.street_lighting.front_tolerance_m == 5
    # Off unless the YAML says so.
    assert not defaults.streets.refine and defaults.streets.fit == {}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p: p.update(fit={"typo": 1}), r"fit: unknown keys \['typo'\]"),
    ],
)
def test_refined_yaml_errors(tmp_path, change, message):
    with pytest.raises(cities.CityConfigError, match=message):
        cities.from_yaml(
            write_city(
                tmp_path,
                "mexicali",
                lambda c: change(c["sources"]["streets"]["params"]),
            )
        )


@pytest.mark.parametrize(
    "bad", [{"min_share": 0}, {"min_share": 1.5}, {"protected_width_m": 0}]
)
def test_bike_lane_params_are_checked(tmp_path, bad):
    with pytest.raises(cities.CityConfigError, match="must be"):
        cities.from_yaml(
            write_city(
                tmp_path,
                "mexicali",
                lambda c: c["sources"]["bike_lanes"]["params"].update(bad),
            )
        )


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
    assert len(tables["reimagina_baseline_elements_amenities"]) == 2
    assert len(tables["reimagina_baseline_elements_establishments"]) == 3
    assert len(tables["reimagina_baseline_elements_right_of_way"]) == 3


# ------------------------------------------------------------------ city YAMLs

MEXICALI_YAML = cities.FOLDER / "mexicali.yaml"


def test_mexicali_yaml():
    city = CITIES["mexicali"]
    assert city.cve_code == "02002"
    assert city.key_columns == KEYS
    assert city.normativa_columns["USOSUELO24"] == "zoning"
    assert city.denue_municipio == "Mexicali"
    assert city.lot_max_distance_m == 50.0
    assert city.floor_share_by_land_use["mixed_use"] == 80
    assert city.amenities_uri.endswith("imip/equipamiento")
    assert city.amenities_normalizer == "imip"
    assert str(city.existing_land_use_uri).endswith(
        "uso_de_suelo/usos_de_suelo_finales.geojson"
    )
    assert city.existing_land_use_column == "USOSUELO24"
    assert city.streets is not None
    assert city.streets.custom_filter.startswith('["highway"]["area"!~"yes"]')
    assert city.streets.retain_all
    assert city.streets.output_prefix.endswith("cleaned/vialidades")

    assert city.street_lighting.max_distance_m == 25
    assert city.street_lighting.source.member.endswith("INV2020_IND_EU_FTE_02.shp")
    # Both files are in the lake, beside the city's other sources.
    assert city.street_lighting.source.uri.startswith("az://raw/inegi/")
    assert city.bike_lanes.source.uri.endswith(
        "imip/ciclovia_exist/ciclovia_exist.geojson"
    )
    assert city.bike_lanes.type_column == "TIPO"


def test_streets_are_optional(tmp_path):
    def drop(config):
        # Lighting and bike lanes are over the network: they go with it.
        for name in ("streets", "street_lighting", "bike_lanes"):
            config["sources"].pop(name)

    preset = cities.from_yaml(write_city(tmp_path, "mexicali", drop))
    assert (preset.streets, preset.street_lighting, preset.bike_lanes) == (None,) * 3


def test_existing_land_use_is_optional(tmp_path):
    path = write_city(
        tmp_path, "mexicali", lambda c: c["sources"].pop("existing_land_use")
    )
    assert cities.from_yaml(path).existing_land_use_uri is None


def write_city(tmp_path: Path, name: str, change) -> Path:
    config = yaml.safe_load(MEXICALI_YAML.read_text(encoding="utf-8"))
    change(config)
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: c["sources"].pop("denue"), "sources.denue: uri is required"),
        (lambda c: c["params"].pop("space_allocation"), "housing.floor_share is required"),
        (lambda c: c["sources"]["existing_land_use"]["params"].pop("column"),
         "existing_land_use.params: column is required"),
        (lambda c: c["sources"]["existing_land_use"]["params"].update(typo=1),
         r"existing_land_use.params: unknown keys \['typo'\]"),
        (lambda c: c["sources"]["amenities"]["params"].update(normalizer="sigcdmx"),
         "unknown normalizer 'sigcdmx'"),
        (lambda c: c["sources"]["amenities"]["params"].update(categories=["parks"]),
         r"unknown categories \['parks'\]"),
        (lambda c: c["sources"]["amenities"]["params"].pop("categories"),
         "categories is required"),
        (lambda c: c["params"]["space_allocation"]["housing.floor_share"].update(park=1),
         r"unknown land uses \['park'\]"),
        (lambda c: c["params"]["space_allocation"]["housing.floor_share"].update(amenity=120),
         "must be 0-100"),
        (lambda c: c["sources"]["denue"]["params"].pop("municipio"), "municipio"),
        (lambda c: c["params"]["lots"].update(typo=1), r"params.lots: unknown keys \['typo'\]"),
        (lambda c: c["sources"].update(roads={"uri": "x"}), r"sources: unknown keys \['roads'\]"),
        (lambda c: c.update(city="tijuana"), "must be tijuana.yaml"),
        (lambda c: c["sources"]["streets"].update(uri="osm://drive"), "uri must be osm://overpass"),
        (lambda c: c["sources"]["streets"]["params"].pop("output_prefix"), "output_prefix is required"),
        (lambda c: c["sources"]["streets"]["params"].update(typo=1), r"streets.params: unknown keys \['typo'\]"),
    ],
)  # fmt: skip
def test_city_yaml_errors(tmp_path, change, message):
    with pytest.raises(cities.CityConfigError, match=message):
        cities.from_yaml(write_city(tmp_path, "mexicali", change))


def test_cities_loads_every_yaml(tmp_path):
    def rename(config):
        config.update(city="tijuana", name="Tijuana")

    write_city(tmp_path, "tijuana", rename)
    assert list(cities.load(tmp_path)) == ["tijuana"]
    assert cities.load(tmp_path)["tijuana"].lots_uri == CITIES["mexicali"].lots_uri
