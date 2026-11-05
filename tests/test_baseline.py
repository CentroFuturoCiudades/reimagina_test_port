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
from shapely.geometry import Point, box

from reimagina_test_port import plugin
from reimagina_test_port.common import lake
from reimagina_test_port.ingestion import (
    baseline,
    cadastral,
    cities,
    prep,
    raw_amenities,
)
from reimagina_test_port.ingestion.cities import CITIES

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
    return {
        "lots_uri": str(root / "lots" / "PREDIOS.shp"),
        "normativa_uri": str(root / "normativa.geojson"),
        "existing_land_use_uri": str(root / "existing_land_use.geojson"),
        "amenities_uri": str(root / "imip"),
        "denue_uri": str(root / "denue.csv"),
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
    assert blocks.loc[CVEGEO[0], "people_per_household"] == 3.25
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
