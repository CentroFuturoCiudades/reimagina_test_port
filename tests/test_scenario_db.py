"""Scenario, POI and road network datasets against PostGIS (minimal EAV schema).

REIMAGINA_TEST_DSN points to a server where the user may create databases; each
test creates and drops its own. See docs/verification.md for the container.
"""

import os
from collections.abc import Iterator
from uuid import uuid4

import geopandas as gpd
import numpy as np
import pandas as pd
import psycopg
import pytest
from lyra.sdk import GraphExecutionError, LocalRunner
from lyra.sdk.errors import MetricInputError
from psycopg import conninfo, sql
from shapely.geometry import box

from reimagina_test_port import accessibility, plugin, scenario

pytestmark = pytest.mark.postgis

BASELINE = "10000000-0000-0000-0000-000000000000"
FORK = "20000000-0000-0000-0000-000000000000"
PROJECT = "30000000-0000-0000-0000-000000000000"
BLOCK = "a0000000-0000-0000-0000-000000000001"
BLOCK_2 = "a0000000-0000-0000-0000-000000000002"
HOUSE = "b0000000-0000-0000-0000-000000000001"
SHOP = "b0000000-0000-0000-0000-000000000002"
FLATS = "b0000000-0000-0000-0000-000000000003"
NEW = "b0000000-0000-0000-0000-000000000004"
GONE = "b0000000-0000-0000-0000-000000000005"
SQUARE = "POLYGON((0 0, 0 0.001, 0.001 0.001, 0.001 0, 0 0))"
SMALL = "POLYGON((0 0, 0 0.0001, 0.0001 0.0001, 0.0001 0, 0 0))"

# Columns the dataset reads, with the API's types (alembic, dev).
SCHEMA_DDL = """
CREATE EXTENSION postgis;
CREATE SCHEMA reimagina_urbano;
SET search_path = reimagina_urbano, public;
CREATE TYPE element_type AS ENUM ('block', 'lot', 'building', 'tree');
CREATE TYPE attribute_value_type AS ENUM ('number', 'integer', 'text', 'enum', 'json');
CREATE TABLE projects (id uuid PRIMARY KEY, road_network_uri text);
CREATE TABLE scenarios (
    id uuid PRIMARY KEY,
    project_id uuid REFERENCES projects (id),
    parent_scenario_id uuid REFERENCES scenarios (id),
    deleted_at timestamptz
);
CREATE TABLE elements (
    id uuid PRIMARY KEY,
    parent_element_id uuid REFERENCES elements (id),
    type element_type NOT NULL,
    source_record_ref text,
    deleted_at timestamptz
);
CREATE TABLE element_state (
    scenario_id uuid NOT NULL REFERENCES scenarios (id),
    element_id uuid NOT NULL REFERENCES elements (id),
    parent_element_id uuid REFERENCES elements (id),
    geometry geometry(Geometry, 4326) NOT NULL,
    deleted_at timestamptz,
    PRIMARY KEY (scenario_id, element_id)
);
CREATE TABLE attribute_catalog (
    id serial PRIMARY KEY,
    element_type element_type NOT NULL,
    key text NOT NULL,
    value_type attribute_value_type NOT NULL,
    UNIQUE (element_type, key)
);
CREATE TABLE element_state_attribute_value (
    scenario_id uuid NOT NULL,
    element_id uuid NOT NULL,
    attribute_id integer NOT NULL REFERENCES attribute_catalog (id),
    value_num double precision,
    value_text text,
    value_json jsonb,
    PRIMARY KEY (scenario_id, element_id, attribute_id),
    FOREIGN KEY (scenario_id, element_id)
        REFERENCES element_state (scenario_id, element_id)
);
INSERT INTO attribute_catalog (element_type, key, value_type) VALUES
    ('lot', 'land_use', 'enum'), ('lot', 'cos', 'number'),
    ('lot', 'num_levels', 'integer'), ('lot', 'area', 'number'),
    ('lot', 'unit_size_m2', 'number'), ('lot', 'people_per_household', 'number'),
    ('lot', 'occupancy_rate', 'number'), ('lot', 'amenity_type', 'enum');
"""

HOUSE_ATTRS = {
    "land_use": "single_family",
    "area": 200,
    "cos": 0.5,
    "num_levels": 2,
    "people_per_household": 3.5,
    "occupancy_rate": 0.9,
}
SHOP_ATTRS = {"land_use": "commerce", "area": 100, "cos": 1, "num_levels": 1}
FLATS_ATTRS = {
    "land_use": "multifamily",
    "area": 1000,
    "cos": 0.6,
    "num_levels": 4,
    "unit_size_m2": 80,
    "people_per_household": 3,
}


@pytest.fixture
def network_uri(tmp_path) -> str:
    """A 20 m grid H5 over the test geometries, in their UTM zone."""
    corner = gpd.GeoSeries([box(-0.0005, -0.0005, 0.0015, 0.0015)], crs=4326)
    xmin, ymin, xmax, ymax = corner.to_crs(corner.estimate_utm_crs()).total_bounds
    xs, ys = np.arange(xmin, xmax, 20), np.arange(ymin, ymax, 20)
    ids = np.arange(len(xs) * len(ys))
    side = len(xs)
    right = ids[(ids % side) < side - 1]
    up = ids[ids < side * (len(ys) - 1)]
    nodes = pd.DataFrame({"id": ids, "x": xs[ids % side], "y": ys[ids // side]})
    edges = pd.DataFrame(
        {
            "from": np.concatenate([right, up]),
            "to": np.concatenate([right + 1, up + side]),
            "distance": 20.0,
            "two_way": True,
        }
    )
    path = tmp_path / "net.h5"
    accessibility.build_network(nodes, edges).save_hdf5(str(path))
    return str(path)


@pytest.fixture
def dsn(monkeypatch, network_uri) -> Iterator[str]:
    admin_dsn = os.environ.get("REIMAGINA_TEST_DSN")
    if not admin_dsn:
        pytest.skip("REIMAGINA_TEST_DSN is not set")
    name = f"reimagina_test_{uuid4().hex[:12]}"
    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    test_dsn = conninfo.make_conninfo(admin_dsn, dbname=name, sslmode="disable")
    try:
        with psycopg.connect(test_dsn) as conn:
            conn.execute(SCHEMA_DDL)
            seed(conn, network_uri)
        monkeypatch.setenv(scenario.DSN_ENV_VAR, test_dsn)
        yield test_dsn
    finally:
        with psycopg.connect(admin_dsn, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                    sql.Identifier(name)
                )
            )


def add_state(conn, scenario_id, element_id, wkt=SQUARE, parent=None, deleted=False):
    conn.execute(
        "INSERT INTO element_state VALUES "
        "(%s, %s, %s, ST_GeomFromText(%s, 4326), CASE WHEN %s THEN NOW() END)",
        (scenario_id, element_id, parent, wkt, deleted),
    )


def set_attrs(conn, scenario_id, element_id, attrs):
    for key, value in attrs.items():
        text = value if isinstance(value, str) else None
        num = None if isinstance(value, str) else value
        conn.execute(
            "INSERT INTO element_state_attribute_value "
            "(scenario_id, element_id, attribute_id, value_num, value_text) "
            "SELECT %s, %s, id, %s, %s FROM attribute_catalog WHERE key = %s",
            (scenario_id, element_id, num, text, key),
        )


def seed(conn, network_uri: str) -> None:
    """Baseline with two blocks and four lots; a fork that edits it."""
    conn.execute("SET search_path = reimagina_urbano, public")
    conn.execute("INSERT INTO projects VALUES (%s, %s)", (PROJECT, network_uri))
    conn.execute(
        "INSERT INTO scenarios VALUES (%s, %s, NULL, NULL)", (BASELINE, PROJECT)
    )
    conn.execute(
        "INSERT INTO scenarios VALUES (%s, %s, %s, NULL)", (FORK, PROJECT, BASELINE)
    )
    for block in (BLOCK, BLOCK_2):
        conn.execute(
            "INSERT INTO elements VALUES (%s, NULL, 'block', %s, NULL)",
            (block, f"cvegeo:{block[-1]}"),
        )
        add_state(conn, BASELINE, block)
    for lot, block, attrs in (
        (HOUSE, BLOCK, HOUSE_ATTRS),
        (SHOP, BLOCK, SHOP_ATTRS),
        (FLATS, BLOCK, FLATS_ATTRS),
        (GONE, BLOCK_2, SHOP_ATTRS),
    ):
        conn.execute(
            "INSERT INTO elements VALUES (%s, %s, 'lot', %s, NULL)",
            (lot, block, f"lot:{lot[-1]}"),
        )
        add_state(conn, BASELINE, lot)
        set_attrs(conn, BASELINE, lot, attrs)
    # Amenities: the flats hold a clinic, GONE a pharmacy.
    set_attrs(conn, BASELINE, FLATS, {"amenity_type": "clinic"})
    set_attrs(conn, BASELINE, GONE, {"amenity_type": "pharmacy"})
    # Fork: the shop becomes a house; the flats get an all-NULL unit size row; the
    # house moves to the second block with a new geometry; GONE is deleted; NEW is
    # a lot that exists only in the fork.
    add_state(conn, FORK, SHOP)
    set_attrs(conn, FORK, SHOP, {"land_use": "single_family", "num_levels": 2})
    set_attrs(conn, FORK, SHOP, {"amenity_type": "park"})
    add_state(conn, FORK, FLATS)
    set_attrs(conn, FORK, FLATS, {"unit_size_m2": None, "amenity_type": None})
    add_state(conn, FORK, HOUSE, wkt=SMALL, parent=BLOCK_2)
    add_state(conn, FORK, GONE, deleted=True)
    conn.execute(
        "INSERT INTO elements VALUES (%s, %s, 'lot', NULL, NULL)", (NEW, BLOCK_2)
    )
    add_state(conn, FORK, NEW)
    set_attrs(conn, FORK, NEW, SHOP_ATTRS)


def run(scenario_id: str):
    runner = LocalRunner.from_definitions(
        {"reimagina-test-port": plugin.create_plugin()}
    )
    return runner.run(
        "reimagina_built_form", {"parameters": {"scenario_id": scenario_id}}
    )


def by_id(table):
    return {row["id"]: row for row in table.to_dict("records")}


def test_baseline_tables(dsn):
    tables = plugin.read_scenario(BASELINE)
    lots = tables["reimagina_scenario_lots"]
    blocks = tables["reimagina_scenario_blocks"]
    assert lots.id.tolist() == [HOUSE, SHOP, FLATS, GONE]
    assert blocks.id.tolist() == [BLOCK, BLOCK_2]
    assert lots.crs == "EPSG:4326"
    assert lots.parent_id.tolist() == [BLOCK, BLOCK, BLOCK, BLOCK_2]
    assert lots.source_record_ref.tolist() == ["lot:1", "lot:2", "lot:3", "lot:5"]
    house = by_id(lots)[HOUSE]
    assert house["land_use"] == "single_family"
    assert house["occupancy_rate"] == pytest.approx(0.9)
    assert house["geometry"].bounds == pytest.approx((0, 0, 0.001, 0.001))
    assert lots[plugin.NUMERIC_ATTRIBUTES].dtypes.eq("float64").all()


def test_baseline_built_form(dsn):
    results = run(BASELINE)
    lots = by_id(results["reimagina_built_form_lots"])
    # 1 home on 200 m2: 50 viv/ha, as indicator_catalog.
    assert lots[HOUSE]["num_housing_units"] == 1
    assert lots[HOUSE]["population"] == 3
    assert lots[HOUSE]["density"] == pytest.approx(50)
    assert lots[SHOP]["num_housing_units"] == 0
    # round(0.6 * 4 * 1000 / 80) = 30 homes on 1000 m2.
    assert lots[FLATS]["num_housing_units"] == 30
    assert lots[FLATS]["density"] == pytest.approx(300)
    blocks = by_id(results["reimagina_built_form_blocks"])
    assert blocks[BLOCK]["num_housing_units"] == 31
    assert blocks[BLOCK]["area"] == pytest.approx(1300)
    assert blocks[BLOCK_2]["num_housing_units"] == 0


def test_fork_falls_back_to_baseline(dsn):
    tables = plugin.read_scenario(FORK)
    lots = by_id(tables["reimagina_scenario_lots"])
    # Deleted in the fork: absent; only in the fork: present.
    assert set(lots) == {HOUSE, SHOP, FLATS, NEW}
    assert lots[NEW]["land_use"] == "commerce"
    # Scenario value wins; untouched keys come from the baseline.
    assert lots[SHOP]["land_use"] == "single_family"
    assert lots[SHOP]["num_levels"] == 2
    assert lots[SHOP]["area"] == 100
    # An all-NULL scenario row hides the baseline value.
    assert pd.isna(lots[FLATS]["unit_size_m2"])
    # Geometry and parent from the fork's own state row.
    assert lots[HOUSE]["parent_id"] == BLOCK_2
    assert lots[HOUSE]["geometry"].bounds == pytest.approx((0, 0, 0.0001, 0.0001))
    assert lots[HOUSE]["area"] == 200

    results = run(FORK)
    computed = by_id(results["reimagina_built_form_lots"])
    assert computed[SHOP]["num_housing_units"] == 1
    assert pd.isna(computed[FLATS]["num_housing_units"])
    blocks = by_id(results["reimagina_built_form_blocks"])
    # SHOP (now 1 home) and FLATS (no unit size) in BLOCK; HOUSE and NEW in BLOCK_2.
    assert blocks[BLOCK]["num_housing_units"] == 1
    assert blocks[BLOCK_2]["num_housing_units"] == 1


def test_baseline_is_not_modified_by_fork(dsn):
    assert (
        by_id(plugin.read_scenario(BASELINE)["reimagina_scenario_lots"])[SHOP][
            "land_use"
        ]
        == "commerce"
    )


def test_unknown_scenario(dsn):
    with pytest.raises(GraphExecutionError) as failure:
        run(str(uuid4()))
    assert isinstance(failure.value.__cause__, MetricInputError)
    assert failure.value.node == "reimagina_scenario"


def test_missing_dsn(monkeypatch):
    monkeypatch.delenv(scenario.DSN_ENV_VAR, raising=False)
    with pytest.raises(RuntimeError, match=scenario.DSN_ENV_VAR):
        scenario.resolve_dsn()


@pytest.mark.parametrize(
    ("dsn_value", "expected"),
    [
        ("postgresql://h/db", "postgresql://h/db?sslmode=require"),
        ("postgresql://h/db?a=1", "postgresql://h/db?a=1&sslmode=require"),
        ("postgresql://h/db?sslmode=disable", "postgresql://h/db?sslmode=disable"),
    ],
)
def test_dsn_requires_tls_by_default(monkeypatch, dsn_value, expected):
    monkeypatch.setenv(scenario.DSN_ENV_VAR, dsn_value)
    assert scenario.resolve_dsn() == expected


def test_pois_fall_back_to_baseline(dsn):
    baseline = scenario.read_pois(BASELINE).set_index("id")
    assert baseline["amenity_type"].to_dict() == {FLATS: "clinic", GONE: "pharmacy"}
    assert baseline.crs == "EPSG:4326"
    # Fork: SHOP gains a park, the flats' clinic is hidden by an all-NULL row and
    # GONE is deleted.
    fork = scenario.read_pois(FORK).set_index("id")
    assert fork["amenity_type"].to_dict() == {SHOP: "park"}
    assert fork["capacity"].isna().all()


def test_road_network_uri(dsn, network_uri):
    assert scenario.read_road_network_uri(FORK) == network_uri
    with psycopg.connect(dsn) as conn:
        conn.execute("UPDATE reimagina_urbano.projects SET road_network_uri = ''")
    assert scenario.read_road_network_uri(BASELINE) is None
    runner = LocalRunner.from_definitions(
        {"reimagina-test-port": plugin.create_plugin()}
    )
    with pytest.raises(GraphExecutionError) as failure:
        runner.run("reimagina_accessibility", {"parameters": {"scenario_id": BASELINE}})
    assert isinstance(failure.value.__cause__, MetricInputError)
    assert failure.value.node == "reimagina_road_network"


def test_fork_accessibility(dsn):
    runner = LocalRunner.from_definitions(
        {"reimagina-test-port": plugin.create_plugin()}
    )
    results = runner.run(
        "reimagina_accessibility", {"parameters": {"scenario_id": FORK}}
    )
    blocks = by_id(results["reimagina_accessibility_blocks"])
    lots = by_id(results["reimagina_accessibility_lots"])
    assert set(blocks) == {BLOCK, BLOCK_2}
    for block in blocks.values():
        # Only the fork's park is a destination.
        assert block["farthest_amenity"] == "park"
        assert block["minutes"] == block["distance_to_park"]
        assert pd.isna(block["distance_to_clinic"])
        assert pd.isna(block["distance_to_pharmacy"])
    assert lots[HOUSE]["parent_id"] == BLOCK_2
    assert lots[HOUSE]["distance_to_park"] == blocks[BLOCK_2]["distance_to_park"]
    assert lots[SHOP]["distance_to_park"] == blocks[BLOCK]["distance_to_park"]
