"""Scenario datasets and the indicator writer against PostGIS (minimal EAV schema).

REIMAGINA_TEST_DSN points to a server where the user may create databases; each
test creates and drops its own. See docs/verification.md for the container.
"""

import hashlib
import importlib.util
import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import geopandas as gpd
import numpy as np
import pandas as pd
import psycopg
import pytest
from lyra.sdk import GraphExecutionError, LocalRunner
from lyra.sdk.errors import MetricInputError
from psycopg import conninfo, sql
from shapely.geometry import LineString, box

from reimagina_test_port import plugin
from reimagina_test_port.common import lake, schema
from reimagina_test_port.indicators.accessibility import model as accessibility
from reimagina_test_port.indicators.accessibility import nodes as accessibility_nodes
from reimagina_test_port.indicators.composites import calibration as calibrations
from reimagina_test_port.indicators.composites import catalog as composite_catalog
from reimagina_test_port.indicators.composites import composite_metric
from reimagina_test_port.indicators.land_use import calculation as land_use_calculation
from reimagina_test_port.indicators.population_access import (
    nodes as population_access_nodes,
)
from reimagina_test_port.indicators.streets import calculation as streets_calculation
from reimagina_test_port.ingestion import baseline_writer, catalogs
from reimagina_test_port.ingestion import nodes as ingestion_nodes
from reimagina_test_port.scenario import contract, writer
from reimagina_test_port.scenario import nodes as scenario_nodes
from reimagina_test_port.scenario import source as scenario

pytestmark = pytest.mark.postgis
ROOT = Path(__file__).resolve().parents[1]

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
CLINIC = "f0000000-0000-0000-0000-000000000001"
PHARMACY = "f0000000-0000-0000-0000-000000000002"
PARK = "f0000000-0000-0000-0000-000000000003"
SQUARE = "POLYGON((0 0, 0 0.001, 0.001 0.001, 0.001 0, 0 0))"
SMALL = "POLYGON((0 0, 0 0.0001, 0.0001 0.0001, 0.0001 0, 0 0))"

# Columns the dataset reads, with the API's types (alembic, dev).
SCHEMA_DDL = """
CREATE EXTENSION postgis;
CREATE SCHEMA reimagina_urbano;
SET search_path = reimagina_urbano, public;
CREATE TYPE element_type AS ENUM (
    'block', 'lot', 'building', 'tree', 'amenity', 'establishment', 'right_of_way',
    'bus_stop', 'station'
);
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
    origin_scenario_id uuid,
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
    enum_values text[],
    unit text,
    applies_when jsonb,
    status text NOT NULL DEFAULT 'active',
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
CREATE TYPE indicator_catalog_status AS ENUM ('active', 'deprecated', 'suspended');
CREATE TYPE indicator_scope AS ENUM ('diagnostic', 'scenario');
CREATE TABLE job_run (id uuid PRIMARY KEY);
CREATE TABLE indicator_catalog (
    id serial PRIMARY KEY,
    element_type element_type NOT NULL,
    key text NOT NULL,
    name text NOT NULL,
    unit text,
    value_type attribute_value_type NOT NULL,
    scope indicator_scope NOT NULL,
    job_code text,
    status indicator_catalog_status NOT NULL DEFAULT 'active',
    UNIQUE (element_type, key)
);
CREATE TABLE indicator_input (
    id serial PRIMARY KEY,
    indicator_id integer NOT NULL REFERENCES indicator_catalog (id),
    input_attribute_id integer,
    input_indicator_id integer REFERENCES indicator_catalog (id),
    external_source text
);
CREATE TABLE element_state_indicator_value (
    scenario_id uuid NOT NULL REFERENCES scenarios (id),
    element_id uuid NOT NULL REFERENCES elements (id),
    indicator_id integer NOT NULL REFERENCES indicator_catalog (id),
    value_num double precision,
    value_text text,
    job_run_id uuid REFERENCES job_run (id),
    computed_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    updated_at timestamptz NOT NULL DEFAULT NOW(),
    PRIMARY KEY (scenario_id, element_id, indicator_id)
);
CREATE TABLE scenario_file (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario_id uuid NOT NULL REFERENCES scenarios (id) ON DELETE CASCADE,
    kind text NOT NULL,
    uri text NOT NULL,
    content_hash text,
    job_run_id uuid REFERENCES job_run (id) ON DELETE SET NULL,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    updated_at timestamptz NOT NULL DEFAULT NOW(),
    UNIQUE (scenario_id, kind)
);
INSERT INTO attribute_catalog (element_type, key, value_type) VALUES
    ('lot', 'land_use', 'enum'), ('lot', 'cos', 'number'),
    ('lot', 'num_levels', 'integer'), ('lot', 'area', 'number'),
    ('lot', 'housing.unit_size_m2', 'number'),
    ('lot', 'housing.people_per_household', 'number'),
    ('lot', 'occupancy_rate', 'number'),
    ('lot', 'commerce_type', 'text'), ('block', 'vivtot', 'number'),
    ('block', 'housing.people_per_household', 'number'),
    ('amenity', 'category', 'enum'), ('amenity', 'scian_code', 'text'),
    ('establishment', 'category', 'enum'), ('establishment', 'scian_code', 'text');
CREATE TABLE poi_catalog (
    poi_type text PRIMARY KEY,
    rule_criteria jsonb NOT NULL,
    params jsonb NOT NULL DEFAULT '{}',
    status text NOT NULL DEFAULT 'active'
);
-- Rules as the API's poi_catalog by category; "park" is active but has no
-- accessibility outputs, "care" is inactive.
INSERT INTO poi_catalog (poi_type, rule_criteria, status) VALUES
    ('health', '{"any": [{"type": "amenity", "category": {"in": ["health_center",
        "clinic", "hospital"]}},
        {"type": "establishment", "category": "pharmacy"}]}', 'active'),
    ('establishment', '{"type": "establishment"}', 'active'),
    ('public_space', '{"type": "amenity", "category": "park"}', 'active'),
    ('convenience_store', '{"type": "establishment", "category":
        "convenience_store"}', 'active'),
    ('park', '{"type": "amenity", "category": "park"}', 'active'),
    ('care', '{"type": "amenity", "category": "daycare"}', 'inactive');
UPDATE attribute_catalog SET enum_values = ARRAY[
    'single_family', 'multifamily', 'mixed_use', 'amenity', 'commerce', 'office',
    'industrial'
] WHERE key = 'land_use';
"""

HOUSE_ATTRS = {
    "land_use": "single_family",
    "area": 200,
    "cos": 0.5,
    "num_levels": 2,
    "housing.people_per_household": 3.5,
    "occupancy_rate": 0.9,
}
SHOP_ATTRS = {"land_use": "commerce", "area": 100, "cos": 1, "num_levels": 1}
FLATS_ATTRS = {
    "land_use": "multifamily",
    "area": 1000,
    "cos": 0.6,
    "num_levels": 4,
    "housing.unit_size_m2": 80,
    "housing.people_per_household": 3,
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
def dsn(monkeypatch, network_uri, tmp_path) -> Iterator[str]:
    admin_dsn = os.environ.get("REIMAGINA_TEST_DSN")
    if not admin_dsn:
        pytest.skip("REIMAGINA_TEST_DSN is not set")
    # A baseline's write saves its composite calibration: here, not the lake.
    monkeypatch.setenv(calibrations.PREFIX_ENV, str(tmp_path / "calibration"))
    name = f"reimagina_test_{uuid4().hex[:12]}"
    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    test_dsn = conninfo.make_conninfo(admin_dsn, dbname=name, sslmode="disable")
    try:
        with psycopg.connect(test_dsn) as conn:
            conn.execute(SCHEMA_DDL)
            seed(conn, network_uri)
            seed_catalog(conn)
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
            "SELECT %s, %s, c.id, %s, %s FROM attribute_catalog c "
            "JOIN elements e ON e.id = %s AND e.type::text = c.element_type::text "
            "WHERE c.key = %s",
            (scenario_id, element_id, num, text, element_id, key),
        )


def add_point(conn, scenario_id, element_id, element_type, lot, attrs):
    conn.execute(
        "INSERT INTO elements VALUES (%s, %s, %s, NULL, NULL, NULL)",
        (element_id, lot, element_type),
    )
    add_state(conn, scenario_id, element_id, parent=lot)
    set_attrs(conn, scenario_id, element_id, attrs)


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
            "INSERT INTO elements VALUES (%s, NULL, 'block', NULL, %s, NULL)",
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
            "INSERT INTO elements VALUES (%s, %s, 'lot', NULL, %s, NULL)",
            (lot, block, f"lot:{lot[-1]}"),
        )
        add_state(conn, BASELINE, lot)
        set_attrs(conn, BASELINE, lot, attrs)
    # Contents: the flats hold a clinic (an amenity), GONE a pharmacy (an
    # establishment).
    clinic = {"category": "health_center", "scian_code": "621112"}
    add_point(conn, BASELINE, CLINIC, "amenity", FLATS, clinic)
    pharmacy = {"category": "pharmacy", "scian_code": "464111"}
    add_point(conn, BASELINE, PHARMACY, "establishment", GONE, pharmacy)
    # Fork: the shop becomes a house; the flats get an all-NULL unit size row; the
    # house moves to the second block with a new geometry; GONE is deleted; NEW is
    # a lot that exists only in the fork.
    add_state(conn, FORK, SHOP)
    set_attrs(conn, FORK, SHOP, {"land_use": "single_family", "num_levels": 2})
    add_point(conn, FORK, PARK, "amenity", SHOP, {"category": "park"})
    add_state(conn, FORK, FLATS)
    set_attrs(conn, FORK, FLATS, {"housing.unit_size_m2": None})
    add_state(conn, FORK, CLINIC, parent=FLATS)
    set_attrs(conn, FORK, CLINIC, {"category": None, "scian_code": None})
    add_state(conn, FORK, HOUSE, wkt=SMALL, parent=BLOCK_2)
    add_state(conn, FORK, GONE, deleted=True)
    add_state(conn, FORK, PHARMACY, parent=GONE, deleted=True)
    conn.execute(
        "INSERT INTO elements VALUES (%s, %s, 'lot', NULL, NULL, NULL)", (NEW, BLOCK_2)
    )
    add_state(conn, FORK, NEW)
    set_attrs(conn, FORK, NEW, SHOP_ATTRS)


def seed_catalog(conn) -> None:
    keys = [
        *schema.UNITS,
        "minutes",
        *accessibility_nodes.DISTANCE_KEYS,
        *accessibility_nodes.SCORE_KEYS,
        *population_access_nodes.KEYS,
        *population_access_nodes.DIVERSITY_KEYS,
        population_access_nodes.REACHABLE_JOBS_KEY,
        *land_use_calculation.KEYS,
        *streets_calculation.KEYS,
    ]
    for element_type in ("lot", "block"):
        for key in keys:
            conn.execute(
                "INSERT INTO reimagina_urbano.indicator_catalog "
                "(element_type, key, name, value_type, scope) "
                "VALUES (%s, %s, %s, 'number', 'scenario')",
                (element_type, key, key),
            )
        conn.execute(
            "INSERT INTO reimagina_urbano.indicator_catalog "
            "(element_type, key, name, value_type, scope) "
            "VALUES (%s, 'farthest_amenity', 'farthest_amenity', 'text', 'scenario')",
            (element_type,),
        )


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
    tables = scenario_nodes.read_scenario(BASELINE)
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
    assert lots[schema.NUMERIC_ATTRIBUTES].dtypes.eq("float64").all()


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
    tables = scenario_nodes.read_scenario(FORK)
    lots = by_id(tables["reimagina_scenario_lots"])
    # Deleted in the fork: absent; only in the fork: present.
    assert set(lots) == {HOUSE, SHOP, FLATS, NEW}
    assert lots[NEW]["land_use"] == "commerce"
    # Scenario value wins; untouched keys come from the baseline.
    assert lots[SHOP]["land_use"] == "single_family"
    assert lots[SHOP]["num_levels"] == 2
    assert lots[SHOP]["area"] == 100
    # An all-NULL scenario row hides the baseline value.
    assert pd.isna(lots[FLATS]["housing.unit_size_m2"])
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
        by_id(scenario_nodes.read_scenario(BASELINE)["reimagina_scenario_lots"])[SHOP][
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
    baseline = scenario.read_pois(BASELINE)
    # One row per poi_catalog type whose rule the element meets.
    assert list(zip(baseline["id"], baseline["poi_type"], strict=True)) == [
        (CLINIC, "health"),
        (PHARMACY, "establishment"),
        (PHARMACY, "health"),
    ]
    # Each row carries its element's category.
    assert baseline["category"].tolist() == ["health_center", "pharmacy", "pharmacy"]
    # And its type; the seed has no num_workers, so they are missing.
    assert baseline["element_type"].tolist() == [
        "amenity",
        "establishment",
        "establishment",
    ]
    assert baseline["num_workers"].isna().all()
    assert baseline.crs == "EPSG:4326"
    # Fork: SHOP gains a park, the clinic is hidden by all-NULL rows and the
    # pharmacy is deleted. "park" (no outputs) and inactive types are ignored.
    fork = scenario.read_pois(FORK).set_index("id")
    assert fork["poi_type"].to_dict() == {PARK: "public_space"}
    assert fork["capacity"].isna().all()
    assert fork["category"].tolist() == ["park"]


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
        assert block["farthest_amenity"] == "public_space"
        assert block["minutes"] == block["distance_to_public_space"]
        assert pd.isna(block["distance_to_health"])
        assert pd.isna(block["distance_to_establishment"])
    assert lots[HOUSE]["parent_id"] == BLOCK_2
    assert (
        lots[HOUSE]["distance_to_public_space"]
        == blocks[BLOCK_2]["distance_to_public_space"]
    )
    assert (
        lots[SHOP]["distance_to_public_space"]
        == blocks[BLOCK]["distance_to_public_space"]
    )


def run_metric(name: str, scenario_id: str, **options) -> dict:
    runner = LocalRunner.from_definitions(
        {"reimagina-test-port": plugin.create_plugin()}
    )
    result = runner.run(name, {"parameters": {"scenario_id": scenario_id, **options}})
    return result[name].iloc[0].to_dict()


def write(scenario_id: str, **options) -> dict:
    result = run_metric("reimagina_scenario_write", scenario_id, **options)
    # Sending only what may have changed leaves what sending everything would.
    check = run_metric("reimagina_scenario_write", scenario_id, full=True, dry_run=True)
    if not result["dry_run"]:
        assert (check["values_written"], check["values_cleared"]) == (0, 0)
    return result


def stored(dsn: str, scenario_id: str) -> dict:
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            "SELECT v.element_id::text, c.key, v.value_num, v.value_text, "
            "v.job_run_id::text, v.computed_at "
            "FROM reimagina_urbano.element_state_indicator_value v "
            "JOIN reimagina_urbano.indicator_catalog c ON c.id = v.indicator_id "
            "WHERE v.scenario_id = %s",
            (scenario_id,),
        ).fetchall()
    return {(e, k): (num, text, job, at) for e, k, num, text, job, at in rows}


def test_write_baseline_is_idempotent(dsn):
    first = write(BASELINE)
    assert first["elements"] == 6
    assert first["values_written"] > 0
    assert first["values_cleared"] == 0
    values = stored(dsn, BASELINE)
    assert values[(HOUSE, "density")][0] == pytest.approx(50)
    assert values[(BLOCK, "num_housing_units")][0] == 31
    # Baseline POIs: a clinic and a pharmacy.
    assert values[(BLOCK, "farthest_amenity")][1] in {"health", "establishment"}
    assert values[(BLOCK, "distance_to_health")][0] > 0
    # No POI: never written.
    assert (BLOCK, "distance_to_public_space") not in values
    # Composites: each one's index, on blocks.
    assert 40 < values[(BLOCK, "composite_index")][0] < 160
    assert values[(BLOCK, "composite_social_infrastructure")][0] > 0
    assert not any(e == HOUSE and k.startswith("composite_") for e, k in values)
    # Accessibility is stored on blocks only; a lot shows its block's.
    assert not any(
        element == HOUSE and key in accessibility_nodes.ACCESSIBILITY_KEYS
        for element, key in values
    )
    # Null values are not stored in a baseline: the shop has no unit size.
    assert (SHOP, "unit_size_m2") not in values
    again = write(BASELINE)
    assert (again["values_written"], again["values_cleared"]) == (0, 0)
    assert stored(dsn, BASELINE) == values


def test_write_fork_stores_only_differences(dsn):
    write(BASELINE)
    baseline = stored(dsn, BASELINE)
    first = write(FORK)
    fork = stored(dsn, FORK)
    assert first["values_written"] == len(fork)
    for key, (num, text, _, _) in fork.items():
        if key in baseline:
            assert (num, text) != baseline[key][:2]
    # The house kept its attributes: no built-form override.
    assert (HOUSE, "density") not in fork
    # The shop became a house: overridden.
    assert fork[(SHOP, "num_housing_units")][0] == 1
    # The flats lost their unit size: a NULL row hides the baseline value.
    assert fork[(FLATS, "num_housing_units")][:2] == (None, None)
    # GONE is not in the fork: no rows.
    assert not any(element == GONE for element, _ in fork)
    again = write(FORK)
    assert (again["values_written"], again["values_cleared"]) == (0, 0)
    assert stored(dsn, FORK) == fork


def test_write_registers_missing_indicator_keys(dsn):
    # A key new to the plugin, missing from the catalog until the next baseline.
    with psycopg.connect(dsn) as conn:
        conn.execute(
            "DELETE FROM reimagina_urbano.indicator_catalog "
            "WHERE key IN ('distance_m_health', 'reachable_jobs')"
        )
    write(BASELINE)
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            "SELECT element_type::text, key, job_code, status::text "
            "FROM reimagina_urbano.indicator_catalog "
            "WHERE key IN ('distance_m_health', 'reachable_jobs') ORDER BY 1, 2"
        ).fetchall()
    assert rows == [
        ("block", "distance_m_health", "population_access", "active"),
        ("block", "reachable_jobs", "diversity", "active"),
    ]
    # Its values are written with the rest.
    assert stored(dsn, BASELINE)[(BLOCK, "distance_m_health")][0] > 0


def test_register_indicators_counts_the_rows_added(dsn):
    keys, types = ["distance_m_health", "population", "unknown"], ["lot", "block"]
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM reimagina_urbano.indicator_catalog "
            "WHERE key = 'distance_m_health'"
        )
        # Sent together: only the two missing rows count; unknown keys are skipped.
        assert catalogs.register_indicators(cur, keys, types) == 2
        assert catalogs.register_indicators(cur, keys, types) == 0
        assert catalogs.register_indicators(cur, [], types) == 0


def test_write_population_access(dsn):
    with psycopg.connect(dsn) as conn:
        conn.execute(
            "UPDATE reimagina_urbano.poi_catalog SET params = '{\"coverage_m\": 300}' "
            "WHERE poi_type IN ('health', 'establishment')"
        )
    write(BASELINE)
    values = stored(dsn, BASELINE)
    assert values[(BLOCK, "population_with_access_health")][0] == 93
    assert values[(BLOCK, "distance_m_health")][0] > 0
    # Only blocks are written.
    assert (HOUSE, "distance_m_health") not in values
    # No POIs of the type: no distance.
    assert (BLOCK, "distance_m_public_space") not in values
    # A partial run keeps the values of the other types.
    run_metric(
        "reimagina_scenario_population_access", BASELINE, poi_types=["establishment"]
    )
    assert stored(dsn, BASELINE) == values


def test_population_access_without_coverage_uses_the_default(dsn, caplog):
    # The seeded poi_catalog has no coverage_m.
    with caplog.at_level(logging.WARNING):
        write(BASELINE)
    assert "without coverage_m" in caplog.text
    access = stored(dsn, BASELINE)[(BLOCK, "population_with_access_health")]
    assert access[0] == 93


def test_write_diversity(dsn, monkeypatch):
    # A set over the seeded categories: the clinic's, the pharmacy's and one absent.
    subcategories = {
        "clinic": ("health_center",),
        "pharmacy": ("pharmacy",),
        "other": ("financial",),
    }
    monkeypatch.setitem(
        population_access_nodes.DIVERSITY_SETS, "fresh_food", subcategories
    )
    write(BASELINE)
    values = stored(dsn, BASELINE)
    # One of each: the pharmacy counts once, though of two poi_types; k = 3.
    expected = np.log(2) / np.log(3)
    key = "diversity_fresh_food"
    assert values[(BLOCK, key)][0] == pytest.approx(expected, abs=1e-4)
    # Only blocks are written.
    assert (HOUSE, key) not in values
    # The fork keeps only the park, in no subcategory: 0 where the house now is.
    write(FORK)
    assert stored(dsn, FORK)[(BLOCK_2, key)][0] == 0


def test_accessibility_tolerance_keeps_the_baseline_value(dsn):
    write(BASELINE)
    baseline = stored(dsn, BASELINE)
    key = next(
        k for e, k in baseline if e == BLOCK and k in accessibility_nodes.SCORE_KEYS
    )
    value = baseline[(BLOCK, key)][0]

    def write_score(delta: float, tolerance: float | None = 0.001) -> tuple[int, int]:
        table = pd.DataFrame({"id": [BLOCK], key: [value + delta]})
        return writer.write_indicators(
            FORK,
            writer.long_values(table, "block", {key: "number"}),
            {key: "number"},
            tolerances=None if tolerance is None else {key: tolerance},
        )

    # A shift below the tolerance is not stored: the fork shows the baseline's.
    assert write_score(-0.0009) == (0, 0)
    assert (BLOCK, key) not in stored(dsn, FORK)
    # At the tolerance it is.
    assert write_score(0.001) == (1, 0)
    assert stored(dsn, FORK)[(BLOCK, key)][0] == pytest.approx(value + 0.001)
    # Back within the tolerance: the stored value is cleared.
    assert write_score(0.0005) == (0, 1)
    assert (BLOCK, key) not in stored(dsn, FORK)
    # Without a tolerance any difference is stored.
    assert write_score(0.0001, tolerance=None) == (1, 0)


def test_write_accessibility_uses_the_tolerance():
    assert set(scenario_nodes.ACCESSIBILITY_WRITE_TOLERANCES) == set(
        accessibility_nodes.SCORE_KEYS
    )
    assert set(scenario_nodes.ACCESSIBILITY_WRITE_TOLERANCES.values()) == {0.001}


@pytest.mark.parametrize(
    ("keys", "tolerances"),
    [
        ({"farthest_amenity": "text"}, {"farthest_amenity": 0.001}),
        ({"minutes": "number"}, {"density": 0.001}),
        ({"minutes": "number"}, {"minutes": -1.0}),
        ({"minutes": "number"}, {"minutes": float("nan")}),
    ],
)
def test_write_rejects_invalid_tolerances(keys, tolerances):
    values = pd.DataFrame(columns=["id", "element_type", "parent_id", "key"])
    with pytest.raises(ValueError, match="Tolerances"):
        writer.write_indicators(str(uuid4()), values, keys, tolerances=tolerances)


def test_write_clears_values_back_to_baseline(dsn):
    write(BASELINE)
    write(FORK)
    with psycopg.connect(dsn) as conn:
        conn.execute(
            "DELETE FROM reimagina_urbano.element_state_attribute_value "
            "WHERE scenario_id = %s AND element_id = %s",
            (FORK, SHOP),
        )
    result = write(FORK)
    assert result["values_cleared"] > 0
    fork = stored(dsn, FORK)
    assert (SHOP, "num_housing_units") not in fork


def test_write_after_reverting_a_moved_lot(dsn):
    # The fork moved HOUSE to BLOCK_2; undoing that leaves its fork values stale.
    write(BASELINE)
    write(FORK)
    with psycopg.connect(dsn) as conn:
        conn.execute(
            "DELETE FROM reimagina_urbano.element_state "
            "WHERE scenario_id = %s AND element_id = %s",
            (FORK, HOUSE),
        )
    write(FORK)  # write() checks the result against a full write
    # Lots store no accessibility, so none is left stale on the house.
    values = stored(dsn, FORK)
    assert not any(
        element == HOUSE and key in accessibility_nodes.ACCESSIBILITY_KEYS
        for element, key in values
    )


def test_dry_run_and_job_run(dsn):
    dry = write(BASELINE, dry_run=True)
    assert dry["dry_run"]
    assert dry["values_written"] > 0
    assert stored(dsn, BASELINE) == {}
    job = str(uuid4())
    with psycopg.connect(dsn) as conn:
        conn.execute("INSERT INTO reimagina_urbano.job_run VALUES (%s)", (job,))
    write(BASELINE, job_run_id=job)
    assert {row[2] for row in stored(dsn, BASELINE).values()} == {job}
    # An unknown job_run is not recorded; unchanged rows keep the old one.
    with psycopg.connect(dsn) as conn:
        conn.execute(
            "UPDATE reimagina_urbano.element_state_attribute_value "
            "SET value_num = 3 WHERE scenario_id = %s AND element_id = %s "
            "AND attribute_id = (SELECT id FROM reimagina_urbano.attribute_catalog "
            "WHERE key = 'num_levels')",
            (BASELINE, HOUSE),
        )
    write(BASELINE, job_run_id=str(uuid4()))
    values = stored(dsn, BASELINE)
    assert values[(HOUSE, "num_levels")][2] is None
    assert values[(SHOP, "num_levels")][2] == job


def scenario_statistics(dsn: str) -> list[str]:
    """The scenario ids the planner's statistics know in the value table."""
    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            "SELECT most_common_vals::text::text[] FROM pg_stats "
            "WHERE schemaname = 'reimagina_urbano' "
            "AND tablename = 'element_state_indicator_value' "
            "AND attname = 'scenario_id'"
        ).fetchone()
    return list(row[0] or []) if row else []


def test_write_refreshes_the_statistics(dsn):
    # Without them the API plans a fork as ~1 row (a nested loop over the baseline).
    write(BASELINE, dry_run=True)
    assert scenario_statistics(dsn) == []
    write(BASELINE)
    assert scenario_statistics(dsn) == [BASELINE]
    write(FORK)
    assert sorted(scenario_statistics(dsn)) == sorted([BASELINE, FORK])


def test_failed_statistics_refresh_only_warns(caplog):
    class Failing:
        def transaction(self):
            raise psycopg.errors.LockNotAvailable("lock timeout")

    with caplog.at_level(logging.WARNING):
        writer._refresh_statistics(Failing(), logging.getLogger("test"))
    assert "ANALYZE of the indicator values failed" in caplog.text


def test_composites_register_their_catalog_rows(dsn):
    # No composite key is seeded: the composites' write registers them.
    run_metric("reimagina_scenario_composites", BASELINE, dry_run=True)
    with psycopg.connect(dsn) as conn:
        assert not conn.execute(
            "SELECT 1 FROM reimagina_urbano.indicator_catalog "
            "WHERE key LIKE 'composite%%'"
        ).fetchall()  # a dry run rolls the registration back too
        conn.execute(
            "INSERT INTO reimagina_urbano.indicator_catalog "
            "(element_type, key, name, value_type, scope, job_code) VALUES "
            "('block', 'composite_retired', 'x', 'number', 'scenario', 'composites')"
        )
    run_metric("reimagina_scenario_composites", BASELINE)
    run_metric("reimagina_scenario_composites", BASELINE)  # idempotent
    with psycopg.connect(dsn) as conn:
        rows = dict(
            conn.execute(
                "SELECT key, status::text FROM reimagina_urbano.indicator_catalog "
                "WHERE job_code = 'composites'"
            ).fetchall()
        )
        inputs = conn.execute(
            "SELECT ic.element_type::text, ic.key, inp.key "
            "FROM reimagina_urbano.indicator_input ii "
            "JOIN reimagina_urbano.indicator_catalog ic ON ic.id = ii.indicator_id "
            "JOIN reimagina_urbano.indicator_catalog inp "
            "ON inp.id = ii.input_indicator_id"
        ).fetchall()
    names = [row[1] for row in composite_catalog.catalog_rows()]
    assert rows == {**dict.fromkeys(names, "active"), "composite_retired": "deprecated"}
    assert sorted(inputs) == sorted(composite_catalog.input_pairs())
    values = stored(dsn, BASELINE)
    assert any(key == "composite_index" for _, key in values)


def calibration_table(scenario_id: str) -> pd.DataFrame:
    runner = LocalRunner.from_definitions(
        {"reimagina-test-port": plugin.create_plugin()}
    )
    name = composite_metric.CALIBRATION
    return runner.run(name, {"parameters": {"scenario_id": scenario_id}})[name]


def calibration_row(dsn: str, scenario_id: str):
    with psycopg.connect(dsn) as conn:
        return conn.execute(
            "SELECT uri, content_hash, job_run_id::text "
            "FROM reimagina_urbano.scenario_file "
            "WHERE scenario_id = %s AND kind = 'composite_calibration'",
            (scenario_id,),
        ).fetchone()


def test_a_baseline_saves_its_calibration_and_forks_read_it(dsn):
    uri = calibrations.calibration_uri(PROJECT)
    # Before the baseline has one, a fork fails: no scoring against another city.
    for name in (composite_metric.CALIBRATION, "reimagina_scenario_composites"):
        with pytest.raises(GraphExecutionError) as failure:
            run_metric(name, FORK)
        assert failure.value.node == composite_metric.CALIBRATION
        assert isinstance(failure.value.__cause__, MetricInputError)
        assert "no composite calibration" in str(failure.value.__cause__)
    # A dry run saves nothing.
    run_metric("reimagina_scenario_composites", BASELINE, dry_run=True)
    assert calibration_row(dsn, BASELINE) is None and not Path(uri).exists()

    job = str(uuid4())
    with psycopg.connect(dsn) as conn:
        conn.execute("INSERT INTO reimagina_urbano.job_run VALUES (%s)", (job,))
    saved = run_metric(
        "reimagina_scenario_composite_calibration", BASELINE, job_run_id=job
    )
    assert saved["saved"] and saved["uri"] == uri
    text = Path(uri).read_text(encoding="utf-8")
    document = json.loads(text)
    assert document["project_id"] == PROJECT
    assert document["baseline_scenario_id"] == BASELINE
    assert len(document["leaves"]) == saved["leaves"] > 0
    digest = hashlib.sha256(text.encode()).hexdigest()
    assert calibration_row(dsn, BASELINE) == (uri, digest, job)
    assert calibration_row(dsn, FORK) is None  # forks get no row of their own
    fork = run_metric("reimagina_scenario_composite_calibration", FORK)
    assert not fork["saved"] and fork["uri"] == uri

    # The fork reads the project's file and never computes its own: a value
    # changed in the file is the fork's.
    leaf = next(iter(document["leaves"]))
    document["leaves"][leaf]["reference"] = 12345.0
    Path(uri).write_text(json.dumps(document), encoding="utf-8")
    table = calibration_table(FORK)
    assert set(table["origin"]) == {uri}
    assert calibrations.from_table(table) == document["leaves"]
    # Running the baseline again recomputes and overwrites it.
    run_metric("reimagina_scenario_composite_calibration", BASELINE)
    assert (
        json.loads(Path(uri).read_text(encoding="utf-8"))["leaves"][leaf]["reference"]
        != 12345.0
    )


def test_the_baseline_average_of_a_calibrated_leaf_scores_100(dsn):
    table = calibration_table(BASELINE)
    assert set(table["origin"]) == {calibrations.COMPUTED}
    values = calibrations.from_table(table)
    _, composites, _ = composite_metric.check_tree(composite_metric.load_config())
    node, leaf = next(
        (node, child)
        for node in composites
        for child in node["children"]
        if child.get("name") in values and "transform" not in child
        and not child.get("compress")
    )  # fmt: skip
    runner = LocalRunner.from_definitions(
        {"reimagina-test-port": plugin.create_plugin()}
    )
    name = f"reimagina_composite_{node['name']}"
    scores = runner.run(name, {"parameters": {"scenario_id": BASELINE}})[
        f"{name}_blocks"
    ][leaf["name"]]
    assert scores.dropna().mean() == pytest.approx(100)


def test_a_failed_calibration_upload_fails_the_composites_write(dsn, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("the lake is down")

    monkeypatch.setattr(lake, "upload", fail)
    with pytest.raises(GraphExecutionError) as failure:
        run_metric("reimagina_scenario_composites", BASELINE)
    assert failure.value.node == "reimagina_scenario_composite_calibration"
    assert calibration_row(dsn, BASELINE) is None
    assert not any(key.startswith("composite_") for _, key in stored(dsn, BASELINE))


def test_write_checks_the_indicator_catalog(dsn):
    with psycopg.connect(dsn) as conn:
        conn.execute(
            "UPDATE reimagina_urbano.indicator_catalog SET status = 'deprecated' "
            "WHERE key = 'cus' AND element_type = 'block'"
        )
        conn.execute(
            "UPDATE reimagina_urbano.indicator_catalog SET value_type = 'number' "
            "WHERE key = 'farthest_amenity' AND element_type = 'block'"
        )
    # Built-form and accessibility are checked and written separately.
    with pytest.raises(GraphExecutionError) as failure:
        write(BASELINE)
    message = str(failure.value.__cause__)
    assert "missing block.cus (number)" in message
    with pytest.raises(GraphExecutionError) as failure:
        run_metric("reimagina_scenario_accessibility", BASELINE)
    message = str(failure.value.__cause__)
    assert "block.farthest_amenity is number, expected text" in message
    assert stored(dsn, BASELINE) == {}


# ------------------------------------------------------------- baseline writer

NEW_BASELINE = "40000000-0000-0000-0000-000000000000"
NEW_FORK = "50000000-0000-0000-0000-000000000000"
NEW_LOTS = [f"c0000000-0000-0000-0000-00000000000{i}" for i in range(1, 5)]
NEW_AMENITY = "d0000000-0000-0000-0000-000000000001"
NEW_ESTABLISHMENT = "e0000000-0000-0000-0000-000000000001"


def lot_box(i: int, shift: float = 0.0):
    x = i * 0.002 + shift
    return box(x, 0.0, x + 0.001, 0.001)


def new_tables(count: int = 4):
    """Blocks reuse BLOCK and BLOCK_2; four lots, an amenity and an establishment
    with new ids, both in the first lot."""
    blocks = gpd.GeoDataFrame(
        {
            "id": [BLOCK, BLOCK_2],
            "parent_id": [None, None],
            "source_record_ref": ["cvegeo:1", "cvegeo:2"],
            "vivtot": [10.0, None],
            "housing.people_per_household": [3.0, None],
            "lot.size_m2": [None, None],
            "lot.frontage_m": [None, None],
        },
        geometry=[box(0, 0, 0.004, 0.001), box(0.004, 0, 0.008, 0.001)],
        crs=4326,
    )
    lots = gpd.GeoDataFrame(
        {
            "id": NEW_LOTS[:count],
            "parent_id": [BLOCK, BLOCK, BLOCK_2, BLOCK_2][:count],
            "source_record_ref": [f"1-A-{i}" for i in range(count)],
            "land_use": ["single_family", "commerce", "multifamily", None][:count],
            "cos": [0.5, 1.0, 0.6, None][:count],
            "num_levels": [2, 1, 4, None][:count],
            "area": [200.0, 100.0, 1000.0, 50.0][:count],
            "housing.unit_size_m2": [None, None, 80.0, None][:count],
            "housing.people_per_household": [3.5, None, 3.0, None][:count],
            "occupancy_rate": [0.9, None, None, None][:count],
            "housing.floor_share": [100.0, 100.0, 80.0, None][:count],
            "commerce_type": [None, "shop", None, None][:count],
            "tree_coverage": [12.5, None, None, None][:count],
        },
        geometry=[lot_box(i, 0.00002) for i in range(count)],
        crs=4326,
    )
    inside = lot_box(0, 0.00002).centroid
    amenities = gpd.GeoDataFrame(
        {
            "id": [NEW_AMENITY],
            "parent_id": [NEW_LOTS[0]],
            "source_record_ref": ["denue:11"],
            "category": ["health_center"],
            "scian_code": ["621112"],
            "num_workers": [3.0],
            "capacity": [1.0],
            "name": [None],
            "area_m2": [None],
            # A clinic, not a park: no tree coverage.
            "tree_coverage": [None],
        },
        geometry=[inside],
        crs=4326,
    )
    establishments = gpd.GeoDataFrame(
        {
            "id": [NEW_ESTABLISHMENT],
            "parent_id": [NEW_LOTS[0]],
            "source_record_ref": ["denue:101"],
            "category": ["convenience_store"],
            "scian_code": ["461110"],
            "scian_sector": ["46"],
            "num_workers": [None],
            "name": ["Tienda"],
        },
        geometry=[inside],
        crs=4326,
    )
    return {
        "block": blocks,
        "lot": lots,
        "amenity": amenities,
        "establishment": establishments,
    }


@pytest.fixture
def new_baseline(dsn):
    with psycopg.connect(dsn) as conn:
        conn.execute(
            "INSERT INTO reimagina_urbano.scenarios VALUES (%s, %s, NULL, NULL)",
            (NEW_BASELINE, PROJECT),
        )
    return NEW_BASELINE


def write_new_baseline(count: int = 4, **options) -> dict:
    return baseline_writer.write_baseline(
        PROJECT,
        NEW_BASELINE,
        new_tables(count),
        ingestion_nodes.BASELINE_ATTRIBUTES,
        **options,
    )


def counts(dsn: str, scenario_id: str) -> tuple[int, int]:
    with psycopg.connect(dsn) as conn:
        return tuple(
            conn.execute(
                f"SELECT count(*) FROM reimagina_urbano.{table} WHERE scenario_id = %s",
                (scenario_id,),
            ).fetchone()[0]
            for table in ("element_state", "element_state_attribute_value")
        )


def test_baseline_contract(new_baseline, dsn):
    with psycopg.connect(dsn) as conn:
        conn.execute(
            "UPDATE reimagina_urbano.attribute_catalog SET status = 'deprecated' "
            "WHERE key = 'occupancy_rate'"
        )
        conn.execute(
            "UPDATE reimagina_urbano.attribute_catalog "
            "SET enum_values = ARRAY['single_family'] WHERE key = 'land_use'"
        )
    with pytest.raises(contract.ContractError) as failure:
        write_new_baseline()
    assert "missing lot.occupancy_rate (number)" in str(failure.value)
    assert "lot.land_use enum lacks ['commerce', 'multifamily']" in str(failure.value)
    assert counts(dsn, NEW_BASELINE) == (0, 0)


def test_catalog_upsert_needs_the_element_types(dsn):
    with (
        psycopg.connect(dsn) as conn,
        conn.cursor() as cur,
        pytest.raises(contract.ContractError, match=r"lacks \['gate'\]"),
    ):
        catalogs.upsert(cur, {"gate": {"name": "text"}})


def test_write_new_baseline(new_baseline, dsn):
    old = counts(dsn, BASELINE)
    with psycopg.connect(dsn) as conn:
        conn.execute(
            "DELETE FROM reimagina_urbano.indicator_catalog "
            "WHERE element_type = 'block' AND key = 'density'"
        )
        land_use = conn.execute(
            "SELECT id, enum_values FROM reimagina_urbano.attribute_catalog "
            "WHERE element_type = 'lot' AND key = 'land_use'"
        ).fetchone()
    dry = write_new_baseline(dry_run=True)
    assert dry["elements_created"] == 6
    assert counts(dsn, NEW_BASELINE) == (0, 0)
    first = write_new_baseline()
    # The blocks already exist: reused, not created. 23 lot and block values, 4 of
    # the amenity (no name) and 4 of the establishment (no workers). The catalogs
    # get the 12 attributes they lacked (housing.floor_share, tree_coverage, the
    # block's targets, 5 of the amenity and 3 of the establishment), block.density and the
    # composites' keys (the scenario writer registers those too); the dry run
    # rolled its rows back.
    assert first == {
        "elements": 8,
        "elements_created": 6,
        "attribute_values": 32,  # with the first lot's tree_coverage
        "elements_removed": 0,
        "attributes_added": 12,
        "indicators_added": 1 + len(composite_catalog.catalog_rows()),
        "by_type": {"block": 2, "lot": 4, "amenity": 1, "establishment": 1},
    }
    assert counts(dsn, NEW_BASELINE) == (8, 32)
    with psycopg.connect(dsn) as conn:
        # Added rows take the plugin's types and units; existing ones stay.
        rows = dict(
            conn.execute(
                "SELECT element_type || '.' || key, (value_type::text, unit, "
                "enum_values) FROM reimagina_urbano.attribute_catalog"
            ).fetchall()
        )
        density = conn.execute(
            "SELECT name, unit, job_code FROM reimagina_urbano.indicator_catalog "
            "WHERE element_type = 'block' AND key = 'density'"
        ).fetchone()
        assert (
            conn.execute(
                "SELECT id, enum_values FROM reimagina_urbano.attribute_catalog "
                "WHERE element_type = 'lot' AND key = 'land_use'"
            ).fetchone()
            == land_use
        )
    assert rows["lot.housing.floor_share"] == ("number", "%", None)
    assert rows["block.lot.size_m2"] == ("number", "m2", None)
    assert rows["establishment.num_workers"] == ("integer", None, None)
    assert rows["amenity.area_m2"] == ("number", "m2", None)
    assert density == ("Densidad", "viv/ha", "indicators")
    assert counts(dsn, BASELINE) == old
    tables = scenario_nodes.read_scenario(NEW_BASELINE)
    lots = by_id(tables["reimagina_scenario_lots"])
    assert lots[NEW_LOTS[0]]["num_levels"] == 2
    assert lots[NEW_LOTS[0]]["occupancy_rate"] == pytest.approx(0.9)
    assert lots[NEW_LOTS[2]]["parent_id"] == BLOCK_2
    assert lots[NEW_LOTS[1]]["source_record_ref"] == "1-A-1"
    with psycopg.connect(dsn) as conn:
        origin = conn.execute(
            "SELECT origin_scenario_id::text FROM reimagina_urbano.elements "
            "WHERE id = %s",
            (NEW_LOTS[0],),
        ).fetchone()[0]
        points = dict(
            conn.execute(
                "SELECT e.type::text, s.parent_element_id::text "
                "FROM reimagina_urbano.elements e JOIN reimagina_urbano.element_state s "
                "ON s.element_id = e.id AND s.scenario_id = %s WHERE e.id IN (%s, %s)",
                (NEW_BASELINE, NEW_AMENITY, NEW_ESTABLISHMENT),
            ).fetchall()
        )
        clinic = conn.execute(
            "SELECT v.value_text FROM reimagina_urbano.element_state_attribute_value v "
            "JOIN reimagina_urbano.attribute_catalog c ON c.id = v.attribute_id "
            "WHERE v.scenario_id = %s AND v.element_id = %s AND c.key = 'category'",
            (NEW_BASELINE, NEW_AMENITY),
        ).fetchone()[0]
    assert origin == NEW_BASELINE
    assert points == {"amenity": NEW_LOTS[0], "establishment": NEW_LOTS[0]}
    assert clinic == "health_center"
    # Accessibility destinations are the amenity and establishment elements: the
    # clinic and the store count, by their category.
    assert scenario.read_pois(NEW_BASELINE)["poi_type"].tolist() == [
        "health",
        "convenience_store",
        "establishment",
    ]
    # Indicators then come from the scenario writer.
    assert write(NEW_BASELINE)["values_written"] > 0
    # Again without one lot: replaced, and its indicators go too.
    second = write_new_baseline(count=3)
    assert second["elements_removed"] == 1
    assert second["by_type"]["lot"] == 3
    assert counts(dsn, NEW_BASELINE)[0] == 7
    assert not any(e == NEW_LOTS[3] for e, _ in stored(dsn, NEW_BASELINE))


def test_baseline_with_forks_is_refused(new_baseline, dsn):
    with psycopg.connect(dsn) as conn:
        conn.execute(
            "INSERT INTO reimagina_urbano.scenarios VALUES (%s, %s, %s, NULL)",
            (NEW_FORK, PROJECT, NEW_BASELINE),
        )
    with pytest.raises(baseline_writer.BaselineScenarioError, match="has forks"):
        write_new_baseline()
    with pytest.raises(baseline_writer.BaselineScenarioError, match="is a fork"):
        baseline_writer.write_baseline(
            PROJECT, FORK, new_tables(), ingestion_nodes.BASELINE_ATTRIBUTES
        )


def test_migrate_fork_edits(new_baseline, dsn):
    # Distinct old lot shapes; the new lots cover them, shifted by 2 %.
    with psycopg.connect(dsn) as conn:
        for i, lot in enumerate([HOUSE, SHOP, FLATS, GONE]):
            conn.execute(
                "UPDATE reimagina_urbano.element_state "
                "SET geometry = ST_GeomFromText(%s, 4326) "
                "WHERE scenario_id = %s AND element_id = %s",
                (lot_box(i).wkt, BASELINE, lot),
            )
    write_new_baseline()
    with psycopg.connect(dsn) as conn:
        conn.execute(
            "INSERT INTO reimagina_urbano.scenarios VALUES (%s, %s, %s, NULL)",
            (NEW_FORK, PROJECT, NEW_BASELINE),
        )
    spec = importlib.util.spec_from_file_location(
        "migrate_fork_edits", ROOT / "scripts" / "migrate_fork_edits.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dry = module.migrate(dsn, FORK, NEW_FORK, apply=False)
    assert counts(dsn, NEW_FORK) == (0, 0)
    mapping = {row[0]: row for row in module.migrate(dsn, FORK, NEW_FORK, apply=True)}
    assert {k: v[:3] for k, v in mapping.items()} == {
        k: v[:3] for k, v in ((r[0], r) for r in dry)
    }
    expected = {
        HOUSE: NEW_LOTS[0],
        SHOP: NEW_LOTS[1],
        FLATS: NEW_LOTS[2],
        GONE: NEW_LOTS[3],
    }
    for old, new in expected.items():
        assert mapping[old][1:3] == (new, "geometry")
        assert mapping[old][3] > module.MIN_IOU
    # Only in the fork: same id.
    assert mapping[NEW][1:3] == (NEW, "same id")
    fork = scenario_nodes.read_scenario(NEW_FORK)
    lots = by_id(fork["reimagina_scenario_lots"])
    # The shop's edit (now a house) lands on the new lot; GONE stays deleted.
    assert lots[NEW_LOTS[1]]["land_use"] == "single_family"
    assert NEW_LOTS[3] not in lots
    assert lots[NEW]["land_use"] == "commerce"
    # The house was moved to BLOCK_2 in the fork; its parent carries over.
    assert lots[NEW_LOTS[0]]["parent_id"] == BLOCK_2
    with pytest.raises(ValueError, match="already has edits"):
        module.migrate(dsn, FORK, NEW_FORK, apply=False)


def test_baseline_write_metric(new_baseline, dsn, monkeypatch, offline_network):
    from reimagina_test_port.ingestion import baseline, streets, transit

    tables = new_tables()
    monkeypatch.setattr(
        baseline,
        "build",
        lambda *_: {
            "lots": tables["lot"],
            "blocks": tables["block"],
            "amenities": tables["amenity"],
            "establishments": tables["establishment"],
            "report": {},
        },
    )
    street = gpd.GeoDataFrame(
        {
            "id": [str(uuid4())],
            "parent_id": [None],
            "source_record_ref": ["1-2"],
            "bands.use": ['["sidewalk", "protected_bike_lane", "lanes", "sidewalk"]'],
            "bands.share": ["[20, 10, 50, 20]"],
            "streetlights": [50.0],
            "tree_coverage": [30.0],
        },
        # A right of way drawn as a polygon.
        geometry=[LineString([(-115.5, 32.6), (-115.49, 32.6)]).buffer(0.0001)],
        crs=4326,
    )
    monkeypatch.setattr(streets, "build", lambda *_: street)
    stop = gpd.GeoDataFrame(
        {
            "id": [str(uuid4())],
            "parent_id": [None],
            "source_record_ref": ["IMIP:1"],
            "element_type": ["bus_stop"],
            "source": ["IMIP"],
            "name": ["AUTOTRANSPORTES URBANOS"],
        },
        geometry=gpd.points_from_xy([-115.5], [32.6]),
        crs=4326,
    )
    monkeypatch.setattr(transit, "build", lambda *_: stop)
    runner = LocalRunner.from_definitions(
        {"reimagina-test-port": plugin.create_plugin()}
    )
    parameters = {"city": "mexicali", "project_id": PROJECT, "scenario_id": FORK}
    with pytest.raises(GraphExecutionError) as failure:
        runner.run("reimagina_baseline_write", {"parameters": parameters})
    assert isinstance(failure.value.__cause__, MetricInputError)
    parameters["scenario_id"] = NEW_BASELINE
    result = runner.run("reimagina_baseline_write", {"parameters": parameters})
    row = result["reimagina_baseline_write"].iloc[0].to_dict()
    assert (row["elements"], row["elements_created"], row["dry_run"]) == (10, 8, False)
    columns = (
        "blocks", "lots", "amenities", "establishments", "rights_of_way", "bus_stops",
        "stations",
    )  # fmt: skip
    assert [row[c] for c in columns] == [2, 4, 1, 1, 1, 1, 0]
    # The stop has no parent; its source and name are stored as text.
    with psycopg.connect(dsn) as conn:
        parent, source = conn.execute(
            "SELECT s.parent_element_id, v.value_text "
            "FROM reimagina_urbano.element_state s "
            "JOIN reimagina_urbano.element_state_attribute_value v "
            "ON v.element_id = s.element_id AND v.scenario_id = s.scenario_id "
            "JOIN reimagina_urbano.attribute_catalog c ON c.id = v.attribute_id "
            "WHERE s.element_id = %s::uuid AND c.key = 'source'",
            (stop["id"].iloc[0],),
        ).fetchone()
    assert (parent, source) == (None, "IMIP")

    # A street's bands are stored as JSON lists; its streetlights and
    # tree_coverage as numbers. A lot's tree_coverage too.
    def stored(element_id: str) -> dict:
        with psycopg.connect(dsn) as conn:
            return dict(
                conn.execute(
                    "SELECT ac.key, coalesce(v.value_json::text, v.value_num::text) "
                    "FROM reimagina_urbano.element_state_attribute_value v "
                    "JOIN reimagina_urbano.attribute_catalog ac "
                    "ON ac.id = v.attribute_id WHERE v.element_id = %s::uuid",
                    (element_id,),
                ).fetchall()
            )

    assert stored(street["id"].iloc[0]) == {
        "bands.use": '["sidewalk", "protected_bike_lane", "lanes", "sidewalk"]',
        "bands.share": "[20, 10, 50, 20]",
        "streetlights": "50",
        "tree_coverage": "30",
    }
    assert stored(tables["lot"]["id"].iloc[0])["tree_coverage"] == "12.5"
    with psycopg.connect(dsn) as conn:
        catalog = conn.execute(
            "SELECT element_type::text, value_type::text, unit "
            "FROM reimagina_urbano.attribute_catalog WHERE key = 'tree_coverage' "
            "ORDER BY 1"
        ).fetchall()
    # A park's too (its lot's): the amenity has the key, as a percentage.
    assert catalog == [
        ("amenity", "number", "%"),
        ("lot", "number", "%"),
        ("right_of_way", "number", "%"),
    ]


# ------------------------------------------------------------------------ append

APPEND_PARK = "d0000000-0000-0000-0000-000000000002"
APPEND_FIRE = "d0000000-0000-0000-0000-000000000003"
APPEND_STOP = "d0000000-0000-0000-0000-000000000004"


def append_tables() -> dict[str, gpd.GeoDataFrame]:
    """new_tables' four lots and its clinic, plus a park on the second lot, a fire
    station on the fourth and a bus stop."""
    tables = new_tables()
    extra = gpd.GeoDataFrame(
        {
            "id": [APPEND_PARK, APPEND_FIRE],
            "parent_id": [NEW_LOTS[1], NEW_LOTS[3]],
            "source_record_ref": ["IMIP:esp_pub:1", "IMIP:admon_pub:5"],
            "category": ["park", "fire_station"],
            "scian_code": [None, None],
            "num_workers": [None, None],
            "capacity": [1.0, 1.0],
            "name": [None, "Bomberos"],
            "area_m2": [2000.0, None],
            "tree_coverage": [35.0, None],
        },
        geometry=[lot_box(1).centroid, lot_box(3).centroid],
        crs=4326,
    )
    tables["amenity"] = pd.concat([tables["amenity"], extra], ignore_index=True)
    tables["bus_stop"] = gpd.GeoDataFrame(
        {
            "id": [APPEND_STOP],
            "parent_id": [None],
            "source_record_ref": ["IMIP:1"],
            "element_type": ["bus_stop"],
            "source": ["IMIP"],
            "name": ["AUTOTRANSPORTES URBANOS"],
        },
        geometry=gpd.points_from_xy([0.0005], [0.002]),
        crs=4326,
    )
    # The lots' canopy, to update: the first lot's changes, the third gets one.
    tables["lot"] = tables["lot"].assign(tree_coverage=[20.0, None, 5.0, 7.0])
    return tables


@pytest.fixture
def forked_baseline(new_baseline, dsn):
    """new_baseline with three lots and a fork that deletes the second."""
    write_new_baseline(count=3)
    with psycopg.connect(dsn) as conn:
        conn.execute(
            "INSERT INTO reimagina_urbano.scenarios VALUES (%s, %s, %s, NULL)",
            (NEW_FORK, PROJECT, NEW_BASELINE),
        )
        conn.execute("SET search_path = reimagina_urbano, public")
        add_state(conn, NEW_FORK, NEW_LOTS[1], deleted=True)
    return NEW_BASELINE


def append(**options) -> dict:
    tables = append_tables()
    return baseline_writer.append_baseline(
        PROJECT,
        NEW_BASELINE,
        {t: tables[t] for t in ("amenity", "bus_stop")},
        ingestion_nodes.BASELINE_ATTRIBUTES,
        updates={
            "lot": (tables["lot"], ["tree_coverage"]),
            "amenity": (tables["amenity"], ["tree_coverage"]),
        },
        **options,
    )


def outcomes(result: dict) -> list[tuple]:
    return [
        (r["element_type"], r["attribute"], r["outcome"], r["count"], r["examples"])
        for r in result["outcomes"]
    ]


def test_append_to_a_baseline_with_forks(forked_baseline, dsn):
    before = counts(dsn, NEW_BASELINE)
    fork_before = counts(dsn, NEW_FORK)
    expected = [
        ("amenity", None, "inserted", 1, ["IMIP:esp_pub:1"]),
        ("amenity", None, "already_present", 1, ["denue:11"]),
        # The baseline has three lots: the fire station's is not one.
        ("amenity", None, "missing_parent", 1, ["IMIP:admon_pub:5"]),
        # The fork deleted the park's lot; the park is there still.
        ("amenity", None, "parent_deleted_in_fork", 1, ["IMIP:esp_pub:1"]),
        # Updates concern the elements the baseline had: the park is new.
        ("amenity", "tree_coverage", "absent", 1, ["IMIP:esp_pub:1"]),
        ("bus_stop", None, "inserted", 1, ["IMIP:1"]),
        ("lot", "tree_coverage", "added", 1, ["1-A-2"]),
        ("lot", "tree_coverage", "updated", 1, ["1-A-0"]),
        ("lot", "tree_coverage", "absent", 1, ["1-A-3"]),
    ]
    assert "public_space" not in set(scenario.read_pois(NEW_FORK)["poi_type"])
    dry = append(dry_run=True)
    assert outcomes(dry) == expected
    assert counts(dsn, NEW_BASELINE) == before
    result = append()
    assert outcomes(result) == expected
    # The stops' source and name join the catalog; the amenity's keys were there.
    assert (result["attributes_added"], result["indicators_added"]) == (2, 0)
    # Two elements: the park with category, capacity, area and canopy, the stop
    # with source and name; one lot value added.
    assert counts(dsn, NEW_BASELINE) == (before[0] + 2, before[1] + 7)
    assert counts(dsn, NEW_FORK) == fork_before
    with psycopg.connect(dsn) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM reimagina_urbano.elements WHERE id = %s",
                (APPEND_FIRE,),
            ).fetchone()[0]
            == 0
        )
        parent = conn.execute(
            "SELECT parent_element_id::text FROM reimagina_urbano.element_state "
            "WHERE scenario_id = %s AND element_id = %s",
            (NEW_BASELINE, APPEND_PARK),
        ).fetchone()[0]
    assert parent == NEW_LOTS[1]
    lots = by_id(
        scenario.read_scenario(NEW_BASELINE, ["land_use", "tree_coverage"])["lot"]
    )
    assert lots[NEW_LOTS[0]]["tree_coverage"] == 20
    assert lots[NEW_LOTS[2]]["tree_coverage"] == 5
    # Nothing else changed.
    assert lots[NEW_LOTS[0]]["land_use"] == "single_family"
    # The fork sees the new park through the baseline.
    assert "public_space" in set(scenario.read_pois(NEW_FORK)["poi_type"])
    # Again: nothing new.
    again = {(r[0], r[1], r[2]): r[3] for r in outcomes(append())}
    assert again == {
        ("amenity", None, "already_present"): 2,
        ("amenity", None, "missing_parent"): 1,
        ("amenity", "tree_coverage", "unchanged"): 1,
        ("bus_stop", None, "already_present"): 1,
        ("lot", "tree_coverage", "unchanged"): 2,
        ("lot", "tree_coverage", "absent"): 1,
    }
    assert counts(dsn, NEW_BASELINE) == (before[0] + 2, before[1] + 7)


def test_append_needs_a_baseline_and_known_attributes(forked_baseline):
    tables = append_tables()
    with pytest.raises(baseline_writer.BaselineScenarioError, match="is a fork"):
        baseline_writer.append_baseline(
            PROJECT,
            NEW_FORK,
            {"amenity": tables["amenity"]},
            ingestion_nodes.BASELINE_ATTRIBUTES,
        )
    with pytest.raises(baseline_writer.BaselineScenarioError, match="no attributes"):
        baseline_writer.append_baseline(
            PROJECT,
            NEW_BASELINE,
            {},
            ingestion_nodes.BASELINE_ATTRIBUTES,
            updates={"lot": (tables["lot"], ["canopy"])},
        )


def test_baseline_append_metric(forked_baseline, dsn, monkeypatch):
    from reimagina_test_port.ingestion import baseline, streets, transit

    tables = append_tables()
    monkeypatch.setattr(
        baseline,
        "build",
        lambda *_: {
            "lots": tables["lot"],
            "blocks": tables["block"],
            "amenities": tables["amenity"],
            "establishments": tables["establishment"],
            "report": {},
        },
    )
    monkeypatch.setattr(transit, "build", lambda *_: tables["bus_stop"])

    def no_streets(*_):
        raise AssertionError("appending does not fit the streets")

    monkeypatch.setattr(streets, "build", no_streets)
    runner = LocalRunner.from_definitions(
        {"reimagina-test-port": plugin.create_plugin()}
    )
    parameters = {
        "city": "mexicali",
        "project_id": PROJECT,
        "scenario_id": NEW_BASELINE,
        "element_types": ["amenity"],
        "categories": ["fire_station", "park"],
        "update_attributes": ["lot.tree_coverage"],
        "dry_run": True,
    }
    result = runner.run("reimagina_baseline_append", {"parameters": parameters})
    rows = result["reimagina_baseline_append"]
    assert rows[["element_type", "attribute", "outcome", "count"]].fillna(
        "-"
    ).values.tolist() == [
        ["amenity", "-", "inserted", 1],
        ["amenity", "-", "missing_parent", 1],
        ["amenity", "-", "parent_deleted_in_fork", 1],
        ["lot", "tree_coverage", "added", 1],
        ["lot", "tree_coverage", "updated", 1],
        ["lot", "tree_coverage", "absent", 1],
    ]
    assert rows["examples"].iloc[1] == "IMIP:admon_pub:5"
    assert rows["dry_run"].all()
    with psycopg.connect(dsn) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM reimagina_urbano.elements WHERE id = %s",
                (APPEND_PARK,),
            ).fetchone()[0]
            == 0
        )
    for field, value in (
        ("categories", ["fire"]),
        ("update_attributes", ["tree_coverage"]),
        ("element_types", []),
    ):
        bad = {**parameters, field: value}
        if field == "element_types":
            bad["update_attributes"] = []
        with pytest.raises(GraphExecutionError) as failure:
            runner.run("reimagina_baseline_append", {"parameters": bad})
        assert isinstance(failure.value.__cause__, MetricInputError)


# ------------------------------------------------------------------ street network


@pytest.fixture
def offline_network(monkeypatch, tmp_path):
    """Mexicali with a local lake folder, one square of blocks and OSM a triangle."""
    import dataclasses

    from reimagina_test_port.common import lake
    from reimagina_test_port.ingestion import network
    from reimagina_test_port.ingestion.cities import CITIES

    city = CITIES["mexicali"]
    lake_folder = tmp_path / "lake"
    monkeypatch.setitem(
        CITIES,
        "mexicali",
        dataclasses.replace(
            city,
            streets=dataclasses.replace(city.streets, output_prefix=str(lake_folder)),
        ),
    )
    square = gpd.GeoSeries([box(-115.46, 32.61, -115.45, 32.62)], crs=4326)
    monkeypatch.setattr(
        lake,
        "read_parquet",
        lambda uri, columns: pd.DataFrame({"geometry": square.to_wkb()}),
    )
    nodes = pd.DataFrame(
        {"x": [0.0, 100.0, 0.0], "y": [0.0, 0.0, 100.0]},
        index=pd.Index([1, 2, 3], name="id"),
    )
    edges = pd.DataFrame(
        {"from": [1, 2, 3], "to": [2, 3, 1], "length": [100, 141.4, 100]}
    )
    monkeypatch.setattr(
        network,
        "osm_network",
        lambda blocks, crs, params, boundary, **_: (nodes, edges),
    )
    return lake_folder


def write_network(project_id: str, *, dry_run: bool) -> dict:
    runner = LocalRunner.from_definitions(
        {"reimagina-test-port": plugin.create_plugin()}
    )
    result = runner.run(
        "reimagina_street_network_write",
        {
            "parameters": {
                "city": "mexicali",
                "project_id": project_id,
                "dry_run": dry_run,
            }
        },
    )
    return result["reimagina_street_network_write"].iloc[0].to_dict()


def test_street_network_write(dsn, network_uri, offline_network):
    from reimagina_test_port.indicators.accessibility.network import read_network

    dry = write_network(PROJECT, dry_run=True)
    assert dry["previous_road_network_uri"] == network_uri
    assert dry["road_network_uri"].startswith(str(offline_network))
    assert (dry["nodes"], dry["edges"]) == (3, 3)
    assert not offline_network.exists()  # nothing uploaded
    assert scenario.read_road_network_uri(BASELINE) == network_uri

    written = write_network(PROJECT, dry_run=False)
    assert written["road_network_uri"] == dry["road_network_uri"]  # named by contents
    assert scenario.read_road_network_uri(FORK) == written["road_network_uri"]
    tables = read_network(Path(written["road_network_uri"]))
    assert len(tables["nodes"]) == 3
    # Again: the same file is kept, the project already points at it.
    again = write_network(PROJECT, dry_run=False)
    assert again["previous_road_network_uri"] == written["road_network_uri"]


def test_street_network_write_unknown_project(dsn, offline_network):
    with pytest.raises(GraphExecutionError) as failure:
        write_network(str(uuid4()), dry_run=True)
    assert isinstance(failure.value.__cause__, MetricInputError)
