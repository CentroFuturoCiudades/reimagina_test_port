"""Scenario datasets and the indicator writer against PostGIS (minimal EAV schema).

REIMAGINA_TEST_DSN points to a server where the user may create databases; each
test creates and drops its own. See docs/verification.md for the container.
"""

import importlib.util
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
from shapely.geometry import box

from reimagina_test_port import (
    accessibility,
    baseline_writer,
    contract,
    plugin,
    scenario,
)

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
CREATE TABLE job_run (id uuid PRIMARY KEY);
CREATE TABLE indicator_catalog (
    id serial PRIMARY KEY,
    element_type element_type NOT NULL,
    key text NOT NULL,
    value_type attribute_value_type NOT NULL,
    status indicator_catalog_status NOT NULL DEFAULT 'active',
    UNIQUE (element_type, key)
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
INSERT INTO attribute_catalog (element_type, key, value_type) VALUES
    ('lot', 'land_use', 'enum'), ('lot', 'cos', 'number'),
    ('lot', 'num_levels', 'integer'), ('lot', 'area', 'number'),
    ('lot', 'unit_size_m2', 'number'), ('lot', 'people_per_household', 'number'),
    ('lot', 'occupancy_rate', 'number'), ('lot', 'amenity_type', 'enum'),
    ('lot', 'commerce_type', 'text'), ('block', 'vivtot', 'number'),
    ('block', 'people_per_household', 'number');
UPDATE attribute_catalog SET enum_values = ARRAY[
    'single_family', 'multifamily', 'mixed_use', 'amenity', 'commerce', 'office'
] WHERE key = 'land_use';
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
        "INSERT INTO elements VALUES (%s, %s, 'lot', NULL, NULL, NULL)", (NEW, BLOCK_2)
    )
    add_state(conn, FORK, NEW)
    set_attrs(conn, FORK, NEW, SHOP_ATTRS)


def seed_catalog(conn) -> None:
    keys = [
        *plugin.UNITS,
        "minutes",
        *plugin.DISTANCE_KEYS,
        *plugin.SCORE_KEYS,
    ]
    for element_type in ("lot", "block"):
        for key in keys:
            conn.execute(
                "INSERT INTO reimagina_urbano.indicator_catalog "
                "(element_type, key, value_type) VALUES (%s, %s, 'number')",
                (element_type, key),
            )
        conn.execute(
            "INSERT INTO reimagina_urbano.indicator_catalog "
            "(element_type, key, value_type) VALUES (%s, 'farthest_amenity', 'text')",
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
    assert values[(BLOCK, "farthest_amenity")][1] in {"clinic", "pharmacy"}
    assert values[(HOUSE, "distance_to_clinic")][0] > 0
    # No POI: never written.
    assert (HOUSE, "distance_to_park") not in values
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
    # Back in BLOCK, the house repeats that block's fork accessibility.
    values = stored(dsn, FORK)
    access = {k for k in plugin.ACCESSIBILITY_KEYS}
    house = {k: v[:2] for (e, k), v in values.items() if e == HOUSE and k in access}
    block = {k: v[:2] for (e, k), v in values.items() if e == BLOCK and k in access}
    assert house == block


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


def test_write_checks_the_indicator_catalog(dsn):
    with psycopg.connect(dsn) as conn:
        conn.execute(
            "UPDATE reimagina_urbano.indicator_catalog SET status = 'deprecated' "
            "WHERE key = 'cus' AND element_type = 'block'"
        )
        conn.execute(
            "UPDATE reimagina_urbano.indicator_catalog SET value_type = 'number' "
            "WHERE key = 'farthest_amenity' AND element_type = 'lot'"
        )
    # Built-form and accessibility are checked and written separately.
    with pytest.raises(GraphExecutionError) as failure:
        write(BASELINE)
    message = str(failure.value.__cause__)
    assert "missing block.cus (number)" in message
    with pytest.raises(GraphExecutionError) as failure:
        run_metric("reimagina_scenario_accessibility", BASELINE)
    message = str(failure.value.__cause__)
    assert "lot.farthest_amenity is number, expected text" in message
    assert stored(dsn, BASELINE) == {}


# ------------------------------------------------------------- baseline writer

NEW_BASELINE = "40000000-0000-0000-0000-000000000000"
NEW_FORK = "50000000-0000-0000-0000-000000000000"
NEW_LOTS = [f"c0000000-0000-0000-0000-00000000000{i}" for i in range(1, 5)]


def lot_box(i: int, shift: float = 0.0):
    x = i * 0.002 + shift
    return box(x, 0.0, x + 0.001, 0.001)


def new_tables(count: int = 4):
    """Blocks reuse BLOCK and BLOCK_2; four lots with new ids."""
    blocks = gpd.GeoDataFrame(
        {
            "id": [BLOCK, BLOCK_2],
            "parent_id": [None, None],
            "source_record_ref": ["cvegeo:1", "cvegeo:2"],
            "vivtot": [10.0, None],
            "people_per_household": [3.0, None],
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
            "unit_size_m2": [None, None, 80.0, None][:count],
            "people_per_household": [3.5, None, 3.0, None][:count],
            "occupancy_rate": [0.9, None, None, None][:count],
            "amenity_type": [None, None, "clinic", None][:count],
            "commerce_type": [None, "shop", None, None][:count],
        },
        geometry=[lot_box(i, 0.00002) for i in range(count)],
        crs=4326,
    )
    return {"lot": lots, "block": blocks}


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
        PROJECT, NEW_BASELINE, new_tables(count), plugin.BASELINE_ATTRIBUTES, **options
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


def test_write_new_baseline(new_baseline, dsn):
    old = counts(dsn, BASELINE)
    dry = write_new_baseline(dry_run=True)
    assert dry["elements_created"] == 4
    assert counts(dsn, NEW_BASELINE) == (0, 0)
    first = write_new_baseline()
    # The blocks already exist: reused, not created.
    assert first == {
        "elements": 6,
        "elements_created": 4,
        "attribute_values": 21,
        "elements_removed": 0,
    }
    assert counts(dsn, NEW_BASELINE) == (6, 21)
    assert counts(dsn, BASELINE) == old
    tables = plugin.read_scenario(NEW_BASELINE)
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
    assert origin == NEW_BASELINE
    # Indicators then come from the scenario writer.
    assert write(NEW_BASELINE)["values_written"] > 0
    # Again without one lot: replaced, and its indicators go too.
    second = write_new_baseline(count=3)
    assert second["elements_removed"] == 1
    assert counts(dsn, NEW_BASELINE)[0] == 5
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
            PROJECT, FORK, new_tables(), plugin.BASELINE_ATTRIBUTES
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
    fork = plugin.read_scenario(NEW_FORK)
    lots = by_id(fork["reimagina_scenario_lots"])
    # The shop's edit (now a house) lands on the new lot; GONE stays deleted.
    assert lots[NEW_LOTS[1]]["land_use"] == "single_family"
    assert NEW_LOTS[3] not in lots
    assert lots[NEW]["land_use"] == "commerce"
    # The house was moved to BLOCK_2 in the fork; its parent carries over.
    assert lots[NEW_LOTS[0]]["parent_id"] == BLOCK_2
    with pytest.raises(ValueError, match="already has edits"):
        module.migrate(dsn, FORK, NEW_FORK, apply=False)


def test_baseline_write_metric(new_baseline, dsn, monkeypatch):
    from reimagina_test_port import baseline

    tables = new_tables()
    monkeypatch.setattr(
        baseline,
        "build",
        lambda *_: {"lots": tables["lot"], "blocks": tables["block"], "report": {}},
    )
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
    assert (row["elements"], row["elements_created"], row["dry_run"]) == (6, 4, False)
