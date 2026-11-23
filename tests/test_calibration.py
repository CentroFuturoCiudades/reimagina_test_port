"""A project's calibration of the composite leaves (composites/calibration.py)."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from reimagina_test_port import plugin
from reimagina_test_port.indicators.composites import calibration
from reimagina_test_port.indicators.composites import composite_metric as nodes
from reimagina_test_port.indicators.composites.composite import (
    compress_outliers,
    compression_parameters,
    normalize_indicators,
    soft_threshold,
)

COUNT = {"name": "jobs", "source": "diversity", "variable": "reachable_jobs",
         "compress": True, "reference": 342.1, "goalposts": {"min": 0, "max": 2000}}  # fmt: skip
SHARE = {"name": "parks", "source": "population_access",
         "variable": "distance_m_public_space", "transform": {"soft_threshold": 300},
         "reference": 0.5, "goalposts": {"min": 0, "max": 1}}  # fmt: skip


@pytest.mark.parametrize(
    ("value", "expected"),
    [(133.4, 200), (313.8, 500), (1524.5, 2000), (1, 1), (0.3, 0.5), (0, None)],
)
def test_nice_ceiling(value, expected):
    assert calibration.nice_ceiling(value) == expected


def test_a_count_gets_reference_compress_and_goalposts():
    values = pd.Series([float(v) for v in range(1, 101)] + [5000.0, np.nan])
    result = calibration.leaf_calibration(COUNT, values)
    # The reference is the transformed average, before compressing.
    assert result["reference"] == pytest.approx(values.mean())
    assert result["compress"] == compression_parameters(values)
    top = compress_outliers(values.dropna(), **result["compress"]).quantile(0.99)
    assert result["goalposts"] == {"min": 0.0, "max": calibration.nice_ceiling(top)}


def test_a_share_gets_only_its_reference():
    distances = pd.Series([100.0, 300.0, 900.0])
    result = calibration.leaf_calibration(SHARE, distances)
    assert result == {"reference": pytest.approx(soft_threshold(distances, 300).mean())}


def test_a_leaf_without_values_keeps_the_yaml():
    assert calibration.leaf_calibration(SHARE, pd.Series([np.nan, np.nan])) == {}
    assert calibration.calibrated(SHARE, {}) is SHARE


def test_calibrated_sets_only_what_the_leaf_takes():
    values = {
        "reference": 0.3,
        "compress": {"mean": 1, "sd": 2, "minimum": 0, "maximum": 9},
        "goalposts": {"min": 0, "max": 5},
    }
    share = calibration.calibrated(SHARE, {"parks": values})
    assert share["reference"] == 0.3
    assert "compress" not in share and share["goalposts"] == SHARE["goalposts"]
    count = calibration.calibrated(COUNT, {"jobs": values})
    assert count["compress"] == {"mean": 1, "sd": 2, "minimum": 0, "maximum": 9}
    assert count["goalposts"] == {"min": 0, "max": 5}
    assert COUNT["goalposts"] == {"min": 0, "max": 2000}  # not modified in place
    mpi = calibration.calibrated({**SHARE, "method": "mpi"}, {"parks": values})
    assert mpi["reference"] == SHARE["reference"]


def test_the_table_round_trip():
    values = {
        "jobs": {
            "reference": 300.0,
            "compress": {"mean": 300.0, "sd": 50.0, "minimum": 0.0, "maximum": 900.0},
            "goalposts": {"min": 0.0, "max": 1000.0},
        },
        "parks": {"reference": 0.4},
    }
    table = calibration.to_table(values, "az://x/p.json")
    assert set(table["origin"]) == {"az://x/p.json"}
    assert calibration.from_table(table) == values
    assert calibration.from_table(calibration.to_table({}, "")) == {}
    document = calibration.document(values, "p", "s", "blocks")
    assert calibration.from_document(document) == values
    with pytest.raises(TypeError):
        calibration.from_document({"leaves": []})


def test_the_uri_is_fixed_per_project(monkeypatch):
    monkeypatch.delenv(calibration.PREFIX_ENV, raising=False)
    assert calibration.calibration_uri("p") == (
        "az://sandbox/projects/EXE/cleaned/composite_calibration/p.json"
    )
    monkeypatch.setenv(calibration.PREFIX_ENV, "/tmp/cal/")
    assert calibration.calibration_uri("p") == "/tmp/cal/p.json"


def blocks(values, column="reachable_jobs"):
    n = len(values)
    return gpd.GeoDataFrame(
        {"id": [f"b{i}" for i in range(n)], "parent_id": None, "source_record_ref": None,
         column: values},
        geometry=[box(i, 0, i + 1, 1) for i in range(n)],
        crs=4326,
    )  # fmt: skip


def test_the_baseline_average_scores_100():
    node = {"name": "jobs_index", "children": [COUNT]}
    baseline = blocks([100.0, 200.0, 300.0, 400.0, 20_000.0])
    values = calibration.calibrate([COUNT], {"diversity": baseline})
    table = nodes.composite_table(node, {"diversity": baseline}, calibration=values)
    reference = values["jobs"]["reference"]
    compressed = compress_outliers(
        baseline["reachable_jobs"], **values["jobs"]["compress"]
    )
    expected = 100 + 60 * (compressed - reference) / values["jobs"]["goalposts"]["max"]
    assert table["jobs"].tolist() == pytest.approx(expected.tolist())
    # A fork with an extreme block: the others keep their score.
    fork = baseline.copy()
    fork.loc[0, "reachable_jobs"] = 1_000_000.0
    again = nodes.composite_table(node, {"diversity": fork}, calibration=values)
    assert again["jobs"].tolist()[1:] == table["jobs"].tolist()[1:]
    # Without a calibration, the YAML's.
    yaml = nodes.composite_table(node, {"diversity": baseline})
    assert yaml["jobs"].iloc[0] == pytest.approx(
        normalize_indicators(
            baseline["reachable_jobs"], **nodes.leaf_arguments(COUNT)
        ).iloc[0]
    )


def test_a_leaf_the_calibration_lacks_is_missing():
    node = {"name": "jobs_index", "min_valid": 1, "children": [COUNT, SHARE]}
    tables = {"diversity": blocks([100.0, 200.0]),
              "population_access": blocks([100.0, 900.0], "distance_m_public_space")}  # fmt: skip
    values = calibration.calibrate([COUNT], tables)
    table = nodes.composite_table(node, tables, calibration=values)
    assert table["jobs"].notna().all() and table["parks"].isna().all()
    assert table["n"].tolist() == [1, 1]
    # Without a calibration (the snapshot graph), the YAML's.
    assert nodes.composite_table(node, tables)["parks"].notna().all()


def test_a_boundary_bounds_the_calibration():
    baseline = blocks([100.0, 200.0, 10_000.0])
    area = gpd.GeoDataFrame(geometry=[box(0, 0, 2, 1)], crs=4326)
    values = calibration.calibrate([COUNT], {"diversity": baseline}, area)
    assert values["jobs"]["reference"] == pytest.approx(150)


def test_the_scenario_graph_wires_the_calibration():
    manifests = {m.name: m for m in plugin.create_plugin().node_manifests}
    _, composites, _ = nodes.check_tree(nodes.load_config())
    calibrate = manifests[nodes.CALIBRATION].inputs
    sources = {leaf["source"] for leaf in nodes.tree_leaves(composites)}
    assert set(calibrate) == sources
    for composite in composites:
        inputs = manifests[f"reimagina_composite_{composite['name']}"].inputs
        assert inputs["calibration"].output == nodes.CALIBRATION
    save = manifests["reimagina_scenario_composite_calibration"].inputs
    assert save["calibration"].output == nodes.CALIBRATION


def test_the_snapshot_graph_keeps_the_yaml():
    manifests = {m.name: m for m in plugin.create_snapshot_plugin().node_manifests}
    assert nodes.CALIBRATION not in manifests
    assert "calibration" not in manifests["reimagina_composite_index"].inputs


def test_calibration_is_a_reserved_name():
    config = nodes.load_config()
    config["trees"][0]["children"][0]["name"] = "calibration"
    with pytest.raises(nodes.CompositeConfigError, match="name reserved"):
        nodes.check_tree(config)
