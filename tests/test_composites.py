"""Composite indicators: the normalisation math, the tree as Lyra nodes, and the
tree run over the synthetic snapshot's accessibility and built-form indicators."""

import copy

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from lyra.sdk import LocalRunner, PluginDefinition
from shapely.geometry import box

from reimagina_test_port import plugin, snapshot
from reimagina_test_port.indicators.accessibility.nodes import accessibility_metric
from reimagina_test_port.indicators.built_form.nodes import built_form_metrics
from reimagina_test_port.indicators.composites import composite_metric as nodes
from reimagina_test_port.indicators.composites.composite import (
    build_composite,
    normalize_indicators,
    soft_threshold,
    stepped_score,
)
from reimagina_test_port.local import run_snapshot


def test_normalisation():
    x = pd.Series([0.25, 0.5, 0.75])
    ampi = {"method": "ampi", "reference": 0.5, "goalposts": {"min": 0, "max": 1}}
    assert normalize_indicators(x, **ampi).tolist() == [85, 100, 115]
    assert normalize_indicators(x, polarity="negative", **ampi).tolist() == [
        115,
        100,
        85,
    ]
    mpi = normalize_indicators(x, method="mpi", mean=0.5, sd=0.25)
    assert mpi.tolist() == [90, 100, 110]
    # Closer is better: half at the threshold, unreachable (missing) is 0.
    near = soft_threshold(pd.Series([10.0, 0.0, np.nan]), 10)
    assert near.iloc[0] == 0.5 and near.iloc[1] > 0.99 and near.iloc[2] == 0
    steps = stepped_score(pd.Series([3.0, 8.0, 20.0]), [[5, 1.0], [10, 0.5]])
    assert steps.tolist() == [1.0, 0.5, 0.0]


def test_mazziotta_pareto():
    a, b = pd.Series([100.0, 100.0, np.nan]), pd.Series([120.0, 100.0, 110.0])
    out = build_composite([a, b])
    # mean 110, SD 10: penalty 10 × 10 / 110, taken off for a positive phenomenon.
    assert out["index"].iloc[0] == pytest.approx(110 - 100 / 110)
    assert out["index"].iloc[1] == 100  # balanced: no penalty
    assert np.isnan(out["index"].iloc[2])  # a child missing, all required
    assert build_composite([a, b], phenomenon="negative")["index"].iloc[0] == (
        pytest.approx(110 + 100 / 110)
    )
    assert build_composite([a, b], min_valid=1)["index"].iloc[2] == 110
    weighted = build_composite([a, b], weights=[3, 1])
    assert weighted["mean"].iloc[0] == pytest.approx(105)


def composite_manifests():
    return {
        m.name: m
        for m in plugin.create_snapshot_plugin().node_manifests
        if m.name.startswith("reimagina_composite_")
    }


def test_one_node_per_composite_wired_as_the_tree():
    manifests = composite_manifests()
    inputs = {
        name: {k: v.output for k, v in m.inputs.items()}
        for name, m in manifests.items()
    }
    # The bundled tree names no boundary: composites read every block.
    assert inputs == {
        "reimagina_composite_health": {
            "accessibility": "reimagina_accessibility_blocks"
        },
        "reimagina_composite_proximity": {
            "health": "reimagina_composite_health_blocks",
            "accessibility": "reimagina_accessibility_blocks",
        },
        "reimagina_composite_urban_form": {"built_form": "reimagina_built_form_blocks"},
        "reimagina_composite_index": {
            "proximity": "reimagina_composite_proximity_blocks",
            "urban_form": "reimagina_composite_urban_form_blocks",
        },
    }
    columns = [c.name for c in manifests["reimagina_composite_health"].outputs[
        "reimagina_composite_health_blocks"].columns]  # fmt: skip
    assert columns[3:] == ["clinic", "pharmacy", "index", "mean", "penalty", "n"]
    # The production graph has the same composites.
    production = {m.name for m in plugin.create_plugin().node_manifests}
    assert set(manifests) <= production


def test_the_tree_over_the_synthetic_indicators():
    index = run_snapshot("synthetic", "reimagina_composite_index")[
        "reimagina_composite_index_blocks"
    ].set_index("id")
    proximity = run_snapshot("synthetic", "reimagina_composite_proximity")[
        "reimagina_composite_proximity_blocks"
    ].set_index("id")
    health = run_snapshot("synthetic", "reimagina_composite_health")[
        "reimagina_composite_health_blocks"
    ].set_index("id")
    access = run_snapshot("synthetic", "reimagina_accessibility")[
        "reimagina_accessibility_blocks"
    ].set_index("id")
    # A leaf: the indicator, transformed and normalised as composite_config.yaml says.
    clinic = normalize_indicators(
        access["distance_to_clinic"].reindex(health.index),
        transform={"soft_threshold": 10},
        reference=0.5,
        goalposts={"min": 0, "max": 1},
    )
    assert health["clinic"].tolist() == pytest.approx(clinic.tolist())
    # A parent reads its child composite's index.
    assert proximity["health"].tolist() == pytest.approx(health["index"].tolist())
    expected = build_composite([index["proximity"], index["urban_form"]], min_valid=1)[
        "index"
    ]
    assert index["index"].tolist() == pytest.approx(expected.tolist())
    # A block without homes has no urban form; the root needs one child.
    empty = index.loc["block-empty"]
    assert np.isnan(empty["urban_form"]) and empty["n"] == 1
    assert empty["index"] == pytest.approx(empty["proximity"])
    assert index["index"].between(60, 140).all()


def tree():
    return copy.deepcopy(nodes.load_config())


def health(config):
    return config["trees"][0]["children"][0]["children"][0]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: c.update(element="streets"), "element must be one of"),
        (lambda c: health(c)["children"][0].update(source="osm"), "source must be one of"),
        (lambda c: health(c)["children"][0].update(variable="lighting_score"),
         "accessibility has no column 'lighting_score'"),
        (lambda c: health(c)["children"][0].pop("goalposts"), "ampi needs"),
        (lambda c: health(c)["children"][0].update(transform={"radius": 5}),
         "transform needs"),
        (lambda c: health(c)["children"][1].update(name="clinic"), "name repeated"),
        (lambda c: health(c)["children"][1].update(name="mean"), "name reserved"),
        (lambda c: health(c)["children"][0].update(polarty="negative"),
         r"unknown keys \['polarty'\]"),
        (lambda c: health(c).update(min_valid=3), "min_valid must be"),
        (lambda c: health(c).update(children=[]), "a composite needs children"),
        (lambda c: health(c).update(phenomenon="neutral"), "phenomenon must be"),
    ],
)  # fmt: skip
def test_config_errors_name_the_node(change, message):
    config = tree()
    change(config)
    with pytest.raises(nodes.CompositeConfigError, match=message):
        nodes.composite_metrics(snapshot.SnapshotParameters, config)


# ---------------------------------------------------------------- a DAG of composites

LEAF = {"source": "accessibility", "transform": {"soft_threshold": 15},
        "reference": 0.5, "goalposts": {"min": 0, "max": 1}}  # fmt: skip


def dag():
    """The bundled tree, and a second index reusing its health composite."""
    config = tree()
    config["trees"].append(
        {"name": "care", "children": [
            {"ref": "health", "weight": 2},
            {"name": "school", "variable": "distance_to_primary_school", **LEAF},
        ]}
    )  # fmt: skip
    return config


def test_a_composite_reused_by_two_indices():
    config = dag()
    _, order, _ = nodes.check_tree(config)
    names = [n["name"] for n in order]
    assert names.count("health") == 1  # computed once
    assert names.index("health") < names.index("care")
    assert names.index("health") < names.index("proximity")
    manifests = {
        m.name: m for m in PluginDefinition(
            nodes=nodes.composite_metrics(snapshot.SnapshotParameters, config)
        ).node_manifests
    }  # fmt: skip
    care = {
        k: v.output for k, v in manifests["reimagina_composite_care"].inputs.items()
    }
    assert care == {
        "health": "reimagina_composite_health_blocks",
        "accessibility": "reimagina_accessibility_blocks",
    }
    runner = snapshot_runner(config)
    params = {"parameters": {"snapshot": "synthetic"}}
    out = runner.run("reimagina_composite_care", params)[
        "reimagina_composite_care_blocks"
    ]
    health_index = runner.run("reimagina_composite_health", params)[
        "reimagina_composite_health_blocks"
    ].set_index("id")["index"]
    out = out.set_index("id")
    assert out["health"].tolist() == pytest.approx(
        health_index.reindex(out.index).tolist()
    )
    expected = build_composite([out["health"], out["school"]], weights=[2, 1])["index"]
    assert out["index"].tolist() == pytest.approx(expected.tolist())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: c["trees"][1]["children"][0].update(ref="welfare"),
         "no composite named 'welfare'"),
        (lambda c: c["trees"][1]["children"][0].update(phenomenon="negative"),
         "a ref only takes a weight"),
        (lambda c: c["trees"][1]["children"].append({"ref": "health"}), "a child appears twice"),
        (lambda c: c["trees"][1].update(weight=2), "a tree's root has no parent"),
        (lambda c: health(c)["children"].append({"ref": "care"}),
         "health > care > health: a cycle"),
        (lambda c: c.update(name="index"), r"unknown keys \['name'\]"),
        (lambda c: c.update(trees=[]), "trees must list"),
        (lambda c: c["trees"].append({"name": "lone", "source": "accessibility"}),
         "a tree's root must be a composite"),
    ],
)  # fmt: skip
def test_dag_errors(change, message):
    config = dag()
    change(config)
    with pytest.raises(nodes.CompositeConfigError, match=message):
        nodes.check_tree(config)


def test_errors_say_where():
    config = tree()
    health(config)["children"][0].update(source="osm")
    with pytest.raises(
        nodes.CompositeConfigError, match="index > proximity > health > clinic"
    ):
        nodes.check_tree(config)


# ---------------------------------------------------------------- boundary


def snapshot_runner(config):
    """The snapshot graph with the composites of ``config``."""
    definition = PluginDefinition(
        nodes=[
            snapshot.prepared_snapshot,
            snapshot.prepared_accessibility,
            *built_form_metrics(
                snapshot.SnapshotParameters, "reimagina_prepared_lots",
                "reimagina_prepared_blocks",
            ),
            accessibility_metric(
                snapshot.SnapshotParameters, "reimagina_prepared_pois",
                "reimagina_prepared_network_nodes", "reimagina_prepared_network_edges",
            ),
            *nodes.composite_metrics(snapshot.SnapshotParameters, config),
        ]
    )  # fmt: skip
    return LocalRunner.from_definitions({"composites": definition})


def test_a_boundary_keeps_the_blocks_inside_it(tmp_path):
    # Two polygons in UTM, the synthetic blocks' CRS: z and a inside, block-empty
    # (y >= 3,610,200) out. Merged and read with the file's own CRS.
    path = tmp_path / "city.parquet"
    gpd.GeoDataFrame(
        geometry=[
            box(639900, 3609900, 640500, 3610095),
            box(639900, 3610095, 640500, 3610195),
        ],
        crs=32611,
    ).to_parquet(path)
    config = tree() | {"boundary": str(path)}
    runner = snapshot_runner(config)
    index = runner.run(
        "reimagina_composite_index", {"parameters": {"snapshot": "synthetic"}}
    )
    assert sorted(index["reimagina_composite_index_blocks"]["id"]) == [
        "block-a",
        "block-z",
    ]
    manifests = {
        m.name: m for m in PluginDefinition(
            nodes=nodes.composite_metrics(snapshot.SnapshotParameters, config)
        ).node_manifests
    }  # fmt: skip
    for name, manifest in manifests.items():
        if name != nodes.BOUNDARY:
            assert manifest.inputs["boundary"].output == nodes.BOUNDARY
    boundary = nodes.read_boundary(str(path))
    assert len(boundary) == 1 and boundary.crs == "EPSG:4326"


def test_a_boundary_that_cannot_be_read(tmp_path):
    pd.DataFrame({"geometry": [box(0, 0, 1, 1).wkb]}).to_parquet(
        tmp_path / "plain.parquet"
    )
    for uri in (str(tmp_path / "plain.parquet"), str(tmp_path / "missing.parquet")):
        runner = snapshot_runner(tree() | {"boundary": uri})
        with pytest.raises(Exception, match="reimagina_composite_boundary"):
            runner.run(
                "reimagina_composite_index", {"parameters": {"snapshot": "synthetic"}}
            )
    with pytest.raises(nodes.CompositeConfigError, match="boundary must be"):
        nodes.check_tree(tree() | {"boundary": 3})


# ---------------------------------------------------------------- writing


def test_composites_written_as_one_value_per_element():
    from reimagina_test_port.scenario import nodes as scenario_nodes
    from reimagina_test_port.scenario.writer import long_values

    health = pd.DataFrame(
        {"id": ["a", "b"], "parent_id": None, "index": [101.5, 98.25]}
    )
    index = pd.DataFrame({"id": ["b", "c"], "parent_id": None, "index": [97.0, np.nan]})
    table = scenario_nodes.composite_values({"health": health, "index": index})
    assert list(table["id"]) == ["a", "b", "c"]
    assert table["composite_health"].tolist()[:2] == [101.5, 98.25]
    assert np.isnan(table["composite_health"].iloc[2])
    assert table["composite_index"].tolist()[1] == 97.0
    keys = {"composite_health": "number", "composite_index": "number"}
    values = long_values(table, "block", keys)
    # Every element and key, missing ones as None so stale values are cleared.
    assert len(values) == 6 and set(values["element_type"]) == {"block"}


def test_the_write_reads_every_composite():
    manifests = {m.name: m for m in plugin.create_plugin().node_manifests}
    write = manifests["reimagina_scenario_composites"].inputs
    assert {k: v.output for k, v in write.items()} == {
        f"composite_{name}": f"reimagina_composite_{name}_blocks"
        for name in ("health", "proximity", "urban_form", "index")
    }
    assert "composites" in manifests["reimagina_scenario_write"].inputs


def test_a_baseline_registers_the_composite_keys():
    from reimagina_test_port.ingestion import catalogs

    rows = {(row[0], row[1]): row for row in catalogs.indicator_rows()}
    for name in ("health", "proximity", "urban_form", "index"):
        *_, unit, value_type, job = rows[("block", f"composite_{name}")]
        assert (unit, value_type, job) == ("score", "number", "composites")
    assert ("lot", "composite_index") not in rows  # the tree's element only
