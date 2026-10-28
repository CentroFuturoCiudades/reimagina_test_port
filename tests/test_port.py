"""Reference parity, acquisition, and actual SDK graph behavior."""

import json
from pathlib import Path
from unittest.mock import Mock

import geopandas as gpd
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from lyra.sdk import GraphExecutionError, LocalRunner
from lyra.sdk.errors import MetricInputError

from reimagina_test_port import plugin
from reimagina_test_port.local import run_snapshot

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = json.loads((ROOT / "tests/expected.json").read_text())


def test_reference_parity_and_shared_acquisition(monkeypatch):
    reader = Mock(wraps=plugin.read_snapshot)
    calculation = Mock(wraps=plugin.lot_indicators)
    monkeypatch.setattr(plugin, "read_snapshot", reader)
    monkeypatch.setattr(plugin, "lot_indicators", calculation)
    runner = LocalRunner.from_definitions(
        {"reimagina-test-port": plugin.create_plugin()}
    )
    for invocation in range(2):
        results = runner.run(
            "reimagina_built_form", {"parameters": {"snapshot": "synthetic"}}
        )
        assert reader.call_count == invocation + 1
        assert calculation.call_count == (invocation + 1) * len(EXPECTED["lots"])
        for kind in ("lots", "blocks"):
            table = results[f"reimagina_built_form_{kind}"]
            assert table.id.tolist() == list(EXPECTED[kind])
            for row in table.to_dict("records"):
                for key, expected in EXPECTED[kind][row["id"]].items():
                    if expected is None:
                        assert pd.isna(row[key])
                    else:
                        assert row[key] == pytest.approx(expected)
    standalone = run_snapshot("synthetic", "reimagina_lot_built_form")
    assert_geodataframe_equal(
        standalone["reimagina_lot_indicators"], results["reimagina_built_form_lots"]
    )


def test_handlers_preserve_inputs():
    source = plugin.read_snapshot("synthetic")
    copies = {key: value.copy(deep=True) for key, value in source.items()}
    parameters = plugin.SnapshotParameters(snapshot="synthetic")
    lots = plugin.lot_built_form(parameters, source["reimagina_prepared_lots"])
    before = lots.copy(deep=True)
    plugin.built_form(parameters, lots, source["reimagina_prepared_blocks"])
    assert_geodataframe_equal(lots, before)
    for key in source:
        assert_geodataframe_equal(source[key], copies[key])


def snapshot(tmp_path, monkeypatch):
    folder = tmp_path / "custom"
    folder.mkdir()
    for kind in ("lot", "block"):
        frame = gpd.read_parquet(
            ROOT / f"src/reimagina_test_port/data/synthetic/{kind}.parquet"
        )
        frame.to_parquet(folder / f"{kind}.parquet")
    monkeypatch.setenv("REIMAGINA_SNAPSHOT_ROOT", str(tmp_path))
    return folder


@pytest.mark.parametrize(
    "problem",
    [
        "missing_file",
        "duplicate_id",
        "null_id",
        "empty_id",
        "missing_id",
        "missing_parent",
        "missing_crs",
        "corrupt",
    ],
)
def test_invalid_snapshots(tmp_path, monkeypatch, problem):
    folder = snapshot(tmp_path, monkeypatch)
    path = folder / "lot.parquet"
    frame = gpd.read_parquet(path)
    if problem == "missing_file":
        path.unlink()
    elif problem == "corrupt":
        path.write_bytes(b"invalid parquet")
    else:
        if problem == "duplicate_id":
            frame.loc[1, "id"] = frame.loc[0, "id"]
        elif problem == "null_id":
            frame.loc[0, "id"] = None
        elif problem == "empty_id":
            frame.loc[0, "id"] = ""
        elif problem == "missing_id":
            frame = frame.drop(columns="id")
        elif problem == "missing_parent":
            frame = frame.drop(columns="parent_id")
        else:
            frame = frame.set_crs(None, allow_override=True)
        frame.to_parquet(path)
    with pytest.raises(GraphExecutionError) as failure:
        run_snapshot("custom")
    assert isinstance(failure.value.__cause__, MetricInputError)
    assert failure.value.node == "reimagina_prepared_snapshot"
    assert failure.value.path[0] == "reimagina_built_form"


def test_empty_and_absent_attributes(tmp_path, monkeypatch):
    folder = snapshot(tmp_path, monkeypatch)
    path = folder / "lot.parquet"
    frame = gpd.read_parquet(path).drop(columns=plugin.ATTRIBUTES)
    frame.to_parquet(path)
    results = run_snapshot("custom")
    assert results["reimagina_built_form_lots"].population.isna().all()
    frame.iloc[:0].to_parquet(path)
    results = run_snapshot("custom")
    assert results["reimagina_built_form_lots"].empty
    assert results["reimagina_built_form_blocks"][list(plugin.UNITS)].isna().all().all()
    gpd.read_parquet(folder / "block.parquet").iloc[:0].to_parquet(
        folder / "block.parquet"
    )
    assert all(table.empty for table in run_snapshot("custom").values())


@pytest.mark.parametrize("name", ["", "../synthetic", "/tmp", "a/b"])
def test_snapshot_selector(name):
    with pytest.raises(ValueError):
        plugin.SnapshotParameters(snapshot=name)


def test_manifest_freshness():
    definition = plugin.create_plugin()
    from lyra.sdk.models.plugin import PluginInfo

    manifest = definition.manifest(
        plugin=PluginInfo(name="reimagina-test-port", version="0.1.0"),
        factory="reimagina_test_port.plugin:create_plugin",
    )
    assert json.loads((ROOT / "lyra.plugin.json").read_text()) == manifest.model_dump(
        mode="json"
    )
    assert (ROOT / "lyra.plugin.json").read_bytes() == (
        ROOT / "data/share/lyra/plugins/reimagina-test-port/lyra.plugin.json"
    ).read_bytes()
