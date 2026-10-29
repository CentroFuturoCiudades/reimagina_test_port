"""Accessibility over the synthetic snapshot: parity, inheritance and edge cases."""

import json
from pathlib import Path

import pandana as pdna
import pandas as pd
import pytest
from lyra.sdk import GraphExecutionError

from reimagina_test_port import accessibility, network, plugin
from reimagina_test_port.local import run_snapshot

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = json.loads((ROOT / "tests/expected_accessibility.json").read_text())


def inputs():
    built = run_snapshot("synthetic")
    tables = plugin.read_snapshot_accessibility("synthetic")
    return (
        built["reimagina_built_form_lots"],
        built["reimagina_built_form_blocks"],
        tables["reimagina_prepared_pois"],
        tables["reimagina_prepared_network_nodes"],
        tables["reimagina_prepared_network_edges"],
    )


def present(row: dict) -> dict:
    return {key: value for key, value in row.items() if not pd.isna(value)}


def test_parity_with_reimagina_lyra():
    results = run_snapshot("synthetic", "reimagina_accessibility")
    for kind in ("lots", "blocks"):
        table = results[f"reimagina_accessibility_{kind}"]
        assert table.id.tolist() == list(EXPECTED[kind])
        assert list(table.columns) == [
            *plugin.IDENTITY,
            "geometry",
            *plugin.ACCESSIBILITY_KEYS,
        ]
        for row in table.drop(columns=plugin.IDENTITY[1:] + ["geometry"]).to_dict(
            "records"
        ):
            values = present(row)
            expected = EXPECTED[kind][values.pop("id")]
            assert values.keys() == expected.keys()
            for key, value in expected.items():
                assert values[key] == (
                    value if isinstance(value, str) else pytest.approx(value)
                )


def test_lots_inherit_their_block():
    results = run_snapshot("synthetic", "reimagina_accessibility")
    blocks = results["reimagina_accessibility_blocks"].set_index("id")
    for row in results["reimagina_accessibility_lots"].to_dict("records"):
        values = present(row)
        if row["parent_id"] in blocks.index:
            block = present(blocks.loc[row["parent_id"]].to_dict())
            for key in plugin.ACCESSIBILITY_KEYS:
                assert values.get(key) == block.get(key)
        else:
            # Orphans and lots without a parent reach nothing.
            assert not set(values) & set(plugin.ACCESSIBILITY_KEYS)


def test_minutes_is_the_slowest_reachable_amenity():
    blocks = run_snapshot("synthetic", "reimagina_accessibility")[
        "reimagina_accessibility_blocks"
    ]
    for row in blocks.to_dict("records"):
        distances = {
            key.removeprefix("distance_to_"): row[key]
            for key in plugin.DISTANCE_KEYS
            if not pd.isna(row[key])
        }
        assert row["farthest_amenity"] == max(distances, key=distances.get)
        assert row["minutes"] == distances[row["farthest_amenity"]]
    # Unreachable or absent categories (and the unknown "casino") stay null.
    assert blocks["distance_to_hospital"].isna().all()
    scores = blocks[plugin.SCORE_KEYS].stack().dropna()
    assert scores.between(0, 1, inclusive="right").all()


def test_no_demand_or_no_pois_gives_nulls():
    lots, blocks, pois, nodes, edges = inputs()
    empty = blocks.assign(population=0)
    tables = plugin.accessibility_tables(lots, empty, pois, nodes, edges)
    assert (
        tables["reimagina_accessibility_blocks"][plugin.ACCESSIBILITY_KEYS]
        .isna()
        .all()
        .all()
    )
    tables = plugin.accessibility_tables(lots, blocks, pois.iloc[:0], nodes, edges)
    assert (
        tables["reimagina_accessibility_lots"][plugin.ACCESSIBILITY_KEYS]
        .isna()
        .all()
        .all()
    )


def test_network_in_another_crs_fails():
    lots, blocks, pois, nodes, edges = inputs()
    far = nodes.assign(x=nodes.x + 500_000)
    with pytest.raises(ValueError, match="ACCESS_CRS"):
        plugin.accessibility_tables(lots, blocks, pois, far, edges)


def test_missing_snapshot_inputs(tmp_path, monkeypatch):
    for kind in ("lot", "block"):
        source = ROOT / f"src/reimagina_test_port/data/synthetic/{kind}.parquet"
        (tmp_path / "custom").mkdir(exist_ok=True)
        (tmp_path / "custom" / f"{kind}.parquet").write_bytes(source.read_bytes())
    monkeypatch.setenv("REIMAGINA_SNAPSHOT_ROOT", str(tmp_path))
    with pytest.raises(GraphExecutionError) as failure:
        run_snapshot("custom", "reimagina_accessibility")
    assert failure.value.node == "reimagina_prepared_accessibility"


def test_read_network_round_trip(tmp_path):
    _, _, _, nodes, edges = inputs()
    source = tmp_path / "source" / "net.h5"
    source.parent.mkdir()
    accessibility.build_network(nodes, edges).save_hdf5(str(source))
    copied = network.download(str(source), tmp_path / "job")
    assert copied == tmp_path / "job" / "net.h5"
    tables = network.read_network(copied)
    pd.testing.assert_frame_equal(tables["nodes"], nodes, check_dtype=False)
    pd.testing.assert_frame_equal(tables["edges"], edges, check_dtype=False)
    rebuilt = accessibility.build_network(tables["nodes"], tables["edges"])
    assert isinstance(rebuilt, pdna.Network)
    assert len(rebuilt.edges_df) == len(edges)
