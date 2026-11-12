"""The street network step: OpenStreetMap's graph as tables, a file of streets
sewn into it, and the H5 the indicators read back."""

import dataclasses
import hashlib
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import shapely
from lyra.sdk import LocalRunner
from shapely.geometry import LineString, box

from reimagina_test_port import plugin
from reimagina_test_port.common import lake
from reimagina_test_port.indicators.accessibility.network import read_network
from reimagina_test_port.ingestion import network
from reimagina_test_port.ingestion.cities import CITIES, NetworkParams

CRS = "EPSG:32611"
PARAMS = NetworkParams(output_prefix="unused")


def grid(size: int = 5, step: float = 100.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A ``size`` x ``size`` grid of nodes ``step`` m apart, both directions of each
    street (as osmnx's walk network), node ids 1..size**2."""
    ids = np.arange(size * size)
    nodes = pd.DataFrame(
        {"x": (ids % size) * step, "y": (ids // size) * step},
        index=pd.Index(ids + 1, name="id"),
    )
    right = ids[(ids % size) < size - 1]
    up = ids[ids < size * (size - 1)]
    a, b = np.concatenate([right, up]) + 1, np.concatenate([right + 1, up + size]) + 1
    edges = pd.DataFrame(
        {"from": np.concatenate([a, b]), "to": np.concatenate([b, a]), "length": step}
    )
    return nodes, edges


def node_at(nodes: pd.DataFrame, x: float, y: float) -> int:
    hit = nodes.index[(nodes["x"] - x).abs().lt(0.01) & (nodes["y"] - y).abs().lt(0.01)]
    assert len(hit) == 1, (x, y, list(hit))
    return int(hit[0])


def walk(nodes: pd.DataFrame, edges: pd.DataFrame, a: tuple, b: tuple) -> float:
    graph = nx.Graph()
    for row in edges.itertuples(index=False):
        graph.add_edge(row[0], row[1], length=row[2])
    return nx.shortest_path_length(
        graph, node_at(nodes, *a), node_at(nodes, *b), weight="length"
    )


def pieces(nodes: pd.DataFrame, edges: pd.DataFrame) -> int:
    graph = nx.Graph(list(zip(edges["from"], edges["to"], strict=True)))
    graph.add_nodes_from(nodes.index)
    return nx.number_connected_components(graph)


# ------------------------------------------------------------------ step 1


def osm_like_graph() -> nx.MultiDiGraph:
    """Three nodes in EPSG:4326, a two-way street and a parallel longer edge."""
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    for node, (x, y) in {
        10: (-115.45, 32.62),
        20: (-115.449, 32.62),
        30: (-115.449, 32.621),
    }.items():
        graph.add_node(node, x=x, y=y)
    for u, v, length in [
        (10, 20, 94.0),
        (20, 10, 94.0),
        (20, 30, 111.0),
        (20, 30, 150.0),
        (30, 30, 5.0),
    ]:
        graph.add_edge(u, v, length=length)
    # OSM tags: a list where osmnx joined several ways.
    graph.edges[10, 20, 0].update(highway="residential", name="Calle Uno", lanes="2")
    graph.edges[20, 30, 0].update(highway=["primary", "secondary"], lanes=["4", "2"])
    return graph


def test_graph_tables():
    nodes, edges = network.graph_tables(osm_like_graph(), CRS)
    assert list(nodes.index) == [10, 20, 30]
    expected = gpd.GeoSeries.from_xy([-115.45], [32.62], crs=4326).to_crs(CRS)
    assert nodes.loc[10, "x"] == pytest.approx(expected.x[0])
    # The parallel edge keeps the shortest; the loop is dropped.
    # OSM's highway, name and lanes ride along as text: a list gives its first,
    # a missing tag is null.
    assert edges[["from", "to", "length"]].to_dict("records") == [
        {"from": 10, "to": 20, "length": 94.0},
        {"from": 20, "to": 10, "length": 94.0},
        {"from": 20, "to": 30, "length": 111.0},
    ]
    assert edges["highway"].tolist()[0::2] == ["residential", "primary"]
    assert edges["lanes"].tolist()[0::2] == ["2", "4"]
    assert edges["name"].tolist()[0] == "Calle Uno"
    assert edges[["highway", "name", "lanes"]].iloc[1].isna().all()
    assert pd.isna(edges["name"].iloc[2])
    # Each edge keeps its shape, in the same CRS as the nodes.
    start = edges.loc[0, "geometry"].coords[0]
    assert start == pytest.approx((nodes.loc[10, "x"], nodes.loc[10, "y"]))


def test_osm_download_is_cached_in_the_lake(monkeypatch, tmp_path):
    calls = []

    def download(area, params):
        calls.append(area)
        return osm_like_graph()

    monkeypatch.setattr(network, "osm_graph", download)
    params = dataclasses.replace(PARAMS, output_prefix=str(tmp_path / "lake"))
    blocks = gpd.GeoSeries([box(-115.451, 32.619, -115.448, 32.622)], crs=4326)

    def build(**options):
        return network.osm_network(
            blocks, CRS, params, cve_code="02002", workdir=tmp_path / "w", **options
        )

    first = build()
    files = sorted(p.name for p in (tmp_path / "lake").iterdir())
    assert len(files) == 2 and files[0].startswith("osm_02002_")
    # The next run reads the lake's copy, with the same tables and shapes.
    second = build()
    assert len(calls) == 1
    pd.testing.assert_frame_equal(second[0], first[0])
    pd.testing.assert_frame_equal(
        second[1].drop(columns="geometry"), first[1].drop(columns="geometry")
    )
    assert all(
        a.equals_exact(b, 1e-6)
        for a, b in zip(second[1]["geometry"], first[1]["geometry"], strict=True)
    )
    # refresh downloads again and replaces the copy; another filter is another file.
    build(refresh=True)
    assert len(calls) == 2
    network.osm_network(
        blocks,
        CRS,
        dataclasses.replace(params, custom_filter='["highway"]'),
        cve_code="02002",
        workdir=tmp_path / "w",
    )
    assert len(calls) == 3
    assert len(list((tmp_path / "lake").iterdir())) == 4


def test_an_old_cache_is_not_read():
    # The hash carries the tags' version: a cache written without them is another file.
    params = dataclasses.replace(PARAMS, output_prefix="lake")
    area = box(-115.451, 32.619, -115.448, 32.622)
    (_, edges) = network.osm_cache_uris("lake", "02002", area, params)
    digest = hashlib.sha256()
    for part in (
        shapely.to_wkb(area),
        str(params.custom_filter).encode(),
        str(params.retain_all).encode(),
    ):
        digest.update(part)
    assert f"osm_02002_{digest.hexdigest()[:10]}_edges" not in edges


def test_without_cve_code_nothing_is_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(network, "osm_graph", lambda area, params: osm_like_graph())
    params = dataclasses.replace(PARAMS, output_prefix=str(tmp_path / "lake"))
    blocks = gpd.GeoSeries([box(-115.451, 32.619, -115.448, 32.622)], crs=4326)
    network.osm_network(blocks, CRS, params)
    assert not (tmp_path / "lake").exists()


def test_study_area_covers_blocks_and_margin():
    block = box(640000, 3610000, 640100, 3610100)
    blocks = gpd.GeoSeries([block], crs=CRS).to_crs(4326)
    area = gpd.GeoSeries([network.study_area(blocks, CRS, 200)], crs=4326).to_crs(CRS)
    assert area.iloc[0].contains(block.buffer(190))
    assert not area.iloc[0].contains(block.buffer(210))


def test_study_area_holds_the_roads_between_localities():
    # Two localities 5 km apart: the road between them is inside the area.
    near, far = (
        box(640000, 3610000, 640100, 3610100),
        box(645000, 3610000, 645100, 3610100),
    )
    blocks = gpd.GeoSeries([near, far], crs=CRS).to_crs(4326)
    area = gpd.GeoSeries([network.study_area(blocks, CRS, 200)], crs=4326).to_crs(CRS)
    assert area.iloc[0].contains(LineString([(640100, 3610050), (645000, 3610050)]))


def test_study_area_stops_at_the_boundary():
    block = box(640000, 3610000, 640100, 3610100)
    blocks = gpd.GeoSeries([block], crs=CRS).to_crs(4326)
    # A boundary that ends 50 m north of the block: the margin stops there.
    boundary = gpd.GeoSeries([box(639000, 3609000, 641000, 3610150)], crs=CRS)
    area = network.study_area(blocks, CRS, 200, boundary.to_crs(4326))
    _, _, _, top = gpd.GeoSeries([area], crs=4326).to_crs(CRS).total_bounds
    assert top == pytest.approx(3610150, abs=1)


# ------------------------------------------------------------------ step 2


def test_streets_replace_their_zone():
    nodes, edges = grid()
    # A cross through the middle: its hull covers the four streets at (200, 200).
    streets = gpd.GeoSeries(
        [LineString([(200, 100), (200, 300)]), LineString([(100, 200), (300, 200)])],
        crs=CRS,
    )
    merged_nodes, merged_edges, report = network.merge_streets(
        nodes, edges, streets, PARAMS
    )
    assert report["removed_edges"] == 8  # four streets, both directions
    assert report["new_pieces"] == 4  # noded where they cross
    assert report["new_nodes"] == 1  # the crossing; the ends reuse grid nodes
    assert node_at(merged_nodes, 200, 200) >= network.NODE_ID_FLOOR
    assert pieces(merged_nodes, merged_edges) == 1
    assert walk(merged_nodes, merged_edges, (200, 100), (200, 300)) == pytest.approx(
        200
    )


def test_new_streets_keep_their_shape():
    nodes, edges = grid()
    # A curved street across a block, corner to corner: its edge is the curve,
    # not the chord.
    curve = LineString([(100, 100), (120, 180), (200, 200)])
    streets = gpd.GeoSeries([curve], crs=CRS)
    params = dataclasses.replace(PARAMS, split_max_m=0, connector_max_m=0)
    merged_nodes, merged_edges, _ = network.merge_streets(
        nodes, edges, streets, params, zone=box(0, 0, 1, 1)
    )
    lines = network.shapes(merged_nodes, merged_edges)
    new = merged_edges.index[merged_edges["length"].eq(curve.length)]
    assert len(new) == 1
    assert lines[new[0]].equals_exact(curve, 0.01)
    # The grid's edges have no shape here: the straight line between their nodes.
    assert lines[0].length == pytest.approx(100)


def test_a_loose_end_splits_the_nearest_street():
    nodes, edges = grid()
    # Mid-block, 10 m short of both streets: both ends hang where they meet them.
    streets = gpd.GeoSeries([LineString([(150, 10), (150, 90)])], crs=CRS)
    merged_nodes, merged_edges, report = network.merge_streets(
        nodes, edges, streets, PARAMS
    )
    assert report["removed_edges"] == 0  # a line's hull has no area
    assert report["loose_ends_hung"] == 2
    assert pieces(merged_nodes, merged_edges) == 1
    # From the corner, 50 m along the street, 10 m to the new street's end.
    assert walk(merged_nodes, merged_edges, (100, 0), (150, 10)) == pytest.approx(60)


def test_a_piece_apart_is_bridged():
    nodes, edges = grid()
    params = dataclasses.replace(PARAMS, split_max_m=0, connector_max_m=5)
    streets = gpd.GeoSeries([LineString([(150, 103), (150, 140)])], crs=CRS)
    merged_nodes, merged_edges, report = network.merge_streets(
        nodes, edges, streets, params
    )
    assert report["loose_ends_hung"] == 0
    assert report["pieces_bridged"] == 1
    assert pieces(merged_nodes, merged_edges) == 1
    assert walk(merged_nodes, merged_edges, (100, 100), (150, 140)) == pytest.approx(
        50 + 3 + 37
    )


def test_a_piece_too_far_stays_apart(caplog):
    nodes, edges = grid()
    params = dataclasses.replace(PARAMS, split_max_m=0, connector_max_m=2)
    streets = gpd.GeoSeries([LineString([(150, 103), (150, 140)])], crs=CRS)
    merged_nodes, merged_edges, report = network.merge_streets(
        nodes, edges, streets, params
    )
    assert report["pieces_bridged"] == 0
    assert pieces(merged_nodes, merged_edges) == 2
    assert "not joined" in caplog.text


# ------------------------------------------------------------------ output


def test_h5_reads_back(tmp_path):
    nodes, edges = grid(3)
    path = network.save_h5(nodes, edges, tmp_path / "net.h5")
    tables = read_network(path)
    assert len(tables["nodes"]) == 9
    assert tables["edges"]["distance"].eq(100).all()
    assert tables["edges"]["two_way"].all()


def test_h5_name_follows_contents():
    nodes, edges = grid(3)
    name = network.h5_name("02002", nodes, edges)
    assert name.startswith("vialidades_02002_") and name.endswith(".h5")
    assert network.h5_name("02002", nodes, edges) == name
    assert network.h5_name("02002", nodes, edges.assign(length=99.0)) != name


def test_upload_never_overwrites(tmp_path):
    source = tmp_path / "a.h5"
    source.write_bytes(b"x")
    target = str(tmp_path / "lake" / "a.h5")
    assert lake.upload(source, target) == target
    assert lake.exists(target)
    with pytest.raises(FileExistsError):
        lake.upload(source, target)
    # Unless told to (the OSM cache's refresh).
    source.write_bytes(b"y")
    lake.upload(source, target, overwrite=True)
    assert Path(target).read_bytes() == b"y"


# ------------------------------------------------------------------ Lyra nodes


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """Mexicali with a local output folder, its blocks one square and OSM a grid."""
    city = dataclasses.replace(
        CITIES["mexicali"],
        streets=dataclasses.replace(
            CITIES["mexicali"].streets, output_prefix=str(tmp_path / "lake")
        ),
    )
    monkeypatch.setitem(CITIES, "mexicali", city)
    square = gpd.GeoSeries([box(0, 0, 400, 400)], crs=CRS).to_crs(4326)
    monkeypatch.setattr(
        lake,
        "read_parquet",
        lambda uri, columns: pd.DataFrame({"geometry": square.to_wkb()}),
    )
    monkeypatch.setattr(
        network, "osm_network", lambda blocks, crs, params, boundary, **_: grid()
    )
    return city


def test_street_network_dataset(offline, tmp_path):
    streets = tmp_path / "streets.parquet"
    gpd.GeoDataFrame(
        geometry=[
            LineString([(200, 100), (200, 300)]),
            LineString([(100, 200), (300, 200)]),
        ],
        crs=CRS,
    ).to_crs(4326).to_parquet(streets)
    runner = LocalRunner.from_definitions(
        {"reimagina-test-port": plugin.create_plugin()}
    )
    tables = runner.run(
        "reimagina_street_network_tables",
        {"parameters": {"city": "mexicali", "streets_uri": str(streets)}},
    )
    nodes = tables["reimagina_street_network_tables_nodes"]
    edges = tables["reimagina_street_network_tables_edges"]
    assert len(nodes) == 25  # the crossing is a new node, the old one went
    assert set(edges.columns) >= {"from", "to", "distance", "two_way", "geometry"}
    # Every edge has a shape: the grid's are straight lines between their nodes.
    assert edges.geometry.notna().all()
    assert edges["two_way"].all()
