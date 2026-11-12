"""``street_fit`` and the refined ``streets.build`` on real, small maps of Condesa.

``tests/data/*.gpkg`` are cut from reimagina_pipelines_lab's ``experiments/junctions/data``
(EPSG:32611; layers ``roads``, as ``street_fit.ROAD_COLUMNS`` and ``pedestrian`` ones, and
``lots``, geometry only):

- ``test_area``: 600 m around Calle Casanova × Av. Escobar (one-way pairs, links, a
  diagonal), residential and tertiary streets.
- ``lazaro_cardenas``: 800 m of Boulevard Lázaro Cárdenas, a trunk with a median and its
  frontage roads, service roads and footways.

Unlike the synthetic networks of test_baseline, these have the mess of a real city. The
checks are invariants (bands share 100, keys are unique, every vehicular edge is in a right
of way), not exact geometries, so they hold while ``street_fit`` is tuned.
"""

import json
from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

from reimagina_test_port.ingestion import street_fit, streets
from reimagina_test_port.ingestion.cities import CITIES

DATA = Path(__file__).parent / "data"
AREAS = ["test_area", "lazaro_cardenas"]


def load(area: str) -> tuple[gpd.GeoDataFrame, gpd.GeoSeries]:
    path = DATA / f"{area}.gpkg"
    roads = gpd.read_file(path, layer="roads")
    lots = gpd.read_file(path, layer="lots")
    for column in ("road_key", "from_node", "to_node", "road_type", "name"):
        roads[column] = roads[column].astype(object)
    return roads, lots.geometry


def network_of(roads: gpd.GeoDataFrame) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Nodes and edges as ``reimagina_street_network`` returns them (EPSG:4326, each way
    both ways, OSM's highway, name and lanes) from the area's roads, whose ids are uuids
    and whose lines start at their ``from_node``."""
    xy: dict = {}
    for line, a, b in zip(
        roads.geometry, roads["from_node"], roads["to_node"], strict=True
    ):
        xy.setdefault(a, line.coords[0])
        xy.setdefault(b, line.coords[-1])
    ids = {node: i for i, node in enumerate(xy, 1)}
    nodes = pd.DataFrame(
        {
            "id": list(ids.values()),
            "x": [c[0] for c in xy.values()],
            "y": [c[1] for c in xy.values()],
        }
    )
    rows = []
    for road, line in zip(roads.itertuples(), roads.geometry, strict=True):
        a, b = ids[road.from_node], ids[road.to_node]
        highway = "footway" if road.road_type == "pedestrian" else road.road_type
        lanes = str(road.lanes) if road.lanes_tagged else None
        name = road.name if isinstance(road.name, str) else None
        rows += [
            (a, b, highway, name, lanes, line),
            (b, a, highway, name, lanes, shapely.reverse(line)),
        ]
    edges = gpd.GeoDataFrame(
        {
            "from": [r[0] for r in rows],
            "to": [r[1] for r in rows],
            "distance": [r[5].length for r in rows],
            "two_way": True,
            "highway": [r[2] for r in rows],
            "name": [r[3] for r in rows],
            "lanes": [r[4] for r in rows],
        },
        geometry=[r[5] for r in rows],
        crs=roads.crs,
    ).to_crs(4326)
    return nodes, edges


def vehicular(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return roads[roads["road_type"] != "pedestrian"].reset_index(drop=True)


@pytest.fixture(params=AREAS)
def area(request) -> str:
    return request.param


def test_the_areas_are_small_but_real():
    for name, at_least in zip(AREAS, (200, 400), strict=True):
        roads, lots = load(name)
        assert len(roads) >= at_least and len(lots) > 300
        assert roads.crs.to_epsg() == 32611 == lots.crs.to_epsg()
        assert set(street_fit.ROAD_COLUMNS) <= set(roads.columns)


def test_fit_on_the_area(area):
    roads, lots = load(area)
    roads = vehicular(roads)
    out = street_fit.fit(roads, lots)
    street_fit.check(out)
    assert 0 < len(out) < len(roads)  # corridors join segments
    assert out["width_m"].between(3, 80).all()
    # Every segment is in a right of way, or the fit lost it.
    used = {k for keys in out["road_keys"] for k in keys}
    assert len(set(roads["road_key"]) - used) <= 0.02 * len(roads)
    assert (out["measured_sides"] >= 1).mean() > 0.5  # lots on at least one side


def test_the_boulevard_has_a_median():
    roads, lots = load("lazaro_cardenas")
    out = street_fit.fit(vehicular(roads), lots)
    dual = out[out["carriageways"] == 2]
    assert len(dual) > 0
    assert all("median" in use for use in dual["bands.use"])
    assert (dual["median_m"] > 0).all()


def test_build_refined_on_the_area(area, monkeypatch, tmp_path):
    roads, lots = load(area)
    nodes, edges = network_of(roads)
    monkeypatch.setattr(streets, "lot_shapes", lambda city, workdir: lots)
    city = CITIES["mexicali"]
    city = replace(
        city,
        streets=replace(city.streets, refine=True),
        street_lighting=None,
        bike_lanes=None,
    )
    table = streets.build(city, nodes, edges, tmp_path)

    assert table["id"].is_unique and table["source_record_ref"].is_unique
    assert table.crs.to_epsg() == 4326 and not table.geometry.is_empty.any()
    for use, share in zip(table["bands.use"], table["bands.share"], strict=True):
        use, share = json.loads(use), np.array(json.loads(share), dtype=float)
        assert len(use) == len(share) and abs(share.sum() - 100) < 1e-6
    fit = table.attrs["fit"]
    pedestrian = int((roads["road_type"] == "pedestrian").sum())
    assert fit["fitted"] > 0 and fit["fitted"] + fit["unfitted"] == len(table)
    assert fit["unfitted"] >= pedestrian * 0.9  # footways keep their own profile
    assert len(table) < len(roads)


def test_refined_builds_the_same_twice(monkeypatch, tmp_path):
    roads, lots = load("test_area")
    nodes, edges = network_of(roads)
    monkeypatch.setattr(streets, "lot_shapes", lambda city, workdir: lots)
    city = CITIES["mexicali"]
    city = replace(
        city,
        streets=replace(city.streets, refine=True),
        street_lighting=None,
        bike_lanes=None,
    )
    first = streets.build(city, nodes, edges, tmp_path)
    second = streets.build(city, nodes, edges, tmp_path)
    assert first["id"].tolist() == second["id"].tolist()
    assert first["bands.share"].tolist() == second["bands.share"].tolist()


def test_unnamed_streets_pair_whatever_the_name_dtype():
    """On pandas 3 a ``string`` name column kept NaN (truthy) through ``where(..., None)``,
    so an unnamed carriageway was never paired with a named one."""
    roads, lots = load("test_area")
    roads = vehicular(roads)
    assert roads["name"].isna().any() and roads["name"].notna().any()
    as_object = street_fit.fit(roads, lots)
    as_string = street_fit.fit(roads.assign(name=roads["name"].astype("string")), lots)
    assert len(as_object) == len(as_string)
    assert (as_object["carriageways"] == 2).sum() == (
        as_string["carriageways"] == 2
    ).sum()


# ------------------------------------------------------------------ footprints


def footprints_of(area: str):
    roads, lots = load(area)
    roads = vehicular(roads)
    fitted = street_fit.fit(roads, lots)
    return roads, lots, fitted, street_fit.footprints(fitted, roads)


def test_strip_length_and_width():
    straight = shapely.box(0, 0, 100, 12)
    curved = shapely.Point(0, 0).buffer(56).difference(shapely.Point(0, 0).buffer(44))
    curved = curved.intersection(shapely.box(0, 0, 100, 100))  # a quarter ring, 12 m
    two = shapely.MultiPolygon([straight, shapely.box(0, 50, 30, 62)])
    length = street_fit.strip_length(
        [straight, curved, two, shapely.LineString([(0, 0), (7, 0)])]
    )
    assert length == pytest.approx([100, np.pi / 2 * 50, 130, 7], rel=0.01)
    assert street_fit.strip_width([straight, curved, two]) == pytest.approx(
        [12, 12, 12], rel=0.01
    )


def test_footprints_are_single_valid_polygons(area):
    _, _, fitted, shapes = footprints_of(area)
    assert shapes.index.equals(fitted.index)
    assert (shapes.geom_type == "Polygon").all() and shapes.is_valid.all()
    assert not shapes.is_empty.any()


def overlaps(shapes: gpd.GeoSeries) -> list[float]:
    """Areas where two footprints overlap, neither inside a junction."""
    g, inside = shapes.to_numpy(), set(shapes.attrs["inside"])
    return [
        g[i].intersection(g[j]).area
        for i, j in zip(
            *shapely.STRtree(g).query(g, predicate="intersects"), strict=True
        )
        if i < j and i not in inside and j not in inside
    ]


@pytest.mark.parametrize(
    "name",
    [
        "test_area",
        pytest.param(
            "lazaro_cardenas",
            marks=pytest.mark.xfail(
                strict=True,
                reason="parallel streets sharing no node overlap (service roads up to 470 m²)",
            ),
        ),
    ],
)
def test_footprints_do_not_overlap(name):
    _, _, _, shapes = footprints_of(name)
    assert max(overlaps(shapes), default=0) < 1.0


def test_footprints_stay_off_the_lots(area):
    roads, lots, fitted, shapes = footprints_of(area)
    lot_shapes = shapely.make_valid(lots.to_numpy())
    # Lots a network axis runs through aren't street sides.
    crossed = shapely.STRtree(lot_shapes).query(
        roads.geometry.to_numpy(), predicate="crosses"
    )[1]
    sides = shapely.union_all(np.delete(lot_shapes, np.unique(crossed)))
    street = shapely.union_all(
        shapes[(fitted["road_class"] != "service").to_numpy()].to_numpy()
    )
    assert street.intersection(sides).area < 0.01 * street.area


@pytest.mark.xfail(
    strict=True,
    reason="W = A / L is within ±5 % on 25-40 % of the straight stretches (median error 9-12 %)",
)
def test_the_width_is_read_from_the_footprint(area):
    _, _, fitted, shapes = footprints_of(area)
    axes = fitted.geometry.to_numpy()
    chord = shapely.distance(shapely.get_point(axes, 0), shapely.get_point(axes, -1))
    straight = chord / shapely.length(axes) > 0.999
    straight &= ~np.isin(np.arange(len(shapes)), shapes.attrs["inside"])
    width = street_fit.strip_width(shapes.to_numpy())
    error = np.abs(width / fitted["width_m"].to_numpy(dtype=float) - 1)
    assert straight.sum() > 20
    assert (error[straight] < 0.05).all()


def test_unrefined_coverage_is_as_with_lines(monkeypatch, tmp_path):
    """The streets indicator weighs the polygons by strip_length as the lines by their
    length: within ±2 % on an unrefined network."""
    from reimagina_test_port.indicators.streets.calculation import coverage_by_block

    roads, _lots = load("test_area")
    nodes, edges = network_of(roads)
    city = replace(CITIES["mexicali"], street_lighting=None, bike_lanes=None)
    city = replace(city, streets=replace(city.streets, refine=False))
    table = streets.build(city, nodes, edges, tmp_path).set_index("source_record_ref")
    lines = streets.network_streets(nodes, edges, city.target_crs).reindex(table.index)
    rng = np.random.default_rng(0)
    flags = {
        "bike_lane": rng.random(len(table)) < 0.4,
        "protected_bike_lane": rng.random(len(table)) < 0.2,
        "streetlights": rng.choice([0.0, 50.0, 100.0], len(table)),
    }
    crs = roads.crs
    middle = shapely.union_all(lines.to_numpy()).centroid
    blocks = gpd.GeoDataFrame(
        geometry=[
            shapely.box(
                middle.x + dx, middle.y + dy, middle.x + dx + 40, middle.y + dy + 40
            )
            for dx in (-300, 0, 250)
            for dy in (-250, 0, 200)
        ],
        crs=crs,
    )
    with_lines = coverage_by_block(
        blocks, gpd.GeoDataFrame(flags, geometry=lines.to_numpy(), crs=crs), crs
    )
    with_shapes = coverage_by_block(
        blocks,
        gpd.GeoDataFrame(flags, geometry=table.geometry.to_numpy(), crs=4326),
        crs,
    )
    assert set(table.geom_type) == {"Polygon"}
    pd.testing.assert_frame_equal(with_shapes, with_lines, atol=0.02, rtol=0)
