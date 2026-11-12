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
    for line, a, b in zip(roads.geometry, roads["from_node"], roads["to_node"], strict=True):
        xy.setdefault(a, line.coords[0])
        xy.setdefault(b, line.coords[-1])
    ids = {node: i for i, node in enumerate(xy, 1)}
    nodes = pd.DataFrame(
        {"id": list(ids.values()), "x": [c[0] for c in xy.values()], "y": [c[1] for c in xy.values()]}
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
        city, streets=replace(city.streets, refine=True), street_lighting=None, bike_lanes=None
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
    assert (as_object["carriageways"] == 2).sum() == (as_string["carriageways"] == 2).sum()
