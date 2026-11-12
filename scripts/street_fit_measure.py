"""How well ``street_fit.footprints``' polygons give back their rights of way's width.

    uv run python scripts/street_fit_measure.py test_area lazaro_cardenas
    uv run python scripts/street_fit_measure.py path/to/city.gpkg

Each argument is a name under ``tests/data`` or a gpkg with layers ``roads`` and ``lots``. Footways are
dropped. Prints, per area: ``W = A / L`` (``strip_width``) against ``width_m`` (share within ±5 / ±10 %, median
error), the strips shorter than wide (the formula swaps their length and width), those left ``inside`` a
junction, overlaps >= 1 m² between footprints (sharing a node or not) and the share of the untrimmed strips'
area kept. The baseline for ``docs/right_of_way_perpendicular_trim_plan.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely

from reimagina_test_port.ingestion import street_fit

DATA = Path(__file__).parent.parent / "tests" / "data"


def load(arg: str) -> tuple[gpd.GeoDataFrame, gpd.GeoSeries]:
    path = Path(arg) if arg.endswith(".gpkg") else DATA / f"{arg}.gpkg"
    roads = gpd.read_file(path, layer="roads")
    lots = gpd.read_file(path, layer="lots")
    for column in ("road_key", "from_node", "to_node", "road_type", "name"):
        roads[column] = roads[column].astype(object)
    roads = roads[roads["road_type"] != "pedestrian"].reset_index(drop=True)
    return roads, lots.geometry


def measure(arg: str) -> None:
    roads, lots = load(arg)
    fitted = street_fit.fit(roads, lots)
    shapes = street_fit.footprints(fitted, roads)
    g = shapes.to_numpy()
    n = len(g)
    width = fitted["width_m"].to_numpy(dtype=float)
    inside = np.zeros(n, bool)
    inside[shapes.attrs["inside"]] = True
    area = shapely.area(g)
    got = street_fit.strip_width(g)
    error = got / width - 1
    short = area / width < width  # shorter than wide: length and width swap
    ok = ~inside
    print(f"\n== {arg}: {n} rights of way")
    print(
        f"  W within ±5 %:  {np.mean(np.abs(error) < 0.05):6.1%}   ±10 %: {np.mean(np.abs(error) < 0.10):6.1%}"
    )
    print(
        f"  W within ±5 %, not shorter than wide: {np.mean(np.abs(error[~short]) < 0.05):6.1%}"
    )
    print(f"  median error of W: {np.median(error):+.1%}")
    print(
        f"  shorter than wide: {short.sum()} ({short.mean():.0%});  inside a junction: {inside.sum()} ({inside.mean():.0%})"
    )
    nodes_of = {
        k: {a, b}
        for k, a, b in zip(
            roads["road_key"], roads["from_node"], roads["to_node"], strict=True
        )
    }
    nodes = [
        set().union(*(nodes_of.get(k, set()) for k in keys))
        for keys in fitted["road_keys"]
    ]
    shared, apart = [], []
    for i, j in zip(*shapely.STRtree(g).query(g, predicate="intersects"), strict=True):
        if i >= j or not ok[i] or not ok[j]:
            continue
        a = shapely.intersection(g[i], g[j]).area
        if a >= 1.0:
            (shared if nodes[i] & nodes[j] else apart).append(a)
    print(
        f"  overlaps >= 1 m²:  sharing a node {len(shared)} ({sum(shared):,.0f} m²);  not {len(apart)} ({sum(apart):,.0f} m²)"
    )
    plain = np.array(
        [
            street_fit.ribbon(a, w)[0].area
            for a, w in zip(fitted.geometry, width, strict=True)
        ]
    )
    print(f"  area kept of the untrimmed strips: {area.sum() / plain.sum():.0%}")
    trims = np.array(shapes.attrs["trims"]).ravel()
    cut = trims[trims > 0]
    if len(cut):
        print(
            f"  ends trimmed: {len(cut)} of {len(trims)}; median {np.median(cut):.1f} m, p90 {np.percentile(cut, 90):.1f} m"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("areas", nargs="+")
    args = parser.parse_args()
    for area in args.areas:
        measure(area)
