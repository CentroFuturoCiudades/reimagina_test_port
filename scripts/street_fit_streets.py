"""``street_fit``'s rights of way drawn as streets, as the lab draws them: sidewalks, lanes, median and junctions.

    uv run python scripts/street_fit_streets.py lazaro_cardenas                    # tests/data/<name>.gpkg
    uv run python scripts/street_fit_streets.py path/to/city.gpkg --out city.html
    uv run python scripts/street_fit_streets.py --roads roads.parquet --lots lot.parquet --out x.html
    ... --lab ../reimagina_pipelines_lab/output/condesa                             # compare with the lab

The input is a gpkg with layers ``roads`` and ``lots`` (see ``docs/street_fit_test_data.md``), or the two as
separate files (``--roads`` as ``streets.osm_roads`` writes them). Footways (``road_type == "pedestrian"``) are
dropped, as in an OSM ``drive`` network. ``--lab`` is a folder with the lab's ``right_of_way.parquet``: each
right of way is paired with the lab's one under its midpoint and the page compares their widths and sections.

The rights of way are ``street_fit.footprints``' polygons, as ``streets.build`` writes them, and their width is
read from the shape (``strip_width``), as one reading the base would: what is drawn is what is stored. The bands
(and lanes) are strips across the axis carried into its junctions, ``share`` of that width each, cut to the polygon.
Junctions are ``footprints``' own.

It also prints (and the page shows) how good the fit is, without the lab: how much of the rights of way lies on
lots, how much of the street space by the network they leave out, and how plausible their bands are.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon

from reimagina_test_port.ingestion.street_fit import (
    DEFAULTS,
    extend,
    fit,
    footprints,
    strip_width,
)

DATA = Path(__file__).parents[1] / "tests/data"
TEMPLATE = Path(__file__).with_suffix(".html")
KIND = {
    "sidewalk": 0,
    "lanes": 1,
    "median": 2,
    "shared_lanes": 3,
    "protected_bike_lane": 4,
}


# ------------------------------------------------------------------------------------------------ geometry
BROKEN: set = set()  # axes GEOS couldn't offset as they are (they cross themselves)


def one_side(axis, d: float):
    try:
        return axis.buffer(d, single_sided=True, cap_style="flat")
    except shapely.errors.GEOSException:
        BROKEN.add(axis.wkb)
        return shapely.simplify(axis, 0.5).buffer(
            d, single_sided=True, cap_style="flat"
        )


def strip(axis, outer: float, inner: float, side: int):
    far = one_side(axis, side * outer)
    if inner > 0:
        return far.difference(one_side(axis, side * inner))
    return far


def between(axis, a: float, b: float):
    """From ``a`` to ``b`` metres off the axis (left is positive, a < b)."""
    if a >= 0:
        return strip(axis, b, a, 1)
    if b <= 0:
        return strip(axis, -a, -b, -1)
    return shapely.union_all([strip(axis, b, 0, 1), strip(axis, -a, 0, -1)])


def areas_only(g):
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    parts = [
        p for p in getattr(g, "geoms", []) if p.geom_type in ("Polygon", "MultiPolygon")
    ]
    return shapely.union_all(parts) if parts else Polygon()


def streets(port: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> dict:
    """Rights of way, bands and lanes, junctions and lane marks, as polygons and lines."""
    half = DEFAULTS["default_half_width_m"]
    widest_at: dict = defaultdict(float)
    nodes_of_road = {}
    for key, a, b, kind in zip(
        roads["road_key"],
        roads["from_node"],
        roads["to_node"],
        roads["road_type"],
        strict=True,
    ):
        nodes_of_road[key] = (a, b)
        for n in (a, b):
            widest_at[n] = max(widest_at[n], half.get(kind, 4))

    shapes = footprints(port, roads)
    width = strip_width(shapes.to_numpy())
    rows, parts = [], []
    for i, r in port.iterrows():
        nodes = {n for k in r.road_keys for n in nodes_of_road[k]}
        reach = max((widest_at[n] for n in nodes), default=0)
        line = extend(r.geometry, reach) if reach else r.geometry
        shape = shapes.iloc[i]
        rows.append({"nodes": nodes, "shape": shape})
        w_total = float(width[i])
        top = w_total / 2
        per_cw = max(1, int(r.lanes) // int(r.carriageways))
        for b, (use, share) in enumerate(
            zip(r["bands.use"], r["bands.share"], strict=True)
        ):
            w = w_total * share / 100
            n = per_cw if use in ("lanes", "shared_lanes") else 1
            for k in range(n):
                strip_ = between(line, top - w / n * (k + 1), top - w / n * k)
                parts.append(
                    {
                        "row": i,
                        "kind": KIND[use],
                        "band": b,
                        "lane": k,
                        "width_m": round(w / n, 2),
                        "shape": areas_only(strip_.intersection(shape)),
                    }
                )
            top -= w
    whole = shapes.attrs["junctions"]
    junctions = []
    for g in shapely.get_parts(whole):
        if g.area <= 1.0:
            continue
        near = shapes.sindex.query(g, predicate="intersects")
        junctions.append({"shape": g, "streets": len(near)})
    extended = list(shapes)
    parts = [p for p in parts if p["shape"].area > 0.5]
    inside = set(shapes.attrs["inside"])

    # Lines between neighbouring lanes of one band (the lab's dashed ones).
    lanes_by = defaultdict(list)
    for p in parts:
        if p["kind"] in (1, 3):
            lanes_by[(p["row"], p["band"])].append(p)
    marks = []
    for group in lanes_by.values():
        group.sort(key=lambda p: p["lane"])
        for a, b in itertools.pairwise(group):
            edge = shapely.intersection(b["shape"].boundary, a["shape"].buffer(0.05))
            lines = [
                g
                for g in shapely.get_parts(edge)
                if g.geom_type == "LineString" and not g.is_empty
            ]
            if lines:
                merged = shapely.line_merge(shapely.MultiLineString(lines))
                marks += [g for g in shapely.get_parts(merged) if g.length >= 1.0]
    return {
        "rows": rows,
        "parts": parts,
        "junctions": junctions,
        "inside": inside,
        "marks": marks,
        "extended": extended,
        "whole": whole,
    }


# ------------------------------------------------------------------------------------------------- quality
def quality(
    port: gpd.GeoDataFrame, roads: gpd.GeoDataFrame, lots: gpd.GeoSeries, drawn: dict
) -> dict:
    """How good the fit is, without the lab.

    - on lots: share of the rights of way (axis ± half the width, not carried into junctions) that lies on a lot;
      should be ~0. Only lots no network axis runs through; service roads apart (they often run inside a lot).
    - street space left out: of the ground within ``probe`` m of a network axis that isn't a lot (surely street),
      the share no right of way or junction covers; holes and misplaced axes show here.
    - per right of way: lanes under 2.6 m, sidewalks under 1 m, a median band wider than all its lanes, wider than
      50 m, no side measured against lots, and a median between carriageways wider than the right of way.
    """
    probe = 4.0
    # Lots a network axis runs through aren't street sides (the cadastre has parcels over streets): as in fit_axis,
    # they don't count.
    shapes = shapely.make_valid(lots.to_numpy())
    crossed = shapely.STRtree(shapes).query(
        roads.geometry.to_numpy(), predicate="crosses"
    )[1]
    lot_union = shapely.union_all(np.delete(shapes, np.unique(crossed)))
    # Service roads (parking aisles, roads inside a lot) lie on lots by nature: measured apart.
    service = (port["road_class"] == "service").to_numpy()
    plain = [
        shapely.buffer(a, w / 2, cap_style="flat")
        for a, w in zip(port.geometry, port["width_m"], strict=True)
    ]

    def on(shapes) -> str:
        g = shapely.union_all(shapes)
        return (
            f"{100 * g.intersection(lot_union).area / g.area:.1f} %" if g.area else "—"
        )

    street = shapely.union_all([*drawn["extended"], drawn["whole"]])
    axes = roads.geometry[roads["road_type"] != "service"].to_numpy()
    near = shapely.union_all(shapely.buffer(axes, probe, cap_style="flat")).difference(
        lot_union
    )
    left_out = near.difference(street).area / near.area if near.area else 0.0
    by_class = {}
    for kind in port["road_class"].unique():
        rows = [g for g, k in zip(plain, port["road_class"], strict=True) if k == kind]
        by_class[f"  on lots: {kind} ({len(rows)})"] = on(rows)

    def band(r, use):
        return [
            r.width_m * s / 100
            for u, s in zip(r["bands.use"], r["bands.share"], strict=True)
            if u == use
        ]

    flags = defaultdict(int)
    for _, r in port.iterrows():
        lanes_w = sum(band(r, "lanes"))
        if r.lanes and lanes_w / r.lanes < 2.6:
            flags["lanes under 2.6 m"] += 1
        if any(w < 1.0 for w in band(r, "sidewalk")):
            flags["a sidewalk under 1 m"] += 1
        if sum(band(r, "median")) > lanes_w:
            flags["median wider than its lanes"] += 1
        if r.width_m > 50:
            flags["wider than 50 m"] += 1
        if r.measured_sides == 0:
            flags["no side measured"] += 1
        if r.carriageways == 2 and r.median_m > r.width_m:
            flags["carriageways farther apart than its width"] += 1
    n = len(port)
    self_crossing = int((~shapely.is_simple(port.geometry.to_numpy())).sum())
    if self_crossing:
        flags["axis crosses itself"] = self_crossing
    out = {
        "right of way on lots, not service": on(
            [g for g, s in zip(plain, service, strict=True) if not s]
        ),
        **dict(sorted(by_class.items())),
        f"street space ≤{probe:.0f} m from an axis left out (not service)": f"{100 * left_out:.1f} %",
        "both sides measured": f"{100 * (port['measured_sides'] == 2).mean():.0f} %",
    }
    out |= {
        k: f"{v} ({100 * v / n:.1f} %)"
        for k, v in sorted(flags.items(), key=lambda kv: -kv[1])
    }
    return out


# ---------------------------------------------------------------------------------------------------- page
def page(port, roads, lots, lab, drawn, report: dict) -> str:
    crs = port.crs
    mid = gpd.GeoDataFrame(
        geometry=port.geometry.interpolate(0.5, normalized=True), crs=crs
    )
    if lab is not None:
        hit = gpd.sjoin(mid, lab[["geometry"]], predicate="within")
        hit = hit[~hit.index.duplicated()]
        lab_of = hit["index_right"].reindex(port.index)
    else:
        lab = gpd.GeoDataFrame(
            columns=[
                "width_m",
                "width_left_m",
                "width_right_m",
                "sidewalk_left_m",
                "sidewalk_right_m",
                "lanes",
                "lane_width_m",
                "carriageways",
                "median_m",
                "measured_sides",
            ],
            geometry=[],
            crs=crs,
        )
        lab_of = pd.Series(np.nan, index=port.index)
    origin = gpd.GeoSeries([port.union_all().centroid], crs=crs).to_crs(4326).iloc[0]
    x0, y0 = round(origin.x, 3), round(origin.y, 3)

    def ll(geoms):
        return gpd.GeoSeries(list(geoms), crs=crs).to_crs(4326).to_numpy()

    def packed(coords):
        q = np.round((np.asarray(coords)[:, :2] - [x0, y0]) * 1e6).astype(int)
        return q.ravel().tolist()

    def rings(g):
        return [
            packed(p.exterior.coords)
            for p in shapely.get_parts(g)
            if p.geom_type == "Polygon" and not p.is_empty
        ]

    def polys(geoms, simplify=0.2):
        return (
            [rings(g) for g in ll([g.simplify(simplify) for g in geoms])]
            if len(geoms)
            else []
        )

    def num(v, digits=2):
        return (
            None
            if v is None or (isinstance(v, float) and np.isnan(v))
            else round(float(v), digits)
        )

    def whole(v):
        return None if v is None or np.isnan(v) else int(v)

    rows, parts, junctions = drawn["rows"], drawn["parts"], drawn["junctions"]
    data = {
        "origin": [x0, y0],
        "row": {
            "poly": polys([r["shape"] for r in rows]),
            "axis": [packed(g.coords) for g in ll(port.geometry)],
            "key": port["key"].tolist(),
            "name": [n if isinstance(n, str) else None for n in port["name"]],
            "cls": port["road_class"].tolist(),
            "w": [num(v) for v in port["width_m"]],
            "wl": [num(v) for v in port["width_left_m"]],
            "wr": [num(v) for v in port["width_right_m"]],
            "med": [num(v) for v in port["median_m"]],
            "ms": port["measured_sides"].astype(int).tolist(),
            "lanes": port["lanes"].astype(int).tolist(),
            "cw": port["carriageways"].astype(int).tolist(),
            "ang": [num(v) for v in port["angle_adjust_deg"]],
            "ov": [num(v, 1) for v in port["lot_overlap_m2"]],
            "use": [[KIND[u] for u in us] for us in port["bands.use"]],
            "share": [list(map(float, s)) for s in port["bands.share"]],
            "nroads": [len(k) for k in port["road_keys"]],
            "len": [num(v, 1) for v in port.length],
            "lab": [None if np.isnan(v) else int(v) for v in lab_of],
            "inside": sorted(drawn["inside"]),
        },
        "part": {
            "poly": polys([p["shape"] for p in parts], 0.1),
            "row": [p["row"] for p in parts],
            "kind": [p["kind"] for p in parts],
            "w": [p["width_m"] for p in parts],
        },
        "junction": {
            "poly": polys([j["shape"] for j in junctions]),
            "streets": [j["streets"] for j in junctions],
        },
        "marks": [packed(g.coords) for g in ll(drawn["marks"])],
        "lab": {
            "poly": polys(lab.geometry.to_numpy(), 0.3),
            "w": [num(v) for v in lab["width_m"]],
            "wl": [num(v) for v in lab["width_left_m"]],
            "wr": [num(v) for v in lab["width_right_m"]],
            "sl": [num(v) for v in lab["sidewalk_left_m"]],
            "sr": [num(v) for v in lab["sidewalk_right_m"]],
            "lanes": [whole(v) for v in lab["lanes"]],
            "lw": [num(v) for v in lab["lane_width_m"]],
            "cw": [whole(v) for v in lab["carriageways"]],
            "med": [num(v) for v in lab["median_m"]],
            "ms": [whole(v) for v in lab["measured_sides"]],
        },
        "lots": polys(lots.to_numpy(), 0.5),
    }
    matched = lab_of.notna()
    diff = (
        port["width_m"].to_numpy()
        - lab["width_m"].reindex(lab_of.fillna(-1).astype(int)).to_numpy()
    )
    if matched.any():
        d = np.abs(diff[matched.to_numpy()])
        report |= {
            "paired with the lab": f"{100 * matched.mean():.0f} %",
            "same width as the lab ±1 m": f"{100 * (d < 1).mean():.0f} %",
            "within ±3 m": f"{100 * (d < 3).mean():.0f} %",
            "median |difference|": f"{np.median(d):.1f} m",
        }
    data["stats"] = {
        "rows": len(port),
        "lab": len(lab),
        "parts": len(parts),
        "junctions": len(junctions),
        "inside": len(drawn["inside"]),
        "matched": int(matched.sum()),
        "same": int((np.abs(diff[matched.to_numpy()]) < 1).sum()),
        "dual": int((port["carriageways"] == 2).sum()),
        "dual_lab": int((lab["carriageways"] == 2).sum()),
        "outliers": port.index[port["width_m"] > 50].tolist(),
    }
    data["quality"] = report
    return TEMPLATE.read_text(encoding="utf-8").replace(
        "/*DATA*/null", json.dumps(data, separators=(",", ":"))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "area",
        nargs="?",
        help="a name in tests/data, or a gpkg with layers roads and lots",
    )
    parser.add_argument(
        "--roads", type=Path, help="roads (parquet or gpkg), instead of an area"
    )
    parser.add_argument(
        "--lots", type=Path, help="lots (parquet or gpkg), instead of an area"
    )
    parser.add_argument(
        "--lab", type=Path, help="the lab's output folder, with right_of_way.parquet"
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    def read(path: Path, layer: str | None = None):
        return (
            gpd.read_parquet(path)
            if path.suffix == ".parquet"
            else gpd.read_file(path, layer=layer)
        )

    if args.roads:
        roads, lots, name = (
            read(args.roads, "roads"),
            read(args.lots, "lots"),
            args.roads.stem,
        )
    else:
        path = Path(args.area or "test_area")
        path = path if path.suffix == ".gpkg" else DATA / f"{path}.gpkg"
        roads, lots, name = read(path, "roads"), read(path, "lots"), path.stem
    lots = lots.to_crs(roads.crs)
    lots = gpd.GeoSeries(shapely.make_valid(lots.geometry.to_numpy()), crs=roads.crs)
    for column in ("road_key", "from_node", "to_node", "road_type", "name"):
        roads[column] = roads[column].astype(object)
    roads = roads[roads["road_type"] != "pedestrian"].reset_index(drop=True)

    start = time.time()
    port = fit(roads, lots).reset_index(drop=True)
    seconds = time.time() - start
    drawn = streets(port, roads)
    report = {
        "segments": len(roads),
        "rights of way": len(port),
        "street_fit": f"{seconds:.0f} s",
    }
    report |= quality(port, roads, lots, drawn)
    lab = None
    if args.lab:
        lab = (
            gpd.read_parquet(args.lab / "right_of_way.parquet")
            .to_crs(roads.crs)
            .reset_index(drop=True)
        )
    out = args.out or Path(f"street_fit_{name}_streets.html")
    out.write_text(page(port, roads, lots, lab, drawn, report), encoding="utf-8")
    print(f"{out}: {out.stat().st_size / 1e6:.1f} MB")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
