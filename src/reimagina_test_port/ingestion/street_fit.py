"""Streets as rights of way, from intersection to intersection, fitted to the lots.

Ported from reimagina_pipelines_lab, ``pipelines/ingestion/src/reimagina_ingestion/
rights_of_way.py`` at commit ``1ca58c1`` (and three tables of its ``vocabulary.py``).
This repo owns it now: changes are made here, and the lab's later ones aren't followed.
What differs from the lab:

- ``fit`` returns one line per right of way (its axis, without the carrying-on into
  the junctions) with its bands as attributes, not polygons of sidewalks, lanes,
  medians and junctions.
- No internal roads (``internal_classes``, street parcels): an OSM ``drive`` network
  is public streets.
- No topology of its own: the nodes are OSM's, so bridges are not crossings.

A right of way is not a network segment. The network splits a street at every bend,
gives an avenue one line per direction and adds short links at junctions; a right of
way is the space between lot fronts, and may hold several segments:

1. Dual carriageways. Two segments that run parallel (≤ ``max_parallel_deg``),
   ``min_median_m``–``max_median_m`` apart, overlap along at least
   ``min_overlap_share`` of the shorter, share no node and have (almost) no lot
   between them are the two sides of one street; so are two that open from one node
   as a "Y" (≤ ``max_fork_deg``, the gap taken at their open ends). A segment its
   partners together cover less than ``min_overlap_share`` of loses them. One
   corridor whose axis is the midline, as far as either carriageway reaches. Short
   segments joining them (median crossings, U-turns) belong to it.
2. Intersection to intersection. Other segments are chained through nodes where only
   two segments meet and the street carries on (turning less than ``max_bend_deg``);
   a dual corridor is cut where a cross street meets either carriageway (a stretch
   without segments goes with its neighbour; staggered corridors meet halfway along
   their overlap). Chains shorter than ``min_row_length_m`` join a neighbour: the
   street they carry on, whose axis runs over them, else the longest.
3. Each axis is carried into its junctions by the half width of the widest street
   there and fitted: perpendicular rays every ``step_m`` up to ``max_half_width_m``
   stop at the first lot, or halfway to another street running beside it, and a
   side's width is a low ``quantile`` of them; a straight corridor with lots on both
   sides is also tried turned by up to ``±max_angle_deg``; a side without lots takes
   its class's default half width. The axis is then moved halfway between its two
   sides (``width_left_m`` and ``width_right_m`` are as measured before).
4. Junctions. The axes meeting at a node are moved to meet at one point (where they
   cross, carried on); a street ending on a carriageway reaches the avenue's axis.
   Consecutive rights of way of one street, both sides measured, whose widths differ
   by less than ``smooth_width_m`` take one width.

The width is shared out, lot front to lot front, as sidewalk · lanes · sidewalk (a
dual street: sidewalk · lanes · median · lanes · sidewalk), each part by its typical
width; a lane grows to at most ``max_lane_m``, past that the rest goes to sidewalks
and median.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import shapely.affinity
import shapely.ops
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import linemerge, substring

DEFAULTS: dict = {
    "step_m": 3.0,
    "max_half_width_m": 25.0,
    "quantile": 0.2,
    # Below this share of rays meeting a lot, a side isn't measured.
    "min_hit_share": 0.4,
    "max_angle_deg": 4.0,
    "angle_step_deg": 1.0,
    "overlap_penalty": 10.0,
    # A corridor within this of its chord is straight and may turn.
    "straight_tolerance_m": 1.0,
    "min_turn_length_m": 30.0,
    # A lot the axis runs through for this share of its length isn't a side.
    "crossed_share": 0.2,
    # Dual carriageways.
    "max_parallel_deg": 12.0,
    # Two carriageways opening from one node (a "Y") open up to this.
    "max_fork_deg": 25.0,
    "min_median_m": 3.0,
    "max_median_m": 35.0,
    "min_overlap_share": 0.5,
    "max_lots_between_share": 0.1,
    # Shorter chains join a neighbour.
    "min_row_length_m": 12.0,
    # A street carries on through a node of two segments if it turns less than this.
    "max_bend_deg": 60.0,
    # Cuts of a dual corridor this close are one cross street.
    "cut_merge_m": 15.0,
    # Consecutive rights of way of one street (same name, turning less than
    # ``smooth_bend_deg``) whose widths differ by less than ``smooth_width_m`` take
    # one width.
    "smooth_width_m": 2.0,
    "smooth_bend_deg": 30.0,
    "lane_m": 3.5,
    # A sidewalk's typical width: its share of the right of way against the lanes'.
    "sidewalk_weight_m": 1.5,
    # A lane grows to at most this; the rest goes to sidewalks and median.
    "max_lane_m": 4.0,
    "default_half_width_m": {
        "motorway": 15,
        "trunk": 12,
        "primary": 10,
        "secondary": 8,
        "tertiary": 6,
        "residential": 5,
        "service": 3,
        "pedestrian": 2,
    },
}
RANK = [
    "pedestrian",
    "service",
    "residential",
    "tertiary",
    "secondary",
    "primary",
    "trunk",
    "motorway",
]
# OSM ``highway`` → road class. ``road`` (unknown type) and ``busway`` aren't in the
# lab's table; the drive network has them, so they count as residential.
ROAD_CLASS_BY_HIGHWAY = {
    "motorway": "motorway",
    "motorway_link": "motorway",
    "trunk": "trunk",
    "trunk_link": "trunk",
    "primary": "primary",
    "primary_link": "primary",
    "secondary": "secondary",
    "secondary_link": "secondary",
    "tertiary": "tertiary",
    "tertiary_link": "tertiary",
    "residential": "residential",
    "living_street": "residential",
    "unclassified": "residential",
    "road": "residential",
    "busway": "residential",
    "service": "service",
    "pedestrian": "pedestrian",
    "footway": "pedestrian",
    "path": "pedestrian",
    "steps": "pedestrian",
    "cycleway": "pedestrian",
}
# Lanes when OSM doesn't say (both directions together), and lane widths, by class.
DEFAULT_LANES = {
    "motorway": 4,
    "trunk": 4,
    "primary": 4,
    "secondary": 2,
    "tertiary": 2,
    "residential": 2,
    "service": 1,
    "pedestrian": 0,
}
LANE_WIDTH_M = {
    "motorway": 3.6,
    "trunk": 3.5,
    "primary": 3.5,
    "secondary": 3.25,
    "tertiary": 3.25,
    "residential": 3.0,
    "service": 3.0,
    "pedestrian": 0.0,
}
# A street meeting an avenue's axis less than this off it goes on along it.
ALONG_DEG = 30.0
ROAD_COLUMNS = [
    "road_key",
    "from_node",
    "to_node",
    "road_type",
    "name",
    "lanes",
    "lanes_tagged",
    "lane_width_m",
]
COLUMNS = [
    "key",
    "name",
    "road_class",
    "width_m",
    "measured_sides",
    "carriageways",
    "lanes",
    "bands.use",
    "bands.share",
    "road_keys",
    # Diagnostics: not right_of_way attributes.
    "width_left_m",
    "width_right_m",
    "median_m",
    "angle_adjust_deg",
    "lot_overlap_m2",
]


def params_of(overrides: dict) -> dict:
    """``DEFAULTS`` with ``overrides``; an unknown key is an error."""
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown street fit parameters: {sorted(unknown)}")
    return {
        **DEFAULTS,
        **overrides,
        "default_half_width_m": {
            **DEFAULTS["default_half_width_m"],
            **overrides.get("default_half_width_m", {}),
        },
    }


# ------------------------------------------------------ fitting a corridor to the lots


def side_widths(
    axis: LineString,
    barrier: shapely.Geometry,
    p: dict,
    neighbours: shapely.Geometry | None = None,
) -> tuple[float | None, float | None]:
    """(left, right) half widths along ``axis``, to ``barrier``, or halfway to
    ``neighbours`` (other streets' lines) if nearer; None: not measured."""
    n = max(2, int(axis.length // p["step_m"]) + 1)
    at = np.linspace(0.05, 0.95, n) * axis.length
    points = shapely.line_interpolate_point(axis, at)
    before = shapely.get_coordinates(
        shapely.line_interpolate_point(axis, np.maximum(at - 1, 0))
    )
    after = shapely.get_coordinates(
        shapely.line_interpolate_point(axis, np.minimum(at + 1, axis.length))
    )
    tangent = after - before
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True).clip(1e-9)
    # To the left of the direction of travel.
    normal = np.c_[-tangent[:, 1], tangent[:, 0]]
    start = shapely.get_coordinates(points)
    out = []
    for sign in (1, -1):
        end = start + normal * sign * p["max_half_width_m"]
        rays = shapely.linestrings(np.stack([start, end], axis=1))
        # NaN where the ray met nothing.
        lengths = shapely.distance(points, shapely.intersection(rays, barrier))
        # Starting inside a lot measures nothing.
        lengths[shapely.intersects(barrier, points)] = np.nan
        if neighbours is not None and not neighbours.is_empty:
            # Two streets with no lot between them share the space between.
            halfway = (
                shapely.distance(points, shapely.intersection(rays, neighbours)) / 2
            )
            lengths = np.fmin(lengths, halfway)
        met = ~np.isnan(lengths)
        measured = met.mean() >= p["min_hit_share"]
        out.append(
            float(np.quantile(lengths[met], p["quantile"])) if measured else None
        )
    return out[0], out[1]


def band(axis: LineString, left: float, right: float) -> Polygon:
    """The corridor from ``right`` metres right of ``axis`` to ``left`` metres left."""
    # One-sided buffers (left: positive) hold for any axis, curved or zig-zag, where
    # offset outlines break apart.
    sides = [
        axis.buffer(w, single_sided=True, cap_style="flat") for w in (left, -right) if w
    ]
    return shapely.union_all(sides) if sides else Polygon()


def rotated(axis: LineString, degrees: float) -> LineString:
    middle = axis.interpolate(0.5, normalized=True)
    return shapely.affinity.rotate(axis, degrees, origin=middle)


def fit_axis(
    axis: LineString,
    lots_near: np.ndarray,
    default: float,
    p: dict,
    neighbours: shapely.Geometry | None = None,
) -> dict:
    """The fitted ``axis`` (turned, if that fills the gap between the lots better),
    its widths and how they were measured (to the lots, or halfway to
    ``neighbours``)."""
    if len(lots_near):
        inside = shapely.length(shapely.intersection(lots_near, axis))
    else:
        inside = np.array([])
    crossed = inside > min(10.0, p["crossed_share"] * axis.length)
    sides = lots_near[~crossed] if len(lots_near) else lots_near
    barrier = shapely.union_all(sides) if len(sides) else Polygon()
    chord = LineString([axis.coords[0], axis.coords[-1]])
    straight = axis.hausdorff_distance(chord) <= p["straight_tolerance_m"]

    def widths(line: LineString) -> tuple[float | None, float | None]:
        if barrier.is_empty and (neighbours is None or neighbours.is_empty):
            return None, None
        return side_widths(line, barrier, p, neighbours)

    # Turning fills a gap between two rows of lots: only with both sides measured, on
    # corridors long enough.
    level = widths(chord if straight else axis)
    turns = straight and axis.length >= p["min_turn_length_m"] and None not in level
    if turns:
        step = p["angle_step_deg"]
        angles = np.arange(-p["max_angle_deg"], p["max_angle_deg"] + 1e-9, step)
    else:
        angles = [0.0]
    best: tuple | None = None
    for angle in angles:
        line = rotated(chord, angle) if straight else axis
        left, right = widths(line)
        measured = (left is not None, right is not None)
        left = left if left is not None else default
        right = right if right is not None else default
        shape = band(line, left, right)
        overlap = shape.intersection(barrier).area if not barrier.is_empty else 0.0
        # Overlap with lots costs; turning a little costs too, so 0° wins ties.
        score = (
            shape.area - p["overlap_penalty"] * overlap - abs(angle) * 0.01 * shape.area
        )
        if best is None or score > best[0]:
            best = (score, angle, left, right, overlap, measured, line)
    assert best is not None  # angles is never empty
    _, angle, left, right, overlap, measured, line = best
    return {
        "axis": line,
        "width_m": round(left + right, 2),
        "width_left_m": round(left, 2),
        "width_right_m": round(right, 2),
        "angle_adjust_deg": float(angle),
        "lot_overlap_m2": round(overlap, 2),
        "measured_sides": int(sum(measured)),
    }


# --------------------------------------------------- from network segments to corridors


def chord_bearing(g: LineString) -> float:
    (x0, y0), (x1, y1) = g.coords[0], g.coords[-1]
    return math.atan2(y1 - y0, x1 - x0)


def dual_pairs(
    roads: gpd.GeoDataFrame, lot_shapes: np.ndarray, lot_tree: shapely.STRtree, p: dict
) -> list[tuple[int, int]]:
    """Pairs of segments (positions) that are the two carriageways of one street.

    Two segments sharing a node pair only as a "Y": one street drawn as two one-way
    carriageways that open from that node, less than ``max_fork_deg`` apart; their
    gap is measured at the open ends. A segment whose partners together cover less
    than ``min_overlap_share`` of it loses them (see ``covered_pairs``)."""
    geoms = roads.geometry.values
    tree = shapely.STRtree(geoms)
    cos_max = math.cos(math.radians(p["max_parallel_deg"]))
    cos_fork = math.cos(math.radians(p["max_fork_deg"]))
    if "name" in roads:
        names = roads["name"].where(roads["name"].notna(), None).tolist()
    else:
        names = [None] * len(roads)
    ends = list(zip(roads["from_node"], roads["to_node"], strict=True))
    pairs = []
    for i, a in enumerate(geoms):
        for j in tree.query(a.buffer(p["max_median_m"]), predicate="intersects"):
            shared = set(ends[i]) & set(ends[j])
            if j <= i or len(shared) > 1:
                continue
            b = geoms[j]
            short, long_ = (a, b) if a.length <= b.length else (b, a)
            if shared:
                # A "Y": both leave the shared node the same way, and the gap is
                # where they end apart.
                (node,) = shared
                va = leaving(a, ends[i][0] == node, a.length)
                vb = leaving(b, ends[j][0] == node, b.length)
                if np.dot(va, vb) < cos_fork * math.hypot(*va) * math.hypot(*vb):
                    continue
                at_start = ends[i if short is a else j][0] == node
                open_end = Point(short.coords[-1] if at_start else short.coords[0])
                gap = long_.distance(open_end)
            else:
                if abs(math.cos(chord_bearing(a) - chord_bearing(b))) < cos_max:
                    continue
                gap = long_.distance(short.interpolate(0.5, normalized=True))
            if not p["min_median_m"] <= gap <= p["max_median_m"]:
                continue
            reach = sorted(
                long_.project(Point(c)) for c in (short.coords[0], short.coords[-1])
            )
            if reach[1] - reach[0] < p["min_overlap_share"] * short.length:
                continue
            if names[i] and names[j] and names[i] != names[j]:
                continue
            # What lies between the two lines.
            strip = MultiLineString([a, b]).convex_hull
            near = lot_tree.query(strip, predicate="intersects")
            between = shapely.intersection(lot_shapes[near], strip)
            if shapely.area(between).sum() > p["max_lots_between_share"] * max(
                strip.area, 1
            ):
                continue
            pairs.append((i, j))
    return covered_pairs(pairs, geoms, p["min_overlap_share"])


def covered_pairs(
    pairs: list[tuple[int, int]], geoms: np.ndarray, share: float
) -> list[tuple[int, int]]:
    """``pairs`` without those of a segment its partners, all together, cover less
    than ``share`` of: a short segment beside a long one isn't an avenue, five short
    ones along a long one are. Dropping a segment's pairs uncovers its partners, so
    until nothing changes."""
    pairs = list(pairs)
    while True:
        partners = defaultdict(list)
        for i, j in pairs:
            partners[i].append(j)
            partners[j].append(i)
        poor = set()
        for i, others in partners.items():
            g = geoms[i]
            spans = sorted(
                sorted(
                    g.project(Point(c))
                    for c in (geoms[j].coords[0], geoms[j].coords[-1])
                )
                for j in others
            )
            covered, reach = 0.0, 0.0
            for s0, s1 in spans:
                covered += max(0.0, s1 - max(s0, reach))
                reach = max(reach, s1)
            if covered < share * g.length:
                poor.add(i)
        if not poor:
            return pairs
        pairs = [(i, j) for i, j in pairs if i not in poor and j not in poor]


def midline(
    side_a: list, side_b: list, step: float = 4.0
) -> tuple[list[LineString], float]:
    """The line halfway between two carriageways (one per merged part of side A), and
    their median distance (where they run side by side). It reaches as far as either
    carriageway: past an end of side A it follows side B, half the gap at that end
    away from it."""
    a, b = linemerge(side_a), shapely.union_all(side_b)
    mids, halves, gaps = [], [], []
    for part in getattr(a, "geoms", [a]):
        here_mids = []
        for t in np.linspace(0, 1, max(2, int(part.length // step) + 1)):
            here = part.interpolate(t, normalized=True)
            there = shapely.ops.nearest_points(here, b)[1]
            gaps.append(here.distance(there))
            here_mids.append(((here.x + there.x) / 2, (here.y + there.y) / 2))
            if t in (0, 1):
                halves.append(((here.x - there.x) / 2, (here.y - there.y) / 2))
        mids.append(here_mids)
    lines = [LineString(m) for m in mids]
    # Side B past the ends of the midline: (how far past, point), per line and end.
    past: dict = defaultdict(list)
    merged_b = linemerge(side_b)
    for part in getattr(merged_b, "geoms", [merged_b]):
        for t in np.linspace(0, 1, max(2, int(part.length // step) + 1)):
            q = part.interpolate(t, normalized=True)
            k = int(np.argmin([q.distance(line) for line in lines]))
            line = lines[k]
            for end, (e, f) in enumerate(
                ((line.coords[0], line.coords[1]), (line.coords[-1], line.coords[-2]))
            ):
                inward = np.subtract(f, e)
                inward /= np.linalg.norm(inward) or 1
                beyond = -float(np.dot(np.subtract((q.x, q.y), e), inward))
                if beyond > step / 2:
                    dx, dy = halves[2 * k + end]
                    past[k, end].append((beyond, (q.x + dx, q.y + dy)))
    out = []
    for k, line_mids in enumerate(mids):
        before = [xy for _, xy in sorted(past[k, 0], reverse=True)]
        after = [xy for _, xy in sorted(past[k, 1])]
        out.append(LineString([*before, *line_mids, *after]))
    return out, float(np.median(gaps)) if gaps else 0.0


def extend(line: LineString, by: float) -> LineString:
    """``line`` with its first and last pieces carried on by ``by`` metres."""
    if not by:
        return line
    c = list(line.coords)

    def carry(p0: tuple, p1: tuple) -> tuple[float, float]:
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        n = math.hypot(dx, dy) or 1
        return (p1[0] + dx / n * by, p1[1] + dy / n * by)

    return LineString([carry(c[1], c[0]), *c, carry(c[-2], c[-1])])


def leaving(g: LineString, at_start: bool, d: float = 10.0) -> tuple[float, float]:
    """The direction a segment leaves a node in (from its start, or from its end), over
    its first ``d`` metres."""
    a = g.interpolate(0 if at_start else g.length)
    b = g.interpolate(min(d, g.length) if at_start else max(g.length - d, 0))
    return b.x - a.x, b.y - a.y


def bend(g1: LineString, start1: bool, g2: LineString, start2: bool) -> float:
    """Degrees a street turns going from one segment into the other through their
    shared node (0: straight on)."""
    (ax, ay), (bx, by) = leaving(g1, start1), leaving(g2, start2)
    cos = (ax * bx + ay * by) / ((math.hypot(ax, ay) * math.hypot(bx, by)) or 1)
    return 180 - math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def corridors(
    roads: gpd.GeoDataFrame, lot_shapes: np.ndarray, lot_tree: shapely.STRtree, p: dict
) -> list[dict]:
    """[{axis, members (segment positions), carriageways, median_m}], before fitting.
    ``roads`` has a 0..n index and each geometry starts at its ``from_node``."""
    geoms = roads.geometry.values
    ends = list(zip(roads["from_node"], roads["to_node"], strict=True))
    at_node = defaultdict(list)
    for i, (a, b) in enumerate(ends):
        at_node[a].append(i)
        at_node[b].append(i)
    node_xy = {}
    for i, (a, b) in enumerate(ends):
        node_xy.setdefault(a, Point(geoms[i].coords[0]))
        node_xy.setdefault(b, Point(geoms[i].coords[-1]))

    # 1. Dual carriageways, and the short segments between them.
    pairs = dual_pairs(roads, lot_shapes, lot_tree, p)
    up = {}

    def find(x: int) -> int:
        up.setdefault(x, x)
        while up[x] != x:
            up[x] = up[up[x]]
            x = up[x]
        return x

    for i, j in pairs:
        up[find(i)] = find(j)
    groups = defaultdict(set)
    for i, j in pairs:
        groups[find(i)].update((i, j))
    dual_of = {i: g for g, members in groups.items() for i in members}
    # Median crossings and U-turns: short, both ends on the same dual street.
    for i, (a, b) in enumerate(ends):
        if i not in dual_of and geoms[i].length <= 1.5 * p["max_median_m"]:
            at_a = {dual_of.get(k) for k in at_node[a]}
            at_b = {dual_of.get(k) for k in at_node[b]}
            # As the lab builds it: which of several streets ``next(iter())`` picks
            # depends on how the set was made (an in-place ``-=`` picks another).
            both = (at_a & at_b) - {None}
            if both:
                dual_of[i] = next(iter(both))
                groups[dual_of[i]].add(i)

    out: list[dict[str, Any]] = []
    for g, members in groups.items():
        paired = defaultdict(set)
        for i, j in pairs:
            if find(i) == g:
                paired[i].add(j)
                paired[j].add(i)
        side = {}
        for first in paired:  # Two colours: one per carriageway.
            if first in side:
                continue
            side[first], queue = 0, [first]
            while queue:
                x = queue.pop()
                for y in paired[x]:
                    if y not in side:
                        side[y] = 1 - side[x]
                        queue.append(y)
        side_a, side_b = ([geoms[i] for i, s in side.items() if s == k] for k in (0, 1))
        if not side_a or not side_b:
            continue
        lines, median = midline(side_a, side_b)
        # Cross streets.
        cuts = {
            n
            for i in members
            for n in ends[i]
            if any(k not in members for k in at_node[n])
        }
        for line in lines:
            at = sorted(
                line.project(node_xy[n])
                for n in cuts
                if line.distance(node_xy[n]) <= p["max_median_m"]
            )
            stops = [0.0]
            for d in at:
                if (
                    d - stops[-1] > p["cut_merge_m"]
                    and line.length - d > p["cut_merge_m"]
                ):
                    stops.append(d)
            stops.append(line.length)
            # A stretch no segment is in the middle of goes with its neighbour: the
            # avenue has no gap.
            pieces: list[list] = []
            for s0, s1 in itertools.pairwise(stops):
                mine = [
                    i
                    for i in members
                    if s0 - 1
                    <= line.project(geoms[i].interpolate(0.5, normalized=True))
                    <= s1 + 1
                ]
                if pieces and not pieces[-1][2]:
                    pieces[-1] = [pieces[-1][0], s1, mine]
                elif pieces and not mine:
                    pieces[-1][1] = s1
                else:
                    pieces.append([s0, s1, mine])
            for s0, s1, mine in pieces:
                if mine:
                    out.append(
                        {
                            "axis": substring(line, s0, s1),
                            "members": mine,
                            # The whole avenue: its other pieces aren't neighbours.
                            "street": members,
                            "carriageways": 2,
                            "median_m": round(median, 1),
                        }
                    )
    trim_overlaps([c for c in out if c["carriageways"] == 2])

    # 2. Single streets, chained through nodes where only two segments meet.
    seen = set()
    for i in range(len(roads)):
        if i in dual_of or i in seen:
            continue
        chain, frontier = {i}, [i]
        while frontier:
            x = frontier.pop()
            for node in ends[x]:
                if len(at_node[node]) != 2:
                    continue
                for y in at_node[node]:
                    if (
                        y not in chain
                        and y not in dual_of
                        and bend(
                            geoms[x], ends[x][0] == node, geoms[y], ends[y][0] == node
                        )
                        <= p["max_bend_deg"]
                    ):
                        chain.add(y)
                        frontier.append(y)
        seen |= chain
        merged = linemerge([geoms[k] for k in chain])
        for part in getattr(merged, "geoms", [merged]):
            out.append(
                {
                    "axis": part,
                    "members": sorted(chain),
                    "carriageways": 1,
                    "median_m": 0.0,
                }
            )

    # 3. Stubs (turning links, short pieces between junctions) join a neighbour: the
    # longest single street they carry on (its axis then runs over them), else the
    # longest one (they are inside the junction, and not drawn).
    keep = [
        c
        for c in out
        if c["carriageways"] == 2 or c["axis"].length >= p["min_row_length_m"]
    ]
    owner = {m: c for c in keep for m in c["members"]}
    for stub in (c for c in out if c not in keep):
        neighbours = [
            (owner[k], node)
            for m in stub["members"]
            for node in ends[m]
            for k in at_node[node]
            if k in owner
        ]
        if not neighbours:
            keep.append(stub)
            continue
        carried = [
            (host, joined)
            for host, node in neighbours
            if (joined := carried_on(host, stub, node_xy[node], p)) is not None
        ]
        if carried:
            host, joined = max(carried, key=lambda hj: hj[0]["axis"].length)
            host["axis"] = joined
        else:
            host = max((h for h, _ in neighbours), key=lambda c: c["axis"].length)
        host["members"] = sorted(set(host["members"]) | set(stub["members"]))
        for m in stub["members"]:
            owner[m] = host
    return keep


def carried_on(host: dict, stub: dict, node: Point, p: dict) -> LineString | None:
    """``host``'s axis carried on over ``stub``'s, if a single street ending at
    ``node`` goes on into the stub (turning less than ``max_bend_deg``)."""
    a, b = host["axis"], stub["axis"]
    if host["carriageways"] != 1:
        return None
    host_end = [Point(a.coords[k]).distance(node) < 0.01 for k in (0, -1)]
    stub_end = [Point(b.coords[k]).distance(node) < 0.01 for k in (0, -1)]
    if not any(host_end) or not any(stub_end):
        return None
    if bend(a, host_end[0], b, stub_end[0]) > p["max_bend_deg"]:
        return None
    joined = linemerge([a, b])
    return joined if isinstance(joined, LineString) else None


def trim_overlaps(duals: list[dict], tolerance: float = 2.0) -> None:
    """Axes of two dual corridors whose ends run over each other (staggered
    carriageways, cut into different corridors) are both cut at the middle of the
    overlap, in place, so they meet there."""
    if len(duals) < 2:
        return
    axes = [c["axis"] for c in duals]
    tree = shapely.STRtree(axes)
    for i, j in zip(*tree.query(axes, predicate="dwithin", distance=tolerance)):
        if j <= i:
            continue
        a, b = duals[i]["axis"], duals[j]["axis"]
        shared = a.intersection(b.buffer(tolerance))
        # End to end, or across, they share about ``2 * tolerance``.
        if shared.is_empty or shared.length < 2 * tolerance + 1:
            continue
        piece = max(shapely.get_parts(shared), key=lambda g: g.length)
        middle = piece.interpolate(0.5, normalized=True)
        cut = []
        for line in (a, b):
            span = sorted(
                line.project(Point(c)) for c in (piece.coords[0], piece.coords[-1])
            )
            at_end = span[0] < tolerance or span[1] > line.length - tolerance
            # Only end to end: not the whole of one along the other.
            if not at_end or span[1] - span[0] > 0.9 * line.length:
                break
            d = line.project(middle)
            kept = (
                substring(line, 0, d)
                if d > line.length / 2
                else substring(line, d, line.length)
            )
            if kept.length < 1.0:
                break
            coords = list(kept.coords)
            coords[-1 if d > line.length / 2 else 0] = (middle.x, middle.y)
            cut.append(LineString(coords))
        if len(cut) == 2:
            duals[i]["axis"], duals[j]["axis"] = cut


# -------------------------------------------------------------------------- the bands


def layout(
    c: dict, lanes: int, lane_width: float, total: float, p: dict
) -> list[tuple[str, int, int, float]]:
    """The right of way across, from its right edge to its left:
    [(part, carriageway, number, width)], filling ``total``. Parts: ``sidewalk``,
    ``lane``, ``median``. Shares by typical width; lanes capped at ``max_lane_m``."""
    sw = p["sidewalk_weight_m"]
    parts = [("sidewalk", 0, 0, sw)]
    if not lanes:  # A footpath: all of it is for walking.
        parts.append(("sidewalk", 0, 0, sw))
        return [(k, cw, n, w * total / (2 * sw)) for k, cw, n, w in parts]
    if c["carriageways"] == 2:
        side = lanes // 2
        median = max(1.0, c["median_m"] - side * lane_width)
        parts += [("lane", 2, n + 1, lane_width) for n in range(side)]
        parts += [("median", 0, 0, median)]
        parts += [("lane", 1, n + 1, lane_width) for n in range(side)]
    else:
        parts += [("lane", 1, n + 1, lane_width) for n in range(lanes)]
    parts.append(("sidewalk", 0, 0, sw))
    scale = total / sum(w for *_, w in parts)
    widths = [w * scale for *_, w in parts]
    # Lanes stop growing; sidewalks and median take the rest, in proportion.
    if lane_width * scale > p["max_lane_m"]:
        lane_total = sum(1 for k, *_ in parts if k == "lane") * p["max_lane_m"]
        others = sum(w for k, _, _, w in parts if k != "lane")
        widths = [
            p["max_lane_m"] if k == "lane" else w * (total - lane_total) / others
            for k, _, _, w in parts
        ]
    return [
        (k, cw, n, width) for (k, cw, n, _), width in zip(parts, widths, strict=True)
    ]


def lanes_of(c: dict, members: pd.DataFrame) -> tuple[int, float, float]:
    """(lanes in both directions, lane width, metres the traffic takes on each side of
    the axis)."""
    main = members.loc[members.geometry.length.idxmax()]
    width = float(main["lane_width_m"]) if "lane_width_m" in members else 3.0
    lanes = int(main["lanes"]) if "lanes" in members else 2
    # A carriageway is one direction: tagged lanes are its own, a class default both.
    if c["carriageways"] == 2:
        tagged = bool(main.get("lanes_tagged", False))
        side = lanes if tagged else max(1, round(lanes / 2))
        return 2 * side, width, c["median_m"] / 2 + side * width / 2
    return lanes, width, lanes * width / 2


def bands_of(
    parts: list[tuple[str, int, int, float]], total: float
) -> tuple[list[str], list[float]]:
    """``bands.use`` and ``bands.share`` (0-100, summing 100) of ``layout``'s parts,
    left to right, the lanes of a carriageway as one ``lanes`` band."""
    use: list[str] = []
    width: list[float] = []
    for part, _, _, w in reversed(parts):
        name = "lanes" if part == "lane" else part
        if name == "lanes" and use and use[-1] == "lanes":
            width[-1] += w
        else:
            use.append(name)
            width.append(w)
    share = [round(100 * w / total, 2) if total else 0.0 for w in width]
    largest = int(np.argmax(share))
    share[largest] = round(100 - sum(s for k, s in enumerate(share) if k != largest), 2)
    return use, share


def neighbours_of(
    c: dict,
    nodes: set,
    geoms: np.ndarray,
    ends: list[tuple],
    tree: shapely.STRtree,
    p: dict,
) -> shapely.Geometry | None:
    """The segments along corridor ``c`` (within ``max_half_width_m`` of it and less
    than ``ALONG_DEG`` off its direction) that aren't its own (nor its avenue's) and
    touch none of its ``nodes``: another street running beside it, whose axis its width stops halfway
    to."""
    members = set(c["members"]) | set(c.get("street", ()))
    reach = c["axis"].buffer(p["max_half_width_m"])
    cos_along = math.cos(math.radians(ALONG_DEG))
    bearing = chord_bearing(c["axis"])
    near = [
        geoms[j]
        for j in tree.query(reach, predicate="intersects")
        if j not in members
        and not set(ends[j]) & nodes
        and abs(math.cos(chord_bearing(geoms[j]) - bearing)) >= cos_along
    ]
    return shapely.union_all(near) if near else None


# ------------------------------------------------------------ axes after fitting them


def centred(line: LineString, left: float, right: float) -> LineString:
    """``line`` moved ``(left − right) / 2`` to its left (negative: right), halfway
    between the lot fronts; as it is if that breaks it (a tight curve)."""
    shift = (left - right) / 2
    if abs(shift) < 0.01:
        return line
    moved = line.offset_curve(shift)
    if (
        not isinstance(moved, LineString)
        or moved.is_empty
        or moved.length < 0.5 * line.length
    ):
        return line
    return moved


def arriving(line: LineString, at_start: bool, d: float = 10.0) -> np.ndarray:
    """The unit direction ``line`` reaches its start (or end) in."""
    v = -np.asarray(leaving(line, at_start, d), dtype=float)
    return v / (np.linalg.norm(v) or 1)


def crossing(
    p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray, min_sin: float = 0.5
) -> np.ndarray | None:
    """Where the line through ``p1`` along ``d1`` meets the one through ``p2`` along
    ``d2``; None if they are less than 30° apart."""
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < min_sin:
        return None
    w = p2 - p1
    return p1 + d1 * (w[0] * d2[1] - w[1] * d2[0]) / cross


def end_of(line: LineString, at_start: bool) -> np.ndarray:
    return np.asarray(line.coords[0 if at_start else -1], dtype=float)


def with_end(line: LineString, at_start: bool, xy: np.ndarray) -> LineString:
    """``line`` with its first (or last) vertex at ``xy``; as it is if that leaves it
    shorter than a metre."""
    coords = list(line.coords)
    coords[0 if at_start else -1] = (float(xy[0]), float(xy[1]))
    moved = LineString(coords)
    return moved if moved.length >= 1.0 else line


def meet_at_junctions(
    rows: list[dict], node_xy: dict, p: dict
) -> list[list[tuple[int, bool]]]:
    """Moves the ends of the fitted axes (``rows``' ``geometry``) in place so those
    meeting at a junction meet at one point, and returns the junctions: the ends
    (row, at its start) that met there before fitting (``was``).

    The point is where the axes, carried on, cross (their mean, if several cross;
    the ends' mean, if they are almost parallel); where none was moved, the node. A
    single street ending on a carriageway is carried on to the avenue's axis. An end
    is left as it is if that moves it more than ``max_half_width_m``."""
    limit = p["max_half_width_m"]
    avenues_at = defaultdict(list)
    for k, r in enumerate(rows):
        if r["carriageways"] == 2:
            for n in r["nodes"]:
                avenues_at[n].append(k)
    ends, to_avenue = [], []
    for k, r in enumerate(rows):
        for at_start in (True, False):
            was = Point(end_of(r["was"], at_start))
            on = [
                n
                for n in r["nodes"]
                if n in avenues_at and node_xy[n].distance(was) < 0.01
            ]
            if r["carriageways"] == 1 and on:
                to_avenue.append((k, at_start, on[0]))
            else:
                ends.append((k, at_start, was))

    # Ends that were one point: union by distance.
    up = list(range(len(ends)))

    def find(x: int) -> int:
        while up[x] != x:
            up[x] = up[up[x]]
            x = up[x]
        return x

    points = [w for *_, w in ends]
    for a, b in zip(*shapely.STRtree(points).query(points, "dwithin", distance=0.5)):
        up[find(a)] = find(b)
    groups = defaultdict(list)
    for x in range(len(ends)):
        groups[find(x)].append(x)
    junctions = []
    for members in groups.values():
        if len(members) < 2:
            continue
        junction = [(ends[x][0], ends[x][1]) for x in members]
        junctions.append(junction)
        now = [end_of(rows[k]["geometry"], s) for k, s in junction]
        was = [np.array([ends[x][2].x, ends[x][2].y]) for x in members]
        if all(np.linalg.norm(n - w) < 0.05 for n, w in zip(now, was, strict=True)):
            continue
        heading = [arriving(rows[k]["geometry"], s) for k, s in junction]
        crossings = [
            xy
            for i, j in itertools.combinations(range(len(junction)), 2)
            if (xy := crossing(now[i], heading[i], now[j], heading[j])) is not None
        ]
        point = np.mean(crossings or now, axis=0)
        for (k, s), n in zip(junction, now, strict=True):
            if np.linalg.norm(point - n) <= limit:
                rows[k]["geometry"] = with_end(rows[k]["geometry"], s, point)

    # A street onto an avenue: on to its axis. Where the avenue's axis ends there
    # (its carriageways join, or it stops at the street), a street going on along it
    # meets that end, and one across meets the axis carried on, which then reaches it.
    for k, s, n in to_avenue:
        line = rows[k]["geometry"]
        here, heading = end_of(line, s), arriving(line, s)
        ray = LineString([here - heading, here + heading * p["max_median_m"]])
        hits = []
        for a in avenues_at[n]:
            hits += [
                (h, None)
                for h in shapely.get_parts(ray.intersection(rows[a]["geometry"]))
            ]
        hits = [(h, a) for h, a in hits if isinstance(h, Point)]
        if not hits:
            hits = avenue_end(rows, avenues_at[n], here, heading, ray, p)
        if not hits:
            continue
        hit, avenue = min(hits, key=lambda h: h[0].distance(Point(here)))
        if hit.distance(Point(here)) > limit:
            continue
        xy = np.array([hit.x, hit.y])
        rows[k]["geometry"] = with_end(line, s, xy)
        if avenue is not None:
            a, at_start = avenue
            rows[a]["geometry"] = with_end(rows[a]["geometry"], at_start, xy)
    return junctions


def avenue_end(
    rows: list[dict],
    avenues: list[int],
    here: np.ndarray,
    heading: np.ndarray,
    ray: LineString,
    p: dict,
) -> list[tuple[Point, tuple[int, bool] | None]]:
    """Where a street ending at ``here`` (arriving along ``heading``) meets an avenue
    whose axis ends near it: [(point, (avenue row, its start) when its axis has to be
    carried on to it, else None)]."""
    out = []
    for a in avenues:
        axis = rows[a]["geometry"]
        t = axis.project(Point(here))
        if 1.0 < t < axis.length - 1.0:
            continue
        at_start = t <= 1.0
        end = end_of(axis, at_start)
        along = arriving(axis, at_start)
        if abs(float(np.dot(along, heading))) > math.cos(math.radians(ALONG_DEG)):
            out.append((Point(end), None))  # Along it: they meet at its end.
            continue
        carried = LineString([end, end + along * p["max_median_m"]])
        hit = ray.intersection(carried)
        if isinstance(hit, Point):
            out.append((hit, (a, at_start)))
    return out


def smooth_widths(
    rows: list[dict], junctions: list[list[tuple[int, bool]]], p: dict
) -> None:
    """Consecutive rights of way of one street, both sides measured, whose widths
    differ by less than ``smooth_width_m`` take their mean width (by length), in
    place. The smallest differences join first, and a group whose widths would span
    ``smooth_width_m`` or more isn't joined: a street that changes width keeps it."""
    joints = []
    for junction in junctions:
        for (i, si), (j, sj) in itertools.combinations(junction, 2):
            a, b = rows[i], rows[j]
            if (
                i == j
                or not a["name"]
                or a["name"] != b["name"]
                or a["carriageways"] != b["carriageways"]
                or a["measured_sides"] != 2
                or b["measured_sides"] != 2
            ):
                continue
            cos = float(
                np.dot(arriving(a["geometry"], si), arriving(b["geometry"], sj))
            )
            turn = 180 - math.degrees(math.acos(max(-1.0, min(1.0, cos))))
            gap = abs(a["width_m"] - b["width_m"])
            if turn < p["smooth_bend_deg"] and gap < p["smooth_width_m"]:
                joints.append((gap, i, j))
    group = {k: [k] for k in range(len(rows))}
    for _, i, j in sorted(joints):
        gi, gj = group[i], group[j]
        if gi is gj:
            continue
        widths = [rows[k]["width_m"] for k in gi + gj]
        if max(widths) - min(widths) >= p["smooth_width_m"]:
            continue
        gi.extend(gj)
        for k in gj:
            group[k] = gi
    for members in {id(g): g for g in group.values()}.values():
        if len(members) < 2:
            continue
        length = np.array([rows[k]["geometry"].length for k in members])
        width = np.array([rows[k]["width_m"] for k in members])
        mean = round(float(np.average(width, weights=length.clip(1e-9))), 2)
        for k in members:
            rows[k]["width_m"] = mean


# ------------------------------------------------------------------------------- fit


def empty(crs: object = None) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(columns=COLUMNS, geometry=[], crs=crs)


def fit(
    roads: gpd.GeoDataFrame, lots: gpd.GeoSeries, params: dict | None = None
) -> gpd.GeoDataFrame:
    """The rights of way of ``roads`` (``ROAD_COLUMNS``, each line starting at its
    ``from_node``) between ``lots``, both in the same metric CRS: one row per right of
    way, its geometry the fitted axis from junction to junction."""
    p = params_of(params or {})
    if roads.empty:
        return empty(roads.crs)
    roads = gpd.GeoDataFrame(roads.reset_index(drop=True))
    # Cadastres carry self-intersecting outlines.
    lot_shapes = shapely.make_valid(np.asarray(lots.geometry.values))
    lot_shapes = lot_shapes[~shapely.is_empty(lot_shapes)]
    lot_tree = shapely.STRtree(lot_shapes)

    def half(kind: str) -> float:
        return p["default_half_width_m"].get(kind, 4)

    # Node → half width of the widest street meeting there, by class.
    widest_at: dict = defaultdict(float)
    for a, b, kind in zip(
        roads["from_node"], roads["to_node"], roads["road_type"], strict=True
    ):
        for node in (a, b):
            widest_at[node] = max(widest_at[node], half(kind))
    rows = []
    used: Counter = Counter()
    road_geoms = roads.geometry.to_numpy()
    road_nodes = list(zip(roads["from_node"], roads["to_node"], strict=True))
    road_tree = shapely.STRtree(road_geoms)
    found = corridors(roads, lot_shapes, lot_tree, p)
    for c in found:
        members = roads.iloc[c["members"]]
        kind = max(
            members["road_type"], key=lambda k: RANK.index(k) if k in RANK else 0
        )
        dual = c["carriageways"] == 2
        default = max(half(kind), c["median_m"] / 2 + p["lane_m"] if dual else 0)
        nodes = set(members["from_node"]) | set(members["to_node"])
        reach = max((widest_at[n] for n in nodes), default=0)
        # Into the junctions, as far as the widest street there.
        axis = extend(c["axis"], reach)
        near = lot_tree.query(
            axis.buffer(p["max_half_width_m"]), predicate="intersects"
        )
        keys = sorted(members["road_key"])
        used[keys[0]] += 1
        key = keys[0] + (f"#{used[keys[0]]}" if used[keys[0]] > 1 else "")
        others = neighbours_of(c, nodes, road_geoms, road_nodes, road_tree, p)
        fitted = fit_axis(axis, lot_shapes[near], default, p, others)
        line = centred(
            fitted.pop("axis"), fitted["width_left_m"], fitted["width_right_m"]
        )
        # The right of way itself, without its carrying-on into the junctions.
        line = substring(line, reach, line.length - reach) if reach else line
        lanes, lane_width, _ = lanes_of(c, members)
        name = members.loc[members.geometry.length.idxmax(), "name"]
        rows.append(
            {
                "key": key,
                "name": name if isinstance(name, str) and name else None,
                "road_class": kind,
                "carriageways": c["carriageways"],
                "lanes": lanes,
                "lane_width": lane_width,
                "road_keys": keys,
                "median_m": c["median_m"],
                "nodes": nodes,
                "was": c["axis"],
                **fitted,
                "geometry": line,
            }
        )
    if not rows:
        return empty(roads.crs)
    node_xy = {}
    for a, b, g in zip(
        roads["from_node"], roads["to_node"], roads.geometry, strict=True
    ):
        node_xy.setdefault(a, Point(g.coords[0]))
        node_xy.setdefault(b, Point(g.coords[-1]))
    junctions = meet_at_junctions(rows, node_xy, p)
    smooth_widths(rows, junctions, p)
    for row, c in zip(rows, found, strict=True):
        parts = layout(c, row["lanes"], row.pop("lane_width"), row["width_m"], p)
        row["bands.use"], row["bands.share"] = bands_of(parts, row["width_m"])
        del row["nodes"], row["was"]
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=roads.crs)
    check(out)
    return gpd.GeoDataFrame(
        out[[*COLUMNS, "geometry"]], geometry="geometry", crs=roads.crs
    )


def check(out: gpd.GeoDataFrame) -> None:
    """Every row's bands match and share 100; keys are unique; no empty axis."""
    for key, use, share in zip(
        out["key"], out["bands.use"], out["bands.share"], strict=True
    ):
        if len(use) != len(share) or min(share) < 0 or abs(sum(share) - 100) > 1e-6:
            raise ValueError(f"right of way {key}: bands {use} {share}")
    if out["key"].duplicated().any():
        raise ValueError("right of way keys repeat")
    if (out.geometry.is_empty | out.geometry.isna()).any():
        raise ValueError("a right of way has no axis")
