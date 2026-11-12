"""A city's streets as ``right_of_way`` elements (urban_fabric.md, *Right of way*),
over its street network (``reimagina_street_network``, ``ingestion/network.py``).

- One element per street of the network: a pair of nodes, whatever the direction,
  keyed ``<lower id>-<higher id>``, with the edge's shape (OSM's way, or the
  straight line between its nodes where the network has none).
- ``sources.street_lighting`` (optional): a file of block faces with a lighting
  answer (its ``column``). A street's ``streetlights`` (0-100) is the share of lit
  faces among the answered ones whose midpoint lies within ``max_distance_m`` of
  it; without one it has no value. Mexicali: INEGI's 2020 housing inventory.
- ``sources.bike_lanes`` (optional): the city's own bike lane lines, matched to a
  street whose midpoint lies within ``max_distance_m`` of one. A protected lane
  (its ``type_column`` in ``protected``) takes a band of its own,
  ``protected_bike_lane``, next to the sidewalk; any other is shared: the street's
  lanes, used by bikes too, so its ``lanes`` band is ``shared_lanes``.

The traffic lanes are one band, ``lanes``: their total space, not lane by lane.
``bands.use`` and ``bands.share`` are written as JSON lists (Lyra tables have no
list column).

Unrefined (the default), they are a fixed profile. With ``sources.streets.params.
refine``, the vehicular streets (every class but ``pedestrian``) are fitted to the
cadastre's lots (``street_fit``): one right of way per stretch from intersection to
intersection (a dual carriageway is one), its axis centred between the lot fronts and
its bands the measured width's (sidewalk, lanes, median...). The pedestrian ones stay
as above. The fitted streets also change two matches, by their width (``width_m``):

- streetlights: faces within ``max(max_distance_m, width_m / 2 + front_tolerance_m)``;
- bike lanes: a street has one if at least ``min_share`` of its axis lies within
  ``max(max_distance_m, width_m / 2)`` of it (protected first), on the side the lane
  is. A shared lane makes that side's ``lanes`` band ``shared_lanes``; a protected
  one takes ``protected_width_m`` (at most half) from it, as a band between it and
  that side's sidewalk.

``width_m``, ``lanes``, ``name`` are used here and not written.
"""

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import shapely.ops

from reimagina_test_port.common import lake
from reimagina_test_port.ingestion import street_fit
from reimagina_test_port.ingestion.cadastral import stable_element_uuid
from reimagina_test_port.ingestion.cities import (
    BikeLanes,
    CityPreset,
    Source,
    StreetLighting,
)

logger = logging.getLogger(__name__)

ATTRIBUTES = {"bands.use": "json", "bands.share": "json", "streetlights": "number"}
PROFILES = {
    None: (["sidewalk", "lanes", "sidewalk"], [20, 60, 20]),
    "shared_lanes": (["sidewalk", "shared_lanes", "sidewalk"], [20, 60, 20]),
    "protected_bike_lane": (
        ["sidewalk", "protected_bike_lane", "lanes", "sidewalk"],
        [20, 10, 50, 20],
    ),
}


def read(source: Source, workdir: Path, columns: list[str]) -> gpd.GeoDataFrame:
    """A file GDAL reads (``member``: the file inside a .zip), from the lake or a
    local path."""
    path = lake.download(source.uri, workdir)
    if source.member:
        path = f"zip://{path}!{source.member}"
    return gpd.read_file(path, columns=columns)


def network_streets(
    nodes: pd.DataFrame, edges: pd.DataFrame, crs: str
) -> gpd.GeoSeries:
    """One line per pair of nodes the network joins (both directions are one
    street), in ``crs``, indexed by ``<lower id>-<higher id>``: the edge's
    ``geometry``, or the straight line between its nodes without one."""
    a, b = edges["from"].astype("int64"), edges["to"].astype("int64")
    keep = (
        pd.DataFrame({"lo": np.minimum(a, b), "hi": np.maximum(a, b)})
        .query("lo != hi")
        .drop_duplicates()
    )
    xy = nodes.set_index(nodes["id"].astype("int64"))[["x", "y"]]
    lines = shapely.linestrings(
        np.stack([xy.loc[keep["lo"]].to_numpy(), xy.loc[keep["hi"]].to_numpy()], 1)
    )
    if isinstance(edges, gpd.GeoDataFrame) and edges.crs is not None:
        own = edges.geometry.to_crs(crs).loc[keep.index].to_numpy()
        present = ~(shapely.is_missing(own) | shapely.is_empty(own))
        lines[present] = own[present]
    keys = keep["lo"].astype(str) + "-" + keep["hi"].astype(str)
    return gpd.GeoSeries(lines, index=pd.Index(keys.to_numpy(), name="key"), crs=crs)


def streetlights(
    streets: gpd.GeoSeries,
    faces: gpd.GeoDataFrame,
    lighting: StreetLighting,
    widths: np.ndarray | None = None,
) -> pd.Series:
    """Per street: the share (0-100) of the answered faces within
    ``lighting.max_distance_m`` whose ``column`` is in ``lit``. With ``widths`` (a
    fitted street's ``width_m``; NaN for the others), within ``width_m / 2 +
    front_tolerance_m`` if that is farther."""
    for column, value in lighting.where.items():
        faces = faces[faces[column].astype(str).str.strip() == value]
    answer = faces[lighting.column].astype(str).str.strip()
    value = answer.map(
        {**dict.fromkeys(lighting.unlit, 0.0), **dict.fromkeys(lighting.lit, 100.0)}
    )
    middles = gpd.GeoDataFrame(
        {"lit": value[value.notna()].to_numpy()},
        geometry=faces[value.notna()]
        .geometry.to_crs(streets.crs)
        .interpolate(0.5, normalized=True)
        .to_numpy(),
        crs=streets.crs,
    )
    reach = np.full(len(streets), lighting.max_distance_m, dtype=float)
    if widths is not None:
        wide = np.asarray(widths, dtype=float) / 2 + lighting.front_tolerance_m
        reach = np.fmax(reach, wide)  # fmax: a NaN width keeps max_distance_m
    near = gpd.GeoDataFrame(
        geometry=shapely.buffer(streets.to_numpy(), reach),
        index=streets.index,
        crs=streets.crs,
    )
    hit = gpd.sjoin(near, middles, predicate="contains")
    return hit.groupby(level=0)["lit"].mean().reindex(streets.index)


def lane_bands(
    streets: gpd.GeoSeries, lanes: gpd.GeoDataFrame, bike: BikeLanes
) -> pd.Series:
    """Per street: ``shared_lanes``, ``protected_bike_lane`` or None, by the nearest
    lane within ``bike.max_distance_m`` of its midpoint."""
    if streets.empty or lanes.empty:
        return pd.Series(None, index=streets.index, dtype=object)
    middle = streets.interpolate(0.5, normalized=True)
    near = gpd.sjoin_nearest(
        gpd.GeoDataFrame(geometry=middle),
        lanes[[bike.type_column, "geometry"]].to_crs(streets.crs),
        how="left",
        max_distance=bike.max_distance_m,
    )
    near = near[~near.index.duplicated()]
    kind = np.where(
        near[bike.type_column].isin(bike.protected),
        "protected_bike_lane",
        "shared_lanes",
    )
    return pd.Series(kind, index=near.index).where(near["index_right"].notna())


def roads_from_network(
    nodes: pd.DataFrame, edges: pd.DataFrame, crs: str
) -> gpd.GeoDataFrame:
    """The vehicular streets of the network as ``street_fit``'s roads
    (``street_fit.ROAD_COLUMNS``) in ``crs``: one per pair of nodes (the network
    holds each way both ways), keyed ``<lower id>-<higher id>`` as ``network_streets``.

    Vehicular: every class but ``pedestrian``. An edge without ``highway`` (the
    network's own, from merged streets) is ``residential``; a tag ``street_fit`` does
    not know (``track``...) is not vehicular. Each line starts at its ``from_node``.
    """
    a, b = edges["from"].astype("int64"), edges["to"].astype("int64")
    pairs = (
        pd.DataFrame({"lo": np.minimum(a, b), "hi": np.maximum(a, b)})
        .query("lo != hi")
        .drop_duplicates()
    )
    own = edges.loc[pairs.index]

    def tag(name: str) -> pd.Series:
        if name in own:
            return own[name]
        return pd.Series(None, index=own.index, dtype=object)

    highway = tag("highway")
    road_type = highway.map(street_fit.ROAD_CLASS_BY_HIGHWAY).where(
        highway.notna(), "residential"
    )
    vehicular = (road_type.notna() & (road_type != "pedestrian")).to_numpy()
    pairs, own, road_type = pairs[vehicular], own[vehicular], road_type[vehicular]
    if pairs.empty:
        return gpd.GeoDataFrame(
            {column: [] for column in street_fit.ROAD_COLUMNS},
            geometry=[],
            crs=crs,
        )
    xy = nodes.set_index(nodes["id"].astype("int64"))[["x", "y"]]
    lo_xy, hi_xy = xy.loc[pairs["lo"]].to_numpy(), xy.loc[pairs["hi"]].to_numpy()
    lines = shapely.linestrings(np.stack([lo_xy, hi_xy], 1))
    if isinstance(edges, gpd.GeoDataFrame) and edges.crs is not None:
        shapes = edges.geometry.to_crs(crs).loc[pairs.index].to_numpy()
        present = ~(shapely.is_missing(shapes) | shapely.is_empty(shapes))
        lines[present] = shapes[present]
    # Starts where its geometry does: at the lower node, unless nearer the higher.
    start = shapely.get_point(lines, 0)
    flipped = shapely.distance(start, shapely.points(hi_xy)) < shapely.distance(
        start, shapely.points(lo_xy)
    )
    lanes_tagged = tag("lanes").map(_lanes)
    names = tag("name")
    keys = pairs["lo"].astype(str) + "-" + pairs["hi"].astype(str)
    return gpd.GeoDataFrame(
        {
            "road_key": keys.to_numpy(),
            "from_node": np.where(flipped, pairs["hi"], pairs["lo"]),
            "to_node": np.where(flipped, pairs["lo"], pairs["hi"]),
            "road_type": road_type.to_numpy(),
            "name": names.where(names.notna(), None).to_numpy(),
            "lanes": [
                int(t) if pd.notna(t) else street_fit.DEFAULT_LANES[c]
                for t, c in zip(lanes_tagged, road_type, strict=True)
            ],
            "lanes_tagged": lanes_tagged.notna().to_numpy(),
            "lane_width_m": road_type.map(street_fit.LANE_WIDTH_M).to_numpy(),
        },
        geometry=shapely.force_2d(lines),
        crs=crs,
    )


def _lanes(value: object) -> float:
    """OSM's ``lanes`` tag as a number; NaN when missing or not one."""
    try:
        return float(int(float(str(value))))
    except ValueError:
        return np.nan


def lot_shapes(city: CityPreset, workdir: Path) -> gpd.GeoSeries:
    """The cadastre's lots, geometry only, valid, in ``city.target_crs``: what the
    vehicular streets are fitted between."""
    path = lake.download_shapefile(city.lots_uri, workdir / "lots")
    lots = gpd.read_file(path, engine="pyogrio", columns=[]).to_crs(city.target_crs)
    return gpd.GeoSeries(
        shapely.make_valid(lots.geometry.to_numpy()), crs=city.target_crs
    )


def refined_lane_bands(
    axes: gpd.GeoSeries, widths: np.ndarray, lanes: gpd.GeoDataFrame, bike: BikeLanes
) -> pd.DataFrame:
    """Per fitted street (``axes``, in metres, and their ``width_m``): its bike lane
    (``kind``: ``protected_bike_lane``, ``shared_lanes`` or None) and its ``side``
    (``left`` or ``right`` of the axis, walking from its first vertex). It has one if
    at least ``bike.min_share`` of its axis lies within ``max(max_distance_m,
    width_m / 2)`` of the lanes of that kind; protected first."""
    out = pd.DataFrame({"kind": None, "side": None}, index=axes.index, dtype=object)
    if lanes.empty or axes.empty:
        return out
    lines = lanes[[bike.type_column, "geometry"]].to_crs(axes.crs)
    lines = lines[lines.geometry.notna() & ~lines.geometry.is_empty]
    geoms = lines.geometry.to_numpy()
    protected = lines[bike.type_column].isin(bike.protected).to_numpy()
    axis = axes.to_numpy()
    reach = np.maximum(bike.max_distance_m, np.asarray(widths, dtype=float) / 2)
    street, lane = shapely.STRtree(geoms).query(
        shapely.buffer(axis, reach), predicate="intersects"
    )
    for i in np.unique(street):
        near = lane[street == i]
        for kind, of_kind in (
            ("protected_bike_lane", protected[near]),
            ("shared_lanes", ~protected[near]),
        ):
            if not of_kind.any():
                continue
            ways = shapely.union_all(geoms[near[of_kind]])
            covered = axis[i].intersection(ways.buffer(reach[i]))
            if covered.length >= bike.min_share * axis[i].length:
                out.iloc[i] = [kind, side_of(axis[i], covered, ways)]
                break
    return out


def side_of(
    axis: shapely.LineString, covered: shapely.Geometry, ways: shapely.Geometry
) -> str:
    """``left`` or ``right``: where ``ways`` are from ``axis``, at the middle of its
    longest ``covered`` stretch (the sign of the cross product of the axis's
    direction and the vector to the nearest way). A way along the axis: right."""
    piece = max(shapely.get_parts(covered), key=lambda g: g.length)
    at = axis.project(piece.interpolate(0.5, normalized=True))
    a = axis.interpolate(max(at - 1, 0))
    b = axis.interpolate(min(at + 1, axis.length))
    middle = axis.interpolate(at)
    _, way = shapely.ops.nearest_points(middle, ways)
    cross = (b.x - a.x) * (way.y - middle.y) - (b.y - a.y) * (way.x - middle.x)
    return "left" if cross > 1e-9 else "right"


def with_bike_lane(
    use: list[str],
    share: list[float],
    kind: str | None,
    side: str | None,
    width_m: float,
    protected_width_m: float,
) -> tuple[list[str], list[float]]:
    """A fitted street's bands with its bike lane on ``side``: that side's ``lanes``
    band is ``shared_lanes``, or gives ``protected_width_m`` (at most half of it) to
    a ``protected_bike_lane`` between it and its sidewalk. Bands left to right; the
    shares still sum 100."""
    use, share = list(use), list(share)
    if not isinstance(kind, str) or "lanes" not in use:  # None, or pandas' NaN
        return use, share
    at = (
        use.index("lanes")
        if side == "left"
        else len(use) - 1 - use[::-1].index("lanes")
    )
    if kind == "shared_lanes":
        use[at] = "shared_lanes"
        return use, share
    wanted = 100 * protected_width_m / width_m if width_m > 0 else share[at]
    lane = round(min(wanted, share[at] / 2), 2)
    share[at] = round(share[at] - lane, 2)
    where = at if side == "left" else at + 1
    use.insert(where, "protected_bike_lane")
    share.insert(where, lane)
    return use, share


def fit_streets(
    city: CityPreset, nodes: pd.DataFrame, edges: pd.DataFrame, workdir: Path
) -> gpd.GeoDataFrame:
    """The network's vehicular streets as rights of way (``street_fit.fit``), indexed
    by their key, in ``city.target_crs``."""
    roads = roads_from_network(nodes, edges, city.target_crs)
    if roads.empty:
        return street_fit.empty(city.target_crs).set_index("key")
    fitted = street_fit.fit(
        roads,
        lot_shapes(city, workdir),
        dict(city.streets.fit),  # ty: ignore
    )
    lost = set(roads["road_key"]) - {k for keys in fitted["road_keys"] for k in keys}
    if lost:
        logger.warning("Streets: %d vehicular edges are in no right of way", len(lost))
    return fitted.set_index("key")


def build(
    city: CityPreset, nodes: pd.DataFrame, edges: pd.DataFrame, workdir: Path
) -> gpd.GeoDataFrame:
    """The city's ``right_of_way`` elements over its network's ``nodes`` and
    ``edges`` (in its metric CRS), EPSG:4326.

    With ``city.streets.refine``, the vehicular streets are fitted to the lots and
    the rest are as ever (see the module's docstring); ``table.attrs["fit"]`` says
    how many of each."""
    streets = network_streets(nodes, edges, city.target_crs)
    fitted = None
    if city.streets is not None and city.streets.refine:
        fitted = fit_streets(city, nodes, edges, workdir)
        streets = streets.drop(
            [k for keys in fitted["road_keys"] for k in keys], errors="ignore"
        )
        streets = pd.concat([fitted.geometry, streets])
        streets = gpd.GeoSeries(streets, crs=city.target_crs)
    widths = np.full(len(streets), np.nan)
    if fitted is not None:
        widths[: len(fitted)] = fitted["width_m"].to_numpy(dtype=float)
    lit = pd.Series(np.nan, index=streets.index)
    if (lighting := city.street_lighting) is not None:
        columns = [lighting.column, *lighting.where]
        faces = read(lighting.source, workdir / "street_lighting", columns)
        lit = streetlights(
            streets, faces, lighting, widths if fitted is not None else None
        )
    lanes = pd.Series(None, index=streets.index, dtype=object)
    sides = pd.Series(None, index=streets.index, dtype=object)
    if (bike := city.bike_lanes) is not None:
        lines = read(bike.source, workdir / "bike_lanes", [bike.type_column])
        n = 0 if fitted is None else len(fitted)
        if n:
            found = refined_lane_bands(streets.iloc[:n], widths[:n], lines, bike)
            lanes.iloc[:n], sides.iloc[:n] = found["kind"], found["side"]
        rest = streets.iloc[n:]
        lanes.iloc[n:] = lane_bands(rest, lines, bike).reindex(rest.index).to_numpy()
    n = 0 if fitted is None else len(fitted)
    protected_width = bike.protected_width_m if bike is not None else 0.0
    bands = (
        [
            with_bike_lane(use, share, kind, side, width, protected_width)
            for use, share, kind, side, width in zip(
                fitted["bands.use"],
                fitted["bands.share"],
                lanes.iloc[:n],
                sides.iloc[:n],
                widths[:n],
                strict=True,
            )
        ]
        if n
        else []
    )
    bands += [PROFILES[None if pd.isna(lane) else lane] for lane in lanes.iloc[n:]]
    frame = pd.DataFrame(
        {
            "id": [
                stable_element_uuid(city.key, "right_of_way", k) for k in streets.index
            ],
            "parent_id": None,
            "source_record_ref": streets.index.to_numpy(),
            "bands.use": [json.dumps(use) for use, _ in bands],
            "bands.share": [json.dumps(share) for _, share in bands],
            "streetlights": lit.to_numpy(),
        }
    )
    table = gpd.GeoDataFrame(
        frame,
        geometry=shapely.force_2d(streets.to_crs(4326).to_numpy()),
        crs=4326,
    )
    table.attrs["fit"] = {
        "fitted": n,
        "both_sides": 0
        if fitted is None
        else int((fitted["measured_sides"] == 2).sum()),
        "unfitted": len(streets) - n,
    }
    return table
