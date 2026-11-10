"""Access to types of POI over the street network (ULI sub-variables "Average distance
to closest <poi>" and "Percentage of population with access to <poi> within X
meters"), following GHSCI's method with lots as origins.

Lots and POIs are reached through their entries. A point is its own entry; a polygon
can be entered anywhere along its outline: a point every ``ENTRY_SPACING_M`` metres,
keeping those within ``ENTRY_TOLERANCE_M`` of a street (the closest one when none
is). Each entry joins the street at its projection on the nearest segment. The
distance from a lot to the nearest POI is the shortest walk from one of its entries to
one of the POIs' entries: straight to the street, along the network (every street
two-way) and straight from the street. Null beyond ``max_distance``.

``population_access`` counts the built-form population of lots within each type's
coverage distance as having access. A block adds up its lots and averages their
distances weighted by population.

The POIs arrive already selected; what they are does not matter here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

# GHSCI: a point every 20 m along an open space's outline, kept within 30 m of a road.
ENTRY_SPACING_M = 20.0
ENTRY_TOLERANCE_M = 30.0
# Origins this far from every street mean the network is in another CRS.
MAX_MEDIAN_OFFSET_M = 1_000.0
# scipy's graphs take a zero weight as no edge.
_MIN_WEIGHT = 1e-6


@dataclass(frozen=True)
class StreetNetwork:
    """Two-way street segments between node positions, one per pair of nodes."""

    nodes: int
    start: np.ndarray
    end: np.ndarray
    length: np.ndarray
    segments: np.ndarray
    tree: shapely.STRtree


def street_network(nodes: pd.DataFrame, edges: pd.DataFrame) -> StreetNetwork:
    """Segments from node (id, x, y) and edge (from, to, distance) tables: straight
    between their nodes, with the edge's length; the shortest edge of each pair."""
    if nodes.empty or edges.empty:
        msg = "The road network has no nodes or edges."
        raise ValueError(msg)
    position = pd.Series(np.arange(len(nodes)), index=nodes["id"].astype("int64"))
    a = position.loc[edges["from"].astype("int64")].to_numpy()
    b = position.loc[edges["to"].astype("int64")].to_numpy()
    pairs = (
        pd.DataFrame(
            {
                "start": np.minimum(a, b),
                "end": np.maximum(a, b),
                "length": edges["distance"].to_numpy("float64"),
            }
        )
        .query("start != end")
        .groupby(["start", "end"], as_index=False)["length"]
        .min()
    )
    xy = nodes[["x", "y"]].to_numpy("float64")
    start, end = pairs["start"].to_numpy(), pairs["end"].to_numpy()
    segments = shapely.linestrings(np.stack([xy[start], xy[end]], axis=1))
    return StreetNetwork(
        nodes=len(nodes),
        start=start,
        end=end,
        length=pairs["length"].to_numpy(),
        segments=segments,
        tree=shapely.STRtree(segments),
    )


def entries(geometry: gpd.GeoSeries, network: StreetNetwork) -> pd.DataFrame:
    """Entries of each geometry and where they join the street.

    One row per entry: ``owner`` (position in ``geometry``), ``segment``,
    ``along_start`` / ``along_end`` (metres along it to its start / end node) and
    ``offset`` (straight-line metres to it).
    """
    values = geometry.to_numpy()
    present = np.flatnonzero(~(shapely.is_missing(values) | shapely.is_empty(values)))
    parts, part_owner = shapely.get_parts(values[present], return_index=True)
    part_owner = present[part_owner]
    polygon = shapely.get_type_id(parts) == shapely.GeometryType.POLYGON
    rings = shapely.get_exterior_ring(parts[polygon])
    count = np.maximum(1, np.ceil(shapely.length(rings) / ENTRY_SPACING_M)).astype(int)
    ring = np.repeat(np.arange(len(rings)), count)
    step = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    points = np.concatenate(
        [
            # Points are their own entry (other shapes, a point on them).
            shapely.point_on_surface(parts[~polygon]),
            shapely.line_interpolate_point(
                rings[ring], step / count[ring], normalized=True
            ),
        ]
    )
    owner = np.concatenate([part_owner[~polygon], part_owner[polygon][ring]])
    outline = np.concatenate(
        [np.zeros((~polygon).sum(), bool), np.ones(len(ring), bool)]
    )
    found, nearest = network.tree.query_nearest(points, all_matches=False)
    segment = np.empty(len(points), dtype="int64")
    segment[found] = nearest
    lines = network.segments[segment]
    fraction = np.nan_to_num(shapely.line_locate_point(lines, points, normalized=True))
    frame = pd.DataFrame(
        {
            "owner": owner,
            "outline": outline,
            "segment": segment,
            "along_start": fraction * network.length[segment],
            "along_end": (1 - fraction) * network.length[segment],
            "offset": shapely.distance(points, lines),
        }
    )
    near = frame["offset"] <= ENTRY_TOLERANCE_M
    reachable = frame.loc[frame["outline"] & near, "owner"].unique()
    closest = frame["offset"] == frame.groupby("owner")["offset"].transform("min")
    keep = ~frame["outline"] | near | (~frame["owner"].isin(reachable) & closest)
    return frame[keep].drop(columns="outline").reset_index(drop=True)


def _from_destinations(
    network: StreetNetwork, destinations: pd.DataFrame, max_distance: float
) -> np.ndarray:
    """Per node, metres to the nearest destination entry (inf beyond
    ``max_distance``): one Dijkstra from a source joined to every entry's nodes."""
    source = network.nodes
    seeds = pd.concat(
        [
            pd.DataFrame(
                {
                    "node": network.start[destinations["segment"]],
                    "cost": destinations["along_start"] + destinations["offset"],
                }
            ),
            pd.DataFrame(
                {
                    "node": network.end[destinations["segment"]],
                    "cost": destinations["along_end"] + destinations["offset"],
                }
            ),
        ]
    )
    seeds = seeds.groupby("node")["cost"].min()
    rows = np.concatenate([network.start, network.end, np.full(len(seeds), source)])
    columns = np.concatenate([network.end, network.start, seeds.index.to_numpy()])
    weights = np.maximum(
        np.concatenate([network.length, network.length, seeds.to_numpy()]),
        _MIN_WEIGHT,
    )
    graph = coo_matrix(
        (weights, (rows, columns)), shape=(source + 1, source + 1)
    ).tocsr()
    return dijkstra(graph, directed=True, indices=source, limit=max_distance)[:source]


def _to_nearest(
    origins: pd.DataFrame,
    destinations: pd.DataFrame,
    network: StreetNetwork,
    max_distance: float,
    size: int,
) -> np.ndarray:
    """Per owner of ``origins`` (``size`` of them): metres to the nearest destination
    entry; NaN without one within ``max_distance``."""
    result = np.full(size, np.nan)
    if origins.empty or destinations.empty:
        return result
    to_poi = _from_destinations(network, destinations, max_distance)
    start = network.start[origins["segment"]]
    end = network.end[origins["segment"]]
    walk = np.minimum(
        to_poi[start] + origins["along_start"].to_numpy(),
        to_poi[end] + origins["along_end"].to_numpy(),
    )
    # A POI on the same segment is reached along it, without going to a node.
    same = origins.reset_index().merge(
        destinations, on="segment", suffixes=("", "_poi")
    )
    direct = (same["along_start"] - same["along_start_poi"]).abs() + same["offset_poi"]
    walk = np.fmin(
        walk, direct.groupby(same["index"]).min().reindex(origins.index).to_numpy()
    )
    distance = (
        pd.Series(walk + origins["offset"].to_numpy())
        .groupby(origins["owner"].to_numpy())
        .min()
    )
    distance = distance.where(distance <= max_distance)
    result[distance.index.to_numpy()] = distance.to_numpy()
    return result


def nearest_distances(
    lots: gpd.GeoDataFrame,
    pois: Mapping[str, gpd.GeoSeries],
    network: StreetNetwork,
    crs: str,
    max_distance: float,
) -> dict[str, np.ndarray]:
    """``{poi_type: metres}``: per lot, in order, the walk over ``network`` (in
    ``crs``) to the nearest POI of the type; NaN without geometry or without one
    within ``max_distance``. The lots' entries are built once."""
    origins = entries(
        gpd.GeoSeries(lots.geometry.to_numpy(), crs=lots.crs).to_crs(crs), network
    )
    median_offset = origins.groupby("owner")["offset"].min().median()
    if median_offset > MAX_MEDIAN_OFFSET_M:
        msg = (
            f"Lots lie a median {median_offset:,.0f} m from the road network in {crs}; "
            "set ACCESS_CRS to the network's CRS."
        )
        raise ValueError(msg)
    return {
        poi_type: _to_nearest(
            origins,
            entries(geometry.to_crs(crs), network),
            network,
            max_distance,
            len(lots),
        )
        for poi_type, geometry in pois.items()
    }


def population_access(
    lots: gpd.GeoDataFrame,
    blocks: gpd.GeoDataFrame,
    distances: Mapping[str, np.ndarray],
    coverage_m: Mapping[str, float],
) -> dict[str, pd.DataFrame]:
    """``distance_m_<type>`` and ``population_with_access_<type>`` by lot and by block.

    ``distances`` is ``nearest_distances`` of ``lots``. A block's distance is its
    lots' average weighted by population, over lots with a distance; its residents
    with access are those of its lots. Blocks without lots stay null.
    """
    population = pd.to_numeric(lots["population"]).astype("float64")
    by_lot, weights = {}, {}
    for poi_type, values in distances.items():
        distance = pd.Series(values, index=lots.index, dtype="float64")
        reached = population.where(distance.notna())
        by_lot[f"distance_m_{poi_type}"] = distance
        # A lot without population stays null, not zero; beyond reach has no access.
        by_lot[f"population_with_access_{poi_type}"] = population * (
            distance <= coverage_m[poi_type]
        )
        weights[f"weighted_{poi_type}"] = reached * distance
        weights[f"reached_{poi_type}"] = reached
    lot_values = pd.DataFrame(by_lot, index=lots.index)
    sums = (
        lot_values.filter(like="population_with_access_")
        .join(pd.DataFrame(weights, index=lots.index))
        .groupby(lots["parent_id"])
        .sum(min_count=1)
        .reindex(blocks["id"])
    )
    by_block = {}
    for poi_type in distances:
        by_block[f"distance_m_{poi_type}"] = (
            sums[f"weighted_{poi_type}"] / sums[f"reached_{poi_type}"]
        ).to_numpy()
        by_block[f"population_with_access_{poi_type}"] = sums[
            f"population_with_access_{poi_type}"
        ].to_numpy()
    return {"lots": lot_values, "blocks": pd.DataFrame(by_block, index=blocks.index)}
