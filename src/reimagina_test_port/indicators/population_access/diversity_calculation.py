"""Diversity of the POIs within walking distance of each lot: the Shannon entropy of
the subcategories of the POIs reached, over its maximum,

    D = -sum(p_i ln p_i) / ln k,

where p_i is the share of the POIs reached that are of subcategory i and k is how
many subcategories the set has, fixed. A lot reaching nothing, or only one
subcategory, scores 0; one reaching as many POIs of each of the k scores 1. A POI
counts once, however many entries it has.

The same walk adds up a weight of the POIs reached, such as an establishment's
workers (reachable jobs).

A POI is reached when the walk to it is within the radius, walked as in
``access_calculation``: from one of the lot's entries to one of the POI's, over the
street network. A block averages its lots weighted by population.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra

from reimagina_test_port.indicators.population_access.access_calculation import (
    _MIN_WEIGHT,
    StreetNetwork,
    entries,
    lot_entries,
)

# POIs per Dijkstra call; each returns a dense row over every node.
CHUNK = 256
# Side of the grid cells that group the POIs of a chunk.
CELL_M = 1_000.0


def reachable(
    origins: pd.DataFrame,
    destinations: pd.DataFrame,
    network: StreetNetwork,
    radius: float,
) -> Iterator[pd.DataFrame]:
    """Pairs of ``origin`` and ``destination`` owners whose walk is within
    ``radius``, for ``CHUNK`` destinations at a time: one Dijkstra per destination,
    up to ``radius``, from a node of its own joined to its entries' nodes."""
    for origin, destination in _pairs(origins, destinations, network, radius):
        yield pd.DataFrame(
            {"origin": origin, "destination": destination}
        ).drop_duplicates(ignore_index=True)


def _pairs(
    origins: pd.DataFrame,
    destinations: pd.DataFrame,
    network: StreetNetwork,
    radius: float,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """``reachable``'s pairs as (origin, destination) arrays, possibly repeated:
    a pair joined through several nodes comes once per node."""
    owners = np.unique(destinations["owner"])
    source = network.nodes + np.searchsorted(owners, destinations["owner"])
    segment = destinations["segment"].to_numpy()
    seeds = (
        pd.DataFrame(
            {
                "source": np.tile(source, 2),
                "node": np.concatenate([network.start[segment], network.end[segment]]),
                "cost": np.concatenate(
                    [
                        destinations["along_start"] + destinations["offset"],
                        destinations["along_end"] + destinations["offset"],
                    ]
                ),
            }
        )
        .groupby(["source", "node"], as_index=False)["cost"]
        .min()
    )
    # Each destination's first entry: where it is, to chunk them by place.
    located = segment[np.unique(source, return_index=True)[1]]
    size = network.nodes + len(owners)
    graph = coo_matrix(
        (
            np.maximum(
                np.concatenate([network.length, network.length, seeds["cost"]]),
                _MIN_WEIGHT,
            ),
            (
                np.concatenate([network.start, network.end, seeds["source"]]),
                np.concatenate([network.end, network.start, seeds["node"]]),
            ),
        ),
        shape=(size, size),
    ).tocsr()
    # Each origin entry walks to the two nodes of its segment.
    segment = origins["segment"].to_numpy()
    ends = pd.DataFrame(
        {
            "node": np.concatenate([network.start[segment], network.end[segment]]),
            "origin": np.tile(origins["owner"].to_numpy(), 2),
            "cost": np.concatenate(
                [
                    origins["along_start"] + origins["offset"],
                    origins["along_end"] + origins["offset"],
                ]
            ),
        }
    )
    ends = (
        ends[ends["cost"] <= radius]
        .groupby(["node", "origin"], as_index=False)["cost"]
        .min()
    )
    # Sorted by node: the ends of node n are rows first[n]:first[n + 1].
    end_origin = ends["origin"].to_numpy()
    end_cost = ends["cost"].to_numpy()
    first_end = np.searchsorted(ends["node"].to_numpy(), np.arange(network.nodes + 1))
    # A POI on the same segment is reached along it, without going to a node.
    same = origins.merge(destinations, on="segment", suffixes=("", "_poi"))
    direct = (
        (same["along_start"] - same["along_start_poi"]).abs()
        + same["offset"]
        + same["offset_poi"]
    )
    same = same.loc[direct <= radius]
    same_origin = same["owner"].to_numpy()
    same_destination = same["owner_poi"].to_numpy()
    # Destinations near each other share a chunk, so the nodes the chunk reaches
    # are few: in grid cells of CELL_M by the first node of their first entry.
    xy = np.zeros((network.nodes, 2))
    xy[network.start] = shapely.get_coordinates(shapely.get_point(network.segments, 0))
    xy[network.end] = shapely.get_coordinates(shapely.get_point(network.segments, -1))
    cell = np.floor(xy[network.start[located]] / CELL_M)
    order = np.lexsort((cell[:, 1], cell[:, 0]))
    for first in range(0, len(owners), CHUNK):
        chunk = order[first : first + CHUNK]
        sources = network.nodes + chunk
        # Only the nodes some destination of the chunk reaches are searched: a
        # dense row over every node took most of the time.
        nearest = dijkstra(
            graph, directed=True, indices=sources, limit=radius, min_only=True
        )
        nodes = np.flatnonzero(np.isfinite(nearest[: network.nodes]))
        walk = dijkstra(graph, directed=True, indices=sources, limit=radius)[:, nodes]
        row, column = np.nonzero(np.isfinite(walk))
        node = nodes[column]
        # Each (destination, node) reached, joined to that node's ends.
        start, count = first_end[node], np.diff(first_end)[node]
        pair = np.repeat(np.arange(len(node)), count)
        end = np.arange(len(pair)) - np.repeat(np.cumsum(count) - count - start, count)
        near = walk[row[pair], column[pair]] + end_cost[end] <= radius
        direct = np.isin(same_destination, owners[chunk])
        yield (
            np.concatenate([end_origin[end[near]], same_origin[direct]]),
            np.concatenate([owners[chunk[row[pair[near]]]], same_destination[direct]]),
        )


def reached(
    lots: gpd.GeoDataFrame,
    pois: gpd.GeoSeries,
    types: Sequence[str],
    network: StreetNetwork,
    crs: str,
    radius: float,
    origins: pd.DataFrame | None = None,
    weights: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """POIs reached by each lot within ``radius``: ``(counts, totals, measured)``.
    ``counts`` is lots x subcategories (``types``, in order of first appearance),
    ``totals`` adds up the ``weights`` of the POIs reached (missing weights count
    0; zeros when omitted) and ``measured`` marks the lots with an entry. Rows at
    the same place are walked once. ``origins`` is ``lot_entries`` of ``lots``,
    built here when omitted."""
    if origins is None:
        origins = lot_entries(lots, network, crs)
    measured = np.zeros(len(lots), dtype=bool)
    measured[origins["owner"].unique()] = True
    kind, kinds = pd.factorize(np.asarray(types))
    counts = np.zeros((len(lots), len(kinds)))
    totals = np.zeros(len(lots))
    if len(pois) == 0:
        return counts, totals, measured
    place, places = pd.factorize(pois.to_wkb())
    weight = (
        np.zeros(len(place))
        if weights is None
        else pd.to_numeric(pd.Series(weights), errors="coerce").fillna(0.0).to_numpy()
    )
    # Per place, its POIs by type and their weights: one column per type, then one.
    by_place = np.zeros((len(places), len(kinds) + 1))
    np.add.at(by_place, (place, kind), 1)
    np.add.at(by_place, (place, len(kinds)), weight)
    # POIs reached by lot and type, added up a chunk of places at a time: the lots x
    # places matrix of the pairs, each once, times by_place.
    for origin, destination in _pairs(
        origins,
        entries(gpd.GeoSeries.from_wkb(places, crs=pois.crs).to_crs(crs), network),
        network,
        radius,
    ):
        hits = csr_matrix(
            (np.ones(len(origin)), (origin, destination)),
            shape=(len(lots), len(places)),
        )
        hits.data[:] = 1
        found = hits @ by_place
        counts += found[:, :-1]
        totals += found[:, -1]
    return counts, totals, measured


def entropy(counts: np.ndarray, measured: np.ndarray, k: int) -> np.ndarray:
    """D per lot from ``reached``'s counts; NaN where not measured."""
    values = np.where(measured, 0.0, np.nan)
    if counts.shape[1] < 2:
        return values
    total = counts.sum(axis=1)
    hit = total > 0
    share = counts[hit] / total[hit, None]
    # As sum(p ln 1/p), 0 where p = 0: a negated sum of zeros would give -0.0.
    surprise = np.log(np.divide(1, share, where=share > 0, out=np.ones_like(share)))
    values[hit] = (share * surprise).sum(axis=1) / np.log(k)
    return values


def poi_diversity(
    lots: gpd.GeoDataFrame,
    pois: gpd.GeoSeries,
    types: Sequence[str],
    network: StreetNetwork,
    crs: str,
    radius: float,
    k: int,
    origins: pd.DataFrame | None = None,
) -> np.ndarray:
    """D per lot, in order; NaN without geometry. ``pois`` has one row per POI,
    ``types`` its subcategory, ``radius`` is in metres and ``k`` is how many
    subcategories the set has. Rows at the same place are walked once.
    ``origins`` is ``lot_entries`` of ``lots``, built here when omitted."""
    if len(pd.unique(np.asarray(types))) < 2:
        # Nothing to walk to: every measured lot scores 0.
        pois, types = pois.iloc[:0], []
    counts, _, measured = reached(lots, pois, types, network, crs, radius, origins)
    return entropy(counts, measured, k)


def diversity(
    lots: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame, values: np.ndarray
) -> dict[str, np.ndarray]:
    """By lot and by block, in order. ``values`` is ``poi_diversity`` of ``lots``.
    A block averages its lots weighted by population, over lots with a value;
    blocks without residents stay null."""
    population = pd.to_numeric(lots["population"]).astype("float64")
    by_lot = pd.Series(values, index=lots.index)
    weighted = (by_lot * population).groupby(lots["parent_id"]).sum()
    weights = population.where(by_lot.notna()).groupby(lots["parent_id"]).sum()
    by_block = (weighted / weights).reindex(blocks["id"])
    return {"lots": by_lot.to_numpy(), "blocks": by_block.to_numpy()}
