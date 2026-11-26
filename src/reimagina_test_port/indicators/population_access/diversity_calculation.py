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
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from reimagina_test_port.indicators.population_access.access_calculation import (
    _MIN_WEIGHT,
    StreetNetwork,
    entries,
    lot_entries,
)

# POIs per Dijkstra call; each returns a dense row over every node.
CHUNK = 256


def reachable(
    origins: pd.DataFrame,
    destinations: pd.DataFrame,
    network: StreetNetwork,
    radius: float,
) -> Iterator[pd.DataFrame]:
    """Pairs of ``origin`` and ``destination`` owners whose walk is within
    ``radius``, for ``CHUNK`` destinations at a time: one Dijkstra per destination,
    up to ``radius``, from a node of its own joined to its entries' nodes."""
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
    # A POI on the same segment is reached along it, without going to a node.
    same = origins.merge(destinations, on="segment", suffixes=("", "_poi"))
    direct = (
        (same["along_start"] - same["along_start_poi"]).abs()
        + same["offset"]
        + same["offset_poi"]
    )
    same = same.loc[direct <= radius, ["owner", "owner_poi"]].set_axis(
        ["origin", "destination"], axis=1
    )
    for first in range(0, len(owners), CHUNK):
        sources = network.nodes + np.arange(first, min(first + CHUNK, len(owners)))
        walk = dijkstra(graph, directed=True, indices=sources, limit=radius)
        row, node = np.nonzero(np.isfinite(walk[:, : network.nodes]))
        near = pd.DataFrame(
            {
                "destination": owners[first + row],
                "node": node,
                "walk": walk[row, node],
            }
        ).merge(ends, on="node")
        chunk = owners[first : first + CHUNK]
        yield pd.concat(
            [
                near.loc[
                    near["walk"] + near["cost"] <= radius, ["origin", "destination"]
                ],
                same[same["destination"].isin(chunk)],
            ]
        ).drop_duplicates(ignore_index=True)


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
    kind, kinds = pd.factorize(np.asarray(types))
    table = pd.DataFrame(np.eye(len(kinds))[kind], index=pois.index)
    table["weight"] = 0.0 if weights is None else np.asarray(weights, dtype=object)
    sums, measured = reached_sums(lots, pois, table, network, crs, radius, origins)
    return sums[:, :-1], sums[:, -1], measured


def reached_sums(
    lots: gpd.GeoDataFrame,
    pois: gpd.GeoSeries,
    weights: pd.DataFrame,
    network: StreetNetwork,
    crs: str,
    radius: float,
    origins: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Each column of ``weights`` (one row per POI, in order) added up over the
    POIs each lot reaches within ``radius``: ``(sums, measured)``, ``sums`` lots x
    columns (missing weights count 0) and ``measured`` the lots with an entry.
    Rows at the same place are walked once. ``origins`` is ``lot_entries`` of
    ``lots``, built here when omitted."""
    if origins is None:
        origins = lot_entries(lots, network, crs)
    measured = np.zeros(len(lots), dtype=bool)
    measured[origins["owner"].unique()] = True
    sums = np.zeros((len(lots), weights.shape[1]))
    if len(pois) == 0:
        return sums, measured
    place, places = pd.factorize(pois.to_wkb())
    values = weights.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    rows = pd.DataFrame({"destination": place, "row": np.arange(len(place))})
    # Weights reached by lot, added up a chunk of places at a time.
    for pairs in reachable(
        origins,
        entries(gpd.GeoSeries.from_wkb(places, crs=pois.crs).to_crs(crs), network),
        network,
        radius,
    ):
        found = pairs.merge(rows, on="destination")
        origin = found["origin"].to_numpy()
        reached_values = values[found["row"].to_numpy()]
        for j in range(sums.shape[1]):
            sums[:, j] += np.bincount(origin, reached_values[:, j], minlength=len(lots))
    return sums, measured


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
    return weighted_by(lots, blocks, values, "population")


def weighted_by(
    lots: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame, values: np.ndarray, weight: str
) -> dict[str, np.ndarray]:
    """By lot and by block, in order: a block averages its lots' ``values``
    weighted by their ``weight`` column, over lots with a value; blocks whose
    lots weigh nothing stay null."""
    weights = pd.to_numeric(lots[weight]).astype("float64")
    by_lot = pd.Series(values, index=lots.index)
    weighted = (by_lot * weights).groupby(lots["parent_id"]).sum()
    total = weights.where(by_lot.notna()).groupby(lots["parent_id"]).sum()
    by_block = (weighted / total.where(total > 0)).reindex(blocks["id"])
    return {"lots": by_lot.to_numpy(), "blocks": by_block.to_numpy()}
