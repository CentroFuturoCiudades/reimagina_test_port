"""Diversity of the POIs within walking distance of each lot, by poi_type: the
Shannon entropy of the types of the POIs reached, over its maximum,

    D = -sum(p_i ln p_i) / ln k,

where p_i is the share of the POIs reached that are of type i and k is how many
types are counted. A lot reaching nothing, or only one type, scores 0; one reaching
as many POIs of each of the k types scores 1. A POI of several types counts once in
each, however many entries it has.

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


def poi_diversity(
    lots: gpd.GeoDataFrame,
    pois: gpd.GeoSeries,
    types: Sequence[str],
    network: StreetNetwork,
    crs: str,
    radius: float,
    k: int,
) -> np.ndarray:
    """D per lot, in order; NaN without geometry. ``pois`` has one row per POI and
    type, ``types`` its poi_type; ``radius`` is in metres and ``k`` is how many
    types are counted. Rows at the same place are walked once."""
    origins = entries(
        gpd.GeoSeries(lots.geometry.to_numpy(), crs=lots.crs).to_crs(crs), network
    )
    values = np.full(len(lots), np.nan)
    values[origins["owner"].unique()] = 0.0
    if pois.empty or k < 2:
        return values
    place, places = pd.factorize(pois.to_wkb())
    kind, kinds = pd.factorize(np.asarray(types))
    rows = pd.DataFrame({"destination": place, "kind": kind})
    # POIs reached by lot and type, added up a chunk of places at a time.
    counts = np.zeros((len(lots), len(kinds)))
    for pairs in reachable(
        origins,
        entries(gpd.GeoSeries.from_wkb(places, crs=pois.crs).to_crs(crs), network),
        network,
        radius,
    ):
        found = pairs.merge(rows, on="destination")
        counts += np.bincount(
            found["origin"].to_numpy() * len(kinds) + found["kind"].to_numpy(),
            minlength=counts.size,
        ).reshape(counts.shape)
    total = counts.sum(axis=1)
    reached = total > 0
    share = counts[reached] / total[reached, None]
    # As sum(p ln 1/p), 0 where p = 0: a negated sum of zeros would give -0.0.
    surprise = np.log(np.divide(1, share, where=share > 0, out=np.ones_like(share)))
    values[reached] = (share * surprise).sum(axis=1) / np.log(k)
    return values


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
