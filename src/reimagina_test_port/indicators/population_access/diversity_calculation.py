"""Diversity of a type's POIs within walking distance of each lot (ULI "Diversidad
de <tipo> a 500 m"), as GHSCI's ``normalised_entropy``: the Shannon entropy of the
subcategories of the POIs reached, over its maximum,

    D = -sum(p_i ln p_i) / ln k,

where p_i is the share of the POIs reached that are of subcategory i and k is how
many subcategories the type has. A lot reaching nothing, or only one subcategory,
scores 0; one reaching as many POIs of each of the k scores 1. A POI counts once,
however many entries it has.

A POI is reached when the walk to it is within the type's radius, walked as in
``access_calculation``: from one of the lot's entries to one of the POI's, over the
street network. A block averages its lots weighted by population.
"""

from __future__ import annotations

from collections.abc import Mapping

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
) -> pd.DataFrame:
    """Pairs of ``origin`` and ``destination`` owners whose walk is within
    ``radius``: one Dijkstra per destination, up to ``radius``, from a node of its
    own joined to its entries' nodes."""
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
    ends = ends[ends["cost"] <= radius]
    pairs = []
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
        pairs.append(
            near.loc[near["walk"] + near["cost"] <= radius, ["origin", "destination"]]
        )
    # A POI on the same segment is reached along it, without going to a node.
    same = origins.merge(destinations, on="segment", suffixes=("", "_poi"))
    direct = (
        (same["along_start"] - same["along_start_poi"]).abs()
        + same["offset"]
        + same["offset_poi"]
    )
    pairs.append(
        same.loc[direct <= radius, ["owner", "owner_poi"]].set_axis(
            ["origin", "destination"], axis=1
        )
    )
    return pd.concat(pairs).drop_duplicates(ignore_index=True)


def poi_diversity(
    lots: gpd.GeoDataFrame,
    pois: Mapping[str, gpd.GeoDataFrame],
    network: StreetNetwork,
    crs: str,
    params: Mapping[str, tuple[float, int]],
) -> dict[str, np.ndarray]:
    """``{poi_type: D}`` per lot, in order; NaN without geometry. ``pois`` holds each
    type's POIs with their ``subcategory``, ``params`` its radius in metres and how
    many subcategories it has. The lots' entries are built once."""
    origins = entries(
        gpd.GeoSeries(lots.geometry.to_numpy(), crs=lots.crs).to_crs(crs), network
    )
    located = np.full(len(lots), np.nan)
    located[origins["owner"].unique()] = 0.0
    result = {}
    for poi_type, frame in pois.items():
        radius, count = params[poi_type]
        values = located.copy()
        result[poi_type] = values
        if frame.empty or count < 2:
            continue
        pairs = reachable(
            origins, entries(frame.geometry.to_crs(crs), network), network, radius
        )
        if pairs.empty:
            continue
        subcategory = frame["subcategory"].to_numpy()[pairs["destination"].to_numpy()]
        reached = pd.crosstab(pairs["origin"].to_numpy(), subcategory)
        share = reached.div(reached.sum(axis=1), axis=0)
        # As sum(p ln 1/p): a negated sum of zeros would give -0.0.
        entropy = (share * np.log(1 / share.where(share > 0))).sum(axis=1)
        values[entropy.index.to_numpy()] = entropy.to_numpy() / np.log(count)
    return result


def diversity(
    lots: gpd.GeoDataFrame,
    blocks: gpd.GeoDataFrame,
    values: Mapping[str, np.ndarray],
) -> dict[str, pd.DataFrame]:
    """``diversity_<type>`` by lot and by block. ``values`` is ``poi_diversity`` of
    ``lots``. A block averages its lots weighted by population, over lots with a
    value; blocks without residents stay null."""
    population = pd.to_numeric(lots["population"]).astype("float64")
    by_lot = pd.DataFrame(
        {f"diversity_{poi_type}": value for poi_type, value in values.items()},
        index=lots.index,
    )
    weights = by_lot.notna().mul(population, axis=0)
    weighted = by_lot.mul(population, axis=0).groupby(lots["parent_id"]).sum()
    by_block = weighted / weights.groupby(lots["parent_id"]).sum()
    return {
        "lots": by_lot,
        "blocks": by_block.reindex(blocks["id"]).set_axis(blocks.index),
    }
