"""What the streets around each block are like: bike lane coverage
(uli_mexicali_calculations_notebooks, ``inidcadores_finales_export.ipynb``,
``calculate_cycling_street_proportion``), street lighting coverage and street
connectivity.

Streets are ``right_of_way`` elements (polygons, lot front to lot front; older
baselines, lines), with their make-up as attributes (urban_fabric.md). A street's
length within a block's neighbourhood is ``street_fit.strip_length`` of its polygon
clipped to it (a line's, its length):

- ``bike_lane_coverage``: share of street length with a bike lane band
  (``bands.use`` has ``shared_lanes`` or ``protected_bike_lane``);
  ``protected_bike_lane_coverage`` counts the protected ones alone.
- ``street_lighting_coverage``: share of street length that is lit, each street's
  ``streetlights`` (0-100, share of its length lit) weighted by its length. Streets
  without the value are left out.

The notebook measures the streets inside each polygon; a block has none inside
(they run around it), so a block measures the streets within ``NEIGHBOURHOOD_M`` of
it. All are 0-1, null without streets (with a value) around the block.

- ``intersection_density`` (ULI "Street connectivity", over the road network with
  pandana): intersections (nodes joining three or more streets) within
  ``CONNECTIVITY_M`` along the network of the node nearest the block's
  representative point, per km² (GHSCI's walkable neighbourhood of 1 km). Null
  without block geometry.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from reimagina_test_port.ingestion.street_fit import strip_length

if TYPE_CHECKING:
    import pandana as pdna

NEIGHBOURHOOD_M = 500.0
BIKE_BANDS = ("shared_lanes", "protected_bike_lane")
LANE_KEYS = ["bike_lane_coverage", "protected_bike_lane_coverage"]
COVERAGE_KEYS = [*LANE_KEYS, "street_lighting_coverage"]
KEYS = [*COVERAGE_KEYS, "intersection_density"]
CONNECTIVITY_M = 1_000
# shortcut: divides by the 1 km circle, not GHSCI's buffered network area, and does
# not consolidate intersections within 12 m; upgrade if it must match GHSCI's values.
CONNECTIVITY_KM2 = math.pi * (CONNECTIVITY_M / 1_000) ** 2


def lane_flags(bands_use: pd.Series) -> pd.DataFrame:
    """``bike_lane`` / ``protected_bike_lane``: whether each street's ``bands.use``
    has a bike lane, shared (``shared_lanes``) or protected / a protected one."""
    bands = bands_use.map(lambda value: value if isinstance(value, list) else [])
    return pd.DataFrame(
        {
            "bike_lane": bands.map(lambda b: any(use in BIKE_BANDS for use in b)),
            "protected_bike_lane": bands.map(lambda b: "protected_bike_lane" in b),
        },
        index=bands_use.index,
    ).astype(bool)


def coverage_by_block(
    blocks: gpd.GeoDataFrame, streets: gpd.GeoDataFrame, crs: str
) -> pd.DataFrame:
    """``COVERAGE_KEYS`` per block, in order; lengths in ``crs``."""
    shapes = streets.geometry.to_crs(crs).to_numpy()
    # Reprojected polygons may come out slightly invalid.
    shapes = np.where(shapely.is_valid(shapes), shapes, shapely.make_valid(shapes))
    areas = shapely.buffer(blocks.geometry.to_crs(crs).to_numpy(), NEIGHBOURHOOD_M)
    block, street = shapely.STRtree(shapes).query(areas, predicate="intersects")
    length = strip_length(shapely.intersection(shapes[street], areas[block]))
    lit = pd.to_numeric(streets["streetlights"]).to_numpy("float64")[street] / 100

    def per_block(weights: np.ndarray) -> np.ndarray:
        return np.bincount(block, weights, minlength=len(blocks))

    total = per_block(length)
    result = pd.DataFrame(
        {
            key: per_block(length * streets[flag].fillna(False).to_numpy(bool)[street])
            / np.where(total > 0, total, np.nan)
            for key, flag in zip(
                LANE_KEYS, ["bike_lane", "protected_bike_lane"], strict=True
            )
        }
    )
    measured = per_block(np.where(np.isnan(lit), 0.0, length))
    result["street_lighting_coverage"] = per_block(
        np.where(np.isnan(lit), 0.0, length * lit)
    ) / np.where(measured > 0, measured, np.nan)
    return result


def intersections(edges: pd.DataFrame) -> pd.Index:
    """Ids of the nodes with three or more distinct neighbours."""
    pairs = pd.concat(
        [
            edges[["from", "to"]].set_axis(["node", "other"], axis=1),
            edges[["to", "from"]].set_axis(["node", "other"], axis=1),
        ]
    ).query("node != other")
    degree = pairs.drop_duplicates().groupby("node").size()
    return degree.index[degree >= 3]


def intersection_density(
    blocks: gpd.GeoDataFrame, network: pdna.Network, edges: pd.DataFrame
) -> np.ndarray:
    """Per block, in order: intersections per km² within ``CONNECTIVITY_M``."""
    # pandana (through the model) only where connectivity runs.
    from reimagina_test_port.indicators.accessibility.model import (
        check_network_coverage,
        resolve_access_crs,
    )

    result = np.full(len(blocks), np.nan)
    present = (blocks.geometry.notna() & ~blocks.geometry.is_empty).to_numpy()
    if not present.any():
        return result
    crs = resolve_access_crs(blocks)
    origins = gpd.GeoDataFrame(
        geometry=blocks.geometry[present].to_crs(crs).representative_point()
    )
    check_network_coverage(network, origins, crs)
    nodes = network.get_node_ids(origins.geometry.x, origins.geometry.y).to_numpy()
    crossings = intersections(edges)
    if not len(crossings):
        result[present] = 0.0
        return result
    network.precompute(CONNECTIVITY_M)
    network.set(pd.Series(crossings), name="intersections")
    count = network.aggregate(
        CONNECTIVITY_M, type="sum", decay="flat", name="intersections"
    ).loc[nodes]
    result[present] = count.to_numpy() / CONNECTIVITY_KM2
    return result
