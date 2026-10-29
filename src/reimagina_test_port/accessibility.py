"""Walking accessibility per block, inherited by its lots.

Ported from reimagina-lyra ``accessibility.py`` (exe_reimagina's model job), reading
tables instead of the store. Blocks are the origins, weighted by this scenario's
modelled population and homes; destinations are elements with an ``amenity_type``.
The CFC model (``accessibility_metrics``) is copied unchanged.

Per block: ``minutes`` / ``farthest_amenity`` (slowest reachable amenity type),
``accessibility_<amenity>`` (gravity score as a 0-1 percentile rank among the run's
blocks) and ``distance_to_<amenity>`` (walking minutes to the nearest one). A missing
value means none is reachable within ``MAX_DISTANCE``; it is never written as 0.
"""

import logging
import math
import os
from typing import Any

import geopandas as gpd
import pandana as pdna
import pandas as pd

from reimagina_test_port.accessibility_fastpath import apply_fastpath
from reimagina_test_port.accessibility_metrics import (
    compute_pois_accessibility_scores,
    create_origin_destination_mappings,
)
from reimagina_test_port.vocabulary import (
    AMENITY_ACCESSIBILITY_PARAMS,
    AMENITY_TYPES,
    normalize_amenity_type,
)

logger = logging.getLogger(__name__)

# Vectorized get_accessibility_metrics; ACCESSIBILITY_FASTPATH=0 turns it off.
apply_fastpath()

POB_COL = "population"
VIV_COL = "num_housing_units"
ADJUSTMENT_FACTOR = 3
MAX_DISTANCE = 10_000  # m
NUM_POIS = 5
WALK_SPEED_KMH = 4.8  # 80 m/min
DEFAULT_POI_CAPACITY = 1
MIN_NETWORK_COVERAGE = 0.5
NETWORK_BBOX_MARGIN_M = 1_000

# Radii capped to MAX_DISTANCE, as in reimagina-lyra.
POIS_CONFIG = {
    amenity: {
        **params,
        "radius": min(params["radius"], MAX_DISTANCE),
        "POB_COL": POB_COL,
        "VIV_COL": VIV_COL,
    }
    for amenity, params in AMENITY_ACCESSIBILITY_PARAMS.items()
}

SCORE_KEYS = [f"accessibility_{amenity}" for amenity in AMENITY_TYPES]
DISTANCE_KEYS = [f"distance_to_{amenity}" for amenity in AMENITY_TYPES]
KEYS = ["minutes", "farthest_amenity", *SCORE_KEYS, *DISTANCE_KEYS]


def build_network(nodes: pd.DataFrame, edges: pd.DataFrame) -> pdna.Network:
    """Pandana network from node (id, x, y) and edge (from, to, distance) tables."""
    if nodes.empty or edges.empty:
        msg = "The road network has no nodes or edges."
        raise ValueError(msg)
    node_index = nodes["id"].astype("int64")
    return pdna.Network(
        pd.Series(nodes["x"].to_numpy(), index=node_index),
        pd.Series(nodes["y"].to_numpy(), index=node_index),
        edges["from"].astype("int64"),
        edges["to"].astype("int64"),
        edges[["distance"]],
        twoway=bool(edges["two_way"].iloc[0]),
    )


def resolve_access_crs(blocks: gpd.GeoDataFrame) -> str:
    """ACCESS_CRS if set, else the blocks' UTM zone (the network's CRS)."""
    if crs := os.getenv("ACCESS_CRS"):
        return crs
    return blocks.estimate_utm_crs().to_string()


def check_network_coverage(
    network: pdna.Network, origins: gpd.GeoDataFrame, crs: str
) -> None:
    """Fail when blocks fall outside the network: its CRS is not ``crs``."""
    nodes = network.nodes_df
    margin = NETWORK_BBOX_MARGIN_M
    points = origins.geometry.centroid
    inside = points.x.between(
        nodes["x"].min() - margin, nodes["x"].max() + margin
    ) & points.y.between(nodes["y"].min() - margin, nodes["y"].max() + margin)
    coverage = float(inside.mean())
    if coverage < MIN_NETWORK_COVERAGE:
        msg = (
            f"Only {coverage:.1%} of blocks fall within the road network in {crs}; "
            "set ACCESS_CRS to the network's CRS."
        )
        raise ValueError(msg)


def _category(value: Any) -> str | None:
    try:
        return normalize_amenity_type(value)
    except ValueError:
        return None


def load_pois(pois: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    """Destinations in the model's contract: index id, category, capacity, centroid."""
    categories = pois["amenity_type"].map(_category)
    unknown = sorted(set(pois.loc[categories.isna(), "amenity_type"].dropna()))
    if unknown:
        logger.warning("Unknown amenity_type, POIs ignored: %s", unknown)
    keep = categories.isin(POIS_CONFIG)
    selected = pois[keep].reset_index(drop=True)
    # Missing, zero or non-finite capacity counts as one, as `to_number(v) or 1`.
    capacity = [
        value
        if isinstance(value, int | float) and math.isfinite(value) and value
        else DEFAULT_POI_CAPACITY
        for value in selected["capacity"]
    ]
    result = gpd.GeoDataFrame(
        {
            "id": selected["id"].astype(str),
            "category": categories[keep].reset_index(drop=True),
            "capacity": capacity,
        },
        geometry=selected.geometry.to_crs(crs).to_numpy(),
        crs=crs,
    ).set_index("id")
    # Polygons: the destination is the centroid, in metres.
    result["geometry"] = result.geometry.centroid
    return result


def load_origins(blocks: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    """Every block with geometry; no modelled demand is weight 0, not excluded."""
    origins = blocks.loc[blocks.geometry.notna() & ~blocks.geometry.is_empty]
    return gpd.GeoDataFrame(
        {
            "id": origins["id"].astype(str).to_numpy(),
            POB_COL: pd.to_numeric(origins[POB_COL]).fillna(0.0).to_numpy(),
            VIV_COL: pd.to_numeric(origins[VIV_COL]).fillna(0.0).to_numpy(),
        },
        geometry=origins.geometry.to_crs(crs).to_numpy(),
        crs=crs,
    )


def worst_amenity_by_origin(scores: pd.DataFrame) -> pd.DataFrame:
    """``minutes`` / ``farthest_amenity``: the slowest reachable category."""
    times = scores.reset_index()[["origin_id", "time", "poi_category"]].dropna(
        subset=["time"]
    )
    if times.empty:
        return pd.DataFrame(columns=["time", "poi_category"])
    worst = times.loc[times.groupby("origin_id")["time"].idxmax()]
    return worst.set_index("origin_id")[["time", "poi_category"]]


def category_metrics_by_origin(scores: pd.DataFrame) -> dict[Any, dict[str, Any]]:
    """Percentile-ranked score and minutes to the nearest POI, per category."""
    rows = scores.reset_index()[
        ["origin_id", "poi_category", "accessibility_score", "time"]
    ].copy()
    rows["accessibility_score"] = rows.groupby("poi_category")[
        "accessibility_score"
    ].rank(pct=True)
    metrics: dict[Any, dict[str, Any]] = {}
    for row in rows.itertuples(index=False):
        entry = metrics.setdefault(row.origin_id, {})
        if pd.notna(row.accessibility_score):
            entry[f"accessibility_{row.poi_category}"] = float(row.accessibility_score)
        if pd.notna(row.time):
            entry[f"distance_to_{row.poi_category}"] = float(row.time)
    return metrics


def accessibility_by_block(
    blocks: gpd.GeoDataFrame,
    pois: gpd.GeoDataFrame,
    network: pdna.Network,
) -> dict[str, dict[str, Any]]:
    """``{block_id: {key: value}}``; blocks reaching nothing are absent."""
    if blocks.empty or pois.empty:
        return {}
    crs = resolve_access_crs(blocks)
    origins = load_origins(blocks, crs)
    if origins.empty:
        return {}
    check_network_coverage(network, origins, crs)
    if not origins[POB_COL].gt(0).any():
        # No demand: the gravity model drops every destination.
        logger.warning("No block has %s > 0; no accessibility.", POB_COL)
        return {}
    destinations = load_pois(pois, crs)
    df_accessibility, _, id_to_origin_map = create_origin_destination_mappings(
        network, origins, [POB_COL, VIV_COL]
    )
    # A category without POIs breaks pandana's kd-tree; absent means unreachable.
    available = set(destinations["category"])
    active = {name: p for name, p in POIS_CONFIG.items() if name in available}
    if not active:
        return {}
    scores = compute_pois_accessibility_scores(
        POIS_CONFIG=active,
        df_pois=destinations,
        net=network,
        df_accessibility=df_accessibility,
        weight_col="POB_COL",
        max_distance=MAX_DISTANCE,
        num_pois=NUM_POIS,
        adjustment_factor=ADJUSTMENT_FACTOR,
        travel_speed=WALK_SPEED_KMH,
        include_poi_snap_distance=True,
    )
    worst = worst_amenity_by_origin(scores)
    per_category = category_metrics_by_origin(scores)
    by_block: dict[str, dict[str, Any]] = {}
    for block_id, origin_id in id_to_origin_map.items():
        entry: dict[str, Any] = {}
        if origin_id in worst.index and pd.notna(worst.loc[origin_id, "time"]):
            entry["minutes"] = float(worst.loc[origin_id, "time"])
            entry["farthest_amenity"] = worst.loc[origin_id, "poi_category"]
        entry.update(per_category.get(origin_id, {}))
        if entry:
            by_block[str(block_id)] = entry
    return by_block


def inherit_to_lots(
    lots: pd.DataFrame, by_block: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Each lot gets its block's values; orphan lots get none."""
    by_lot = {}
    for lot_id, parent_id in zip(lots["id"], lots["parent_id"], strict=True):
        if parent_id is None or (
            isinstance(parent_id, float) and math.isnan(parent_id)
        ):
            continue
        if entry := by_block.get(str(parent_id)):
            by_lot[str(lot_id)] = dict(entry)
    return by_lot
