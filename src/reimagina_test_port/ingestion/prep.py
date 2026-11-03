"""Blocks from census and INEGI geometry; lots from cadastre, normativa and blocks.

Moved from exe_reimagina ``src/data_prep`` (``blocks.py``, ``lots.py`` and the
assignment rules of ``normativa.py``).
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BLOCK_CENSUS_COLUMNS = [
    "pobtot", "vivtot", "prom_ocup", "graproes", "vivpar_hab", "tvivpar", "tvivparhab",
    "vph_autom", "vph_pc", "vph_tinaco", "p_0a2", "p_3a5", "p_6a11", "p_12a14",
    "p_15a17", "p_18a24", "p_25a59", "p_60ymas", "puntuaje_hogar_digno",
]  # fmt: skip
BLOCK_OUTPUT_COLUMNS = ["cvegeo", *BLOCK_CENSUS_COLUMNS, "geometry"]
# Census variables repeated on every lot of a block.
BLOCK_LOT_COLUMNS = ["cvegeo", "pobtot", "vivtot", "prom_ocup", "tvivparhab"]
# p_25a59 is derived; puntuaje_hogar_digno has no source yet.
CENSUS_SOURCE_COLUMNS = [
    "loc", "ageb", "mza", "p_18ymas",
    *[c for c in BLOCK_CENSUS_COLUMNS if c not in {"p_25a59", "puntuaje_hogar_digno"}],
]  # fmt: skip
LOT_OUTPUT_COLUMNS = [
    "lot_id", "cvegeo", "lot_area", "zoning", "max_cos", "max_cus", "max_density",
    "pobtot", "vivtot", "prom_ocup", "tvivparhab", "Giro", "geometry",
]  # fmt: skip
NORMATIVA_LIMIT_COLUMNS = ["max_cos", "max_cus", "max_density"]
NORMATIVA_OUTPUT_COLUMNS = ["zoning", *NORMATIVA_LIMIT_COLUMNS]


# ------------------------------------------------------------------------ blocks


def prepare_census(census: pd.DataFrame, ent_code: str, muni_code: str) -> pd.DataFrame:
    census = census.copy()
    census["loc"] = census["loc"].astype(str).str.zfill(4)
    census["ageb"] = census["ageb"].astype(str).str.zfill(4)
    census["mza"] = census["mza"].astype(str).str.zfill(3)
    census["cvegeo"] = (
        ent_code + muni_code + census["loc"] + census["ageb"] + census["mza"]
    )
    census["p_25a59"] = census["p_18ymas"] - census["p_18a24"] - census["p_60ymas"]
    # prom_ocup stays decimal: it is the modelled population factor.
    census["graproes"] = census["graproes"].round()
    census["cvegeo"] = census["cvegeo"].astype(str).str.lower().str.strip()
    return census


def prepare_blocks(
    census: pd.DataFrame, blocks: pd.DataFrame, *, ent_code: str, muni_code: str
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Join census to block geometry; return blocks (EPSG:4326) and lot columns."""
    census = prepare_census(census, ent_code, muni_code)
    geometry = gpd.GeoSeries.from_wkb(blocks["geometry"], crs="EPSG:4326")
    frame = blocks.drop(columns="geometry").rename(columns={"CVEGEO": "cvegeo"})
    frame["cvegeo"] = frame["cvegeo"].astype(str).str.lower().str.strip()
    merged = gpd.GeoDataFrame(frame, geometry=geometry, crs="EPSG:4326").merge(
        census, on="cvegeo", how="left"
    )
    for column in BLOCK_OUTPUT_COLUMNS:
        if column not in merged:
            merged[column] = np.nan
    merged = gpd.GeoDataFrame(
        merged[BLOCK_OUTPUT_COLUMNS], geometry="geometry", crs="EPSG:4326"
    )
    logger.info("Blocks: %s", f"{len(merged):,}")
    return merged, merged[BLOCK_LOT_COLUMNS].copy()


# --------------------------------------------------------------------- normativa


def prepare_normativa(
    normativa: gpd.GeoDataFrame, *, target_crs: str, columns: dict[str, str]
) -> gpd.GeoDataFrame:
    """Reproject, rename to the internal vocabulary and make limits numeric."""
    missing = [source for source in columns if source not in normativa.columns]
    if missing:
        msg = f"Normativa lacks the configured columns: {missing}"
        raise ValueError(msg)
    if "zoning" not in columns.values():
        msg = "The normativa zoning column must be configured."
        raise ValueError(msg)
    normativa = normativa.to_crs(target_crs).rename(columns=columns)
    keep = [n for n in NORMATIVA_OUTPUT_COLUMNS if n in normativa.columns]
    normativa = normativa[[*keep, "geometry"]].reset_index(drop=True)
    for name in NORMATIVA_LIMIT_COLUMNS:
        if name in normativa:
            normativa[name] = pd.to_numeric(normativa[name], errors="coerce")
    score = pd.Series(0, index=normativa.index, dtype=int)
    for name in keep:
        score += normativa[name].notna().astype(int)
    normativa["_norm_rank"] = score
    normativa["_norm_order"] = np.arange(len(normativa))
    return normativa


def _best_match(joined: gpd.GeoDataFrame, lot_index: pd.Index) -> gpd.GeoDataFrame:
    """One row per lot: most complete polygon, then nearest, then first in source."""
    keys, ascending = ["_norm_rank"], [False]
    if "_dist" in joined:
        keys.append("_dist")
        ascending.append(True)
    keys.append("_norm_order")
    ascending.append(True)
    joined = joined.sort_values(keys, ascending=ascending, kind="mergesort")
    joined = joined[~joined.index.duplicated(keep="first")]
    return joined.reindex(lot_index)


def assign_normativa(
    lots: gpd.GeoDataFrame, normativa: gpd.GeoDataFrame, *, max_distance: float
) -> gpd.GeoDataFrame:
    """Normativa containing the lot's representative point, else the nearest one."""
    if not lots.index.is_unique:
        msg = "Lot index must be unique."
        raise ValueError(msg)
    value_columns = [
        c
        for c in normativa.columns
        if c not in ("geometry", "_norm_rank", "_norm_order")
    ]
    points = gpd.GeoDataFrame(
        {"lot_id": lots["lot_id"].to_numpy()},
        geometry=lots.geometry.representative_point().to_numpy(),
        crs=lots.crs,
        index=lots.index,
    )
    within = _best_match(
        points.sjoin(normativa, how="left", predicate="within"), lots.index
    )
    matched = within["index_right"].notna()
    unmatched = lots.index[~matched]
    nearest_found = 0
    if len(unmatched) and max_distance > 0:
        nearest = points.loc[unmatched].sjoin_nearest(
            normativa, how="left", max_distance=max_distance, distance_col="_dist"
        )
        nearest = _best_match(nearest, unmatched)
        found = nearest.index[nearest["index_right"].notna()]
        within.loc[found, value_columns] = nearest.loc[found, value_columns]
        nearest_found = len(found)
    logger.info(
        "Normativa: %s by containment, %s nearest within %s m, %s without",
        f"{int(matched.sum()):,}",
        f"{nearest_found:,}",
        max_distance,
        f"{len(unmatched) - nearest_found:,}",
    )
    lots = lots.drop(columns=value_columns, errors="ignore").copy()
    for column in value_columns:
        lots[column] = within[column].to_numpy()
    for column in NORMATIVA_OUTPUT_COLUMNS:
        if column not in lots:
            lots[column] = np.nan
    return lots


# -------------------------------------------------------------------------- lots


def _nearest_block(
    lots: gpd.GeoDataFrame, block_points: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    joined = lots.drop(
        columns=["index_left", "index_right"], errors="ignore"
    ).sjoin_nearest(block_points, how="left", distance_col="dist_block")
    # Deterministic ties: at equal distance the lower block index wins.
    joined = joined.sort_values(["dist_block", "index_right"], kind="mergesort")
    joined = joined[~joined["lot_id"].duplicated(keep="first")]
    return joined.drop(columns=["dist_block", "index_right"], errors="ignore")


def prepare_lots(  # noqa: PLR0913
    lots: gpd.GeoDataFrame,
    normativa: gpd.GeoDataFrame,
    blocks: gpd.GeoDataFrame,
    block_export: pd.DataFrame,
    *,
    target_crs: str,
    normativa_columns: dict[str, str],
    normativa_max_distance: float,
) -> gpd.GeoDataFrame:
    """Lots in ``target_crs`` with area, normativa, nearest block and its census."""
    source_columns = [c for c in ("Giro", "geometry") if c in lots]
    lots = lots[source_columns].to_crs(target_crs).copy()
    lots["lot_area"] = lots.geometry.area
    lots["lot_id"] = lots.index.astype(str)
    normativa = prepare_normativa(
        normativa, target_crs=target_crs, columns=normativa_columns
    )
    lots = assign_normativa(lots, normativa, max_distance=normativa_max_distance)

    block_points = blocks.to_crs(target_crs)[["geometry", "cvegeo"]].copy()
    block_points["geometry"] = block_points.geometry.representative_point()
    lots = _nearest_block(lots.drop(columns=["cvegeo"], errors="ignore"), block_points)
    # Back to source order so the output is deterministic.
    lots = lots.sort_index().merge(block_export, on="cvegeo", how="left")
    for column in LOT_OUTPUT_COLUMNS:
        if column not in lots:
            lots[column] = np.nan
    lots = gpd.GeoDataFrame(
        lots[LOT_OUTPUT_COLUMNS], geometry="geometry", crs=target_crs
    )
    logger.info("Lots: %s", f"{len(lots):,}")
    return lots
