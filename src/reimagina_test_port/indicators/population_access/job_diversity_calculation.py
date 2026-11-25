"""Job diversity of each lot and block, as the ULI's (uli_mexicali_calculations_notebooks,
``inidcadores_finales_export.ipynb``, ``add_economic_sector_diversity``): the
Shannon entropy of the categories of its places over the log of the categories
present,

    D = -sum(p_i ln p_i) / ln k,

where p_i is the share of its places of category i and k is how many categories
are there; one category or none scores 0. The notebook counts the DENUE records
inside each block's polygon by SCIAN sector. Here the places are the amenities and
establishments, by category, each in its lot (its parent, where ingestion placed
it), and a block counts its lots' places: a scenario that changes its places or
lots changes it.
"""

import numpy as np
import pandas as pd


def evenness(owners: pd.Series, categories: pd.Series) -> pd.Series:
    """D by owner, over the owners with a place; places without owner or category
    do not count."""
    counts = (
        pd.DataFrame({"owner": owners.to_numpy(), "category": categories.to_numpy()})
        .dropna()
        .groupby(["owner", "category"])
        .size()
    )
    share = counts / counts.groupby(level="owner").transform("sum")
    entropy = -(share * np.log(share)).groupby(level="owner").sum()
    present = counts.groupby(level="owner").size()
    with np.errstate(divide="ignore", invalid="ignore"):
        return (entropy / np.log(present)).where(present > 1, 0.0)


def job_diversity(
    lots: pd.DataFrame,
    blocks: pd.DataFrame,
    parents: pd.Series,
    categories: pd.Series,
) -> dict[str, np.ndarray]:
    """D by lot and by block, in order. ``parents`` is each place's lot id and
    ``categories`` its category. A lot or block without places scores 0; a block
    without lots is null."""
    lot_ids = lots["id"].astype("string")
    block_ids = blocks["id"].astype("string")
    parents = pd.Series(parents.to_numpy(), dtype="string")
    block_of = pd.Series(lots["parent_id"].astype("string").to_numpy(), index=lot_ids)
    by_lot = evenness(parents, categories).reindex(lot_ids).fillna(0.0)
    by_block = (
        evenness(parents.map(block_of), categories).reindex(block_ids).fillna(0.0)
    )
    with_lots = block_ids.isin(block_of.dropna()).to_numpy()
    return {
        "lots": by_lot.to_numpy(),
        "blocks": by_block.where(with_lots).to_numpy(),
    }
