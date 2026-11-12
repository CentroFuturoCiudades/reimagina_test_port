"""Walkability components of the ULI (uli_mexicali_calculations_notebooks,
``inidcadores_finales_export.ipynb``), per block from its lots instead of from
IMIP's land use polygons:

- ``net_residential_density``: homes per hectare of residential land, the lots that
  hold homes. A lot's homes are built form's ``num_housing_units`` (one for a
  single-family house, its built space over ``housing.unit_size_m2`` for
  multifamily and mixed use), the same homes as ``density`` and ``population``. A
  lot without them is left out. Null without residential land.
- ``commercial_land_share``: share (0-1) of the block's lot area that is
  ``commerce`` or ``office`` (the notebook's *Comercio y servicios*).
- ``land_use_mix``: entropy of the lots' land uses by area, divided by the log of
  the number of uses present (0-1; one use is 0). Null without lots with a land use.

A block's land is the sum of its lots' areas, as in built form.
"""

import numpy as np
import pandas as pd

COMMERCIAL = ("commerce", "office")
KEYS = ["net_residential_density", "commercial_land_share", "land_use_mix"]


def _number(lots: pd.DataFrame, key: str) -> pd.Series:
    return pd.to_numeric(lots[key], errors="coerce").astype("float64")


def land_use_by_block(lots: pd.DataFrame, block_ids: pd.Series) -> pd.DataFrame:
    """``KEYS`` per id of ``block_ids``, in order, from ``lots``' ``parent_id``,
    ``land_use``, ``area`` (m²) and ``num_housing_units`` (built form's). Lots
    without area are left out."""
    area = _number(lots, "area")
    frame = pd.DataFrame(
        {
            "block": lots["parent_id"],
            "land_use": lots["land_use"],
            "area": area.where(area > 0),
            "homes": _number(lots, "num_housing_units"),
        }
    ).dropna(subset=["block", "area"])
    land = frame.groupby("block")["area"].sum()
    residential = frame[frame["homes"] > 0].groupby("block")[["homes", "area"]].sum()
    residential = residential.reindex(land.index)
    by_use = (
        frame.dropna(subset=["land_use"])
        .pivot_table(index="block", columns="land_use", values="area", aggfunc="sum")
        .reindex(land.index)
    )
    share = by_use.div(by_use.sum(axis=1), axis=0)
    present = by_use.gt(0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -(share * np.log(share)).sum(axis=1)
        mix = (entropy / np.log(present)).where(present > 1, 0.0).where(present > 0)
    result = pd.DataFrame(
        {
            "net_residential_density": residential["homes"]
            / (residential["area"] / 10_000),
            "commercial_land_share": by_use.reindex(columns=list(COMMERCIAL)).sum(
                axis=1
            )
            / land,
            "land_use_mix": mix,
        }
    )
    return result.reindex(block_ids.to_numpy()).reset_index(drop=True)
