"""Element ids: blocks by CVEGEO, lots by their cadastral key.

The namespace is exe_reimagina's, so a block keeps the id it already has. A lot id
comes from the cadastre's own key, never the row number, which changes whenever
the file is re-exported.
"""

import uuid
from typing import Any

import geopandas as gpd

# Must match exe_reimagina's (mig_to_bd_elements) and reimagina-lyra's.
ELEMENTS_UUID_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL, "https://exe-reimagina.local/elements"
)
EMPTY_KEY = "sin_clave"


def stable_element_uuid(*parts: Any) -> str:
    """uuid5 of the parts joined with ":" (exe_reimagina ``stable_element_uuid``)."""
    return str(uuid.uuid5(ELEMENTS_UUID_NAMESPACE, ":".join(map(str, parts))))


def block_id(cvegeo: str) -> str:
    return stable_element_uuid("block", cvegeo)


def lot_id(city: str, key: str) -> str:
    return stable_element_uuid(city, "lot", key)


def lot_keys(
    lots: gpd.GeoDataFrame, key_columns: tuple[str, ...]
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """Lots indexed by a unique cadastral key, and a report.

    The key joins ``key_columns`` (stripped) with "-", trailing dashes removed;
    empty is ``sin_clave``. Exact copies (same key and geometry) are dropped. Lots
    still sharing a key get ``#2``, ``#3``... ordered by representative point
    (x, then y) in the lots' CRS, so the same parcel keeps its key.
    """
    missing = [column for column in key_columns if column not in lots]
    if missing:
        msg = f"Cadastral key columns missing from the lots: {missing}"
        raise ValueError(msg)
    text = lots[list(key_columns)].astype("string").fillna("")
    text = text.apply(lambda column: column.str.strip())
    key = text.agg("-".join, axis=1).str.rstrip("-").replace("", EMPTY_KEY)
    lots = lots.assign(_key=key.to_numpy())
    copies = lots.assign(_shape=lots.geometry.to_wkb()).duplicated(["_key", "_shape"])
    lots = lots[~copies]
    point = lots.geometry.representative_point()
    lots = lots.assign(_x=point.x, _y=point.y).sort_values(
        ["_key", "_x", "_y"], kind="mergesort"
    )
    repeat = lots.groupby("_key").cumcount() + 1
    report = {
        "exact_copies_dropped": int(copies.sum()),
        "lots_sharing_a_key": int(lots["_key"].duplicated(keep=False).sum()),
        "lots_without_key": int((lots["_key"] == EMPTY_KEY).sum()),
    }
    lots.index = lots["_key"] + ("#" + repeat.astype(str)).where(repeat > 1, "")
    lots.index.name = None
    return gpd.GeoDataFrame(lots.drop(columns=["_key", "_x", "_y"])), report
