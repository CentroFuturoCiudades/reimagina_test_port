"""A city's streets as ``right_of_way`` elements (urban_fabric.md, *Right of way*),
over its street network (``reimagina_street_network``, ``ingestion/network.py``).

- One element per street of the network: a pair of nodes, whatever the direction,
  keyed ``<lower id>-<higher id>``, with the edge's shape (OSM's way, or the
  straight line between its nodes where the network has none).
- ``sources.street_lighting`` (optional): a file of block faces with a lighting
  answer (its ``column``). A street's ``streetlights`` (0-100) is the share of lit
  faces among the answered ones whose midpoint lies within ``max_distance_m`` of
  it; without one it has no value. Mexicali: INEGI's 2020 housing inventory.
- ``sources.bike_lanes`` (optional): the city's own bike lane lines, matched to a
  street whose midpoint lies within ``max_distance_m`` of one. A protected lane
  (its ``type_column`` in ``protected``) takes a band of its own,
  ``protected_bike_lane``, next to the sidewalk; any other is shared: the street's
  lanes, used by bikes too, so its ``lanes`` band is ``shared_lanes``.

- ``sources.tree_canopy`` (optional): ``tree_coverage`` (0-100), the share of a
  street's area under canopy (``ingestion/canopy.py``). Only a street drawn as a
  polygon has one: today's lines get none.

The traffic lanes are one band, ``lanes``: their total space, not lane by lane.
``bands.use`` and ``bands.share`` are a default profile until street widths are
loaded; both are written as JSON lists (Lyra tables have no list column).
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from reimagina_test_port.common import lake
from reimagina_test_port.ingestion import canopy
from reimagina_test_port.ingestion.cadastral import stable_element_uuid
from reimagina_test_port.ingestion.cities import (
    BikeLanes,
    CityPreset,
    Source,
    StreetLighting,
)

ATTRIBUTES = {
    "bands.use": "json",
    "bands.share": "json",
    "streetlights": "number",
    "tree_coverage": "number",
}
PROFILES = {
    None: (["sidewalk", "lanes", "sidewalk"], [20, 60, 20]),
    "shared_lanes": (["sidewalk", "shared_lanes", "sidewalk"], [20, 60, 20]),
    "protected_bike_lane": (
        ["sidewalk", "protected_bike_lane", "lanes", "sidewalk"],
        [20, 10, 50, 20],
    ),
}


def read(source: Source, workdir: Path, columns: list[str]) -> gpd.GeoDataFrame:
    """A file GDAL reads (``member``: the file inside a .zip), from the lake or a
    local path."""
    path = lake.download(source.uri, workdir)
    if source.member:
        path = f"zip://{path}!{source.member}"
    return gpd.read_file(path, columns=columns)


def network_streets(
    nodes: pd.DataFrame, edges: pd.DataFrame, crs: str
) -> gpd.GeoSeries:
    """One line per pair of nodes the network joins (both directions are one
    street), in ``crs``, indexed by ``<lower id>-<higher id>``: the edge's
    ``geometry``, or the straight line between its nodes without one."""
    a, b = edges["from"].astype("int64"), edges["to"].astype("int64")
    keep = (
        pd.DataFrame({"lo": np.minimum(a, b), "hi": np.maximum(a, b)})
        .query("lo != hi")
        .drop_duplicates()
    )
    xy = nodes.set_index(nodes["id"].astype("int64"))[["x", "y"]]
    lines = shapely.linestrings(
        np.stack([xy.loc[keep["lo"]].to_numpy(), xy.loc[keep["hi"]].to_numpy()], 1)
    )
    if isinstance(edges, gpd.GeoDataFrame) and edges.crs is not None:
        own = edges.geometry.to_crs(crs).loc[keep.index].to_numpy()
        present = ~(shapely.is_missing(own) | shapely.is_empty(own))
        lines[present] = own[present]
    keys = keep["lo"].astype(str) + "-" + keep["hi"].astype(str)
    return gpd.GeoSeries(lines, index=pd.Index(keys.to_numpy(), name="key"), crs=crs)


def streetlights(
    streets: gpd.GeoSeries, faces: gpd.GeoDataFrame, lighting: StreetLighting
) -> pd.Series:
    """Per street: the share (0-100) of the answered faces within
    ``lighting.max_distance_m`` whose ``column`` is in ``lit``."""
    for column, value in lighting.where.items():
        faces = faces[faces[column].astype(str).str.strip() == value]
    answer = faces[lighting.column].astype(str).str.strip()
    value = answer.map(
        {**dict.fromkeys(lighting.unlit, 0.0), **dict.fromkeys(lighting.lit, 100.0)}
    )
    middles = gpd.GeoDataFrame(
        {"lit": value[value.notna()].to_numpy()},
        geometry=faces[value.notna()]
        .geometry.to_crs(streets.crs)
        .interpolate(0.5, normalized=True)
        .to_numpy(),
        crs=streets.crs,
    )
    near = gpd.GeoDataFrame(geometry=streets.buffer(lighting.max_distance_m))
    hit = gpd.sjoin(near, middles, predicate="contains")
    return hit.groupby(level=0)["lit"].mean().reindex(streets.index)


def lane_bands(
    streets: gpd.GeoSeries, lanes: gpd.GeoDataFrame, bike: BikeLanes
) -> pd.Series:
    """Per street: ``shared_lanes``, ``protected_bike_lane`` or None, by the nearest
    lane within ``bike.max_distance_m`` of its midpoint."""
    middle = streets.interpolate(0.5, normalized=True)
    near = gpd.sjoin_nearest(
        gpd.GeoDataFrame(geometry=middle),
        lanes[[bike.type_column, "geometry"]].to_crs(streets.crs),
        how="left",
        max_distance=bike.max_distance_m,
    )
    near = near[~near.index.duplicated()]
    kind = np.where(
        near[bike.type_column].isin(bike.protected),
        "protected_bike_lane",
        "shared_lanes",
    )
    return pd.Series(kind, index=near.index).where(near["index_right"].notna())


def build(
    city: CityPreset, nodes: pd.DataFrame, edges: pd.DataFrame, workdir: Path
) -> gpd.GeoDataFrame:
    """The city's ``right_of_way`` elements over its network's ``nodes`` and
    ``edges`` (in its metric CRS), EPSG:4326."""
    streets = network_streets(nodes, edges, city.target_crs)
    lit = pd.Series(np.nan, index=streets.index)
    if (lighting := city.street_lighting) is not None:
        columns = [lighting.column, *lighting.where]
        faces = read(lighting.source, workdir / "street_lighting", columns)
        lit = streetlights(streets, faces, lighting)
    lanes = pd.Series(None, index=streets.index, dtype=object)
    if (bike := city.bike_lanes) is not None:
        lines = read(bike.source, workdir / "bike_lanes", [bike.type_column])
        lanes = lane_bands(streets, lines, bike)
    trees = pd.Series(np.nan, index=streets.index)
    if city.tree_canopy is not None:
        trees = canopy.tree_coverage(streets, city.tree_canopy, workdir)
    bands = [PROFILES[None if pd.isna(lane) else lane] for lane in lanes]
    frame = pd.DataFrame(
        {
            "id": [
                stable_element_uuid(city.key, "right_of_way", k) for k in streets.index
            ],
            "parent_id": None,
            "source_record_ref": streets.index.to_numpy(),
            "bands.use": [json.dumps(use) for use, _ in bands],
            "bands.share": [json.dumps(share) for _, share in bands],
            "streetlights": lit.to_numpy(),
            "tree_coverage": trees.to_numpy(),
        }
    )
    return gpd.GeoDataFrame(frame, geometry=streets.to_crs(4326).to_numpy(), crs=4326)
