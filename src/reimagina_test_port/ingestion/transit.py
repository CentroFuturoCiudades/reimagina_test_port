"""Public transport stops, as the ULI's ``pt_any`` (``MX_Mexicali_2025_ULI.yml``):
the city's own layer (``sources.transit_stops``) pooled with OpenStreetMap's
(``sources.osm_transit_stops``), without removing the stops both have (the ULI's
``replace: false``).

- The city's layer (Mexicali: IMIP's equipment category ``transp``) keeps the rows
  whose ``column`` is in ``include`` (Mexicali: the local bus terminals, its
  boarding points), every row when empty; those in ``stations`` are stations, the
  rest bus stops.
- OpenStreetMap's are the elements with one of GHSCI's ``pt_any`` tags
  (``osm_destination_definitions.csv``) over the street network's area (the
  blocks' convex hull plus its margin, within the municipality), at a point on
  them: ``public_transport=station``, rail and ferry tags are stations, the rest
  bus stops. ``scripts/osm_transit.py`` extracts them once from a Geofabrik
  extract (``osm_extract``) into the file the city YAML names; Overpass turns away
  automated downloads.

Each stop is a ``bus_stop`` or ``station`` element at its own point, with no
parent, and the ``source`` (``IMIP``, ``OSM``) and ``name`` attributes; the
``transit_stop`` type of ``poi_catalog`` counts both.
"""

import logging
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely

from reimagina_test_port.common import lake
from reimagina_test_port.ingestion.cadastral import stable_element_uuid
from reimagina_test_port.ingestion.cities import CityPreset

logger = logging.getLogger(__name__)

# GHSCI's pt_any: key -> values.
OSM_TAGS = {
    "public_transport": ["platform", "stop_position", "station"],
    "highway": ["bus_stop", "platform"],
    "railway": ["platform", "tram_stop", "stop"],
    "amenity": ["ferry_terminal"],
}
# The extract's columns (geometry as WKB).
STOP_COLUMNS = ["key", "element_type", "source", "name", "geometry"]
TRANSIT_ATTRIBUTES = {"source": "text", "name": "text"}
COLUMNS = ["id", "parent_id", "source_record_ref", "element_type", *TRANSIT_ATTRIBUTES]
# The layers of GDAL's OSM driver with features of these tags.
OSM_LAYERS = ("points", "lines", "multipolygons")


def empty() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        pd.DataFrame(columns=COLUMNS, dtype=object), geometry=[], crs=4326
    )


def _kind(station: pd.Series) -> pd.Series:
    return station.map({True: "station", False: "bus_stop"})


def own_stops(path: Path, city: CityPreset) -> pd.DataFrame:
    """The city's layer as ``STOP_COLUMNS`` (geometry as points, EPSG:4326)."""
    config = city.transit_stops
    raw = (
        gpd.read_parquet(path) if path.suffix == ".geoparquet" else gpd.read_file(path)
    )
    kind = raw[config.column].astype(str).str.strip()
    kept = kind.isin(config.include) if config.include else kind.notna()
    raw, kind = raw[kept], kind[kept]
    names = (
        raw[config.name_column].astype(str).str.strip()
        if config.name_column
        else pd.Series(None, index=raw.index, dtype=object)
    )
    return pd.DataFrame(
        {
            "key": f"{config.source_name}:" + raw[config.id_column].astype(str),
            "element_type": _kind(kind.isin(config.stations)),
            "source": config.source_name,
            "name": names.where(names.ne("") & names.ne("None")),
            "geometry": shapely.force_2d(raw.geometry.to_crs(4326).to_numpy()),
        }
    )


def _tag(other_tags: object, key: str) -> str | None:
    """``key``'s value in the OSM driver's ``other_tags`` (an hstore string)."""
    if not isinstance(other_tags, str):
        return None
    match = re.search(rf'"{re.escape(key)}"=>"([^"]*)"', other_tags)
    return match.group(1) if match else None


def stop_tags(features: pd.DataFrame) -> pd.DataFrame:
    """Each ``OSM_TAGS`` key of GDAL's OSM features: its column, else from
    ``other_tags``."""
    return pd.DataFrame(
        {
            key: features[key]
            if key in features
            else features["other_tags"].map(lambda tags, key=key: _tag(tags, key))
            for key in OSM_TAGS
        },
        index=features.index,
    )


def is_station(tags: pd.DataFrame) -> pd.Series:
    return (
        tags["public_transport"].eq("station")
        | tags["railway"].isin(OSM_TAGS["railway"])
        | tags["amenity"].eq("ferry_terminal")
    )


def osm_extract(pbf: Path, area: shapely.Geometry) -> pd.DataFrame:
    """The stops of an OpenStreetMap extract (``.osm.pbf``) in ``area`` (EPSG:4326)
    as ``STOP_COLUMNS``; keys are ``OSM:<layer>/<id>``."""
    import pyogrio  # only where the extract is made

    frames = []
    for layer in OSM_LAYERS:
        features = pyogrio.read_dataframe(pbf, layer=layer, bbox=area.bounds)
        tags = stop_tags(features)
        hit = pd.concat(
            [tags[key].isin(values) for key, values in OSM_TAGS.items()], axis=1
        ).any(axis=1)
        features, tags = features[hit], tags[hit]
        # Multipolygons from ways have osm_way_id, from relations osm_id.
        ids = features["osm_id"]
        if "osm_way_id" in features:
            ids = ids.fillna(features["osm_way_id"])
        frames.append(
            pd.DataFrame(
                {
                    "key": f"OSM:{layer}/" + ids.astype(str),
                    "element_type": _kind(is_station(tags)),
                    "source": "OSM",
                    "name": features["name"].astype(object),
                    "geometry": shapely.point_on_surface(features.geometry.to_numpy()),
                }
            )
        )
    stops = pd.concat(frames, ignore_index=True)
    return stops[shapely.contains(area, stops["geometry"].to_numpy())].reset_index(
        drop=True
    )


def build(city: CityPreset, workdir: Path) -> gpd.GeoDataFrame:
    """The city's stops as bus_stop and station elements (``COLUMNS``, EPSG:4326);
    empty without either source."""
    frames = []
    if city.transit_stops is not None:
        frames.append(own_stops(lake.download(city.transit_stops.uri, workdir), city))
    if city.osm_transit_uri:
        osm = pd.read_parquet(lake.download(city.osm_transit_uri, workdir))
        frames.append(osm.assign(geometry=shapely.from_wkb(osm["geometry"])))
    if not frames:
        return empty()
    stops = pd.concat(frames, ignore_index=True)
    if repeated := sorted(set(stops["key"][stops["key"].duplicated()])):
        msg = f"Transit stop keys repeated: {repeated[:5]}"
        raise ValueError(msg)
    logger.info(
        "Transit stops: %s",
        stops.groupby(["source", "element_type"]).size().to_dict(),
    )
    frame = pd.DataFrame(
        {
            "id": [stable_element_uuid(city.key, "transit", k) for k in stops["key"]],
            "parent_id": None,
            "source_record_ref": stops["key"].to_numpy(),
            "element_type": stops["element_type"].to_numpy(),
            "source": stops["source"].to_numpy(),
            "name": stops["name"].to_numpy(),
        }
    )
    return gpd.GeoDataFrame(frame, geometry=stops["geometry"].to_numpy(), crs=4326)
