"""Public transport stops: the city's layer, OpenStreetMap's tags and the elements."""

from dataclasses import replace

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import Point

from reimagina_test_port.ingestion import transit
from reimagina_test_port.ingestion.cities import CITIES

LOCAL = "TERMINALES LOCALES DE AUTOBUSES"
PARKING = "ESTACIONAMIENTO"
CENTRAL = "TERMINAL CENTRAL DE PASAJEROS (CENTRAL DE AUTOBUSES DE PASAJEROS)"


def city(tmp_path, *, own: bool = True, osm: bool = True):
    """Mexicali with a three-row transp layer and a two-stop OSM extract."""
    layer = tmp_path / "transp.geoparquet"
    gpd.GeoDataFrame(
        {
            "OBJECTID": [1, 2, 3],
            "TIPO": [LOCAL, PARKING, CENTRAL],
            "NOMBRE": ["AUTOTRANSPORTES URBANOS", "TRACTOPENSION DEL NOROESTE", ""],
        },
        # With Z, as the lake's layers.
        geometry=[
            Point(-115.4, 32.6, 1),
            Point(-115.5, 32.6, 1),
            Point(-115.45, 32.6, 1),
        ],
        crs=4326,
    ).to_parquet(layer)
    extract = tmp_path / "osm.parquet"
    pd.DataFrame(
        {
            "key": ["OSM:points/1", "OSM:points/2"],
            "element_type": ["bus_stop", "station"],
            "source": "OSM",
            "name": [None, "Estación"],
            "geometry": shapely.to_wkb([Point(-115.41, 32.61), Point(-115.42, 32.62)]),
        }
    ).to_parquet(extract)
    preset = CITIES["mexicali"]
    # The central station kept too, to check the stations.
    stops = replace(
        preset.transit_stops,
        uri=str(layer),
        include=(LOCAL, CENTRAL),
        stations=(CENTRAL,),
    )
    return replace(
        preset,
        transit_stops=stops if own else None,
        osm_transit_uri=str(extract) if osm else None,
    )


def test_stops_pool_the_city_layer_and_openstreetmap(tmp_path):
    stops = transit.build(city(tmp_path), tmp_path / "job").set_index(
        "source_record_ref"
    )
    # The parking lot is left out; the central station is a station.
    assert stops["element_type"].to_dict() == {
        "IMIP:1": "bus_stop",
        "IMIP:3": "station",
        "OSM:points/1": "bus_stop",
        "OSM:points/2": "station",
    }
    assert stops["source"].tolist() == ["IMIP", "IMIP", "OSM", "OSM"]
    assert stops.loc["IMIP:1", "name"] == "AUTOTRANSPORTES URBANOS"
    assert pd.isna(stops.loc["IMIP:3", "name"])
    # At their own point, 2D, without a parent.
    assert stops.geometry.has_z.sum() == 0
    assert stops.loc["IMIP:1"].geometry.equals(Point(-115.4, 32.6))
    assert stops["parent_id"].isna().all()
    assert stops["id"].is_unique
    assert stops.crs == "EPSG:4326"


def test_either_source_alone(tmp_path):
    osm = transit.build(city(tmp_path, own=False), tmp_path / "a")
    assert osm["source"].unique().tolist() == ["OSM"]
    own = transit.build(city(tmp_path, osm=False), tmp_path / "b")
    assert own["source"].unique().tolist() == ["IMIP"]
    none = transit.build(city(tmp_path, own=False, osm=False), tmp_path / "c")
    assert none.empty
    assert list(none.columns) == [*transit.COLUMNS, "geometry"]


def test_openstreetmap_tags():
    features = pd.DataFrame(
        {
            "highway": ["bus_stop", None, None, None, "traffic_signals"],
            "other_tags": [
                '"bus"=>"yes"',
                '"public_transport"=>"station","name"=>"Central"',
                '"railway"=>"tram_stop"',
                '"amenity"=>"ferry_terminal"',
                None,
            ],
        }
    )
    tags = transit.stop_tags(features)
    assert tags.loc[1, "public_transport"] == "station"
    assert pd.isna(tags.loc[0, "railway"])
    assert transit.is_station(tags).tolist() == [False, True, True, True, False]
