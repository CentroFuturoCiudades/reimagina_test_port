"""Extract a city's OpenStreetMap public transport stops: the file
``sources.osm_transit_stops`` reads (ingestion/transit.py).

The stops are the elements with one of GHSCI's ``pt_any`` tags
(``transit.OSM_TAGS``), as the ULI counted them, from a Geofabrik extract read with
GDAL's OSM driver; Overpass turns away automated downloads. The area is the street
network's: the city's blocks' convex hull plus ``sources.streets.params.margin_m``,
within its municipality (``boundary_uri``). The ULI used Mexico's extract of
15 August 2026 (https://download.geofabrik.de/north-america/mexico-latest.osm.pbf):

    uv run python scripts/osm_transit.py mexicali mexico-260815.osm.pbf \\
        --out osm_transit_02002.parquet

Upload it to the lake, to the city's ``sources.osm_transit_stops.uri``. A city's
read takes about 4 minutes.
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import shapely

from reimagina_test_port.common import lake
from reimagina_test_port.ingestion import network, transit
from reimagina_test_port.ingestion.cities import CITIES

logger = logging.getLogger("osm_transit")


def area(city: str) -> shapely.Geometry:
    """The street network's area of ``city``, EPSG:4326."""
    preset = CITIES[city]
    streets = preset.streets
    if streets is None:
        msg = f"{preset.name} has no sources.streets: no area to extract."
        raise SystemExit(msg)
    blocks = gpd.GeoSeries.from_wkb(
        lake.read_parquet(preset.blocks_uri, ["geometry"])["geometry"], crs=4326
    )
    boundary = None
    if streets.boundary_uri:
        raw = lake.read_parquet(streets.boundary_uri, ["geometry"])
        boundary = gpd.GeoSeries.from_wkb(raw["geometry"], crs=4326)
    return network.study_area(blocks, preset.target_crs, streets.margin_m, boundary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("city", choices=sorted(CITIES))
    parser.add_argument("pbf", type=Path, help="Geofabrik .osm.pbf extract.")
    parser.add_argument("--out", type=Path, required=True, help="Parquet to write.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    stops = transit.osm_extract(args.pbf, area(args.city))
    stops.assign(geometry=shapely.to_wkb(stops["geometry"].to_numpy())).to_parquet(
        args.out
    )
    logger.info(
        "%d stops written to %s: %s",
        len(stops),
        args.out,
        stops.groupby("element_type").size().to_dict(),
    )


if __name__ == "__main__":
    main()
