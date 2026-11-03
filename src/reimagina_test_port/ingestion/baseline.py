"""A city's baseline from the data lake, with exe_reimagina's rules.

Adapted from reimagina-lyra ``ingestion/pipeline.build_elements``: the same
sources, block and lot preparation, building heights and element attributes
(``prep``, ``footprints``, ``elements``, copied unchanged), returned as tables
instead of written. Lot ids come from the cadastral key (``cadastral.lot_keys``)
instead of the shapefile's row number. Amenities (equipment layer) and
establishments (DENUE) hang from their lot, as in reimagina_pipelines_lab
(``points``).
"""

import logging
from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import pandas as pd

from reimagina_test_port.common import lake
from reimagina_test_port.ingestion import cadastral, footprints, points, prep
from reimagina_test_port.ingestion.cities import CityPreset
from reimagina_test_port.ingestion.elements import (
    amenity_type_by_lot,
    classify_land_use,
    lot_attributes,
    occupancy_rate,
    people_per_household,
    prepare_pois,
)

logger = logging.getLogger(__name__)

Progress = Callable[[str, int, int], None]
STAGES = ("download", "blocks", "lots", "footprints", "elements", "points")
LOT_ATTRIBUTES = [
    "land_use",
    "cos",
    "num_levels",
    "area",
    "unit_size_m2",
    "people_per_household",
    "occupancy_rate",
    "amenity_type",
    "commerce_type",
]
BLOCK_ATTRIBUTES = ["vivtot", "people_per_household"]


def block_table(blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """One row per block: id by CVEGEO, census attributes, EPSG:4326."""
    frame = pd.DataFrame(
        {
            "id": blocks["cvegeo"].map(cadastral.block_id),
            "parent_id": None,
            "source_record_ref": blocks["cvegeo"],
            "vivtot": pd.to_numeric(blocks["vivtot"], errors="coerce"),
            "people_per_household": [
                people_per_household(value) for value in blocks["prom_ocup"]
            ],
        }
    )
    return gpd.GeoDataFrame(
        frame, geometry=blocks.geometry.to_crs(4326).to_numpy(), crs=4326
    )


def lot_table(
    lots: gpd.GeoDataFrame,
    blocks: gpd.GeoDataFrame,
    pois: gpd.GeoDataFrame,
    city: CityPreset,
) -> gpd.GeoDataFrame:
    """One row per lot (``elements.lot_elements`` with cadastral ids), EPSG:4326."""
    missing = sorted(set(lots["cvegeo"].dropna()) - set(blocks["cvegeo"]))
    if missing:
        msg = f"{len(missing)} lot cvegeo without a block, e.g. {missing[:5]}"
        raise ValueError(msg)
    lots = lots.to_crs(epsg=4326)
    zoning = lots["zoning"].astype(str).str.strip().str.lower()
    amenity_types = amenity_type_by_lot(
        lots,
        zoning,
        pois,
        metric_crs=city.target_crs,
        poi_max_distance_m=city.poi_max_distance_m,
    )
    land_use, unit_size = classify_land_use(
        lots, zoning, amenity_types, mf_min_units=city.mf_min_units
    )
    giro = lots["Giro"] if "Giro" in lots else pd.Series(None, index=lots.index)
    rows = [
        lot_attributes(
            use,
            cos,
            levels,
            area,
            size,
            prom,
            occupancy_rate(inhabited, vivtot),
            activity,
            amenity_types.get(key),
        )
        for key, use, cos, levels, area, size, prom, inhabited, vivtot, activity in zip(
            lots["lot_id"],
            land_use,
            lots["cos"],
            lots["num_levels"],
            lots["lot_area"],
            unit_size,
            lots["prom_ocup"],
            lots["tvivparhab"],
            lots["vivtot"],
            giro,
            strict=True,
        )
    ]
    frame = pd.DataFrame(
        {
            "id": [cadastral.lot_id(city.key, key) for key in lots["lot_id"]],
            "parent_id": lots["cvegeo"].map(cadastral.block_id).to_numpy(),
            "source_record_ref": lots["lot_id"].to_numpy(),
        }
    )
    for key in LOT_ATTRIBUTES:
        frame[key] = [row.get(key) for row in rows]
    # num_levels is an integer in attribute_catalog: whole floors, rounded as the
    # API's integer decoding does.
    frame["num_levels"] = [
        None if pd.isna(value) else round(value) for value in frame["num_levels"]
    ]
    return gpd.GeoDataFrame(frame, geometry=lots.geometry.to_numpy(), crs=4326)


def build(
    city: CityPreset, workdir: Path, progress: Progress
) -> dict[str, gpd.GeoDataFrame | dict]:
    """``lots``, ``blocks``, ``amenities``, ``establishments`` and a ``report``.

    The report has the cadastral keys (``lots``) and, for amenities and
    establishments, the elements kept and those outside every lot.
    """
    progress("download", 0, len(STAGES))
    lots_path = lake.download_shapefile(city.lots_uri, workdir / "lots")
    normativa_path = lake.download(city.normativa_uri, workdir)
    pois_path = lake.download(city.pois_uri, workdir)
    denue_path = lake.download(city.denue_uri, workdir)
    manifests_uri = f"{city.manifests_uri}/{city.year}/manifests"
    manifest_paths = [
        lake.download(f"{manifests_uri}/{name}", workdir / "manifests")
        for name in lake.list_names(manifests_uri)
    ]
    census = lake.read_parquet(city.census_uri, prep.CENSUS_SOURCE_COLUMNS)
    raw_blocks = lake.read_parquet(city.blocks_uri, ["CVEGEO", "geometry"])

    progress("blocks", 1, len(STAGES))
    blocks, block_export = prep.prepare_blocks(
        census, raw_blocks, ent_code=city.ent_code, muni_code=city.muni_code
    )

    progress("lots", 2, len(STAGES))
    source = gpd.read_file(lots_path, engine="pyogrio").to_crs(city.target_crs)
    # prep uses the index as lot_id: index by the unique cadastral key.
    source, keys_report = cadastral.lot_keys(source, city.key_columns)
    logger.info("Cadastral keys: %s", keys_report)
    lots = prep.prepare_lots(
        source,
        gpd.read_file(normativa_path, engine="pyogrio"),
        blocks,
        block_export,
        target_crs=city.target_crs,
        normativa_columns=city.normativa_columns,
        normativa_max_distance=city.normativa_max_distance,
    )

    progress("footprints", 3, len(STAGES))
    tile_names = footprints.tiles_for_lots(manifest_paths, lots)
    tiles = [
        lake.download(f"{city.tiles_uri}/{city.year}/{name}", workdir / "tiles")
        for name in tile_names
    ]
    params = footprints.FootprintParams(
        presence_threshold=city.presence_threshold,
        height_percentile=city.height_percentile,
        average_level_height=city.average_level_height,
        gsd_pixel=city.gsd_pixel,
    )
    height_vrt = footprints.write_vrt(
        tiles, footprints.HEIGHT_BAND, workdir / f"{city.cve_code}_height.vrt"
    )
    presence_vrt = (
        footprints.write_vrt(
            tiles, footprints.PRESENCE_BAND, workdir / f"{city.cve_code}_presence.vrt"
        )
        if params.presence_threshold > 0
        else None
    )
    stats = footprints.compute_height_stats(lots, height_vrt, presence_vrt, params)
    lots = footprints.add_urban_metrics(
        lots, stats, average_level_height=params.average_level_height
    )

    progress("elements", 4, len(STAGES))
    equipment = gpd.read_parquet(pois_path)
    lot_rows = lot_table(lots, blocks, prepare_pois(equipment), city)

    progress("points", 5, len(STAGES))
    amenities, amenities_report = points.amenity_table(equipment, lot_rows, city)
    businesses = points.read_denue(
        denue_path, city.denue_municipio, city.denue_encoding
    )
    establishments, establishments_report = points.establishment_table(
        businesses, lot_rows, city
    )
    return {
        "blocks": block_table(blocks),
        "lots": lot_rows,
        "amenities": amenities,
        "establishments": establishments,
        "report": {
            "lots": keys_report,
            "amenities": amenities_report,
            "establishments": establishments_report,
        },
    }
