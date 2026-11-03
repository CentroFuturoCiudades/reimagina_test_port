"""City presets, one YAML per city in this folder (``<city>.yaml``).

The layout is reimagina_pipelines_lab's ``pipelines/ingestion/cities``: the
city's codes and CRS, its ``sources`` (a URI and its ``params`` each) and the
builders' ``params``. Values come from exe_reimagina's
``src/pipeline/cities/*.toml``; see exe_reimagina ``doc/CITIES.md`` for the
meaning of each parameter. Source URIs point at the CFC data lake.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from reimagina_test_port.common.rules import LAND_USES

FOLDER = Path(__file__).parent


class CityConfigError(ValueError):
    """A city YAML lacks a value or has one this preset does not know."""


@dataclass(frozen=True)
class CityPreset:
    name: str
    # Lot id city key: uuid5(NS, f"{key}:lot:{cadastral key}").
    key: str
    ent_code: str
    muni_code: str
    # Metric CRS used for areas, distances and the height rasters.
    target_crs: str
    blocks_uri: str
    census_uri: str
    lots_uri: str
    # Cadastral columns joined with "-" into each lot's key.
    key_columns: tuple[str, ...]
    normativa_uri: str
    # Normativa source column -> internal name (zoning, max_cos, max_cus, ...).
    normativa_columns: dict[str, str]
    # Open Buildings 2.5D temporal tiles and their manifests.
    tiles_uri: str
    manifests_uri: str
    year: str
    # Folder of the city's raw amenity layers, and the raw_amenities normalizer
    # that turns them into the amenities file (raw_amenities.NORMALIZERS).
    amenities_uri: str
    amenities_normalizer: str
    # Raw INEGI DENUE of the state, filtered to the municipality: amenities and
    # establishments by the SCIAN crosswalk.
    denue_uri: str
    denue_municipio: str
    denue_encoding: str = "latin-1"
    normativa_max_distance: float = 3.0
    presence_threshold: float = 0.5
    height_percentile: float = 95.0
    average_level_height: float = 2.7
    gsd_pixel: float = 0.5
    mf_min_units: float = 1.5
    default_capacity: float = 1.0
    # Placement distance (urban_fabric.md): farther from every lot, not loaded.
    lot_max_distance_m: float = 50.0
    # Space allocation by lot type: land_use -> housing.floor_share (0-100), until
    # the allocation is computed per lot. A land use not listed has no value.
    floor_share_by_land_use: dict[str, float] = field(default_factory=dict)

    @property
    def cve_code(self) -> str:
        return f"{self.ent_code}{self.muni_code}"


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        msg = f"{where} must be a mapping"
        raise CityConfigError(msg)
    return dict(value)


class _Section:
    """A YAML mapping read key by key; ``done`` rejects keys nobody read."""

    def __init__(self, value: Any, where: str) -> None:
        self.values = _mapping(value, where)
        self.where = where

    def required(self, key: str) -> Any:
        if self.values.get(key) is None:
            msg = f"{self.where}: {key} is required"
            raise CityConfigError(msg)
        return self.values.pop(key)

    def optional(self, key: str, default: Any) -> Any:
        value = self.values.pop(key, None)
        return default if value is None else value

    def section(self, key: str) -> "_Section":
        return _Section(self.values.pop(key, None), f"{self.where}.{key}")

    def done(self) -> None:
        if self.values:
            msg = f"{self.where}: unknown keys {sorted(self.values)}"
            raise CityConfigError(msg)


def from_yaml(path: Path) -> CityPreset:
    """The preset of one city YAML; missing or unknown keys raise ``CityConfigError``."""
    config = _Section(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)
    sources, params = config.section("sources"), config.section("params")
    defaults = CityPreset.__dataclass_fields__

    def source(name: str) -> tuple[str, _Section]:
        entry = sources.section(name)
        uri = str(entry.required("uri"))
        options = entry.section("params")
        entry.done()
        return uri, options

    def default(field: str) -> Any:
        return defaults[field].default

    blocks_uri, blocks = source("blocks")
    census_uri, census = source("census")
    lots_uri, parcels = source("parcels")
    normativa_uri, zoning = source("zoning")
    tiles_uri, heights = source("heights")
    amenities_uri, raw_amenities = source("amenities")
    denue_uri, denue = source("denue")
    lots, amenities, points, space_allocation = (
        params.section("lots"),
        params.section("amenities"),
        params.section("points"),
        params.section("space_allocation"),
    )
    shares = {
        str(use): float(share)
        for use, share in _mapping(
            space_allocation.required("housing.floor_share"),
            "space_allocation.housing.floor_share",
        ).items()
    }
    if unknown := sorted(set(shares) - set(LAND_USES)):
        msg = f"{path.name}: space_allocation: unknown land uses {unknown}"
        raise CityConfigError(msg)
    if not all(0 <= share <= 100 for share in shares.values()):
        msg = f"{path.name}: space_allocation: housing.floor_share must be 0-100"
        raise CityConfigError(msg)
    preset = CityPreset(
        name=str(config.required("name")),
        key=str(config.required("city")),
        ent_code=str(config.required("ent_code")),
        muni_code=str(config.required("muni_code")),
        target_crs=str(config.required("crs")),
        blocks_uri=blocks_uri,
        census_uri=census_uri,
        lots_uri=lots_uri,
        key_columns=tuple(map(str, parcels.required("key_columns"))),
        normativa_uri=normativa_uri,
        normativa_columns={
            str(k): str(v)
            for k, v in _mapping(zoning.required("columns"), "columns").items()
        },
        normativa_max_distance=float(
            zoning.optional("max_distance_m", default("normativa_max_distance"))
        ),
        tiles_uri=tiles_uri,
        manifests_uri=str(heights.required("manifests_uri")),
        year=str(heights.required("year")),
        amenities_uri=amenities_uri,
        amenities_normalizer=str(raw_amenities.required("normalizer")),
        denue_uri=denue_uri,
        denue_municipio=str(denue.required("municipio")),
        denue_encoding=str(denue.optional("encoding", default("denue_encoding"))),
        default_capacity=float(
            amenities.optional("default_capacity", default("default_capacity"))
        ),
        floor_share_by_land_use=shares,
        lot_max_distance_m=float(
            points.optional("lot_max_distance_m", default("lot_max_distance_m"))
        ),
        **{
            field: float(lots.optional(field, default(field)))
            for field in (
                "presence_threshold",
                "height_percentile",
                "average_level_height",
                "gsd_pixel",
                "mf_min_units",
            )
        },
    )
    for section in (
        blocks, census, parcels, zoning, heights, raw_amenities, denue,
        lots, amenities, points, space_allocation, sources, params, config,
    ):  # fmt: skip
        section.done()
    # Imported here: raw_amenities needs geopandas, the presets do not.
    from reimagina_test_port.ingestion.raw_amenities import NORMALIZERS

    if preset.amenities_normalizer not in NORMALIZERS:
        msg = (
            f"{path.name}: sources.amenities: unknown normalizer "
            f"{preset.amenities_normalizer!r}; known: {sorted(NORMALIZERS)}"
        )
        raise CityConfigError(msg)
    if preset.key != path.stem:
        msg = f"{path.name}: city is {preset.key!r}, the file must be {preset.key}.yaml"
        raise CityConfigError(msg)
    return preset


def load(folder: Path = FOLDER) -> dict[str, CityPreset]:
    """Every ``*.yaml`` preset of ``folder``, by city key."""
    return {path.stem: from_yaml(path) for path in sorted(folder.glob("*.yaml"))}


CITIES = load()
