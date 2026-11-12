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
# Relative source paths are files of the package (its data/ folder ships with it).
PACKAGE = FOLDER.parents[1]


class CityConfigError(ValueError):
    """A city YAML lacks a value or has one this preset does not know."""


# The only street source: OpenStreetMap through Overpass (osmnx).
OSM_STREETS_URI = "osm://overpass"


@dataclass(frozen=True)
class NetworkParams:
    """``sources.streets.params`` of a city YAML."""

    # Overpass filter of the ways kept; None: osmnx's walk network.
    custom_filter: str | None = None
    # Keep pieces not connected to the rest (gated communities).
    retain_all: bool = True
    # Metres of OSM fetched around the blocks.
    margin_m: float = 500.0
    # Lake folder the H5 is uploaded to.
    output_prefix: str = ""
    # (Geo)Parquet of the polygon OSM is fetched within (the municipality); None:
    # the blocks' hull alone.
    boundary_uri: str | None = None
    # merge_streets: a new street end within snap_m of a network node is that node.
    snap_m: float = 15.0
    # merge_streets: a loose end within split_max_m of a network edge splits it.
    split_max_m: float = 60.0
    # merge_streets: gaps up to this length between pieces are bridged.
    connector_max_m: float = 5.0
    # merge_streets: line ends this close are one node.
    precision_m: float = 0.1
    # reimagina_baseline_streets: fit the vehicular streets to the cadastre's lots
    # (ingestion/street_fit.py). ``fit`` overrides street_fit.DEFAULTS.
    refine: bool = False
    fit: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Source:
    """A file: an ``az://`` URI or a local path, relative to the package
    (``data/...``) unless absolute; ``member``, the file inside a .zip."""

    uri: str
    member: str | None = None


@dataclass(frozen=True)
class StreetLighting:
    """Block faces with a lighting answer (``sources.street_lighting``), matched to
    the streets within ``max_distance_m`` of their midpoint."""

    source: Source
    # The answer's column and its values meaning lit / not lit; others don't count.
    column: str
    lit: tuple[str, ...]
    unlit: tuple[str, ...]
    # Rows kept: column -> value (a national file's municipality, say).
    where: dict[str, str] = field(default_factory=dict)
    max_distance_m: float = 25.0
    # A refined street's fronts are width/2 from its axis: faces within
    # max(max_distance_m, width/2 + front_tolerance_m) count.
    front_tolerance_m: float = 5.0


@dataclass(frozen=True)
class BikeLanes:
    """The city's bike lane lines (``sources.bike_lanes``)."""

    source: Source
    type_column: str
    # Lane types that are protected (segregated).
    protected: tuple[str, ...]
    # A street whose midpoint is this close to a lane has it.
    max_distance_m: float = 15.0
    # A refined street has a lane if at least this share (0-1] of its axis lies
    # within max(max_distance_m, width/2) of one.
    min_share: float = 0.5
    # A refined street's protected lane takes this width (at most half its lanes').
    protected_width_m: float = 2.0


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
    # Amenity categories taken from the amenities file only; every other category
    # comes from the DENUE only. One source per kind, so no amenity is loaded twice.
    amenity_file_categories: tuple[str, ...]
    # Raw INEGI DENUE of the state, filtered to the municipality: amenities and
    # establishments by the SCIAN crosswalk.
    denue_uri: str
    denue_municipio: str
    denue_encoding: str = "latin-1"
    # Existing land use (sources.existing_land_use), in the zoning plan's
    # vocabulary: each lot's zoning, the plan's only where it has none. The plan
    # still gives the limits (max_cos...). None: the plan alone.
    existing_land_use_uri: str | None = None
    existing_land_use_column: str = ""
    existing_land_use_max_distance: float = 3.0
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
    # sources.streets: OpenStreetMap's network (ingestion/network.py). None: the
    # city has no street network step.
    streets: NetworkParams | None = None
    # right_of_way elements over that network (ingestion/streets.py): their
    # streetlights and bike lanes. None: not loaded.
    street_lighting: StreetLighting | None = None
    bike_lanes: BikeLanes | None = None

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


def _fit_params(value: Any, name: str) -> dict[str, Any]:
    """``sources.streets.params.fit``: overrides of ``street_fit.DEFAULTS``; an
    unknown key is an error."""
    fit = _mapping(value, f"{name}: sources.streets.params.fit")
    # Imported here: street_fit needs geopandas, the presets do not.
    from reimagina_test_port.ingestion.street_fit import DEFAULTS

    if unknown := sorted(set(fit) - set(DEFAULTS)):
        msg = f"{name}: sources.streets.params.fit: unknown keys {unknown}"
        raise CityConfigError(msg)
    return fit


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
    existing_uri, existing = (
        source("existing_land_use")
        if "existing_land_use" in sources.values
        else (None, _Section(None, "existing_land_use"))
    )
    streets_uri, streets = (
        source("streets")
        if "streets" in sources.values
        else (None, _Section(None, "streets"))
    )
    if streets_uri not in (None, OSM_STREETS_URI):
        msg = f"{path.name}: sources.streets: uri must be {OSM_STREETS_URI}"
        raise CityConfigError(msg)
    network = (
        NetworkParams(
            custom_filter=streets.optional("custom_filter", None),
            retain_all=bool(streets.optional("retain_all", True)),
            output_prefix=str(streets.required("output_prefix")).rstrip("/"),
            refine=bool(streets.optional("refine", False)),
            fit=_fit_params(streets.optional("fit", None), path.name),
            boundary_uri=streets.optional("boundary_uri", None),
            **{
                name: float(
                    streets.optional(
                        name, NetworkParams.__dataclass_fields__[name].default
                    )
                )
                for name in (
                    "margin_m",
                    "snap_m",
                    "split_max_m",
                    "connector_max_m",
                    "precision_m",
                )
            },
        )
        if streets_uri
        else None
    )

    def file(name: str) -> tuple[Source, _Section] | None:
        if name not in sources.values:
            return None
        uri, options = source(name)
        member = options.optional("member", None)
        if "://" not in uri and not Path(uri).is_absolute():
            uri = str(PACKAGE / uri)
        return Source(uri, None if member is None else str(member)), options

    lighting = bike_lanes = None
    if (entry := file("street_lighting")) is not None:
        lighting_source, options = entry
        lighting = StreetLighting(
            lighting_source,
            column=str(options.required("column")),
            lit=tuple(map(str, options.required("lit"))),
            unlit=tuple(map(str, options.required("unlit"))),
            where={
                str(k): str(v)
                for k, v in _mapping(options.optional("where", None), "where").items()
            },
            max_distance_m=float(options.optional("max_distance_m", 25.0)),
            front_tolerance_m=float(options.optional("front_tolerance_m", 5.0)),
        )
        options.done()
    if (entry := file("bike_lanes")) is not None:
        lanes_source, options = entry
        bike_lanes = BikeLanes(
            lanes_source,
            type_column=str(options.required("type_column")),
            protected=tuple(map(str, options.optional("protected", []))),
            max_distance_m=float(options.optional("max_distance_m", 15.0)),
            min_share=float(options.optional("min_share", 0.5)),
            protected_width_m=float(options.optional("protected_width_m", 2.0)),
        )
        if not 0 < bike_lanes.min_share <= 1:
            msg = f"{path.name}: sources.bike_lanes: min_share must be in (0, 1]"
            raise CityConfigError(msg)
        if bike_lanes.protected_width_m <= 0:
            msg = f"{path.name}: sources.bike_lanes: protected_width_m must be > 0"
            raise CityConfigError(msg)
        options.done()
    if (lighting or bike_lanes) and network is None:
        msg = f"{path.name}: street_lighting and bike_lanes need sources.streets"
        raise CityConfigError(msg)
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
        amenity_file_categories=tuple(map(str, raw_amenities.required("categories"))),
        denue_uri=denue_uri,
        denue_municipio=str(denue.required("municipio")),
        denue_encoding=str(denue.optional("encoding", default("denue_encoding"))),
        existing_land_use_uri=existing_uri,
        existing_land_use_column=(
            str(existing.required("column")) if existing_uri else ""
        ),
        existing_land_use_max_distance=float(
            existing.optional(
                "max_distance_m", default("existing_land_use_max_distance")
            )
        ),
        default_capacity=float(
            amenities.optional("default_capacity", default("default_capacity"))
        ),
        floor_share_by_land_use=shares,
        streets=network,
        street_lighting=lighting,
        bike_lanes=bike_lanes,
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
        blocks, census, parcels, zoning, heights, raw_amenities, denue, existing,
        streets, lots, amenities, points, space_allocation, sources, params, config,
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
    from reimagina_test_port.ingestion.crosswalk import AMENITY_CATEGORIES

    if unknown := sorted(set(preset.amenity_file_categories) - set(AMENITY_CATEGORIES)):
        msg = f"{path.name}: sources.amenities: unknown categories {unknown}"
        raise CityConfigError(msg)
    if preset.key != path.stem:
        msg = f"{path.name}: city is {preset.key!r}, the file must be {preset.key}.yaml"
        raise CityConfigError(msg)
    return preset


def load(folder: Path = FOLDER) -> dict[str, CityPreset]:
    """Every ``*.yaml`` preset of ``folder``, by city key."""
    return {path.stem: from_yaml(path) for path in sorted(folder.glob("*.yaml"))}


CITIES = load()
