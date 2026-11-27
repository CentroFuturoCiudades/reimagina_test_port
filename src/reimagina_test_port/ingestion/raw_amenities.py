"""The raw_amenities step: a city's own amenity layers into the amenities file.

Each city's amenities come from a different source (Mexicali: IMIP's equipment
layers; Tlalpan: SIG CDMX's), so a normalizer per source, named in the city YAML
(``sources.amenities.params.normalizer``), reads the raw files and returns the
**amenities file**, the input of ``points.amenity_table``: one row per amenity,
with ``AMENITIES_FILE_COLUMNS``.

- ``source``, ``source_id``: the record's origin; ``source:source_id`` is its key.
- ``category``: one of the amenity categories (``crosswalk.AMENITY_CATEGORIES``).
- ``name``, ``area_m2``: when the source has them (``area_m2`` only for
  ``park``).
- ``geometry``: a point, in any CRS.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd

from reimagina_test_port.common import lake
from reimagina_test_port.ingestion.crosswalk import AMENITY_CATEGORIES

logger = logging.getLogger(__name__)

AMENITIES_FILE_COLUMNS = ["source", "source_id", "category", "name", "area_m2"]


# ---------------------------------------------------------------------- layers


@dataclass(frozen=True)
class Rule:
    """A layer's rows whose ``Layer.column`` matches ``patterns`` get ``category``.

    ``match`` is ``contains`` or ``exact``, case-insensitive; no patterns keeps
    every row.
    """

    category: str
    patterns: tuple[str, ...] = ()
    match: str = "exact"


@dataclass(frozen=True)
class Layer:
    rules: tuple[Rule, ...]
    # The column the rules match, and the record's id in the layer.
    column: str = "TIPO"
    id_column: str = "OBJECTID"
    name_column: str | None = None
    # Read for park rows only: urban_fabric.md's area_m2.
    area_column: str | None = None


def _matches(values: pd.Series, rule: Rule) -> pd.Series:
    if not rule.patterns:
        return pd.Series(True, index=values.index)
    text = values.astype(str).str.lower()
    mask = pd.Series(False, index=values.index)
    for pattern in rule.patterns:
        if rule.match == "contains":
            mask |= text.str.contains(pattern.lower(), regex=False)
        else:
            mask |= text == pattern.lower()
    return mask


def read_layers(
    folder: Path, source: str, layers: dict[str, Layer]
) -> gpd.GeoDataFrame:
    """``layers`` (``<layer>.geoparquet`` in ``folder``) as the amenities file;
    ``source_id`` is ``<layer>:<id>``, and each geometry a point on it."""
    frames = []
    for name, layer in layers.items():
        raw = gpd.read_parquet(folder / f"{name}.geoparquet")
        values = (
            raw[layer.column] if layer.column in raw else pd.Series("", index=raw.index)
        )
        for rule in layer.rules:
            rows = raw[_matches(values, rule)]
            names = (
                rows[layer.name_column].astype(str).str.strip()
                if layer.name_column
                else pd.Series(None, index=rows.index, dtype=object)
            )
            frames.append(
                gpd.GeoDataFrame(
                    {
                        "source": source,
                        "source_id": name + ":" + rows[layer.id_column].astype(str),
                        "category": rule.category,
                        "name": names.where(names.ne("") & names.ne("None")),
                        "area_m2": (
                            pd.to_numeric(rows[layer.area_column], errors="coerce")
                            if layer.area_column and rule.category == "park"
                            else float("nan")
                        ),
                    },
                    geometry=rows.geometry.representative_point().to_crs(4326),
                    crs=4326,
                )
            )
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=4326)


# ------------------------------------------------------------------------ IMIP


PUBLIC_SPACES = ("JARDIN VECINAL", "PARQUE DE BARRIO", "PARQUE URBANO")
# exe_reimagina ``cleaning/equipamientos/mexicali/2_equip_for_exe.ipynb``
# (CATEGORY_CONFIG): its layers and TIPO filters, with the categories of
# crosswalk.AMENITY_CATEGORIES, plus admon_pub's police and fire stations. Two changes: esp_pub's sports spaces are sports (exe
# counted them as parks), and edu_sup's universities are left out (neither the
# indicators sheet nor denue_crosswalk.xlsx counts them). Rows no rule matches are left out (edu_basica's CENDI
# and telesecundarias; esp_pub's medians, roundabouts...).
IMIP_LAYERS = {
    "edu_basica": Layer(
        rules=(
            Rule("primary_education", ("primaria",), "contains"),
            Rule(
                "secondary_education",
                ("ESCUELA SECUNDARIA GENERAL", "ESCUELA SECUNDARIA TECNICA"),
            ),
            Rule("preschool", ("preescolar",), "contains"),
        ),
        name_column="NOM_ESC",
    ),
    "med_sup": Layer((Rule("upper_secondary_education"),), name_column="NOM_ESC"),
    "deporte": Layer((Rule("sports"),), name_column="NOMBRE"),
    "cult_rec": Layer((Rule("cultural_center"),), name_column="NOMBRE"),
    # No name column; Sup_M2 is the space's area (SUP_HA and Shape_Area are the
    # urban boundary's).
    "esp_pub": Layer(
        (
            Rule("park", PUBLIC_SPACES),
            Rule("sports", ("ESPACIO DEPORTIVO",)),
        ),
        area_column="Sup_M2",
    ),
    # Public administration: police and fire stations, as the ULI's
    # (MX_Mexicali_2025_ULI.yml's police and fire destinations).
    "admon_pub": Layer(
        (
            Rule("police_station", ("INSTALACIONES POLICIALES",)),
            Rule("fire_station", ("ESTACION DE BOMBEROS MUNICIPAL",)),
        ),
        name_column="NOMBRE",
    ),
}


def imip(folder: Path) -> gpd.GeoDataFrame:
    """IMIP's equipment layers (``<layer>.geoparquet`` in ``folder``) as the
    amenities file; ``source_id`` is ``<layer>:<OBJECTID>``."""
    return read_layers(folder, "IMIP", IMIP_LAYERS)


# -------------------------------------------------------------------- SIG CDMX

# exe_reimagina ``cleaning/equipamientos/tlalpan/0_2_sigcdmx_data_prep.ipynb``: its
# three layers and mappings (servicios by TIPO, escuelas by nivel, areas_verdes by
# categoria), with the categories of crosswalk.AMENITY_CATEGORIES. The layers in the
# lake are the notebook's GeoJSONs in UTF-8, areas_verdes cut to Tlalpan. Changes:
# basic education comes from escuelas only (as the notebook says, though it also
# took servicios'), "Preescolar - Inicial" is preschool (exe split it in two), and
# servicios' universities and social assistance centres are left out, as are rows
# no rule matches. Its other filters are not kept: every mapped servicio is "En
# Operación", and the two areas_verdes parks named CAMELLON stay.
SPORTS_TIPOS = (
    "Cancha", "Unidad Deportiva", "Alberca Olímpica", "Campo de Golf", "Lienzo Charro",
    "Estadio",
)  # fmt: skip
CULTURE_TIPOS = (
    "Edificación Cultural", "Centro de Espectáculos", "Pirámide", "Plaza de Toros",
)  # fmt: skip
SIGCDMX_LAYERS = {
    "areas_verdes": Layer(
        (Rule("park", ("Parques, arboledas y alamedas", "Plazas y jardines")),),
        column="categoria",
        id_column="gid",
        name_column="nombre",
        area_column="superficie",
    ),
    "servicios": Layer(
        (
            Rule("hospital", ("Hospital",)),
            Rule("health_center", ("Centro de Salud", "Centro de Rehabilitación")),
            Rule("upper_secondary_education", ("Medio Superior",)),
            Rule("park", ("Parque", "Áreas Verdes", "Jardín")),
            Rule("sports", SPORTS_TIPOS),
            Rule("cultural_center", CULTURE_TIPOS),
            Rule("museum", ("Museo",)),
        ),
        id_column="OID",
        name_column="NOMGEO",
    ),
    "escuelas": Layer(
        (
            Rule("daycare", ("INICIAL",)),
            Rule("preschool", ("preescolar",), "contains"),
            Rule("primary_education", ("Primaria",)),
            Rule("secondary_education", ("Secundaria",)),
        ),
        column="nivel",
        id_column="fid",
        name_column="nombre",
    ),
}


def sigcdmx(folder: Path) -> gpd.GeoDataFrame:
    """SIG CDMX's equipment layers (``<layer>.geoparquet`` in ``folder``) as the
    amenities file; ``source_id`` is ``<layer>:<id>``."""
    return read_layers(folder, "SIGCDMX", SIGCDMX_LAYERS)


# ------------------------------------------------------------------ the step


@dataclass(frozen=True)
class Normalizer:
    # File names it reads from the source folder.
    files: tuple[str, ...]
    read: Callable[[Path], gpd.GeoDataFrame]


NORMALIZERS = {
    "imip": Normalizer(tuple(f"{name}.geoparquet" for name in IMIP_LAYERS), imip),
    "sigcdmx": Normalizer(
        tuple(f"{name}.geoparquet" for name in SIGCDMX_LAYERS), sigcdmx
    ),
}


def build(
    normalizer: str, uri: str | None, directory: Path
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """Download the normalizer's files from the ``uri`` folder into ``directory``
    and return the amenities file and its rows per category. Without ``uri`` (a
    city without sources.amenities) the file is empty: every amenity comes from
    the DENUE.

    Raises:
        ValueError: an unknown normalizer, or a result outside the format.
    """
    if uri is None:
        empty = {column: pd.Series(dtype=object) for column in AMENITIES_FILE_COLUMNS}
        return gpd.GeoDataFrame(empty, geometry=gpd.GeoSeries(crs=4326)), {}
    if normalizer not in NORMALIZERS:
        msg = (
            f"Unknown amenities normalizer {normalizer!r}; known: {sorted(NORMALIZERS)}"
        )
        raise ValueError(msg)
    step = NORMALIZERS[normalizer]
    for name in step.files:
        lake.download(f"{uri.rstrip('/')}/{name}", directory)
    table = step.read(directory)
    check(table)
    report = {str(k): int(v) for k, v in table["category"].value_counts().items()}
    logger.info("raw_amenities (%s): %s", normalizer, report)
    return table, report


def check(table: gpd.GeoDataFrame) -> None:
    """Raise ``ValueError`` unless ``table`` is in the amenities file format."""
    if missing := [c for c in AMENITIES_FILE_COLUMNS if c not in table]:
        msg = f"Amenities file lacks columns {missing}"
        raise ValueError(msg)
    if unknown := sorted(set(table["category"]) - set(AMENITY_CATEGORIES)):
        msg = f"Amenities file categories are not amenity categories: {unknown}"
        raise ValueError(msg)
    keys = table["source"].astype(str) + ":" + table["source_id"].astype(str)
    if repeated := sorted(set(keys[keys.duplicated()])):
        msg = f"Amenities file keys repeated: {repeated[:5]}"
        raise ValueError(msg)
    if not (table.geom_type == "Point").all():
        msg = "Amenities file geometries must be points"
        raise ValueError(msg)
