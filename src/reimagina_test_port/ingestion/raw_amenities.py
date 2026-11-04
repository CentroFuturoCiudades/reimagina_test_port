"""The raw_amenities step: a city's own amenity layers into the amenities file.

Each city's amenities come from a different source (Mexicali: IMIP's equipment
layers), so a normalizer per source, named in the city YAML
(``sources.amenities.params.normalizer``), reads the raw files and returns the
**amenities file**, the input of ``points.amenity_table``: one row per amenity,
with ``AMENITIES_FILE_COLUMNS``.

- ``source``, ``source_id``: the record's origin; ``source:source_id`` is its key.
- ``category``: one of urban_fabric.md's amenity categories.
- ``name``, ``area_m2``: when the source has them (``area_m2`` only for
  ``public_space``).
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


# ------------------------------------------------------------------------ IMIP


@dataclass(frozen=True)
class Rule:
    """IMIP rows whose ``TIPO`` matches ``patterns`` get ``category``.

    ``match`` is ``contains`` or ``exact``, case-insensitive; no patterns keeps
    every row.
    """

    category: str
    patterns: tuple[str, ...] = ()
    match: str = "exact"


@dataclass(frozen=True)
class Layer:
    rules: tuple[Rule, ...]
    name_column: str | None = None
    # Read for public_space rows only: urban_fabric.md's area_m2.
    area_column: str | None = None


PUBLIC_SPACES = ("JARDIN VECINAL", "PARQUE DE BARRIO", "PARQUE URBANO")
# exe_reimagina ``cleaning/equipamientos/mexicali/2_equip_for_exe.ipynb``
# (CATEGORY_CONFIG): the same six layers and TIPO filters, with the categories of
# urban_fabric.md. One change: esp_pub's sports spaces are sports_recreation (exe
# counted them as parks). Rows no rule matches are left out (edu_basica's CENDI
# and telesecundarias; esp_pub's medians, roundabouts...).
IMIP_LAYERS = {
    "edu_basica": Layer(
        rules=(
            Rule("primary_education", ("primaria",), "contains"),
            Rule(
                "secondary_education",
                ("ESCUELA SECUNDARIA GENERAL", "ESCUELA SECUNDARIA TECNICA"),
            ),
            Rule("primary_education", ("preescolar",), "contains"),
        ),
        name_column="NOM_ESC",
    ),
    "edu_sup": Layer((Rule("tertiary_education"),), name_column="NOM_ESC"),
    "med_sup": Layer((Rule("secondary_education"),), name_column="NOM_ESC"),
    "deporte": Layer((Rule("sports_recreation"),), name_column="NOMBRE"),
    "cult_rec": Layer((Rule("cultural_facility"),), name_column="NOMBRE"),
    # No name column; Sup_M2 is the space's area (SUP_HA and Shape_Area are the
    # urban boundary's).
    "esp_pub": Layer(
        (
            Rule("public_space", PUBLIC_SPACES),
            Rule("sports_recreation", ("ESPACIO DEPORTIVO",)),
        ),
        area_column="Sup_M2",
    ),
}


def _matches(tipo: pd.Series, rule: Rule) -> pd.Series:
    if not rule.patterns:
        return pd.Series(True, index=tipo.index)
    text = tipo.astype(str).str.lower()
    mask = pd.Series(False, index=tipo.index)
    for pattern in rule.patterns:
        if rule.match == "contains":
            mask |= text.str.contains(pattern.lower(), regex=False)
        else:
            mask |= text == pattern.lower()
    return mask


def imip(folder: Path) -> gpd.GeoDataFrame:
    """IMIP's equipment layers (``<layer>.geoparquet`` in ``folder``) as the
    amenities file; ``source_id`` is ``<layer>:<OBJECTID>``."""
    frames = []
    for name, layer in IMIP_LAYERS.items():
        raw = gpd.read_parquet(folder / f"{name}.geoparquet")
        tipo = raw["TIPO"] if "TIPO" in raw else pd.Series("", index=raw.index)
        for rule in layer.rules:
            rows = raw[_matches(tipo, rule)]
            names = (
                rows[layer.name_column].astype(str).str.strip()
                if layer.name_column
                else pd.Series(None, index=rows.index, dtype=object)
            )
            frames.append(
                gpd.GeoDataFrame(
                    {
                        "source": "IMIP",
                        "source_id": name + ":" + rows["OBJECTID"].astype(str),
                        "category": rule.category,
                        "name": names.where(names.ne("") & names.ne("None")),
                        "area_m2": (
                            pd.to_numeric(rows[layer.area_column], errors="coerce")
                            if layer.area_column and rule.category == "public_space"
                            else float("nan")
                        ),
                    },
                    geometry=rows.geometry.to_crs(4326),
                    crs=4326,
                )
            )
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=4326)


# ------------------------------------------------------------------ the step


@dataclass(frozen=True)
class Normalizer:
    # File names it reads from the source folder.
    files: tuple[str, ...]
    read: Callable[[Path], gpd.GeoDataFrame]


NORMALIZERS = {
    "imip": Normalizer(tuple(f"{name}.geoparquet" for name in IMIP_LAYERS), imip),
}


def build(
    normalizer: str, uri: str, directory: Path
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """Download the normalizer's files from the ``uri`` folder into ``directory``
    and return the amenities file and its rows per category.

    Raises:
        ValueError: an unknown normalizer, or a result outside the format.
    """
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
