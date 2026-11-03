"""Blocks and lots to Reimagina elements: ids, parents, land use and attributes.

Moved from exe_reimagina ``src/migration/elements.py``. Lot attributes follow
``baseline_setup.md``: land_use, cos, num_levels, area, unit_size_m2 (multifamily
and mixed_use), people_per_household and occupancy_rate (residential) and
commerce_type. ``land_use`` is classified from the lot's contents (amenities and
establishments), as reimagina_urbano_api ``ingestion_pipelines.md`` says; the
amenity's category lives only on amenity elements. Block attributes: vivtot and
people_per_household.
"""

import logging
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry

from reimagina_test_port.common.rules import MIN_BUILT_M2
from reimagina_test_port.ingestion.cadastral import stable_element_uuid
from reimagina_test_port.ingestion.crosswalk import is_office

logger = logging.getLogger(__name__)

# Zoning vocabularies of Mexicali (USOSUELO24) and Tlalpan (uso_descri), lower case.
AMENITY_ZONINGS = {
    "equipamiento", "área verde", "area verde", "equipamiento público y privado",
    "equipamiento rural", "equipamiento deportivo", "equipamiento de servicio",
    "equipamiento de servicios", "equipamiento mortuorio", "áreas verdes",
    "espacio abierto", "espacios abiertos", "área de servicios públicos",
    "colegio de ingenieros civiles",
}  # fmt: skip
COMMERCE_ZONINGS = {"comercio y servicios", "almacenamiento y servicios"}
MIXED_ZONINGS = {
    "mixto", "habitacional con comercio en planta baja",
    "habitacional rural con comercio y servicios", "habitacional rural con comercio",
    "habitacional mixto", "habitacional con oficinas", "centro de barrio",
    "habitacional unifamiliar y plurifamiliar, oficinas y comercio en planta baja",
    "habitacional unifamiliar y plurifamiliar, oficinas y comercio",
    "habitacional unifamiliar, plurifamiliar y comercio en planta baja",
    "habitacional unifamiliar comercio en planta baja",
    "habitacional unifamiliar, plurifamiliar y/o oficinas",
}  # fmt: skip
# Pure residential, split into single_family / multifamily by the census rule.
SINGLE_FAM = {
    "habitacional", "habitacional rural", "habitacional rural de baja densidad",
    "habitacional rural tlalpuente", "habitacional unifamiliar",
    "habitacional unifamiliar hasta 500 m²", "habitacional unifamiliar de 501 m² o más",
    "habitacional hasta 20 habitantes por hectárea",
    "habitacional hasta 800 habitantes por hectárea",
}  # fmt: skip
INDUSTRIAL_ZONINGS = {"industrial"}
# One home counts as multifamily in a building at least this big and tall
# (ingestion_pipelines.md; exe_reimagina MF_PHYS_BUILTUP_M2, MF_PHYS_LEVELS).
MF_BUILTUP_M2 = 600.0
MF_LEVELS = 2.5
# Normalized cadastral Giro -> commerce_type (about 5 % coverage in Mexicali).
COMMERCE_TYPE_BY_GIRO = {
    "LOCAL_COMERCIAL_O_PLAZAS": "shop", "TIENDA_COMERCIAL": "shop", "ABARROTES": "shop",
    "TIENDA_DE_AUTOSERVICIO": "shop", "EXPENDIO": "shop", "TORTILLERIA": "shop",
    "FARMACIA": "shop", "LLANTERA": "shop", "LOTE_DE_AUTOS": "shop",
    "CENTRO_COMERCIAL": "mall", "PLAZA_COMERCIAL": "mall", "TIANGUIS": "market",
    "RESTAURANT": "restaurant", "TAQUERIA": "restaurant",
    "TALLER_MECANICO_Y_CARROCERIA": "workshop", "TALLER_DE_SERVICIO": "workshop",
    "LAVADO_DE_AUTOS": "workshop", "YONKE": "workshop",
    "SALA_DE_BELLEZA": "services", "OFICINAS": "services", "BANCO": "services",
    "CONSULTORIO_MEDICO": "clinic", "CLINICA": "clinic", "ESTACIONAMIENTO": "parking",
    "HOTEL": "lodging", "GASOLINERA": "gas_station",
}  # fmt: skip


@dataclass(frozen=True)
class Element:
    id: str
    type: str
    parent_element_id: str | None
    geometry: BaseGeometry
    attributes: dict[str, Any]


def _value_or_none(value: Any) -> Any:  # noqa: ANN401
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _int_or_none(value: Any) -> int | None:  # noqa: ANN401
    value = _value_or_none(value)
    return None if value is None else int(value)


def _float_or_none(value: Any) -> float | None:  # noqa: ANN401
    value = _value_or_none(value)
    return None if value is None else float(value)


def _str_or_none(value: Any) -> str | None:  # noqa: ANN401
    value = _value_or_none(value)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_giro(value: Any) -> str | None:  # noqa: ANN401
    text = _str_or_none(value)
    return None if text is None else text.upper().replace(" ", "_")


def people_per_household(prom_ocup: Any) -> float | None:  # noqa: ANN401
    """Census ``prom_ocup`` with two decimals, as INEGI publishes it."""
    value = _float_or_none(prom_ocup)
    return None if value is None else round(value, 2)


def occupancy_rate(tvivparhab: Any, vivtot: Any) -> float | None:  # noqa: ANN401
    """Inhabited private dwellings over all dwellings of the block (0-1)."""
    inhabited, total = _float_or_none(tvivparhab), _float_or_none(vivtot)
    if inhabited is None or not total:
        return None
    return round(min(inhabited / total, 1.0), 4)


# -------------------------------------------------------------------- land use


def zone_use(zoning: pd.Series) -> pd.Series:
    """The zone's use mapped to the land-use list; None when it has no equivalent."""
    use = pd.Series([None] * len(zoning), index=zoning.index, dtype=object)
    use[zoning.isin(SINGLE_FAM)] = "single_family"
    use[zoning.isin(MIXED_ZONINGS)] = "mixed_use"
    use[zoning.isin(AMENITY_ZONINGS)] = "amenity"
    use[zoning.isin(COMMERCE_ZONINGS)] = "commerce"
    use[zoning.isin(INDUSTRIAL_ZONINGS)] = "industrial"
    return use


def _contents_by_lot(contents: pd.DataFrame) -> pd.DataFrame:
    """Per lot: amenities, establishments, whether all are offices, manufacturing share."""
    establishments = contents[contents["element_type"] == "establishment"]
    workers = establishments["num_workers"].astype(float)
    manufacturing = workers.where(establishments["category"] == "manufacturing", 0.0)
    grouped = pd.DataFrame(
        {
            "lot_id": establishments["lot_id"],
            "office": is_office(establishments["scian_code"]),
            "workers": workers,
            "manufacturing": manufacturing,
        }
    ).groupby("lot_id")
    by_lot = pd.DataFrame(
        {
            "establishments": grouped.size(),
            "all_offices": grouped["office"].all(),
            "manufacturing_share": grouped["manufacturing"].sum()
            / grouped["workers"].sum(),
        }
    )
    amenities = contents.loc[contents["element_type"] == "amenity", "lot_id"]
    return by_lot.join(amenities.value_counts().rename("amenities"), how="outer")


def classify_land_use(
    lots: gpd.GeoDataFrame,
    zoning: pd.Series,
    contents: pd.DataFrame,
    *,
    mf_min_units: float,
) -> tuple[pd.Series, pd.Series]:
    """``land_use`` per lot, from what it holds, and ``unit_size_m2``.

    reimagina_urbano_api ``ingestion_pipelines.md`` (*Land use from contents*), with
    homes estimated from the census until the space allocation exists: a built lot
    (over ``MIN_BUILT_M2``) whose zone allows housing holds homes when its block
    has any, its share of the block's homes by built-up area. ``contents`` has one
    row per amenity or establishment: ``lot_id``, ``element_type``, ``category``,
    ``scian_code`` and ``num_workers``.

    - Nothing: its zone's use (``zone_use``).
    - Only amenities: ``amenity``.
    - Homes and establishments: ``mixed_use``.
    - Homes only: ``multifamily`` with at least ``mf_min_units`` homes, or one home
      in a building of ``MF_BUILTUP_M2`` and ``MF_LEVELS``; else ``single_family``.
    - Establishments only: ``office`` if all are offices, ``industrial`` if mostly
      manufacturing by workers, else ``commerce``.

    Amenities don't change the class of a lot that also holds homes or
    establishments. ``unit_size_m2`` is the block's multifamily and mixed-use
    built-up area over its homes not in single-family lots.
    """
    vivtot = pd.to_numeric(lots["vivtot"], errors="coerce").astype(float)
    builtup = pd.to_numeric(lots["builtup_area"], errors="coerce").fillna(0.0)
    levels = pd.to_numeric(lots["num_levels"], errors="coerce").fillna(0.0)
    cvegeo = lots["cvegeo"]
    is_built = builtup > MIN_BUILT_M2

    housing = zoning.isin(SINGLE_FAM | MIXED_ZONINGS)
    homes = housing & is_built & (vivtot != 0)
    built_homes = builtup.where(homes, 0.0)
    expected_units = vivtot * built_homes / built_homes.groupby(cvegeo).transform("sum")
    has_excess = vivtot > homes.groupby(cvegeo).transform("sum")
    several = homes & has_excess & (expected_units >= mf_min_units)
    big_building = (builtup >= MF_BUILTUP_M2) & (levels >= MF_LEVELS)

    held = lots[["lot_id"]].join(_contents_by_lot(contents), on="lot_id")
    establishments = held["establishments"].fillna(0).to_numpy() > 0
    amenities = held["amenities"].fillna(0).to_numpy() > 0
    all_offices = held["all_offices"].fillna(False).to_numpy(dtype=bool)
    industrial = held["manufacturing_share"].fillna(0).to_numpy() > 0.5

    land_use = zone_use(zoning)
    land_use[amenities & ~homes & ~establishments] = "amenity"
    only_business = establishments & ~homes
    land_use[only_business] = "commerce"
    land_use[only_business & industrial] = "industrial"
    land_use[only_business & all_offices] = "office"
    land_use[homes & establishments] = "mixed_use"
    homes_only = homes & ~establishments
    land_use[homes_only] = "single_family"
    land_use[homes_only & (several | big_building)] = "multifamily"

    is_sf = land_use == "single_family"
    is_mfmx = land_use.isin(["multifamily", "mixed_use"])
    vivtot_by_block = vivtot.groupby(cvegeo).first()
    builtup_mfmx = builtup.where(is_mfmx, 0.0).groupby(cvegeo).sum()
    reserved = (is_sf & is_built).groupby(cvegeo).sum().astype(float)
    mfmx_built = (is_mfmx & is_built).groupby(cvegeo).sum().astype(float)
    remaining = (vivtot_by_block - reserved).clip(lower=mfmx_built)
    unit_size_by_block = (builtup_mfmx / remaining).where(
        (remaining > 0) & (builtup_mfmx > 0)
    )
    unit_size = cvegeo.map(unit_size_by_block).where(is_mfmx)
    logger.info(
        "land_use: %s",
        {str(k): int(v) for k, v in land_use.value_counts(dropna=False).items()},
    )
    return land_use, unit_size


# --------------------------------------------------------------------- elements


def block_elements(blocks: gpd.GeoDataFrame) -> tuple[list[Element], dict[str, str]]:
    """One element per block, and the cvegeo -> id map lots hang from."""
    ids = {
        c: stable_element_uuid("block", c)
        for c in sorted(blocks["cvegeo"].dropna().unique())
    }
    elements = []
    for cvegeo, vivtot, prom_ocup, geometry in zip(
        blocks["cvegeo"],
        blocks["vivtot"],
        blocks["prom_ocup"],
        blocks.geometry,
        strict=True,
    ):
        attributes = {
            "vivtot": _int_or_none(vivtot),
            "people_per_household": people_per_household(prom_ocup),
        }
        elements.append(
            Element(
                id=ids[cvegeo],
                type="block",
                parent_element_id=None,
                geometry=geometry,
                attributes={k: v for k, v in attributes.items() if v is not None},
            )
        )
    return elements, ids


def lot_attributes(  # noqa: PLR0913, PLR0917
    land_use: Any,  # noqa: ANN401
    cos: Any,  # noqa: ANN401
    num_levels: Any,  # noqa: ANN401
    area: Any,  # noqa: ANN401
    unit_size_m2: Any,  # noqa: ANN401
    prom_ocup: Any,  # noqa: ANN401
    occupancy: float | None,
    giro: Any,  # noqa: ANN401
) -> dict[str, Any]:
    lu = _str_or_none(land_use)
    residential = lu in ("single_family", "multifamily", "mixed_use")
    attributes = {
        "land_use": lu,
        "cos": _float_or_none(cos),
        "num_levels": _float_or_none(num_levels),
        "area": _float_or_none(area),
        "unit_size_m2": (
            _float_or_none(unit_size_m2) if lu in ("multifamily", "mixed_use") else None
        ),
        "people_per_household": people_per_household(prom_ocup)
        if residential
        else None,
        "occupancy_rate": occupancy if residential else None,
        "commerce_type": (
            COMMERCE_TYPE_BY_GIRO.get(normalize_giro(giro) or "")
            if lu == "commerce"
            else None
        ),
    }
    return {key: value for key, value in attributes.items() if value is not None}
