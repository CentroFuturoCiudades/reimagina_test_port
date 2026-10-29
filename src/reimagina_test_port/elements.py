"""Blocks and lots to Reimagina elements: ids, parents, land use and attributes.

Moved from exe_reimagina ``src/migration/elements.py`` (and the POI normalization
of ``pois.py``). Lot attributes follow ``baseline_setup.md``: land_use, cos,
num_levels, area, unit_size_m2 (multifamily and mixed_use), people_per_household and
occupancy_rate (residential), amenity_type and commerce_type. Block attributes:
vivtot and people_per_household.
"""

import logging
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry.base import BaseGeometry

from reimagina_test_port.calculation import MIN_BUILT_M2
from reimagina_test_port.cadastral import stable_element_uuid
from reimagina_test_port.vocabulary import normalize_amenity_type

logger = logging.getLogger(__name__)

# Zoning vocabularies of Mexicali (USOSUELO24) and Tlalpan (uso_descri), lower case.
AMENITY_ZONINGS = {
    "equipamiento", "área verde", "area verde", "equipamiento público y privado",
    "equipamiento rural", "equipamiento deportivo", "equipamiento de servicio",
    "equipamiento de servicios", "equipamiento mortuorio", "áreas verdes",
    "espacio abierto", "espacios abiertos", "área de servicios públicos",
    "colegio de ingenieros civiles",
}  # fmt: skip
# Green areas and open spaces without a POI get amenity_type = park.
GREEN_ZONINGS = {
    "área verde",
    "area verde",
    "áreas verdes",
    "espacio abierto",
    "espacios abiertos",
}
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
# Amenities that occupy the whole lot turn it into `amenity`; the rest (pharmacy,
# clinic, daycare, preschool, gym) are neighbourhood services: the lot becomes
# `mixed_use` and keeps its housing.
EXCLUSIVE_AMENITY_TYPES = {
    "park", "primary_school", "secondary_school", "high_school", "university",
    "hospital", "cinema", "recreation",
}  # fmt: skip
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


# ------------------------------------------------------------------------- POIs


def prepare_pois(pois: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """POI points in EPSG:4326 with a canonical ``amenity_type``."""
    pois = pois.to_crs(epsg=4326)
    geometry = shapely.force_2d(pois.geometry.to_numpy())
    if (
        not (shapely.get_type_id(geometry) == 0).all()
        or not shapely.is_valid(geometry).all()
    ):
        msg = "POI source must contain valid Point geometries only."
        raise ValueError(msg)
    return gpd.GeoDataFrame(
        {"amenity_type": pois["category"].astype(str).map(normalize_amenity_type)},
        geometry=geometry,
        crs="EPSG:4326",
    )


def amenity_type_by_lot(
    lots: gpd.GeoDataFrame,
    zoning: pd.Series,
    pois: gpd.GeoDataFrame,
    *,
    metric_crs: str,
    poi_max_distance_m: float,
) -> dict[str, str]:
    """``amenity_type`` of each lot holding a POI, with any zoning.

    Each POI counts for one lot: the one containing it or, outside every lot, the
    nearest within ``poi_max_distance_m``. With several POIs the most frequent type
    wins, then exclusive amenities. Green zonings without a POI become ``park``.
    """
    points = pois.to_crs(metric_crs)
    shapes = lots[["lot_id", "geometry"]].to_crs(metric_crs)
    inside = gpd.sjoin(points, shapes, how="inner", predicate="within")
    inside = inside.sort_values("index_right", kind="mergesort")
    inside = inside[~inside.index.duplicated()]
    near = points.drop(index=inside.index).sjoin_nearest(
        shapes,
        how="inner",
        max_distance=poi_max_distance_m,
        distance_col="poi_distance",
    )
    near = near.sort_values(["poi_distance", "index_right"], kind="mergesort")
    near = near[~near.index.duplicated()]
    assigned = pd.concat(
        [inside[["lot_id", "amenity_type"]], near[["lot_id", "amenity_type"]]]
    )
    counts = (
        assigned.groupby(["lot_id", "amenity_type"]).size().rename("pois").reset_index()
    )
    counts["exclusive"] = counts["amenity_type"].isin(EXCLUSIVE_AMENITY_TYPES)
    best = counts.sort_values(
        ["lot_id", "pois", "exclusive", "amenity_type"],
        ascending=[True, False, False, True],
        kind="mergesort",
    ).drop_duplicates("lot_id")
    by_lot = dict(zip(best["lot_id"], best["amenity_type"], strict=True))
    green_without_poi = lots.loc[
        zoning.isin(GREEN_ZONINGS) & ~lots["lot_id"].isin(by_lot)
    ]
    for lot_id in green_without_poi["lot_id"]:
        by_lot[lot_id] = "park"
    logger.info(
        "POIs: %s | inside a lot: %s | within %s m: %s | lots with an amenity: %s",
        f"{len(points):,}",
        f"{len(inside):,}",
        poi_max_distance_m,
        f"{len(near):,}",
        f"{len(by_lot):,}",
    )
    return by_lot


# -------------------------------------------------------------------- land use


def classify_land_use(
    lots: gpd.GeoDataFrame,
    zoning: pd.Series,
    amenity_types: dict[str, str],
    *,
    mf_min_units: float,
) -> tuple[pd.Series, pd.Series]:
    """``land_use`` per lot and block ``unit_size_m2`` for multifamily and mixed use.

    1. Zoning gives the base use.
    2. A residential or unzoned lot holding a POI becomes ``amenity`` (exclusive
       amenity, or unzoned) or ``mixed_use`` (neighbourhood service).
    3. Remaining residential lots split with the census: the block's dwellings are
       shared among its built residential lots by built-up area, and a lot is
       multifamily if it gets at least ``mf_min_units`` and the block has more
       dwellings than built residential lots.
    """
    vivtot = pd.to_numeric(lots["vivtot"], errors="coerce").astype(float)
    builtup = pd.to_numeric(lots["builtup_area"], errors="coerce").fillna(0.0)
    cvegeo = lots["cvegeo"]
    is_built = builtup > MIN_BUILT_M2

    land_use = pd.Series([None] * len(lots), index=lots.index, dtype=object)
    land_use[zoning.isin(SINGLE_FAM)] = "single_family"
    land_use[zoning.isin(MIXED_ZONINGS)] = "mixed_use"
    land_use[zoning.isin(AMENITY_ZONINGS)] = "amenity"
    land_use[zoning.isin(COMMERCE_ZONINGS)] = "commerce"

    poi_type = lots["lot_id"].map(amenity_types)
    has_poi = poi_type.notna()
    exclusive = poi_type.isin(EXCLUSIVE_AMENITY_TYPES)
    is_hab = land_use == "single_family"
    land_use[has_poi & (land_use.isna() | (is_hab & exclusive))] = "amenity"
    land_use[has_poi & is_hab & ~exclusive] = "mixed_use"

    is_hab = land_use == "single_family"
    residential_built = land_use.isin(["single_family", "mixed_use"]) & is_built
    built_residential = builtup.where(residential_built, 0.0)
    expected_units = (
        vivtot * built_residential / built_residential.groupby(cvegeo).transform("sum")
    )
    has_excess = vivtot > residential_built.groupby(cvegeo).transform("sum")
    multifamily = is_hab & is_built & has_excess & (expected_units >= mf_min_units)
    land_use[multifamily] = "multifamily"

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
    amenity_type: str | None,
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
        "amenity_type": amenity_type,
        "commerce_type": (
            COMMERCE_TYPE_BY_GIRO.get(normalize_giro(giro) or "")
            if lu == "commerce"
            else None
        ),
    }
    return {key: value for key, value in attributes.items() if value is not None}


def lot_elements(  # noqa: PLR0913
    lots: gpd.GeoDataFrame,
    block_ids: dict[str, str],
    pois: gpd.GeoDataFrame,
    *,
    cve_code: str,
    metric_crs: str,
    poi_max_distance_m: float,
    mf_min_units: float,
) -> list[Element]:
    """One element per lot (EPSG:4326 geometry), parented to its block."""
    missing = sorted(set(lots["cvegeo"].dropna()) - set(block_ids))
    if missing:
        msg = f"{len(missing)} lot cvegeo without a block, e.g. {missing[:5]}"
        raise ValueError(msg)
    lots = lots.to_crs(epsg=4326)
    zoning = lots["zoning"].astype(str).str.strip().str.lower()
    amenity_types = amenity_type_by_lot(
        lots, zoning, pois, metric_crs=metric_crs, poi_max_distance_m=poi_max_distance_m
    )
    land_use, unit_size = classify_land_use(
        lots, zoning, amenity_types, mf_min_units=mf_min_units
    )
    giro = lots["Giro"] if "Giro" in lots else pd.Series(None, index=lots.index)
    elements = []
    for row in zip(
        lots["lot_id"],
        lots["cvegeo"],
        land_use,
        lots["cos"],
        lots["num_levels"],
        lots["lot_area"],
        unit_size,
        lots["prom_ocup"],
        lots["tvivparhab"],
        lots["vivtot"],
        giro,
        lots.geometry,
        strict=True,
    ):
        lot_id, cvegeo, lu, cos, levels, area, us, prom, tvph, vivtot, g, geometry = row
        elements.append(
            Element(
                # The city key keeps lot ids distinct across cities in one database.
                id=stable_element_uuid("lot", cve_code, lot_id),
                type="lot",
                parent_element_id=block_ids[cvegeo],
                geometry=geometry,
                attributes=lot_attributes(
                    lu,
                    cos,
                    levels,
                    area,
                    us,
                    prom,
                    occupancy_rate(tvph, vivtot),
                    g,
                    amenity_types.get(lot_id),
                ),
            )
        )
    return elements
