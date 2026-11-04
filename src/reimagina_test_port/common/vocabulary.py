"""
Vocabulario canónico de amenidades.

Una sola definición para el modelo de accesibilidad: sus 13 tipos, y cómo se derivan
de la `category` y el `scian_code` de los elementos amenity y establishment
(`amenity_type_of`). Antes había dos vocabularios — los slugs
de `POIS_CONFIG` (`Parques_recreativos`, `Educacion_primaria`) en los datos, y el enum
de `elements.md` (`park`, `hospital`, `school`) en la documentación.

Se adopta el de `elements.md`: minúsculas, inglés, snake_case. Los cinco niveles
educativos se conservan por separado, no se colapsan en `school`, porque cada uno tiene
su propio radio de cobertura en el modelo de accesibilidad.

Solo stdlib: tiene que importar en la imagen light y desde los notebooks.
"""

from __future__ import annotations

__all__ = [
    "AMENITY_TYPES",
    "AMENITY_TYPE_BY_RAW_CATEGORY",
    "AMENITY_TYPE_BY_LEGACY_SLUG",
    "LEGACY_SLUG_BY_AMENITY_TYPE",
    "AMENITY_ACCESSIBILITY_PARAMS",
    "normalize_amenity_type",
    "amenity_type_of",
]


# =============================================================================
# El enum canónico
# =============================================================================

AMENITY_TYPES = (
    "park",
    "daycare",
    "preschool",
    "primary_school",
    "secondary_school",
    "high_school",
    "university",
    "hospital",
    "clinic",
    "pharmacy",
    "cinema",
    "gym",
    "recreation",
)


# =============================================================================
# Origen: categorías crudas de cleaning/equipamientos (DENUE + IMIP)
# =============================================================================

AMENITY_TYPE_BY_RAW_CATEGORY = {
    "Hospital": "hospital",
    "Consultorio medico": "clinic",
    "Farmacia": "pharmacy",
    "Parques": "park",
    "Deporte": "gym",
    "Cine": "cinema",
    "Cultural recreativo": "recreation",
    "Guarderia": "daycare",
    "Educación Preescolar": "preschool",
    "Educación Primaria": "primary_school",
    "Educación Secundaria": "secondary_school",
    "Educación Media Superior": "high_school",
    "Educación Superior": "university",
}


# =============================================================================
# Migración: los slugs anteriores (llaves de POIS_CONFIG)
#
# Se conserva el mapeo para poder releer datos ya escritos y para construir la lista de
# llaves de indicadores a retirar. Se puede borrar cuando no queden datos con los slugs
# viejos ni en Azure ni en la BD.
# =============================================================================

AMENITY_TYPE_BY_LEGACY_SLUG = {
    "Hospital_general": "hospital",
    "Consultorios_medicos": "clinic",
    "Farmacia": "pharmacy",
    "Parques_recreativos": "park",
    "Clubs_deportivos_y_acondicionamiento_fisico": "gym",
    "Cine": "cinema",
    "Otros_servicios_recreativos": "recreation",
    "Guarderia": "daycare",
    "Educacion_preescolar": "preschool",
    "Educacion_primaria": "primary_school",
    "Educacion_secundaria": "secondary_school",
    "Educacion_media_superior": "high_school",
    "Educacion_superior": "university",
}

LEGACY_SLUG_BY_AMENITY_TYPE = {
    amenity: slug for slug, amenity in AMENITY_TYPE_BY_LEGACY_SLUG.items()
}


# =============================================================================
# Parámetros del modelo de accesibilidad, por amenidad
#
# Vive aquí y no en accessibility.py para que sea legible sin instalar el stack geo, y
# para que el vocabulario y sus parámetros no puedan desincronizarse.
#
# radius: metros de cobertura · time: minutos caminando · weight: peso en el score
# =============================================================================

_EQUAL_WEIGHT = 1 / 13

AMENITY_ACCESSIBILITY_PARAMS = {
    "daycare":          {"radius": 750,    "time": 10, "weight": _EQUAL_WEIGHT},
    "preschool":        {"radius": 750,    "time": 10, "weight": _EQUAL_WEIGHT},
    "primary_school":   {"radius": 1_000,  "time": 15, "weight": _EQUAL_WEIGHT},
    "secondary_school": {"radius": 500,    "time": 10, "weight": _EQUAL_WEIGHT},
    "high_school":      {"radius": 5_000,  "time": 30, "weight": _EQUAL_WEIGHT},
    "university":       {"radius": 50_000, "time": 50, "weight": _EQUAL_WEIGHT},  # TODO: validar
    "park":             {"radius": 670,    "time": 10, "weight": _EQUAL_WEIGHT},
    "cinema":           {"radius": 1_150,  "time": 15, "weight": _EQUAL_WEIGHT},
    "gym":              {"radius": 1_000,  "time": 15, "weight": _EQUAL_WEIGHT},
    "hospital":         {"radius": 50_000, "time": 50, "weight": _EQUAL_WEIGHT},  # TODO: validar
    "clinic":           {"radius": 2_000,  "time": 25, "weight": _EQUAL_WEIGHT},
    "pharmacy":         {"radius": 1_500,  "time": 20, "weight": _EQUAL_WEIGHT},
    "recreation":       {"radius": 50_000, "time": 50, "weight": _EQUAL_WEIGHT},  # TODO: validar
}


def normalize_amenity_type(value: str | None) -> str | None:
    """Acepta el vocabulario canónico, un slug viejo o una categoría cruda.

    Existe para que la migración no sea un big bang: mientras haya geoparquets en Azure
    con los slugs anteriores, la lectura sigue funcionando.
    """
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text in AMENITY_TYPES:
        return text
    if text in AMENITY_TYPE_BY_LEGACY_SLUG:
        return AMENITY_TYPE_BY_LEGACY_SLUG[text]
    if text in AMENITY_TYPE_BY_RAW_CATEGORY:
        return AMENITY_TYPE_BY_RAW_CATEGORY[text]
    raise ValueError(
        f"Amenidad desconocida: {value!r}. Valores válidos: {', '.join(AMENITY_TYPES)}"
    )


# =============================================================================
# Puente: elementos amenity / establishment -> tipo de accesibilidad
#
# Los elementos usan las categorías de urban_fabric.md (`category`) y su código SCIAN;
# el modelo de accesibilidad sigue con los 13 tipos de arriba hasta que sus
# indicadores pasen a categorías. Gana el prefijo SCIAN más largo; sin código (el
# archivo de equipamientos), decide la categoría. La educación sin código SCIAN no
# cuenta: el DENUE ya trae las escuelas y contarlas dos veces duplicaría destinos.
# =============================================================================

AMENITY_TYPE_BY_SCIAN_PREFIX = {
    "61111": "preschool",
    "61112": "primary_school",
    "61113": "secondary_school",
    "61114": "secondary_school",
    "61115": "high_school",
    "61116": "high_school",
    "6112": "university",
    "6113": "university",
    "6244": "daycare",
    "622": "hospital",
    "6211": "clinic",
    "6214": "clinic",
    "46411": "pharmacy",
    "512130": "cinema",
}
AMENITY_TYPE_BY_CATEGORY = {
    "public_space": "park",
    "sports_recreation": "gym",
    "cultural_facility": "recreation",
}
_SCIAN_PREFIXES = sorted(AMENITY_TYPE_BY_SCIAN_PREFIX, key=len, reverse=True)


def amenity_type_of(category: str | None, scian_code: str | None) -> str | None:
    """Tipo de accesibilidad de un elemento amenity o establishment, o None."""
    code = "" if scian_code is None else str(scian_code).strip()
    for prefix in _SCIAN_PREFIXES:
        if code.startswith(prefix):
            return AMENITY_TYPE_BY_SCIAN_PREFIX[prefix]
    return AMENITY_TYPE_BY_CATEGORY.get(category or "")


# Coherencia interna: si alguien añade una amenidad, tiene que declararla en los tres
# lugares. Falla al importar, no en producción.
_missing_params = set(AMENITY_TYPES) - set(AMENITY_ACCESSIBILITY_PARAMS)
_unknown_targets = (
    set(AMENITY_TYPE_BY_RAW_CATEGORY.values())
    | set(AMENITY_TYPE_BY_LEGACY_SLUG.values())
    | set(AMENITY_TYPE_BY_SCIAN_PREFIX.values())
    | set(AMENITY_TYPE_BY_CATEGORY.values())
) - set(AMENITY_TYPES)
if _missing_params:
    raise RuntimeError(f"Amenidades sin parámetros de accesibilidad: {sorted(_missing_params)}")
if _unknown_targets:
    raise RuntimeError(f"Mapeos que apuntan fuera del enum: {sorted(_unknown_targets)}")
