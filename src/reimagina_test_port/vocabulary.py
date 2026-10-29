"""
Vocabulario canónico de amenidades.

Una sola definición para todo el sistema: `lot.amenity_type`, `points_of_interest.type`
y las categorías del modelo de accesibilidad. Antes había dos vocabularios — los slugs
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


# Coherencia interna: si alguien añade una amenidad, tiene que declararla en los tres
# lugares. Falla al importar, no en producción.
_missing_params = set(AMENITY_TYPES) - set(AMENITY_ACCESSIBILITY_PARAMS)
_unknown_targets = (
    set(AMENITY_TYPE_BY_RAW_CATEGORY.values()) | set(AMENITY_TYPE_BY_LEGACY_SLUG.values())
) - set(AMENITY_TYPES)
if _missing_params:
    raise RuntimeError(f"Amenidades sin parámetros de accesibilidad: {sorted(_missing_params)}")
if _unknown_targets:
    raise RuntimeError(f"Mapeos que apuntan fuera del enum: {sorted(_unknown_targets)}")
