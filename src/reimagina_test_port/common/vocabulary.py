"""
Tipos de destino del modelo de accesibilidad.

Son los tipos activos de `poi_catalog` (API), los mismos que devuelve `/pois/search`.
Qué elementos son de cada tipo lo dicen las reglas de `poi_catalog` (`rule_criteria`),
que se leen de la base en cada corrida (`scenario.source.read_pois`). Aquí sólo viven
los tipos, porque las columnas de salida se declaran antes de leer la base, y los
parámetros del modelo de gravedad, que `poi_catalog` todavía no tiene.

Un tipo activo en `poi_catalog` que no esté aquí se ignora con un aviso; uno de aquí
sin destinos (inactivo o sin elementos) queda nulo.

Solo stdlib: tiene que importar en la imagen light y desde los notebooks.
"""

from __future__ import annotations

__all__ = [
    "DESTINATION_ACCESSIBILITY_PARAMS",
    "DESTINATION_COVERAGE_M",
    "DESTINATION_TYPES",
]


DESTINATION_TYPES = (
    "public_space",
    "fresh_food",
    "convenience_store",
    "petrol_station",
    "establishment",
    "manufacturing",
    "education",
    "primary_education",
    "health",
    "care",
    "community_culture_recreation",
    "police_station",
    "fire_station",
    "transit_stop",
    "lpugs",
)

# Metros de cobertura de cada tipo para las corridas sin base (snapshot): los
# `params.coverage_m` de `poi_catalog`. Con base se leen de ahí.
DESTINATION_COVERAGE_M = {
    "public_space": 300,
    "fresh_food": 300,
    "convenience_store": 300,
    "petrol_station": 250,
    "establishment": 500,
    "manufacturing": 1_000,
    "education": 300,
    "primary_education": 300,
    "health": 300,
    "care": 300,
    "community_culture_recreation": 300,
    "police_station": 2_000,
    "fire_station": 3_000,
    "transit_stop": 300,
    "lpugs": 500,
}


# =============================================================================
# Parámetros del modelo de accesibilidad, por tipo de destino
#
# radius: metros de la caída del modelo de gravedad · time: minutos caminando ·
# weight: peso en el score
#
# Provisionales, sin calibrar: los de los tipos viejos equivalentes (`park` 670 m,
# `primary_school` 1 km, `clinic` 2 km, `pharmacy` 1.5 km, `daycare` 750 m, `gym`
# 1 km); policía y bomberos, su umbral de cobertura (2 y 3 km) y, para los demás
# que no tenían, una caminata de 5 a 15 minutos.
# =============================================================================

_EQUAL_WEIGHT = 1 / len(DESTINATION_TYPES)

DESTINATION_ACCESSIBILITY_PARAMS = {
    "public_space": {"radius": 670, "time": 10, "weight": _EQUAL_WEIGHT},
    "fresh_food": {"radius": 1_000, "time": 15, "weight": _EQUAL_WEIGHT},
    "convenience_store": {"radius": 500, "time": 10, "weight": _EQUAL_WEIGHT},
    "petrol_station": {"radius": 1_500, "time": 20, "weight": _EQUAL_WEIGHT},
    "establishment": {"radius": 500, "time": 10, "weight": _EQUAL_WEIGHT},
    "manufacturing": {"radius": 5_000, "time": 30, "weight": _EQUAL_WEIGHT},
    "education": {"radius": 1_000, "time": 15, "weight": _EQUAL_WEIGHT},
    "primary_education": {"radius": 1_000, "time": 15, "weight": _EQUAL_WEIGHT},
    "health": {"radius": 2_000, "time": 25, "weight": _EQUAL_WEIGHT},
    "care": {"radius": 750, "time": 10, "weight": _EQUAL_WEIGHT},
    "community_culture_recreation": {
        "radius": 1_000,
        "time": 15,
        "weight": _EQUAL_WEIGHT,
    },
    "police_station": {"radius": 2_000, "time": 25, "weight": _EQUAL_WEIGHT},
    "fire_station": {"radius": 3_000, "time": 40, "weight": _EQUAL_WEIGHT},
    "transit_stop": {"radius": 500, "time": 5, "weight": _EQUAL_WEIGHT},
    "lpugs": {"radius": 670, "time": 10, "weight": _EQUAL_WEIGHT},
}


# Coherencia interna: falla al importar, no en producción.
if set(DESTINATION_TYPES) != set(DESTINATION_ACCESSIBILITY_PARAMS):
    raise RuntimeError(
        "DESTINATION_TYPES y DESTINATION_ACCESSIBILITY_PARAMS no coinciden"
    )
