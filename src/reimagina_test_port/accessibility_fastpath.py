"""
Reemplazo vectorizado de `accessibility.metrics.get_accessibility_metrics`.

El paquete `accessibility` vive en otro repo. Esta es la misma función, con las dos
`.apply(..., axis=1)` reescritas en operaciones de numpy/pandas — mismos resultados, sin
recorrer el DataFrame fila por fila desde Python.

POR QUÉ IMPORTA
    El frame de proximidades tiene ~1.2M filas (13 categorías × 19,058 orígenes × 5 POIs).
    La segunda `.apply` además hace DOS búsquedas por índice en cada fila (`in ...index` y
    `.loc[...]`). En el job medido, esta función es el grueso de los ~2:48 que tarda la
    accesibilidad.

POR QUÉ MONKEYPATCH Y NO UN FORK
    `compute_pois_accessibility_scores` resuelve `get_accessibility_metrics` como global de
    su módulo en cada llamada, así que sustituir el atributo basta. Se reemplaza SOLO la
    función caliente: el resto del paquete —incluidos sus arreglos futuros— sigue vigente.
    La alternativa era copiar el bucle de `compute_pois_accessibility_scores` aquí, que
    duplica lógica y se desincroniza en silencio.

EL RIESGO Y CÓMO SE CONTIENE
    Un monkeypatch se rompe callado si el paquete cambia. Contra eso:
      1. `apply_fastpath()` verifica la firma antes de sustituir; si no coincide, no parcha
         y avisa — el job sigue corriendo con la versión original, más lenta pero correcta.
      2. `ACCESSIBILITY_FASTPATH=0` lo desactiva sin tocar código ni volver a desplegar.
      3. `scripts/test/test_accessibility_fastpath.py` compara ambas implementaciones sobre
         datos aleatorios y falla si difieren.

EQUIVALENCIA EXACTA, INCLUIDAS LAS RAREZAS
    No se "arreglan" comportamientos de la original aunque parezcan bugs, porque cambiarlos
    movería los indicadores sin avisar:
      · Si `max == min`, la normalización min-max divide 0/0 y devuelve NaN. Se conserva.
      · Un `destination_id` fuera de `destinations_agg` da score 0, no NaN. Se conserva.
      · `capacity` NaN produce `opportunities_ratio` NaN, que luego cae en el filtro
        `> 0`. Se conserva.
"""

from __future__ import annotations

import inspect
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["fast_get_accessibility_metrics", "apply_fastpath"]

_EXPECTED_SIGNATURE = (
    "proximities",
    "destinations",
    "weights",
    "radius",
    "adjustment_factor",
)


def fast_get_accessibility_metrics(
    proximities: pd.DataFrame,
    destinations: pd.DataFrame,
    weights: pd.Series,
    radius: float,
    adjustment_factor: float,
) -> pd.DataFrame:
    """Idéntica a la del paquete, sin `.apply(axis=1)`."""
    from reimagina_test_port.accessibility_metrics import POI_NETWORK_COL, POI_SNAP_DISTANCE_COL

    proximities = proximities.join(weights, on="origin_id")
    proximities = proximities.join(destinations[["capacity"]], on="destination_id")

    proximities["gravity"] = 1 / np.exp(
        1 / (radius / adjustment_factor) * proximities["distance"]
    )
    proximities["reach"] = proximities["gravity"] * proximities[weights.name]

    dest_pob_reach = proximities.groupby("destination_id")["reach"].sum().rename("pob_reach")

    destinations_agg = proximities.groupby("destination_id").agg(
        {"reach": "sum", "capacity": "first"}
    )

    # ORIGINAL: .apply(lambda x: x["capacity"] / x["reach"] if x["reach"] > 0 else 0, axis=1)
    # `where` evita dividir entre cero sin cambiar el resultado: donde reach <= 0 el valor
    # se descarta igual. Un capacity NaN sigue propagando NaN, como en la original.
    reach = destinations_agg["reach"].to_numpy(dtype="float64")
    capacity = destinations_agg["capacity"].to_numpy(dtype="float64")
    positive = reach > 0
    destinations_agg["opportunities_ratio"] = np.where(
        positive, capacity / np.where(positive, reach, 1.0), 0.0
    )
    destinations_agg = destinations_agg[destinations_agg["opportunities_ratio"] > 0]

    # ORIGINAL: .apply(lambda x: ratio.loc[x["destination_id"]] * x["gravity"]
    #                            if x["destination_id"] in ratio.index else 0, axis=1)
    # `map` hace la misma búsqueda de forma vectorizada; los ausentes salen NaN y el
    # fillna(0) reproduce la rama `else 0`.
    ratio_by_destination = proximities["destination_id"].map(
        destinations_agg["opportunities_ratio"]
    )
    proximities["accessibility_score"] = ratio_by_destination.fillna(0.0) * proximities["gravity"]

    proximities = proximities.join(dest_pob_reach, on="destination_id")
    proximities["attraction"] = proximities["capacity"]

    temp_proximities = (
        proximities.reset_index()
        .groupby("origin_id")
        .agg({"accessibility_score": "sum", "time": "min"})
        .assign(
            accessibility_score=lambda df: (
                df["accessibility_score"] - df["accessibility_score"].min()
            )
            / (df["accessibility_score"].max() - df["accessibility_score"].min()),
        )
    )

    detail_col_names = ["distance", "destination_id", "gravity", "pob_reach", "attraction"]
    detail_col_names.extend(
        col for col in (POI_NETWORK_COL, POI_SNAP_DISTANCE_COL) if col in proximities.columns
    )
    detail_cols = proximities.query("num_poi==1").set_index("origin_id")[detail_col_names]
    return temp_proximities.join(detail_cols)


def apply_fastpath() -> bool:
    """Sustituye la función del paquete. Devuelve si el parche quedó activo.

    No revienta si algo no cuadra: prefiere correr lento y correcto a no correr.
    """
    if os.getenv("ACCESSIBILITY_FASTPATH", "1").strip().lower() in {"0", "false", "no"}:
        logger.info("ACCESSIBILITY_FASTPATH=0: se usa la implementación original.")
        return False

    from reimagina_test_port import accessibility_metrics as metrics

    original = getattr(metrics, "get_accessibility_metrics", None)
    if original is None:
        logger.warning("El paquete accessibility no expone get_accessibility_metrics; sin parche.")
        return False
    if getattr(original, "_is_fastpath", False):
        return True

    found = tuple(inspect.signature(original).parameters)
    if found != _EXPECTED_SIGNATURE:
        logger.warning(
            "get_accessibility_metrics cambió de firma (%s, se esperaba %s): NO se parcha. "
            "El job corre con la versión original; revisa src/indicators_core/accessibility_fastpath.py",
            found,
            _EXPECTED_SIGNATURE,
        )
        return False

    fast_get_accessibility_metrics._is_fastpath = True
    fast_get_accessibility_metrics._original = original
    metrics.get_accessibility_metrics = fast_get_accessibility_metrics
    logger.info("get_accessibility_metrics: usando la versión vectorizada (fastpath).")
    return True
