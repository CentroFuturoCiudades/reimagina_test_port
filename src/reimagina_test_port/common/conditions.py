"""Evaluador del lenguaje de *conditions* (`docs/design/3-how/platform/contracts.md`).

Un solo lenguaje para todo "cuando" y todo "debe": `applies_when` de un atributo, las
reglas de `poi_catalog`, las restricciones.

```json
{ "land_use": "amenity" }                                   // igual
{ "land_use": { "in": ["multifamily", "mixed_use"] } }
{ "num_levels": { "lte": 4 } }
{ "cos": { "between": [0.2, 0.8] } }
{ "scian_code": { "exists": true } }
{ "all": [ { "type": "establishment" }, { "category": "food_retail" } ] }
{ "any": [ ... ] }
{ "not": { ... } }
{ "parent.type": { "in": ["block", "development"] } }
```

- Llaves: un atributo del elemento; `type` para su tipo; `parent.<llave>` para las de
  su padre. Varias llaves en un objeto son `all`.
- Un valor solo es `eq`. Varios operadores en un objeto son `all`.
- Un atributo que falta hace falso todo operador salvo `exists` y `ne`.

Copia de `reimagina_urbano_api` `src/shared/conditions.py` (c81d507), sin sus
prefiltros de SQL (`element_types`, `anchors`): el modelo de accesibilidad evalúa las
reglas de `poi_catalog` sobre los elementos que ya leyó. Si cambia allá, cambia aquí.
"""

from collections.abc import Iterable, Mapping
from typing import Any

COMBINATORS = ("all", "any", "not")
OPERATORS = ("eq", "ne", "in", "not_in", "lt", "lte", "gt", "gte", "between", "exists")
PARENT = "parent."

_MISSING = object()


class ConditionError(ValueError):
    """Una condicion mal formada: operador desconocido o argumento invalido."""


def evaluate(
    condition: Mapping[str, Any],
    element: Mapping[str, Any],
    parent: Mapping[str, Any] | None = None,
) -> bool:
    """Si ``element`` (atributos y ``type``) cumple ``condition``.

    ``parent`` son los atributos y el ``type`` del padre, para las llaves
    ``parent.<llave>``; sin padre, esas llaves faltan.
    """
    if not isinstance(condition, Mapping):
        raise ConditionError(f"Una condicion es un objeto, no {condition!r}")
    for key, spec in condition.items():
        if key == "all":
            ok = all(evaluate(c, element, parent) for c in _conditions(key, spec))
        elif key == "any":
            ok = any(evaluate(c, element, parent) for c in _conditions(key, spec))
        elif key == "not":
            ok = not evaluate(spec, element, parent)
        else:
            ok = _test(_lookup(key, element, parent), spec)
        if not ok:
            return False
    return True


def keys(condition: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    """Las llaves que lee ``condition``: (del elemento, del padre), sin ``parent.``."""
    own: set[str] = set()
    parent: set[str] = set()
    for key, spec in condition.items():
        if key in ("all", "any"):
            for c in _conditions(key, spec):
                o, p = keys(c)
                own |= o
                parent |= p
        elif key == "not":
            o, p = keys(spec)
            own |= o
            parent |= p
        elif key.startswith(PARENT):
            parent.add(key.removeprefix(PARENT))
        else:
            own.add(key)
    return own, parent


def _conditions(key: str, spec: Any) -> Iterable[Mapping[str, Any]]:
    if not isinstance(spec, list):
        raise ConditionError(f"{key} lleva una lista de condiciones, no {spec!r}")
    return spec


def _lookup(
    key: str, element: Mapping[str, Any], parent: Mapping[str, Any] | None
) -> Any:
    if key.startswith(PARENT):
        if parent is None:
            return _MISSING
        key, element = key.removeprefix(PARENT), parent
    value = element.get(key, _MISSING)
    return _MISSING if value is None else value


def _test(value: Any, spec: Any) -> bool:
    if not isinstance(spec, Mapping):
        return _apply("eq", value, spec)
    if not spec:
        raise ConditionError("Un objeto de operadores no puede estar vacio")
    return all(_apply(op, value, arg) for op, arg in spec.items())


def _apply(op: str, value: Any, arg: Any) -> bool:
    if op not in OPERATORS:
        raise ConditionError(f"Operador desconocido: {op!r}")
    if op == "exists":
        return (value is not _MISSING) == bool(arg)
    if op == "ne":
        return value is _MISSING or value != arg
    if op in ("in", "not_in", "between") and not isinstance(arg, list):
        raise ConditionError(f"{op} lleva una lista, no {arg!r}")
    if op == "between" and len(arg) != 2:
        raise ConditionError(f"between lleva [minimo, maximo], no {arg!r}")
    if value is _MISSING:
        return False
    if op == "eq":
        return value == arg
    if op == "in":
        return value in arg
    if op == "not_in":
        return value not in arg
    try:
        if op == "lt":
            return value < arg
        if op == "lte":
            return value <= arg
        if op == "gt":
            return value > arg
        if op == "gte":
            return value >= arg
        low, high = arg
        return low <= value <= high
    except TypeError:
        # Comparar texto con numero: la condicion no se cumple.
        return False
