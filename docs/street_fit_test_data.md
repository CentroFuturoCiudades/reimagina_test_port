# Datos de prueba de street_fit (`tests/data/`)

Los tests de `street_fit` (p. ej. `tests/test_street_fit_area.py`) usan dos áreas
de prueba, `test_area` y `lazaro_cardenas`. Los datos **no están en este repo**:
se generan en `reimagina_pipelines_lab` y se recortan a `tests/data/`.

## Origen

Repo hermano: `reimagina_pipelines_lab`

```
experiments/junctions/data/test_area.gpkg
experiments/junctions/data/lazaro_cardenas.gpkg
```

Cada `.gpkg` tiene, entre otras, estas capas:

| Capa    | Columnas usadas |
|---------|-----------------|
| `roads` | `road_key`, `from_node`, `to_node`, `road_type`, `name`, `lanes`, `lane_width_m`, `lanes_tagged`, `geometry` |
| `lots`  | `geometry` |

El mismo directorio tiene otros `.gpkg` (`city`, `dren`, `knot`, `long_junction`,
`boulevard_gaps`) y una carpeta `osm/`; los tests no los usan.

## Cómo obtenerlos

1. Clona `reimagina_pipelines_lab` junto a este repo y consigue los `.gpkg` de
   `experiments/junctions/data/`. Si no están en el clon, pídeselos al equipo.
2. Desde la raíz de este repo, ejecuta (ajusta `SRC` si el lab está en otra ruta):

```python
import os
import geopandas as gpd

SRC = "../reimagina_pipelines_lab/experiments/junctions/data/"
os.makedirs("tests/data", exist_ok=True)

for name in ("test_area", "lazaro_cardenas"):
    out = f"tests/data/{name}.gpkg"
    if os.path.exists(out):
        os.remove(out)
    roads = gpd.read_file(SRC + f"{name}.gpkg", layer="roads")[
        ["road_key", "from_node", "to_node", "road_type", "name",
         "lanes", "lane_width_m", "lanes_tagged", "geometry"]
    ]
    lots = gpd.read_file(SRC + f"{name}.gpkg", layer="lots")[["geometry"]]
    roads.to_file(out, layer="roads")
    lots.to_file(out, layer="lots")
    print(name, len(roads), len(lots), roads.crs)
```

3. Comprueba que existen `tests/data/test_area.gpkg` y
   `tests/data/lazaro_cardenas.gpkg`, cada uno con las capas `roads` y `lots`.

## Notas

- El CRS se conserva tal cual viene del lab. Revísalo con el `print` anterior.
- Decide si `tests/data/*.gpkg` se versiona. Si pesan poco, conviene hacerlo para
  que los tests corran sin el repo del lab. Si no, agrégalos a `.gitignore`.
- La visualización HTML (`street_fit_<area>.html`) es un artefacto temporal que se
  generó aparte a partir de estos mismos datos. No forma parte del repo.
- Los datos de OSM de `ingestion/streets.py` son otra cosa: `osmnx` los descarga a
  `cache/` (sin versionar). Lo mismo aplica al alumbrado de INEGI en
  `data/inegi/`, que no está en este repo.
