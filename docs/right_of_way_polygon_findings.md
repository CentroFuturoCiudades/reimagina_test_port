# Hallazgos: `right_of_way` como polígono

Resultado de ejecutar `docs/right_of_way_polygon_plan.md` y de revisar el visor sobre la Condesa completa. Las
fases 1 a 5 están implementadas y la suite pasa, pero los polígonos todavía tienen defectos visibles. La mayoría
viene de cómo `footprints` recorta los cruces; otra parte viene de `fit` (ya existía) y una menor del visor.

## Estado de la implementación

| Fase | Estado | Dónde |
|---|---|---|
| 1. `street_fit.footprints` | Hecho | `strip_length`, `strip_width`, `ribbon`, `cut_junctions`, `footprints` |
| 2. Calles sin ajustar | Hecho | `streets.footprints`: cinta de `2 × default_half_width_m` de su clase, recortada contra las vehiculares |
| 3. `streets.build` | Hecho | Escribe polígonos en EPSG:4326; casos especiales en `attrs["footprints"]` |
| 4. Indicador y docs | Hecho | `coverage_by_block` pondera con `strip_length`; las líneas de bases viejas siguen funcionando. `docs/design.md` actualizado |
| 5. Tests | Hecho, con 3 xfail | Ver abajo |
| 6. Visor | Hecho | `scripts/street_fit_streets.py` dibuja los polígonos de `footprints` |

Cambios respecto al plan:

- La extensión hacia los cruces solo se usa para calcular los cruces. Cada polígono se recorta desde la cinta
  **sin extender**; si no, las calles sin salida quedaban más largas que su eje.
- Un tramo al que el recorte le deja menos del 25 % de su cinta cuenta como "dentro del cruce" y conserva su
  cinta entera. El plan solo contemplaba los que quedaban vacíos, pero aparecían muchas astillas de 1–13 m².
- `strip_length` mide pieza por pieza y suma. Aplicar la fórmula al conjunto da de más cuando hay varias piezas.
- El indicador corrige con `make_valid` los polígonos que salen inválidos al reproyectar; sin eso había una
  `TopologyException`.
- El test del indicador sin ajustar (±2 %) está en `tests/test_street_fit_area.py`, no en `test_baseline.py`: la
  red sintética de `test_baseline` tiene 3 calles y no alcanza para comparar.

Tests marcados como fallo esperado (`xfail(strict=True)`; si alguno empieza a pasar, el test avisa):

- `test_the_width_is_read_from_the_footprint` en ambas áreas: `W = A/L` cae dentro del ±5 % en solo 21–40 % de
  los tramos rectos.
- `test_footprints_do_not_overlap[lazaro_cardenas]`: calles paralelas sin nodo en común se traslapan, hasta
  470 m² en calles `service`.

## Lo que se ve en el visor

Sobre la Condesa completa (6,066 derechos de vía), con capturas del visor:

- Rectángulos oscuros (cruces) repetidos a lo largo de las avenidas, más grandes que la calle que cruza.
- Bandas más angostas que el contorno del derecho de vía.
- Calles que no llegan a conectarse con su cruce.
- Formas en "V" o en "Y" pintadas encima de otras calles.
- Ejes en zigzag donde una avenida con camellón se separa.

## Diagnóstico

Medido con `fit` sobre `city.gpkg` del lab, sin peatonales.

### 1. `footprints` corta los cruces demasiado grandes y donde no hay cruce (causa principal)

Hoy cada tramo se alarga hacia sus cruces la mitad del ancho **por defecto de la clase** de la calle más ancha
que toca **en cualquiera de sus nodos**, y lo hace igual **en los dos extremos**. Consecuencias:

- **El alargue no depende del extremo.** Una calle residencial que remata en una avenida se alarga 20 m también
  en su otro extremo.
- **No depende de la calle que cruza.** Donde una calle de 10 m llega a una avenida de 40 m, el cruce mide 40 m a
  lo largo de la avenida, no 10 m.
- **Hay cruces donde no cruza nada.** En **622 nodos** solo se juntan dos tramos de la misma calle que sigue de
  frente, y aun así se recorta un pedazo entre ellos.
- **Usa el ancho por defecto, no el medido.** Si una avenida medida es más ancha que el valor de su clase, el
  traslape no la cubre completa y quedan huecos.
- **El ancho deducido falla por los recortes**, no por la fórmula:

  | Tramos | n | `W` dentro del ±5 % | Mediana de `W / width_m − 1` |
  |---|---|---|---|
  | Sin tocar ningún cruce | 1 | 100 % | −0.2 % |
  | Tocando un cruce | 5,440 | 37 % | −10.6 % |

  Los recortes irregulares inflan el perímetro, y con él `L`; por eso `W` sale menor.
- **Demasiados tramos "dentro del cruce"**: 625 (10 %), con una mediana de 18 m de largo. Conservan su cinta
  entera, y 609 de ellos se traslapan con otras calles.

### 2. `fit` (anterior a este cambio)

- 23 ejes se cruzan a sí mismos y hubo que simplificarlos. Son los zigzags donde se separa una avenida con
  camellón.
- 28 ejes cruzan otro eje sin compartir nodo.
- De los 152 traslapes ≥ 1 m² que quedan, **144 son entre calles que no comparten nodo**, sobre todo
  `service`–`service` (42) y `residential`–`residential` (40). Como no comparten nodo, `footprints` nunca las
  recorta entre sí.

### 3. El visor (menor)

- El contorno se dibuja con el polígono, construido con `width_m`, pero las bandas usan el ancho deducido
  (~10 % menor). Por eso sobra borde a los lados. El origen sigue siendo el punto 1.
- Los tramos "dentro del cruce" se pintan encima de las demás calles, lo que exagera el desorden.

## Lo que se probó

Un prototipo fuera del repo (`proto.py` en el scratchpad de la sesión) alarga **cada extremo por separado** la
mitad del **ancho medido** de la calle que cruza ahí, y **nada** si la calle solo sigue de frente:

| Métrica | Actual | Prototipo |
|---|---|---|
| `W` dentro del ±5 % | 37 % | 48 % |
| `W` dentro del ±10 % | — | 59 % |
| Tramos "dentro del cruce" | 10 % | 7 % |
| Área de cruces | 859k m² | 734k m² |
| Traslapes ≥ 1 m² | 152 | 670 ❌ |

Va en la dirección correcta, pero los traslapes subieron: el prototipo decide qué tramos forman un cruce
según si sus extremos quedan a menos de 1 m, y eso deja fuera pares que sí comparten nodo. La corrección **no
está aplicada** en el repo.

## Siguientes pasos

1. **Corregir el alargue en `footprints`** (punto 1):
   - Para cada extremo, buscar el nodo compartido más cercano.
   - Alargar ese extremo la mitad del `width_m` de la calle más ancha que llega a ese nodo **de través**
     (a más de 30° de seguir de frente).
   - Si en ese nodo la calle solo sigue de frente, no alargar.
   - Para formar los cruces, seguir usando el criterio de nodo compartido, que es el que da 152 traslapes y no
     670.
   - Medir otra vez con las tablas de este documento.
2. **Visor**: dibujar las bandas con el mismo ancho que el contorno, o el contorno también con el ancho deducido,
   y pintar los tramos "dentro del cruce" debajo de las demás calles o con otro estilo.
3. **`fit`**, como trabajo aparte: los ejes que se cruzan a sí mismos y las calles paralelas o que se cruzan sin
   nodo en común.

## Cómo visualizarlo

Desde la raíz del repo:

```bash
# Áreas de prueba (tests/data/*.gpkg); sin --out se guarda en la carpeta actual
uv run python scripts/street_fit_streets.py lazaro_cardenas
uv run python scripts/street_fit_streets.py test_area

# Condesa completa, comparada con el lab (~1 min de street_fit; HTML de ~11 MB)
uv run python scripts/street_fit_streets.py \
  ../reimagina_pipelines_lab/experiments/junctions/data/city.gpkg \
  --lab ../reimagina_pipelines_lab/output/condesa \
  --out condesa_streets.html

open condesa_streets.html
```

El script imprime en la terminal (y la página muestra) un reporte de calidad: área sobre lotes, espacio de calle
sin cubrir, lados medidos, anomalías por derecho de vía y, con `--lab`, cuántos tienen el mismo ancho que el lab.
Esa comparación con el lab usa `width_m` del ajuste, no el ancho deducido del polígono.

Para revisar la corrección del punto 1, conviene ver en la Condesa:

- Avenidas con camellón y calles que rematan en ellas: rectángulos de cruce repetidos.
- Bifurcaciones de avenidas: zigzags (estos vienen de `fit`).
- Enlaces cortos en cruces: tramos "dentro del cruce" encimados.
