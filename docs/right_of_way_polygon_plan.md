# Plan: `right_of_way` como polígono

Hoy un `right_of_way` se guarda como **línea** (el eje) con tres atributos: `bands.use`, `bands.share`
(porcentajes) y `streetlights`. El ancho que mide `street_fit` (`width_m`) se usa durante el cálculo, pero
no se guarda. Por eso la base solo tiene proporciones y no metros: una calle de 10 m y una de 30 m con el
mismo reparto se ven iguales.

Este plan cambia la geometría del elemento a un **polígono**, de frente de lote a frente de lote. Los
atributos siguen siendo los mismos, y el ancho queda definido por la forma del polígono.

## Decisiones

1. **Los cruces no tienen dueño.** Cada derecho de vía se recorta donde empieza el cruce. El área del cruce
   se calcula para hacer el recorte, pero no pertenece a ningún elemento ni se guarda.
2. **Los atributos no cambian**: `bands.use`, `bands.share` y `streetlights`. El ancho se deduce del
   polígono.
3. **El indicador de calles sigue ponderando por longitud.** Esa longitud se deduce del polígono con la
   misma fórmula que da el ancho.

### Ancho y largo a partir del polígono

Con los cruces recortados, cada derecho de vía es una cinta. Si se conocen su área `A` y su perímetro `P`,
y la cinta cumple `P ≈ 2L + 2W` y `A ≈ L·W`, se despejan su largo `L` y su ancho `W`:

```
L = P/4 + √(P²/16 − A)
W = A / L
```

Las relaciones se cumplen también en curvas, porque una cinta curva conserva `P ≈ 2L + 2W`. El ancho de
cada banda en metros es `W × share / 100`. La fórmula se escribe una sola vez, como `street_fit.strip_length`
(y `strip_width`), y el indicador, los tests y el visor la importan de ahí.

## Fases

### 1. Polígonos: `street_fit.footprints(fitted, roads) -> GeoSeries`

`fit` sigue devolviendo el eje. Sus tests y su lógica no cambian. La función nueva toma su salida y, para
cada derecho de vía:

1. Alarga el eje hasta sus cruces, la mitad del ancho de la calle más ancha que llega a cada nodo, como hoy
   `fit`.
2. Construye la cinta como eje ± `width_m / 2`. El eje ya está centrado entre los frentes de lote, y
   `width_m` puede haber cambiado en `smooth_widths`. Por eso `width_left_m` y `width_right_m` **no** se
   usan.
3. Calcula los cruces, que son las partes donde se traslapan las cintas de derechos de vía que comparten un
   nodo, y los recorta de cada cinta.
4. Aplica `make_valid`, conserva un solo `Polygon` (la pieza mayor) y lo simplifica a 10 cm.

Casos especiales:

- **Tramo que queda entero dentro de un cruce** (en la Condesa hay unas decenas): conserva su cinta sin
  recortar y se le permite traslapar con el cruce. Así no hay elementos con geometría vacía ni llaves
  perdidas. Se cuentan en el reporte.
- **Eje que se cruza a sí mismo**, o curva que GEOS no puede desplazar (`TopologyException`): se simplifica
  el eje antes de construir la cinta.

### 2. Calles sin ajustar

Las peatonales, las de `refine: false` y las que no tienen lotes se convierten en una cinta con el ancho por
defecto de su clase (`2 × DEFAULTS["default_half_width_m"][clase]`). Esa cinta se recorta con los polígonos
vehiculares para que no se traslapen. Así todas las calles son polígonos y nunca se mezclan tipos de
geometría.

### 3. `streets.build`

- Escribe el polígono en EPSG:4326 como geometría del elemento.
- Sigue usando en memoria el eje y `width_m` para emparejar alumbrado y ciclovías. Ese cálculo no cambia.
- Las llaves (`source_record_ref`) y los `id` no cambian.

### 4. Consumidores y documentación

- `indicators/streets/calculation.py`: hoy pondera con `shapely.length(intersection(línea, área))`, que en
  un polígono daría el perímetro. Pasa a recortar el polígono al área de 500 m y ponderar con
  `strip_length` del recorte.
- `docs/design.md`: los `right_of_way` son polígonos, los cruces no tienen dueño y se documenta la fórmula
  de `L` y `W`.
- `catalogs.py`: sin cambios.

### 5. Tests

En `tests/test_street_fit_area.py`, sobre `test_area` y `lazaro_cardenas`:

- Cada polígono es válido y es un solo `Polygon`.
- El traslape entre derechos de vía es menor a 1 m², salvo en los tramos que quedan dentro de un cruce.
- Menos de ~1 % del área de los derechos de vía (sin contar `service`) cae sobre lotes que ningún eje
  atraviesa.
- **El ancho deducido funciona:** en tramos rectos, `W = A / L` queda dentro de ±5 % del `width_m` de `fit`.

En `tests/test_baseline.py`:

- `streets.build` escribe polígonos con los tres atributos de siempre.
- Con la red sin ajustar, el indicador de calles da lo mismo que antes, con un margen de ±2 %.

### 6. Validación visual

`scripts/street_fit_streets.py` dibuja directamente los polígonos de `footprints`. El ancho y las bandas los
deduce de la forma, igual que lo haría alguien que lee la base, así que lo que se ve es lo que se guarda. Se
revisa con `lazaro_cardenas` y con la Condesa completa
(`reimagina_pipelines_lab/experiments/junctions/data/city.gpkg` o la red OSM `drive`).

## Riesgos

- **Precisión del ancho deducido.** Es menor en tramos muy cortos y en curvas cerradas. Los tests de la
  fase 5 la miden.
- **Huecos en los cruces.** Un mapa hecho desde la base mostrará los cruces vacíos, salvo que el visor los
  dibuje aparte.
- **Peso en la base.** Un polígono pesa más que una línea; por eso se simplifican a 10 cm.
- **Llaves.** Activar `refine` cambia las llaves vehiculares, igual que hoy. Este plan no lo empeora.

## Referencias

- `src/reimagina_test_port/ingestion/street_fit.py`: `fit`, `centred`, `smooth_widths`, `extend`.
- `src/reimagina_test_port/ingestion/streets.py`: `build`, que escribe los elementos.
- `src/reimagina_test_port/ingestion/catalogs.py`: los atributos de `right_of_way`.
- `src/reimagina_test_port/indicators/streets/calculation.py`: el indicador que pondera por longitud.
- `scripts/street_fit_streets.py`: el visor, que ya construye cintas y cruces como propone la fase 1.
- `docs/street_fit_test_data.md`: de dónde salen los datos de prueba.
