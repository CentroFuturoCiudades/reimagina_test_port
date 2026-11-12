# `right_of_way` como polígono: revisión y plan de recorte perpendicular

Continúa `docs/right_of_way_polygon_plan.md` (el plan original) y `docs/right_of_way_polygon_findings.md` (los
hallazgos tras implementarlo). Este documento revisa esos hallazgos con mediciones nuevas, propone cambiar la
forma en que `footprints` recorta los cruces y deja el plan para hacerlo.

Nada de lo que se propone aquí está aplicado en el repo. Las mediciones salen de un prototipo fuera del repo
(ver [Cómo se midió](#cómo-se-midió)).

## En resumen

- El diagnóstico de los hallazgos es correcto: los defectos vienen sobre todo de cómo `footprints` recorta los
  cruces.
- La corrección que proponen (alargar cada extremo según la calle que cruza) no basta. Cambia el **tamaño** del
  cruce, pero se sigue restando un polígono de cruce, y es la **forma** de ese corte la que daña el ancho
  deducido.
- Recortar cada extremo **perpendicular al eje** y construir la cinta desde el eje recortado sube el ancho
  deducido dentro del ±5 % de 37 % a 91 % en la Condesa, con menos traslapes (133 contra 152).
- Hay un límite que ningún recorte arregla: en los tramos que quedan más cortos que anchos, la fórmula
  intercambia largo y ancho. Sumados a los tramos "dentro del cruce", cerca de 16 % de los elementos no puede
  dar su ancho desde el polígono. Eso pide una decisión (ver [Decisiones pendientes](#decisiones-pendientes)).

## Por qué no basta corregir el alargue

Hoy `cut_junctions` alarga cada cinta hacia sus cruces, calcula los traslapes entre cintas que comparten nodo,
los une en un polígono de cruces y lo resta de cada cinta. El resultado de esa resta rara vez es un rectángulo:

- Si la calle que cruza no cubre todo el ancho, queda una **muesca** en el costado.
- Si el cruce es sesgado, el corte queda **en diagonal** o escalonado.
- Si el cruce parte la cinta, quedan **varias piezas** y solo se conserva la mayor.

`W = A / L` supone una cinta con `P ≈ 2L + 2W`. Cada muesca agrega perímetro sin agregar largo, `L` sale de más
y `W` de menos (mediana de −10.6 % en los tramos que tocan un cruce). Ajustar cuánto se alarga cada extremo
reduce el área del cruce, pero el corte sigue teniendo esa forma: el prototipo de los hallazgos llegó a 48 %.

## La alternativa: recorte perpendicular por extremo

En lugar de restar un polígono, se calcula **una distancia de recorte por extremo** y se acorta el eje:

1. Construir la cinta de cada derecho de vía tal como es (eje ± `width_m / 2`), sin alargarla.
2. Para cada extremo, tomar las otras cintas que comparten el nodo de ese extremo. Con cada una, calcular el
   traslape de las dos cintas y quedarse con las partes que están junto al extremo.
3. Proyectar ese traslape sobre el eje. La distancia de recorte del extremo es el punto más lejano que alcanza,
   tomando el máximo entre todas las calles del nodo.
4. Recortar el eje esas dos distancias y construir la cinta desde el eje recortado.

Lo que se gana:

- **Cada polígono es una cinta limpia**, con los extremos perpendiculares al eje. `P ≈ 2L + 2W` se cumple por
  construcción.
- **Siempre es una sola pieza.** Ya no hace falta conservar "la pieza mayor" ni descartar las demás.
- **El recorte depende del extremo y de la calle que cruza**, con su ancho medido. Una residencial que remata en
  una avenida se recorta medio ancho de la avenida solo en ese extremo; la avenida se recorta medio ancho de la
  residencial.
- **Donde la calle sigue de frente, el recorte sale casi cero** sin regla aparte: dos cintas que se continúan
  apenas se traslapan. No hace falta el umbral de 30°.
- **Desaparece el umbral de 25 %.** Un tramo está "dentro del cruce" cuando sus dos recortes suman su largo.

El cruce deja de ser un polígono que se calcula y se resta. Si el visor lo quiere dibujar, es el espacio que
queda entre los extremos recortados de un nodo.

## Resultados medidos

Sobre `city.gpkg` del lab (Condesa completa, 6,066 derechos de vía, sin peatonales):

| Métrica | Repo | Prototipo de los hallazgos | Recorte perpendicular |
|---|---|---|---|
| `W` dentro del ±5 % | 37 % | 48 % | 91 % |
| `W` dentro del ±10 % | — | 59 % | 92 % |
| `W` dentro del ±5 %, sin tramos más cortos que anchos | 40 % | — | 99 % |
| Mediana del error de `W` | −10.6 % | — | 0.0 % |
| Tramos "dentro del cruce" | 625 (10 %) | 7 % | 485 (8 %) |
| Traslapes ≥ 1 m² | 152 | 670 | 133 |
| — entre calles con nodo en común | 8 (4,087 m²) | — | 16 (7,005 m²) |
| — entre calles sin nodo en común | 144 (10,692 m²) | — | 117 (10,770 m²) |
| Área conservada, de las cintas sin recortar | 83 % | — | 83 % |
| Extremos recortados | — | — | 11,548 de 12,132; mediana 6.4 m, p90 13.6 m |

Sobre las áreas de prueba, con el recorte perpendicular:

| Área | `W` dentro del ±5 % | Sin tramos más cortos que anchos | "Dentro del cruce" | Traslapes ≥ 1 m² |
|---|---|---|---|---|
| `test_area` (104) | 77 % (repo: 32 %) | 99 % | 18 (17 %; repo: 27) | 4 (repo: 0) |
| `lazaro_cardenas` (124) | 75 % | 96 % | 49 (40 %; repo: 54) | 9 (repo: 10) |

Se probó también una variante que prolonga la calle que cruza hasta atravesar toda la cinta antes de medir el
traslape. No mejora nada (91 % igual) y deja más tramos dentro del cruce (518 contra 485), así que el plan usa
las cintas tal como son.

## Hallazgos que no estaban en el documento anterior

### 1. La fórmula intercambia largo y ancho en tramos cortos

`L = P/4 + √(P²/16 − A)` devuelve siempre la dimensión **mayor** del rectángulo. Si después del recorte el tramo
mide 9 m de largo y 17 m de ancho, la fórmula dice que mide 17 de largo y 9 de ancho.

- Con el recorte perpendicular son **485 tramos (8 %)** en la Condesa, con mediana de 9.4 m de largo por 16.9 m
  de ancho. Solo 6 % da el ancho correcto. Por clase: 233 `residential`, 98 `tertiary`, 83 `service`,
  40 `trunk`, 31 `primary`.
- En el repo son 369 (6 %), con 1 % correcto.
- Afecta también al indicador: `coverage_by_block` pondera esos tramos con su ancho en lugar de su largo.

Ningún recorte lo corrige, porque un rectángulo no dice cuál de sus lados es el eje.

### 2. Los tramos "dentro del cruce" no desaparecen

Quedan 485 (8 %) en la Condesa. Tienen una mediana de 18.2 m de eje y 16.6 m de ancho; 79 % mide menos de 30 m.
Son enlaces y tramos cortos entre dos cruces: 139 `residential`, 136 `service`, 94 `tertiary`, 88 `trunk`,
28 `primary`. En `lazaro_cardenas` siguen siendo 40 %: ahí el problema es de `fit` (un troncal con camellón y
laterales), no de `footprints`.

### 3. `largest` descarta piezas

En el repo, el recorte parte 189 cintas (3 %) en dos o más piezas y se pierden 3,805 m². Es menor, y con el
recorte perpendicular no ocurre.

### 4. Muchos derechos de vía abarcan varias aristas

Un derecho de vía de `fit` puede reunir varias aristas de la red, así que tiene **nodos intermedios** donde
llegan otras calles. En `test_area` son 62 de 104. Dos consecuencias:

- **Calle que llega a mitad de un corredor.** Con el recorte perpendicular, la calle se recorta hasta el borde
  del corredor y el corredor queda entero. Esa boca queda dentro del corredor, lo que se aparta de "los cruces
  no tienen dueño" (decisión 1 del plan original). Partir el corredor ahí daría varias piezas por elemento.
- **Dos corredores que se cruzan en un nodo intermedio de ambos.** Ninguno tiene un extremo ahí, así que ninguno
  se recorta y se traslapan. Son 14 de los 16 traslapes con nodo en común que deja el recorte perpendicular, y
  la razón por la que ese renglón sube de 8 a 16.

### 5. Los traslapes sin nodo en común siguen igual

117 con el recorte perpendicular (144 en el repo), casi la misma área. Los más frecuentes:
`residential`–`residential` (30), `service`–`service` (29), `trunk`–`trunk` (12), `tertiary`–`tertiary` (10).
Confirma el punto 2 de los hallazgos: son calles paralelas o que se cruzan sin nodo, y se corrigen en `fit`.

## Decisiones pendientes

### A. ¿De dónde sale el ancho?

Entre los tramos más cortos que anchos (8 %) y los que quedan dentro del cruce (8 %), cerca de 16 % de los
elementos no puede dar su ancho desde el polígono. Opciones:

1. **Guardar `width_m` como atributo** (recomendada). El polígono sigue siendo la geometría y la fórmula queda
   como verificación en los tests. Revisa la decisión 2 del plan original ("los atributos no cambian") y toca
   `catalogs.py`. El largo sale de `A / width_m`, que no depende de la forma del corte.
2. **Mantener los atributos** y aceptar que esos tramos reporten mal su ancho y su largo. Sin cambios de esquema,
   pero quien lea la base no puede saber cuáles son.
3. **Fusionar los tramos cortos al cruce.** Dejan de existir como elementos, se pierden sus llaves y sus
   atributos (alumbrado, bandas).

El plan de abajo supone la opción 1. Si se elige otra, la fase 3 cambia y las demás se mantienen.

### B. Calle que llega a mitad de un corredor

¿Se acepta que la boca quede dentro del corredor? El plan supone que sí: es la única forma de conservar un
polígono por elemento, y el área es pequeña.

### C. Tramos "dentro del cruce"

Hoy conservan su cinta entera y se traslapan con las demás. Como siguen siendo 8 %, conviene decidir si eso se
mantiene o si se guardan con otra forma (por ejemplo, la cinta menos los polígonos vecinos). El plan los deja
como están y los marca, para decidirlo con el visor delante.

## Plan

### Fase 0. Traer la medición al repo

- Pasar el script de medición a `scripts/street_fit_measure.py`, con el mismo reporte que las tablas de este
  documento: ancho deducido, tramos más cortos que anchos, dentro del cruce, traslapes con y sin nodo en común,
  área conservada.
- Correrlo sobre las tres áreas y guardar la salida como línea base.

Sirve para comparar cada fase con la anterior sin depender del visor.

### Fase 1. Recorte perpendicular en `cut_junctions`

En `street_fit.py`:

- **Extremos y nodos.** Para cada derecho de vía, identificar el nodo de cada extremo de su eje (el nodo de sus
  `road_keys` más cercano al extremo) y sus nodos intermedios (los demás). `footprints` los pasa a
  `cut_junctions` en lugar del conjunto de nodos sin distinguir.
- **Distancia de recorte.** Función nueva que, para un extremo, recorre las otras cintas del nodo y devuelve la
  proyección más lejana de sus traslapes sobre el eje. Solo cuentan las partes del traslape que tocan el
  entorno del extremo (un radio del mayor de los dos anchos), para no tomar un traslape en el otro extremo.
- **Cinta final.** `ribbon(substring(eje, recorte_inicio, largo − recorte_fin), width_m)`.
- **Dentro del cruce.** Si queda menos de 1 m de eje, el tramo conserva su cinta entera y se anota en
  `inside`, como hoy.
- **Lo que se elimina:** `reach` y `widest_at_nodes` en `footprints`, el umbral `min_kept_share`, y el polígono
  `junctions` como insumo del recorte.
- **`streets.footprints`** (calles sin ajustar): usa el mismo recorte entre ellas. El corte contra las
  vehiculares, que hoy resta `junctions`, pasa a restar solo los polígonos vehiculares.

Criterio de salida: las cifras de la columna "Recorte perpendicular" de este documento, medidas con el script
de la fase 0.

### Fase 2. Corredores que se cruzan en un nodo intermedio

Donde dos derechos de vía comparten un nodo que es intermedio para ambos, ninguno se recorta. Regla propuesta:
el de menor rango (`RANK`; a igual rango, el más angosto) se parte en ese nodo y cada mitad se recorta como
cualquier extremo; se conserva la mitad mayor. Son 14 casos en la Condesa.

Antes de implementarla, revisar esos 14 casos en el visor: si la mayoría son errores de `fit` (dos corredores
que no deberían compartir ese nodo), se corrigen allá y esta fase se omite.

Criterio de salida: traslapes con nodo en común por debajo de los 8 del repo.

### Fase 3. `width_m` como atributo (si se elige la opción A.1)

- `catalogs.py`: agregar `width_m` a los atributos de `right_of_way`.
- `streets.build`: escribir `width_m`; el de la clase para las calles sin ajustar.
- `indicators/streets/calculation.py`: ponderar con `área del recorte / width_m` cuando el atributo existe, y con
  `strip_length` cuando no (bases anteriores).
- `docs/design.md`: el ancho es un atributo; la fórmula queda documentada como verificación.
- El visor dibuja las bandas con `width_m`. Con eso se resuelve también el punto 3 de los hallazgos (bandas más
  angostas que el contorno).

### Fase 4. Tests

En `tests/test_street_fit_area.py`:

- `test_the_width_is_read_from_the_footprint`: quitar el `xfail`. Se evalúa en los tramos rectos que no están
  dentro del cruce y que quedan más largos que anchos; ahí debe cumplir el ±5 %.
- Test nuevo: cada polígono fuera de un cruce es la cinta de su eje recortado (el área coincide con
  `largo recortado × width_m` dentro de 1 %).
- Test nuevo: en un nodo donde la calle solo sigue de frente, el recorte es menor a 1 m.
- `test_footprints_do_not_overlap`: `test_area` pasa hoy con 0 traslapes y el prototipo deja 4. Hay que
  revisarlos antes de dar la fase 1 por buena: o los cubre la fase 2, o son casos de `fit` y el test los
  excluye de forma explícita. `lazaro_cardenas` sigue con `xfail` hasta la fase 6.
- Con la opción A.1: `test_streets_are_polygons_with_their_attributes` incluye `width_m`.

### Fase 5. Visor

- Dibujar el cruce como el espacio entre los extremos recortados de cada nodo.
- Pintar los tramos "dentro del cruce" debajo de las demás calles y con otro estilo.
- Revisar en la Condesa los casos del documento anterior: avenidas con camellón y las calles que rematan en
  ellas, bifurcaciones, enlaces cortos.
- Con eso delante, cerrar la decisión C.

### Fase 6. `fit` (trabajo aparte)

Sin cambios respecto a los hallazgos:

- 117 traslapes entre calles sin nodo en común (paralelas, laterales, `service`).
- 23 ejes que se cruzan a sí mismos y 28 que cruzan otro eje sin compartir nodo.
- `lazaro_cardenas`, con 40 % de tramos dentro del cruce.

## Riesgos

- **Ejes cuyos extremos no llegan al nodo.** El recorte parte de que las cintas de un nodo se tocan. Si `fit`
  deja un extremo lejos de su nodo, el traslape sale vacío y el extremo no se recorta. En la Condesa, 584
  extremos quedan sin recortar; la mayoría deberían ser calles sin salida, pero hay que comprobarlo.
- **Bifurcaciones en ángulo agudo.** Dos cintas que salen casi paralelas del mismo nodo se traslapan a lo largo
  de muchos metros y el recorte sale largo (p90 de 13.6 m en la Condesa, pero la cola es mayor). Puede dejar
  tramos dentro del cruce que a la vista no lo están.
- **Solo se midió con métricas.** El prototipo no se ha visto en el visor. La fase 1 no se da por buena hasta
  revisarla ahí.
- **Las llaves no cambian**, pero con la opción A.1 cambia el esquema de `right_of_way`: las bases anteriores no
  tienen `width_m` y los consumidores deben tolerarlo.

## Cómo se midió

Con un script fuera del repo (la fase 0 lo trae a `scripts/`). Para cada área:

1. Corre `street_fit.fit` sobre las calles no peatonales y los lotes.
2. Calcula los polígonos del repo con `street_fit.footprints` y los del recorte perpendicular con el algoritmo
   descrito arriba.
3. Reporta, para cada conjunto: `strip_width` contra `width_m`, tramos cuya área entre `width_m` da menos que
   `width_m` (más cortos que anchos), tramos dentro del cruce, y traslapes ≥ 1 m² entre pares que no están
   dentro de un cruce, separados según compartan nodo.

Áreas: `tests/data/test_area.gpkg`, `tests/data/lazaro_cardenas.gpkg` y
`../reimagina_pipelines_lab/experiments/junctions/data/city.gpkg`.

Las columnas "Repo" y "Prototipo de los hallazgos" de la tabla principal reproducen las cifras de
`right_of_way_polygon_findings.md` donde coinciden con esta medición (37 %, 625, 152).
