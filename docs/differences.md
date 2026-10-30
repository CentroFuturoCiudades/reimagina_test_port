# Diferencias vs reimagina-lyra

- Agregación de manzana: aquí `cos`, `cus` y `density` se ponderan por el área de cada lote; reimagina-lyra usa promedio simple.
- Ids de lote del baseline: aquí `uuid5(NS, "<ciudad>:lot:<clave catastral>")` (sin copias exactas, repetidas con `#2`, `#3`…); reimagina-lyra usa `uuid5(NS, "lot:<cve_code>:<fila del shapefile>")`, y el baseline cargado de Mexicali usa `uuid5(NS, "lot:<fila>")`.
- Baseline: aquí `reimagina_baseline_write` escribe en un escenario baseline nuevo (creado en la API) y reutiliza sin cambios los elementos que ya existen; reimagina-lyra (`reimagina_ingest_baseline`) hace upsert sobre el baseline y actualiza los elementos existentes, con columnas de `elements` que ya no existen en el esquema actual.
- `num_levels` del baseline: aquí se redondea a pisos enteros (es `integer` en `attribute_catalog`); reimagina-lyra guarda el promedio con decimales.
- Catálogo: aquí, si falta una llave o el tipo no coincide, la escritura falla y dice qué agregar; reimagina-lyra omite esas llaves sin avisar.
- `dry_run`: aquí calcula los cambios y hace rollback, así que reporta cuántos valores se escribirían y borrarían; reimagina-lyra no ejecuta la escritura y reporta 0.
