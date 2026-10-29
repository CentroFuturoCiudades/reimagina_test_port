# Diferencias vs reimagina-lyra

- Agregación de manzana: aquí `cos`, `cus` y `density` se ponderan por el área de cada lote; reimagina-lyra usa promedio simple.
- Escritura: aquí built-form y accesibilidad se escriben en una sola transacción y, si falla cualquiera, no se escribe nada; reimagina-lyra (`reimagina_scenario_model`) escribe built-form primero y lo deja escrito aunque falle accesibilidad.
- Ids de lote del baseline: aquí `uuid5(NS, "<ciudad>:lot:<clave catastral>")` (sin copias exactas, repetidas con `#2`, `#3`…); reimagina-lyra usa `uuid5(NS, "lot:<cve_code>:<fila del shapefile>")`, y el baseline cargado de Mexicali usa `uuid5(NS, "lot:<fila>")`.
- Baseline: aquí `reimagina_baseline_elements` solo devuelve las tablas; reimagina-lyra (`reimagina_ingest_baseline`) también las escribe en la base.
- `dry_run`: aquí calcula los cambios y hace rollback, así que reporta cuántos valores se escribirían y borrarían; reimagina-lyra no ejecuta la escritura y reporta 0.
