# Diferencias vs reimagina-lyra

- Agregación de manzana: aquí `cos`, `cus` y `density` se ponderan por el área de cada lote; reimagina-lyra usa promedio simple.
- Escritura: aquí built-form y accesibilidad se escriben en una sola transacción y, si falla cualquiera, no se escribe nada; reimagina-lyra (`reimagina_scenario_model`) escribe built-form primero y lo deja escrito aunque falle accesibilidad.
- `dry_run`: aquí calcula los cambios y hace rollback, así que reporta cuántos valores se escribirían y borrarían; reimagina-lyra no ejecuta la escritura y reporta 0.
