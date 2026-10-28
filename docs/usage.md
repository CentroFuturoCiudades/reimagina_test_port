# Usage and snapshots

## Offline calculation

From the project directory:

```sh
uv sync
uv run python examples/local_snapshot.py
```

Or construct the SDK runner directly in Python:

```python
from lyra.sdk import LocalRunner
from reimagina_test_port.plugin import create_plugin

runner = LocalRunner.from_definitions({"reimagina-test-port": create_plugin()})
tables = runner.run("reimagina_built_form", {"parameters": {"snapshot": "synthetic"}})
lots = tables["reimagina_built_form_lots"]
blocks = tables["reimagina_built_form_blocks"]
```

To calculate only lots, use `"reimagina_lot_built_form"` as the first argument to
`runner.run()`; its output is `reimagina_lot_indicators`. Reuse a runner across
calls to reuse its graph; each call acquires fresh snapshot data.

For a single call, `reimagina_test_port.local.run_snapshot("synthetic")` creates a
runner and delegates to the same public interface. Its optional `metric` argument
selects the public metric. Both approaches execute locally without database
clients or a Lyra server. `from_definitions()` works directly with this checkout
and does not require an installed shared-data manifest.

## Prepared input contract

`REIMAGINA_SNAPSHOT_ROOT` overrides the bundled data root. Each snapshot is a
subdirectory containing `lot.parquet` and `block.parquet`, matching Reimagina's
prepared element exports. The snapshot parameter is required, and accepts letters,
digits, hyphens, and underscores (starting with a letter or digit). Missing files
fail explicitly; there is no fallback to synthetic data.

Both files require unique, nonempty string `id` values, a nullable `parent_id`
column, and a GeoParquet `geometry` column with a declared CRS. An optional
`source_record_ref` column is carried through, with null substituted when absent.
The lot table consumes `cos`, `num_levels`, `area`, `land_use`, `unit_size_m2`,
`people_per_household`, and `occupancy_rate`. Missing attributes are treated as
null. Extra input columns are not propagated. Areas are supplied square metres;
the plugin does not infer area from geometry or repair geometries.

The bundled `synthetic` snapshot contains invented geometries in EPSG:32611 and
14 calculation cases with three blocks. It is not real city data; polygon sizes
are illustrative and do not define the supplied areas. Custom snapshots can use
any declared CRS. Native graph outputs retain it; SDK public serialization
converts geometry to EPSG:4326.

## Public interfaces

The plugin exposes `reimagina_lot_built_form` and `reimagina_built_form`.
The combined metric's standard submission body is:

```json
{
  "metric": "reimagina_built_form",
  "input": {"parameters": {"snapshot": "synthetic"}}
}
```

Both combined outputs contain `id`, `parent_id`, `source_record_ref`, geometry,
and the following nullable numerical columns:

| Columns | Units |
| --- | --- |
| `cos`, `cus` | ratio |
| `num_levels` | count (floors; block values are averages) |
| `num_housing_units`, `population` | count (homes and residents respectively) |
| `area`, `unit_size_m2` | m2 |
| `density` | count/km2 (homes per square kilometre) |

The Python client's submission, polling, table decoding, descriptor, and download
interfaces are demonstrated in `tests/test_client.py` with fully mocked HTTP
responses. No live submission or server validation is part of this pilot.
