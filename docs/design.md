# Design and provenance

## Graph

`reimagina_prepared_snapshot` is an internal dataset producing prepared lots and
blocks. `reimagina_lot_built_form` depends on its lot output and computes all lot
indicators. `reimagina_built_form` depends on that metric's result and on the
source block output, aggregates lots by parent ID, and returns both tables.

All nodes share the explicit snapshot parameter. Lyra memoizes the dataset within
one invocation, so the two branches share one acquisition. Separate jobs read
fresh snapshots. Imports and the factory perform no acquisition. Snapshot files
should remain unchanged throughout an invocation; this pilot does not provide
atomic multi-file ingestion, persistent caching, or content-addressed provenance.

IDs and source references are columns, independent of pandas indices. Original
row order and geometries are retained. Lots whose parents are absent still receive
lot results, but are not aggregated into any supplied block. Empty blocks receive
null aggregates. Each snapshot is fully recomputed; incremental change processing
and raw-source ingestion are outside this port.

Local execution uses `lyra.sdk.LocalRunner.from_definitions()` with the plugin's
factory definition. The SDK owns graph construction, argument preparation, and
execution. The `run_snapshot()` convenience helper creates a runner per call;
the example constructs one explicitly so callers can reuse it. This plugin needs
no custom context factory.

## Arithmetic

Source: Reimagina `dev` commit `cf17664d169116a2760741ea92737abf9b9ad212`,
`pipelines/indicators/built-form/src/built_form/__init__.py`.

The pure functions in `calculation.py` preserve the original threshold, land-use
branches, Python rounding, missing-data behavior, and aggregation. Local variable
bindings in aggregation permit static type narrowing without changing arithmetic.
The adapter normalizes pandas missing values to Python `None` before computation.

Single-family housing requires built area strictly greater than 20 m2. Block
counts and areas are summed; floors and dwelling sizes are averaged; COS, CUS,
and density are weighted by lot area. Block area is the sum of lot areas, not
block polygon area. Missing occupancy defaults to one; zero occupancy remains zero.

The original lot calculation returns homes/ha. The lot adapter multiplies density
by 100 exactly once. The block calculation consumes homes/km2 and does not convert
again. Metadata declares the SDK's `count/km2` unit. Changing metadata alone would
not perform this conversion.

## Packaging and dependencies

The existing `uv_build` backend is retained. `scripts/build_manifest.py` invokes
the SDK generator and stages identical bytes under
`data/share/lyra/plugins/reimagina-test-port/lyra.plugin.json`. The root manifest
is generated, never edited manually. Regenerate it after declaration changes.

Lyra dependencies come from GitHub and their resolved revisions are in `uv.lock`.
The SDK must provide `LocalRunner` and `Unit.COUNT_PER_SQUARE_KILOMETRE`. This project has no
`lyra-app` dependency and does not import server internals. No MCP integration or
deployment configuration is included.
