# Design and provenance

## Graph

`reimagina_scenario` is an internal dataset producing a Reimagina scenario's
effective lots and blocks from Postgres. `reimagina_lot_built_form` depends on
its lot output and computes all lot indicators. `reimagina_built_form` depends on
that metric's result and on the source block output, aggregates lots by parent
ID, and returns both tables. `reimagina_accessibility` depends on both
built-form tables and on two more datasets: `reimagina_scenario_pois` (present
elements with an `amenity_type`) and `reimagina_road_network` (the project's road
network as node and edge tables). `create_plugin()` registers this graph.

`create_snapshot_plugin()` registers the same metrics over
`reimagina_prepared_snapshot` and `reimagina_prepared_accessibility` instead,
which read prepared GeoParquet files. It is for offline tests and examples; it is
not in the manifest or the image.

## Scenario source

Ported from reimagina-lyra `store.py`, following the API's effective-state rules.
A fork stores only what differs from its baseline (`scenarios.parent_scenario_id`).
An element is present when the scenario has a live `element_state` row, or has
no row and the baseline has a live one; a scenario row with `deleted_at` hides it.
Geometry and `parent_element_id` come from that state row, falling back to
`elements.parent_element_id`. Elements with `elements.deleted_at` are excluded.
Each attribute comes from the scenario's row when it exists, even if all-NULL
(which hides the baseline value), else from the baseline's. Values are decoded by
`attribute_catalog.value_type` as exe_reimagina's `eav` module does. Geometry is
EPSG:4326.

The worker connects with `REIMAGINA_POSTGRES_DSN`, not Lyra's `[database]`.
TLS is required unless the DSN sets `sslmode`. An unknown scenario or one
without lots and blocks fails as an input error on `parameters.scenario_id`.

All nodes share the source's explicit parameter (`scenario_id` or `snapshot`). Lyra memoizes the dataset within
one invocation, so the two branches share one acquisition. Separate jobs read
fresh data. Imports and the factory perform no acquisition. Snapshot files
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

## Accessibility

Ported from reimagina-lyra `accessibility.py` (exe_reimagina's model job). The
CFC model (`accessibility_metrics.py`, its vectorized `accessibility_fastpath.py`)
and the amenity vocabulary (`vocabulary.py`) are copied unchanged and excluded
from ruff so they stay diffable; `accessibility.py` adapts the orchestration to
tables.

Blocks are the origins, weighted by this run's built-form `population` (blocks
without demand are weight 0, not excluded). Destinations are elements with an
`amenity_type` in the 13-type vocabulary, at their polygon centroid; others are
logged and ignored. The model routes up to five nearest destinations per type
within 10 km at 4.8 km/h, with radii capped at 10 km and the POI-to-network snap
distance included. The metric CRS is `ACCESS_CRS`, else the blocks' UTM zone;
it fails if fewer than half the blocks fall within the network's extent.

Per block: `minutes` / `farthest_amenity` (slowest reachable type),
`accessibility_<amenity>` (gravity score as a 0-1 percentile rank among the
run's blocks, so relative to the run) and `distance_to_<amenity>` (walking
minutes to the nearest). Each lot inherits its block's values. Null means
unreachable within 10 km, never 0. Keys match `indicator_catalog`; the catalog
uses minutes ("min"), which Lyra cannot declare, so those columns have no unit.

`reimagina_road_network` reads `projects.road_network_uri` (an `az://` URI in
the CFC lake, or a local path), downloads it into the job's scratch folder with
`DefaultAzureCredential`, and returns the pandana H5's nodes and edges (its first
impedance as `distance`). A project without the URI fails as an input error.

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

Density is published in homes per hectare (viv/ha), as the original calculation
returns it and as Reimagina's `indicator_catalog` defines it; there is no unit
conversion. Lyra's unit vocabulary has no count/ha, so the column declares no
unit and its description states viv/ha.

## Packaging and dependencies

The existing `uv_build` backend is retained. `scripts/build_manifest.py` invokes
the SDK generator and stages identical bytes under
`data/share/lyra/plugins/reimagina-test-port/lyra.plugin.json`. The root manifest
is generated, never edited manually. Regenerate it after declaration changes.

Lyra dependencies come from GitHub and their resolved revisions are in `uv.lock`.
The SDK must provide `LocalRunner`. `psycopg` matches the version in the Lyra
image. Accessibility adds `pandana`, `tables`, `networkx`, `adlfs` and
`azure-identity`; pandana is imported only when accessibility runs. This project has no `lyra-app` dependency and does not import server
internals. Deployment is described in [deployment](deployment.md).
