# Design and provenance

## Package layout

One package holds both the baseline data and the indicators, so an indicator that
needs new baseline data changes both in the same repository.

```
src/reimagina_test_port/
├── plugin.py      create_plugin / create_snapshot_plugin: assembles the nodes
├── scenario/      a Reimagina scenario in Postgres: source.py (read), writer.py,
│                  contract.py, nodes.py (reimagina_scenario*, road network, POIs)
├── snapshot.py    prepared GeoParquet snapshots, the offline sources
├── ingestion/     a city's baseline from the data lake (data, not indicators):
│                  cities/*.yaml, prep, footprints, cadastral, elements,
│                  crosswalk (+ crosswalk.csv), raw_amenities, points,
│                  baseline, catalogs,
│                  baseline_writer, nodes.py (reimagina_baseline*)
├── indicators/    one subpackage per indicator type, each with its nodes.py
│   ├── built_form/      calculation.py
│   └── accessibility/   model.py, metrics.py, fastpath.py, network.py
└── common/        shared: lake, vocabulary, schema (element identity,
                   attributes, units), rules (MIN_BUILT_M2), nodes (timed, column)
```

A new indicator type is a new `indicators/<type>/` with its `nodes.py`, added to
`create_plugin()`; data it needs in the baseline goes into `ingestion/`.

## Graph

`reimagina_scenario` is an internal dataset producing a Reimagina scenario's
effective lots and blocks from Postgres. `reimagina_lot_built_form` depends on
its lot output and computes all lot indicators. `reimagina_built_form` depends on
that metric's result and on the source block output, aggregates lots by parent
ID, and returns both tables. `reimagina_accessibility` depends on both
built-form tables and on two more datasets: `reimagina_scenario_pois` (present
amenities and establishments with an accessibility `amenity_type`) and `reimagina_road_network` (the project's road
network as node and edge tables). `reimagina_scenario_indicators` writes the
built-form tables and `reimagina_scenario_accessibility` the accessibility
tables to Reimagina's database; `reimagina_scenario_write` is the final node and
runs both. `reimagina_baseline_elements` returns a city's baseline
lots and blocks built from the data lake by the `reimagina_baseline` dataset;
nothing is written. `reimagina_baseline_write` writes them into a baseline
scenario. `create_plugin()` registers this graph.

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

Ported from reimagina-lyra `accessibility.py` (exe_reimagina's model job), in
`indicators/accessibility/`. The CFC model (`metrics.py`, its vectorized `fastpath.py`)
and the amenity vocabulary (`common/vocabulary.py`) are copied unchanged and excluded
from ruff so they stay diffable; `model.py` adapts the orchestration to
tables.

Blocks are the origins, weighted by this run's built-form `population` (blocks
without demand are weight 0, not excluded). Destinations are the present amenity
and establishment elements, at their point. Their `category` and `scian_code` give
one of the model's 13 types (`vocabulary.amenity_type_of`): the longest SCIAN
prefix first (`61112` primary school, `46411` pharmacy, `512130` cinema...), else
the category (`public_space` park, `sports_recreation` gym, `cultural_facility`
recreation). Schools without a SCIAN code are left out, as their level is
unknown (Mexicali loads none: its schools come from the DENUE). Elements with no type are ignored. Until the model's
indicators move to the categories, its keys and radii are unchanged. The model routes up to five nearest destinations per type
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

## Writer

Ported from reimagina-lyra `store.write_indicators`. Values go to
`element_state_indicator_value` in two transactions, as reimagina-lyra's model
job does: `reimagina_scenario_indicators` writes the built-form keys and
`reimagina_scenario_accessibility` the accessibility keys, on blocks only (a
lot's accessibility is its block's; readers take it from the lot's
`parent_id`); each leaves the other's keys untouched, and
`reimagina_scenario_write` runs both. Every key
must be active in `indicator_catalog` (see the catalog contract). Floats are
rounded to 4 decimals, so a fork compares exactly against its baseline.

A fork stores only values that differ from its baseline (same tolerance as the
API). Owned keys without a value are sent as NULL, so a fork can hide a
baseline value and a baseline drops stale ones. Scenario rows that match the
baseline again, or belong to an element no longer present, are removed. Rows
whose stored value is unchanged are not rewritten and keep their `job_run_id`
and `computed_at`, so writing the same scenario again changes nothing.

For a fork, values that cannot have changed are not sent to the database at
all. A lot the fork did not edit (no `element_state` row) and that has no stored
fork value has its baseline value, which is what it computes. Blocks are always
sent. For one edited lot this
sends a few hundred thousand rows instead of ~14.6 million. `full` sends
everything, for when the baseline's stored values were computed another way;
tests check after each write that `full` would change nothing.

`job_run_id` is recorded only when that `job_run` exists. `dry_run` computes
and counts the changes, then rolls the transaction back. Each write returns one
row: `elements`, `values_written`, `values_cleared`, `dry_run`;
`reimagina_scenario_write` adds up both. Each node logs its duration and the
writer its stages (`select`, `copy`, `compare`, `delete`, `upsert`) through the
job's logger.

## Baseline from the data lake

Adapted from reimagina-lyra `ingestion/pipeline.build_elements`, with the same
sources (CFC lake, `DefaultAzureCredential`): INEGI 2024 urban blocks and the
2020 census, the IMIP cadastre, the zoning normativa, Open Buildings 2.5D
heights, IMIP's equipment layers and the INEGI DENUE. The code lives in
`ingestion/` (data only; indicators are computed elsewhere). `ingestion/prep.py`
and `footprints.py` are copied unchanged (excluded from ruff): blocks join census
by CVEGEO; lots get area, normativa (containing polygon, else nearest within 3 m)
and the block of their nearest representative point; heights give `cos`,
`num_levels` and built-up area. `elements.py` gives `commerce_type`, the
multifamily `unit_size_m2` and `land_use`, classified from what each lot holds
(below). Blocks also declare `lot.size_m2` and `lot.frontage_m`, targets that stay
unset when the city is loaded (reimagina_urbano_api `urban_fabric.md`).

Ids use exe_reimagina's namespace. A block is `uuid5(NS, "block:<cvegeo>")`,
unchanged. A lot is `uuid5(NS, "<city>:lot:<key>")` (`city` is the preset key,
e.g. `mexicali`), where the key joins the city's cadastral `key_columns`
(Mexicali: `Numdel`, `Colonia`, `Manzana`, `Lote`, `Fraccion`), stripped, with
"-" and trailing dashes removed; empty is `sin_clave`. Exact copies (same key
and geometry) are dropped, and lots still sharing a key get `#2`, `#3`...
ordered by representative point (x, then y) in the city's metric CRS. The
cadastral row number is never used, so re-exporting the shapefile keeps the
ids. The key is the lot's `source_record_ref`.

This changes the ids of the Mexicali baseline already loaded, which used the
row number (`uuid5(NS, "lot:<row>")`): all 388,023 lot ids differ (16 exact
copies drop out, leaving 388,007), and lot rows of its forks would refer to
ids no longer in the baseline. The loaded baseline is not modified: the new one
is written to a new baseline scenario of the same project (below), so both can
be compared.

`num_levels` is an integer in `attribute_catalog`, so the baseline rounds the
raster's average floors to whole floors, as the API's integer decoding does.

Each city is a YAML in `ingestion/cities/` (`<city>.yaml`, the city key), laid
out as reimagina_pipelines_lab's `pipelines/ingestion/cities`: codes and CRS,
`sources` (`blocks`, `census`, `parcels`, `zoning`, `heights`, `amenities`,
`denue`; a `uri` and its `params` each) and the builders' `params` (`lots`,
`amenities`, `points`). Values come from exe_reimagina's
`src/pipeline/cities/*.toml`. A missing or unknown key fails at import, and the
`city` parameter of the baseline nodes accepts every YAML in the folder. Only
Mexicali is configured: Tlalpan's normativa derives its maxima from levels and
open area, which `prep.prepare_normativa` does not do.

Amenities and establishments are elements of their own (`ingestion/points.py`),
placed as `urban_fabric.md` says. Every record of the raw INEGI DENUE CSV whose
`municipio` is the city's is an amenity or an establishment, never both, by the
SCIAN crosswalk (`ingestion/crosswalk.csv`: a prefix, an element type and a
category, the longest prefix winning; no match is an establishment of category
`other`). The crosswalk is a **draft** written from SCIAN 2018 until the API
repository has the official table; rows marked "draft" or "open question" in its
`note` column are guesses.

The other amenities come from each city's own layers, in two steps. The
`raw_amenities` step (`ingestion/raw_amenities.py`) reads the raw layers of
`sources.amenities` (a folder) with the normalizer the YAML names
(`params.normalizer`) and returns the **amenities file**: one point per amenity
with `source`, `source_id`, `category` (an amenity category), `name` and
`area_m2`. The `points` step then places it like the DENUE. A new city with
another source adds a normalizer to `raw_amenities.NORMALIZERS`.

Each amenity category comes from one source only, as exe_reimagina did, so no
amenity is loaded twice: the categories in `sources.amenities.params.categories`
come only from the amenities file, every other one only from the DENUE; the
report counts the rows each source left to the other (`file_skipped`,
`denue_skipped`). Mexicali takes `public_space`, `sports_recreation` and
`cultural_facility` from IMIP (366 sports and 339 culture places, against the
DENUE's 18 and 47) and schools from the DENUE (with their SCIAN level and
workers). Matched within 100 m, about 214 IMIP schools have no DENUE school of
the same level nearby, and about 40 DENUE sports and culture places no IMIP one.

Mexicali's normalizer, `imip`, reads IMIP's equipment layers
(`az://cleaned/por_estado/02_baja_california/002_mexicali/imip/equipamiento/`)
with exe_reimagina's `2_equip_for_exe.ipynb` rules: six of its 15 layers,
`edu_basica` split by `TIPO` (primary and preschool `primary_education`, general
and technical secondary `secondary_education`; CENDI and telesecundaria left
out), `med_sup` `secondary_education`, `edu_sup` `tertiary_education`, `deporte`
`sports_recreation`, `cult_rec` `cultural_facility`, and `esp_pub` filtered to
neighbourhood gardens and parks (`public_space`) and sports spaces
(`sports_recreation`; exe counted them as parks). Names come from `NOM_ESC` or `NOMBRE` (`esp_pub` has none); `area_m2`,
of public spaces only, from `esp_pub`'s `Sup_M2`. It replaces the copy of that
notebook's output exe_reimagina read (`pois_accesibilidad/pois_02002`), which
also lost names and areas.

DENUE records are keyed `denue:<id>`, with `category`, `scian_code`
(`codigo_act`), `num_workers` (the `per_ocu` range's midpoint; the top range, its
floor; required, so an unknown range fails) and `name` (`nom_estab`);
establishments add `scian_sector` (the code's first two digits). Amenities-file
records are keyed `source:source_id` (Mexicali: `IMIP:<layer>:<OBJECTID>`), with
`category`, `name` and `area_m2`. Amenities also carry `capacity`
(`default_capacity`). `tree_coverage` of public spaces is not loaded: Meta's
canopy height is not a source yet. Each record goes to the
lot containing it, else the nearest lot within `lot_max_distance_m` (50 m), as a
point at that lot's representative point; farther ones are not loaded and are
counted in the report. Ids are `uuid5(NS, "<city>:amenity:<key>")` and
`uuid5(NS, "<city>:establishment:<key>")`, the lab's `element_id`.

`land_use` follows `ingestion_pipelines.md` (*Land use from contents*), with
homes estimated from the census until the space allocation exists: a built lot
(over 20 m²) zoned for housing holds homes when its block has any, its share of
the block's homes by built-up area. With homes and establishments a lot is
`mixed_use`; with homes only, `multifamily` from `mf_min_units` homes or one home
in a building of 600 m² and 2.5 floors, else `single_family`; with establishments
only, `industrial` if most workers are in manufacturing, else `commerce`
(`office`, "all offices", is not assigned: the design does not say which
establishments are offices); with amenities only,
`amenity`; with nothing, its zone's use. Amenities don't change the class of a lot
that also holds homes or establishments. Lots no longer carry `amenity_type`: an
amenity's category lives on the amenity element.

`housing.floor_share` (0-100) is the space allocation by lot type, a stand-in
for the per-lot space allocation of `ingestion_pipelines.md` until it is
computed: each city's YAML gives a share per land use
(`params.space_allocation.housing.floor_share`), and each lot takes its land
use's. A land use not listed, or a lot without land use, has no value. Mexicali:
`mixed_use` 80 (80 housing, 20 other), every other land use 100.

## Baseline writer

`reimagina_baseline_write(city, project_id, scenario_id)` fills an existing
baseline scenario (created in the API, without forks) in one transaction:
`elements` (new ids only; an existing element, such as a block shared with
another baseline, is reused unchanged), `element_state` and
`element_state_attribute_value`, for blocks, lots, amenities and
establishments. The API's `element_type` enum must include `amenity` (it has
`establishment`); otherwise the write fails naming the missing types. Before
writing, `ingestion/catalogs.py` adds the `attribute_catalog` rows of
`BASELINE_ATTRIBUTES` and the `indicator_catalog` rows of every built-form and
accessibility indicator (lots and blocks) that are missing, with the API seed's
types, units and names and the categories as enum values (`ON CONFLICT DO
NOTHING`). Rows already there are left as they are, so an existing row that is
deprecated, of another type or whose enum lacks a value the baseline uses (for
example `lot.land_use` without `industrial`) still fails the contract check,
naming it. The write returns `attributes_added` and `indicators_added`; a dry run
rolls them back too. Running it again replaces the scenario's own
rows, removing elements no longer present and their indicator values. Then
`reimagina_scenario_write` computes the baseline's indicators, and after it
each fork's.

Moving the edits of the old baseline's forks is a one-off migration,
`scripts/migrate_fork_edits.py SOURCE_FORK TARGET_FORK [--apply]`: each edited
lot is matched to the new lot with the largest overlap (IoU >= 0.9); blocks and
fork-only elements keep their id. It rolls back unless `--apply`.

## Catalog contract

The API owns `attribute_catalog` and `indicator_catalog` (alembic). Before
writing, `contract.check` verifies that every key the plugin writes is active,
with a compatible value type (number/integer, text/enum) and, for enums with
`enum_values`, the values it writes. A mismatch fails the job naming what to add,
instead of skipping keys.

## Arithmetic

Source: Reimagina `dev` commit `cf17664d169116a2760741ea92737abf9b9ad212`,
`pipelines/indicators/built-form/src/built_form/__init__.py`.

The pure functions in `indicators/built_form/calculation.py` preserve the original threshold, land-use
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
`azure-identity`, and the baseline adds `rasterio`; both are imported only when
those nodes run. This project has no `lyra-app` dependency and does not import server
internals. Deployment is described in [deployment](deployment.md).
