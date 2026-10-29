# Offline verification

Run from the port repository:

```sh
uv run python scripts/build_manifest.py
uv run lyra-plugin describe
uv run lyra-plugin check-manifest
uv run ruff format
uv run ruff check
uv run ty check
uv run coverage run --source=reimagina_test_port -m pytest
uv run coverage report -m
uv run coverage xml
```

Tests forbid socket connections and DNS resolution, including localhost, except
those marked `postgis` (below).
Inputs are local GeoParquet files; the public SDK `LocalRunner` and serialization
execute in process. The acquisition test reuses one runner across calls and
checks that each call reads fresh data while sharing acquisition within its graph. Python client submission, polling, result decoding, descriptors, and
downloads use fake HTTP responses. No database, Redis, cloud, or Lyra instance is
used. The test guard must remain enabled. No distribution smoke checks are needed.

## PostGIS tests

`tests/test_scenario_db.py` runs the scenario dataset and the full graph against
PostGIS with a minimal copy of the API's EAV schema. It is skipped unless
`REIMAGINA_TEST_DSN` is set; each test creates and drops its own database.

```sh
docker run -d --name reimagina-test-postgis -e POSTGRES_PASSWORD=postgres \
  -p 54329:5432 postgis/postgis:16-3.4
REIMAGINA_TEST_DSN=postgresql://postgres:postgres@localhost:54329/postgres \
  uv run pytest tests/test_scenario_db.py
docker rm -f reimagina-test-postgis
```

`tests/expected.json` freezes independently generated original calculation results
for every lot and block, with density in viv/ha. Regenerate the
synthetic GeoParquet and reference values only when deliberately updating fixtures:

```sh
uv run python scripts/generate_fixture.py /path/to/reimagina_pipelines_lab
```

The generator reads the pinned commit from local Git objects, extracts the original
pure functions and literal constants, and loads them in a temporary module. It
neither imports the port calculation nor imports the Reimagina runtime. Tests use
only the committed fixtures and do not require that checkout.

Coverage includes calculation parity, boundary and null cases, orphan lots, empty
blocks and tables, malformed snapshots, one acquisition per graph invocation,
input immutability, standalone lot execution, manifest freshness, and client
round-tripping. Real-data and live-server validation have not been performed.
