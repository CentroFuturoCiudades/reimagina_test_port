# Deployment

The deployable image is Lyra's published image with this plugin installed on
top. API and workers run from the same image.

## Base image

`ghcr.io/rodolfofigueroa/lyra:0.25.0` (Python 3.11, Lyra commit
`2399c8e`, tag `lyra-v0.25.0`). The `[tool.uv.sources]` in `pyproject.toml`
pin `lyra-sdk`, `lyra-api` and `lyra-utils` to the same commit, and the
dependency floors (pandas 3.0.5, numpy 2.4.6, geopandas 1.1.4) match the
versions in that image. `lyra-api` is a dev dependency only; the image does
not include it.

When bumping Lyra, change `LYRA_IMAGE`, the source `rev`s and, if needed, the
floors together, then run `uv lock`.

## Build

```sh
docker build -t reimagina-lyra:$(uv version --short) .
```

The Dockerfile:

- writes a constraint file with every distribution already in the base image
  and installs the plugin with `uv pip install --no-sources --constraint`, so
  no base package is upgraded or replaced (only new ones, such as `pyarrow`,
  are added);
- runs `uv pip check`;
- imports the plugin and loads its installed manifest.

uv and the source tree are mounted only during the build. Override the base
with `--build-arg LYRA_IMAGE=...`. For Azure, build for `linux/amd64` and push
to the registry:

```sh
docker build --platform linux/amd64 -t <ACR>.azurecr.io/reimagina-lyra:<version> .
docker push <ACR>.azurecr.io/reimagina-lyra:<version>
```

## Run locally

`deploy/` contains a Compose stack (API, `interactive` worker, Redis) and the
Lyra config. The config has no `[database]` or `[earth_engine]` section: the
plugin reads Reimagina's database through its own `REIMAGINA_POSTGRES_DSN`
secret, which only the worker receives. TLS is required unless the DSN sets
`sslmode`. All metrics route to the `interactive` queue.

Accessibility downloads the project's road network from the CFC data lake
(`az://`, account `cfcetlsadls`) with `DefaultAzureCredential`: give the
worker a managed identity with read access, or set `AZURE_CLIENT_ID`,
`AZURE_TENANT_ID` and `AZURE_CLIENT_SECRET`. Optional: `ACCESS_CRS` forces the
network's CRS; `ACCESSIBILITY_FASTPATH=0` uses the original, slower model code.

```sh
cd deploy
cp .env.example .env   # set both API keys and REIMAGINA_POSTGRES_DSN
docker compose up -d --wait
```

Submit a job and poll it:

```sh
curl -X POST localhost:5219/jobs \
  -H "Authorization: Bearer $LYRA_AGENT_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"metric":"reimagina_built_form","input":{"parameters":{"scenario_id":"<uuid>"}}}'
curl localhost:5219/jobs/<job_id> -H "Authorization: Bearer $LYRA_AGENT_API_KEY"
```

`/jobs/<job_id>/result` returns `reimagina_built_form_lots` and
`reimagina_built_form_blocks` for the scenario's effective lots and blocks.

`REIMAGINA_LYRA_IMAGE` selects the image (local tag or registry). The
deployed plugin has no snapshot dataset; snapshots run offline only (see
[usage](usage.md)).
