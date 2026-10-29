# syntax=docker/dockerfile:1
# Imagen de Lyra con el plugin reimagina-test-port instalado encima.
#   docker build -t reimagina-lyra:$(uv version --short) .
# La misma imagen corre la API y los workers (ver docs/deployment.md).
ARG LYRA_IMAGE=ghcr.io/rodolfofigueroa/lyra:0.25.0
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.6

FROM ${UV_IMAGE} AS uv

FROM ${LYRA_IMAGE}
ENV UV_LINK_MODE=copy UV_PYTHON=/app/.venv/bin/python
# Fija los paquetes de la imagen base para que el plugin no los actualice.
# uv y el código fuente se montan solo durante el build; no quedan en la imagen.
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=bind,source=.,target=/src,rw \
    --mount=type=cache,target=/root/.cache/uv \
    python -c "import importlib.metadata as m; print('\n'.join(sorted({d.metadata['Name'] + '==' + d.version for d in m.distributions()})))" \
        > /tmp/base-constraints.txt \
    && uv pip install --no-sources --constraint /tmp/base-constraints.txt /src \
    && uv pip check \
    && rm /tmp/base-constraints.txt
RUN python -c "import reimagina_test_port.plugin" \
    && python -c "from lyra.sdk.plugin_loader import load_plugin_manifest; load_plugin_manifest('reimagina-test-port')"
