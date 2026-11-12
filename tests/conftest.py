"""All tests are isolated from service and network connections.

Tests marked ``postgis`` may connect, and only to ``REIMAGINA_TEST_DSN``.
"""

import socket

import pytest


@pytest.fixture(autouse=True)
def no_services(request, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail(
            "External service access is forbidden; mock the acquisition boundary"
        )

    if request.node.get_closest_marker("postgis") is None:
        for name in ("connect", "connect_ex", "sendto"):
            monkeypatch.setattr(socket.socket, name, forbidden)
        for name in ("create_connection", "getaddrinfo"):
            monkeypatch.setattr(socket, name, forbidden)
    monkeypatch.delenv("REIMAGINA_SNAPSHOT_ROOT", raising=False)
    monkeypatch.delenv("REIMAGINA_POSTGRES_DSN", raising=False)
