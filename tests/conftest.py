"""All tests are isolated from service and network connections."""

import socket

import pytest


@pytest.fixture(autouse=True)
def no_services(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail(
            "External service access is forbidden; mock the acquisition boundary"
        )

    for name in ("connect", "connect_ex", "sendto"):
        monkeypatch.setattr(socket.socket, name, forbidden)
    for name in ("create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, forbidden)
    monkeypatch.delenv("REIMAGINA_SNAPSHOT_ROOT", raising=False)
