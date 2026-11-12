"""Exercise SDK serialization and Python client decoding with no HTTP traffic."""

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock

import geopandas as gpd
import pytest
from geopandas.testing import assert_geodataframe_equal
from lyra.api import LyraClient
from lyra.api.client import sync
from lyra.api.client.endpoints import validate_response
from lyra.sdk.models.job import (
    JobRunProvenance,
    TableJobResult,
    build_result_descriptor,
)
from lyra.sdk.models.plugin import PluginInfo
from lyra.sdk.results import serialize_outputs
from lyra.sdk.types import JsonObject

from reimagina_test_port.local import run_snapshot
from reimagina_test_port.plugin import create_snapshot_plugin


@pytest.mark.parametrize("empty", [False, True])
def test_client_submission_polling_tables_descriptor_download(
    monkeypatch, tmp_path, empty
):
    metric = "reimagina_built_form"
    arguments: JsonObject = {"parameters": {"snapshot": "synthetic"}}
    tables = run_snapshot("synthetic")
    if empty:
        tables = {key: table.iloc[:0] for key, table in tables.items()}
    manifest = create_snapshot_plugin().manifest(
        plugin=PluginInfo(name="reimagina-test-port", version="0.1.0"),
        factory="reimagina_test_port.plugin:create_snapshot_plugin",
    )
    contract = next(node for node in manifest.nodes if node.name == metric)
    now = datetime.now(UTC)
    result = TableJobResult.model_validate(
        {
            "job_id": "offline-job",
            "outputs": serialize_outputs(metric, tables),
            "output_metadata": contract.outputs,
            "provenance": JobRunProvenance(
                metric=metric,
                catalog_fingerprint="offline-fixture",
                plugin=manifest.plugin,
                input=arguments,
                outputs=contract.outputs,
                created_at=now,
            ),
        }
    )
    payload = result.model_dump(mode="json")
    descriptor = build_result_descriptor(result, completed_at=now).model_dump(
        mode="json"
    )
    calls = []

    def response_for(method, url, body=None):
        calls.append((method, url))
        if method == "POST":
            assert body == {"metric": metric, "input": arguments}
            return 202, {
                "job_id": "offline-job",
                "metric": metric,
                "status": "queued",
                "links": {
                    "self": "/jobs/offline-job",
                    "result": "/jobs/offline-job/result",
                },
            }
        if url.endswith("/descriptor"):
            return 200, descriptor
        if url.endswith("/jobs/offline-job"):
            return 200, {
                "job_id": "offline-job",
                "status": "succeeded",
                "updated_at": now.isoformat(),
                "created_at": now.isoformat(),
            }
        assert url.endswith(("/result", "/download"))
        return 200, payload

    def fake_request(method, url, **kwargs):
        status, body = response_for(method, url, kwargs.get("json"))
        response = MagicMock()
        response.__enter__.return_value = response
        response.status_code = status
        response.text = json.dumps(body)
        response.iter_content.return_value = [response.text.encode()]
        response.headers = {"content-type": "application/json"}
        return response

    async def fake_poll(spec, url, headers, request_seconds, **kwargs):
        status, body = response_for(spec.method, url, spec.json_body)
        return validate_response(
            spec, status, json.dumps(body), "application/json", None
        )

    monkeypatch.setattr(sync.requests, "request", fake_request)
    monkeypatch.setattr(
        sync.requests, "get", lambda url, **kwargs: fake_request("GET", url, **kwargs)
    )
    monkeypatch.setattr(sync, "request_json", fake_poll)
    client = LyraClient("offline.invalid", agent_api_key="test-only")
    handle = client.raw.submit(metric, arguments)
    assert handle.wait(timeout=5).model_dump(mode="json") == payload
    decoded = client.results.tables(handle.job_id)
    for name, table in decoded.items():
        assert isinstance(table, gpd.GeoDataFrame)
        assert table.crs == "EPSG:4326"
        expected = tables[name].to_crs(4326).reset_index(drop=True)
        assert table.id.tolist() == expected.id.tolist()
        assert_geodataframe_equal(table, expected, check_dtype=False, check_like=True)
    metadata = client.results.descriptor(handle.job_id)
    for output in metadata.outputs.values():
        density = next(
            column for column in output.declaration.columns if column.name == "density"
        )
        assert density.unit is None
        assert "viv/ha" in density.description
    target = tmp_path / "result.json"
    client.results.download(handle.job_id, target)
    assert json.loads(target.read_text()) == payload
    assert sum(method == "POST" for method, _ in calls) == 1
