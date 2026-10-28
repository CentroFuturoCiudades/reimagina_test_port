"""Run prepared snapshots through Lyra's public local executor."""

from lyra.sdk import LocalRunner
from lyra.sdk.results import NativeOutputs

from reimagina_test_port.plugin import create_plugin


def run_snapshot(snapshot: str, metric: str = "reimagina_built_form") -> NativeOutputs:
    """Calculate a snapshot locally using the SDK's default runtime."""
    runner = LocalRunner.from_definitions({"reimagina-test-port": create_plugin()})
    return runner.run(metric, {"parameters": {"snapshot": snapshot}})
