"""Run the bundled example without a Lyra server."""

from lyra.sdk import LocalRunner

from reimagina_test_port.plugin import create_plugin

runner = LocalRunner.from_definitions({"reimagina-test-port": create_plugin()})
tables = runner.run("reimagina_built_form", {"parameters": {"snapshot": "synthetic"}})
for name, table in tables.items():
    print(name)
    print(table.drop(columns="geometry").to_string(index=False))
