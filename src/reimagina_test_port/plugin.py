"""The plugin's two graphs, assembled from the nodes of each part of the package.

- ``scenario``: read a Reimagina scenario from Postgres and write its indicators.
- ``snapshot``: prepared GeoParquet snapshots, the offline sources.
- ``ingestion``: build a city's baseline from the data lake (data, not indicators).
- ``indicators``: one subpackage per indicator type (``built_form``,
  ``accessibility``, ``population_access``).
- ``common``: what ingestion and indicators share.

Production (``create_plugin``) runs over Reimagina scenarios; the snapshot graph
(``create_snapshot_plugin``) runs the same metrics offline.
"""

from lyra.sdk import PluginDefinition

from reimagina_test_port import snapshot
from reimagina_test_port.common.vocabulary import DESTINATION_COVERAGE_M
from reimagina_test_port.indicators.accessibility.nodes import accessibility_metric
from reimagina_test_port.indicators.built_form import nodes as built_form_nodes
from reimagina_test_port.indicators.population_access.nodes import (
    population_access_metrics,
)
from reimagina_test_port.ingestion import nodes as ingestion_nodes
from reimagina_test_port.scenario import nodes as scenario_nodes
from reimagina_test_port.scenario import source as scenario_source


def create_plugin() -> PluginDefinition:
    """Production graph over Reimagina scenarios; registers without connecting."""
    return PluginDefinition(
        nodes=[
            scenario_nodes.scenario_dataset,
            scenario_nodes.scenario_pois,
            scenario_nodes.road_network,
            *built_form_nodes.built_form_metrics(
                scenario_nodes.ScenarioParameters,
                "reimagina_scenario_lots",
                "reimagina_scenario_blocks",
            ),
            accessibility_metric(
                scenario_nodes.ScenarioParameters,
                "reimagina_scenario_pois",
                "reimagina_road_network_nodes",
                "reimagina_road_network_edges",
            ),
            *population_access_metrics(
                scenario_nodes.ScenarioParameters,
                "reimagina_scenario_lots",
                "reimagina_scenario_pois",
                "reimagina_road_network_nodes",
                "reimagina_road_network_edges",
                scenario_source.read_coverage_m,
            ),
            scenario_nodes.scenario_indicators,
            scenario_nodes.scenario_accessibility,
            scenario_nodes.scenario_population_access,
            scenario_nodes.scenario_write,
            ingestion_nodes.baseline_dataset,
            ingestion_nodes.baseline_elements,
            ingestion_nodes.baseline_write,
        ]
    )


def create_snapshot_plugin() -> PluginDefinition:
    """The same metrics over bundled or prepared snapshots, for offline runs."""
    return PluginDefinition(
        nodes=[
            snapshot.prepared_snapshot,
            snapshot.prepared_accessibility,
            *built_form_nodes.built_form_metrics(
                snapshot.SnapshotParameters,
                "reimagina_prepared_lots",
                "reimagina_prepared_blocks",
            ),
            accessibility_metric(
                snapshot.SnapshotParameters,
                "reimagina_prepared_pois",
                "reimagina_prepared_network_nodes",
                "reimagina_prepared_network_edges",
            ),
            *population_access_metrics(
                snapshot.SnapshotParameters,
                "reimagina_prepared_lots",
                "reimagina_prepared_pois",
                "reimagina_prepared_network_nodes",
                "reimagina_prepared_network_edges",
                lambda: DESTINATION_COVERAGE_M,
            ),
        ]
    )
