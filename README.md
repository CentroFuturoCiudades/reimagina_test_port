# Reimagina built-form port

A Lyra plugin that calculates lot and block built-form indicators for a Reimagina
scenario, read from its Postgres database. A prepared GeoParquet snapshot source
and a synthetic example run the same metrics entirely offline.

Run the example with `uv run python examples/local_snapshot.py`.

See [usage and snapshot format](docs/usage.md), [design and provenance](docs/design.md),
[verification](docs/verification.md), and [deployment](docs/deployment.md).

# Diagram

```mermaid
flowchart LR
    n0["<b>Metric: reimagina_built_form</b><br/><br/><b>Inputs</b><br/>#8226;#160;blocks:#160;reimagina_scenario#46;reimagina_scenario_blocks<br/>#8226;#160;lots:#160;reimagina_lot_built_form#46;reimagina_lot_indicators<br/><br/><b>Output: reimagina_built_form_blocks</b><br/>#8226;#160;id:#160;string<br/>#8226;#160;parent_id:#160;string<br/>#8226;#160;source_record_ref:#160;string<br/>#8226;#160;cos:#160;#91;ratio#93;<br/>#8226;#160;num_levels:#160;#91;count#93;<br/>#8226;#160;cus:#160;#91;ratio#93;<br/>#8226;#160;area:#160;#91;m2#93;<br/>#8226;#160;unit_size_m2:#160;#91;m2#93;<br/>#8226;#160;num_housing_units:#160;#91;count#93;<br/>#8226;#160;population:#160;#91;count#93;<br/>#8226;#160;density:#160;number<br/>#8226;#160;Geometry:#160;geometry<br/><br/><b>Output: reimagina_built_form_lots</b><br/>#8226;#160;id:#160;string<br/>#8226;#160;parent_id:#160;string<br/>#8226;#160;source_record_ref:#160;string<br/>#8226;#160;cos:#160;#91;ratio#93;<br/>#8226;#160;num_levels:#160;#91;count#93;<br/>#8226;#160;cus:#160;#91;ratio#93;<br/>#8226;#160;area:#160;#91;m2#93;<br/>#8226;#160;unit_size_m2:#160;#91;m2#93;<br/>#8226;#160;num_housing_units:#160;#91;count#93;<br/>#8226;#160;population:#160;#91;count#93;<br/>#8226;#160;density:#160;number<br/>#8226;#160;Geometry:#160;geometry"]
    n1["<b>Metric: reimagina_lot_built_form</b><br/><br/><b>Inputs</b><br/>#8226;#160;lots:#160;reimagina_scenario#46;reimagina_scenario_lots<br/><br/><b>Output: reimagina_lot_indicators</b><br/>#8226;#160;id:#160;string<br/>#8226;#160;parent_id:#160;string<br/>#8226;#160;source_record_ref:#160;string<br/>#8226;#160;cos:#160;#91;ratio#93;<br/>#8226;#160;num_levels:#160;#91;count#93;<br/>#8226;#160;cus:#160;#91;ratio#93;<br/>#8226;#160;area:#160;#91;m2#93;<br/>#8226;#160;unit_size_m2:#160;#91;m2#93;<br/>#8226;#160;num_housing_units:#160;#91;count#93;<br/>#8226;#160;population:#160;#91;count#93;<br/>#8226;#160;density:#160;number<br/>#8226;#160;Geometry:#160;geometry"]
    n2["<b>Dataset: reimagina_scenario</b><br/><br/><b>Output: reimagina_scenario_blocks</b><br/>#8226;#160;id:#160;string<br/>#8226;#160;parent_id:#160;string<br/>#8226;#160;source_record_ref:#160;string<br/>#8226;#160;Geometry:#160;geometry<br/><b>Output: reimagina_scenario_lots</b><br/>#8226;#160;id:#160;string<br/>#8226;#160;parent_id:#160;string<br/>#8226;#160;source_record_ref:#160;string<br/>#8226;#160;cos:#160;#91;ratio#93;<br/>#8226;#160;num_levels:#160;#91;count#93;<br/>#8226;#160;area:#160;#91;m2#93;<br/>#8226;#160;land_use:#160;string<br/>#8226;#160;unit_size_m2:#160;#91;m2#93;<br/>#8226;#160;people_per_household:#160;#91;count#93;<br/>#8226;#160;occupancy_rate:#160;#91;ratio#93;<br/>#8226;#160;Geometry:#160;geometry"]
    n2 -->|"reimagina_scenario_blocks #8594; blocks"| n0
    n1 -->|"reimagina_lot_indicators #8594; lots"| n0
    n2 -->|"reimagina_scenario_lots #8594; lots"| n1
```