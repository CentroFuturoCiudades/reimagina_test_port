# Reimagina built-form port

A Lyra plugin that calculates lot and block built-form indicators from prepared
GeoParquet snapshots. Includes a synthetic example that runs entirely offline.

Run the example with `uv run python examples/local_snapshot.py`.

See [usage and snapshot format](docs/usage.md), [design and provenance](docs/design.md),
and [verification](docs/verification.md).

# Diagram

```mermaid
flowchart LR
    n0["Metric: reimagina_built_form<br/>Output: reimagina_built_form_blocks<br/>id: string<br/>parent_id: string<br/>source_record_ref: string<br/>cos: number #91;ratio#93;<br/>num_levels: number #91;count#93;<br/>cus: number #91;ratio#93;<br/>area: number #91;m2#93;<br/>unit_size_m2: number #91;m2#93;<br/>num_housing_units: number #91;count#93;<br/>population: number #91;count#93;<br/>density: number #91;count#47;km2#93;<br/>Geometry: geometry<br/>Output: reimagina_built_form_lots<br/>id: string<br/>parent_id: string<br/>source_record_ref: string<br/>cos: number #91;ratio#93;<br/>num_levels: number #91;count#93;<br/>cus: number #91;ratio#93;<br/>area: number #91;m2#93;<br/>unit_size_m2: number #91;m2#93;<br/>num_housing_units: number #91;count#93;<br/>population: number #91;count#93;<br/>density: number #91;count#47;km2#93;<br/>Geometry: geometry"]
    n1["Metric: reimagina_lot_built_form<br/>Output: reimagina_lot_indicators<br/>id: string<br/>parent_id: string<br/>source_record_ref: string<br/>cos: number #91;ratio#93;<br/>num_levels: number #91;count#93;<br/>cus: number #91;ratio#93;<br/>area: number #91;m2#93;<br/>unit_size_m2: number #91;m2#93;<br/>num_housing_units: number #91;count#93;<br/>population: number #91;count#93;<br/>density: number #91;count#47;km2#93;<br/>Geometry: geometry"]
    n2(["Dataset: reimagina_prepared_snapshot<br/>Output: reimagina_prepared_blocks<br/>id: string<br/>parent_id: string<br/>source_record_ref: string<br/>Geometry: geometry<br/>Output: reimagina_prepared_lots<br/>id: string<br/>parent_id: string<br/>source_record_ref: string<br/>cos: number #91;ratio#93;<br/>num_levels: number #91;count#93;<br/>area: number #91;m2#93;<br/>land_use: string<br/>unit_size_m2: number #91;m2#93;<br/>people_per_household: number #91;count#93;<br/>occupancy_rate: number #91;ratio#93;<br/>Geometry: geometry"])
    n2 -->|"reimagina_prepared_blocks #8594; blocks"| n0
    n1 -->|"reimagina_lot_indicators #8594; lots"| n0
    n2 -->|"reimagina_prepared_lots #8594; lots"| n1

```