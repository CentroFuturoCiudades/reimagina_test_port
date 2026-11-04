"""Count a city's baseline elements per type, and how many DENUE records make it.

Builds the baseline from the data lake as ``reimagina_baseline`` does (``az login``
locally) and prints, per element type, the elements built, and for amenities and
establishments the source points and those outside every lot. The DENUE shows the
rows of the state and of the municipality, and how many the SCIAN crosswalk made
amenities and establishments. Lots show their land use.

    uv run python scripts/baseline_counts.py mexicali [--workdir DIR]

The built tables and report are saved in the workdir (``elements/``); --saved
reads them back instead of building again.

With a project and baseline scenario it also writes them and prints the elements
of each type the scenario holds afterwards, read back from the database. It rolls
back unless --apply:

    REIMAGINA_POSTGRES_DSN=... uv run python scripts/baseline_counts.py mexicali \\
        --project PROJECT --scenario SCENARIO [--apply]
"""

import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd

from reimagina_test_port.ingestion import baseline
from reimagina_test_port.ingestion.cities import CITIES
from reimagina_test_port.ingestion.nodes import BASELINE_ATTRIBUTES

# Element type -> the key of its table in baseline.build's result.
TABLES = {
    "block": "blocks",
    "lot": "lots",
    "amenity": "amenities",
    "establishment": "establishments",
}


def row(*cells: object) -> str:
    return f"{cells[0]:<16}" + "".join(f"{c:>14}" for c in cells[1:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("city", choices=sorted(CITIES))
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path(".cache/baseline"),
        help="Downloads are kept here and reused (default: .cache/baseline/<city>).",
    )
    parser.add_argument(
        "--saved", action="store_true", help="Reuse the tables of an earlier run."
    )
    parser.add_argument("--project")
    parser.add_argument("--scenario")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if bool(args.project) != bool(args.scenario):
        parser.error("--project and --scenario go together")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # The lake client logs every request's headers at INFO.
    logging.getLogger("azure").setLevel(logging.WARNING)

    def progress(stage: str, current: int, total: int) -> None:
        print(f"[{current + 1}/{total}] {stage}", flush=True)

    saved = args.workdir / args.city / "elements"
    if args.saved:
        report = json.loads((saved / "report.json").read_text())
        tables = {t: gpd.read_parquet(saved / f"{t}.parquet") for t in TABLES}
    else:
        result = baseline.build(CITIES[args.city], args.workdir / args.city, progress)
        report = result["report"]
        tables = {t: result[name] for t, name in TABLES.items()}
        saved.mkdir(parents=True, exist_ok=True)
        (saved / "report.json").write_text(json.dumps(report, indent=2))
        for element_type, table in tables.items():
            table.to_parquet(saved / f"{element_type}.parquet")

    print()
    print(row("type", "source", "built", "outside lots"))
    for element_type, table in tables.items():
        points = report.get(TABLES[element_type], {})
        print(
            row(
                element_type,
                points.get("source", "-"),
                len(table),
                points.get("outside_every_lot", "-"),
            )
        )
    denue = report["establishments"]
    print()
    print(f"DENUE ({CITIES[args.city].denue_municipio})")
    print(row("rows in state", denue["denue_state"]))
    print(row("in municipio", denue["denue_municipio"]))
    print(row("no coordinates", denue["without_coordinates"]))
    print(row("-> amenities", denue["denue_amenities"]))
    print(row("-> establishm.", denue["denue_establishments"]))
    print(row("  outside lots", denue["outside_every_lot"]))
    print(row("  written", denue["elements"]))
    print()
    print("Land use")
    for use, count in tables["lot"]["land_use"].value_counts(dropna=False).items():
        print(row(str(use), count))
    for element_type in ("amenity", "establishment"):
        print()
        print(f"{element_type} category")
        for category, count in tables[element_type]["category"].value_counts().items():
            print(row(category, count))
    print()
    print(f"Zoning: {report['zoning']}")
    print(f"Cadastral keys: {report['lots']}")

    if args.scenario:
        from reimagina_test_port.ingestion import baseline_writer

        counts = baseline_writer.write_baseline(
            args.project,
            args.scenario,
            tables,
            BASELINE_ATTRIBUTES,
            dry_run=not args.apply,
        )
        by_type = counts.pop("by_type")
        print()
        print("Written" if args.apply else "Dry run (rolled back)", counts)
        # attributes_added / indicators_added: catalog rows the write added.
        print(row("type", "built", "in scenario"))
        for element_type, table in tables.items():
            print(row(element_type, len(table), by_type[element_type]))


if __name__ == "__main__":
    main()
