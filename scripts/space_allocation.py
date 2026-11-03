"""Write a city's space allocation source: housing.floor_share per lot.

A stand-in for the space allocation of reimagina_urbano_api ``ingestion_pipelines.md``
until it is computed: one row per lot, ``lot_key`` (cadastral key) and
``housing.floor_share`` (0-100), from the lots' ``land_use`` of an earlier
baseline build (``scripts/baseline_counts.py`` saves them). Mixed-use lots are
80 % housing, every other lot 100 %.

    uv run python scripts/space_allocation.py mexicali [--workdir DIR] [--upload]

Writes ``<workdir>/<city>/<file name of space_allocation_uri>``; --upload copies
it to the city's ``space_allocation_uri`` in the data lake (``az login``), which
the next build reads.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from reimagina_test_port.common import lake
from reimagina_test_port.ingestion.cities import CITIES

MIXED_USE_SHARE = 80.0
DEFAULT_SHARE = 100.0


def floor_shares(lots: pd.DataFrame) -> pd.DataFrame:
    share = lots["land_use"].map({"mixed_use": MIXED_USE_SHARE}).fillna(DEFAULT_SHARE)
    return pd.DataFrame(
        {"lot_key": lots["source_record_ref"].astype(str), "housing.floor_share": share}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("city", choices=sorted(CITIES))
    parser.add_argument("--workdir", type=Path, default=Path(".cache/baseline"))
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    city = CITIES[args.city]
    folder = args.workdir / args.city
    lots = gpd.read_parquet(folder / "elements" / "lot.parquet")
    table = floor_shares(lots)
    target = folder / city.space_allocation_uri.rsplit("/", 1)[-1]
    table.to_parquet(target, index=False)
    counts = table["housing.floor_share"].value_counts().sort_index().to_dict()
    print(f"{len(table)} lots -> {target}: {counts}")
    if args.upload:
        lake.filesystem().put(str(target), city.space_allocation_uri)
        print(f"Uploaded to {city.space_allocation_uri}")


if __name__ == "__main__":
    main()
