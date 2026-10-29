"""Regenerate synthetic inputs and golden outputs from pinned upstream arithmetic.

Usage: uv run python scripts/generate_fixture.py /path/to/reimagina_pipelines_lab
This developer command reads local Git objects; it never fetches or calls services.
"""

import argparse
import ast
import json
import runpy
import subprocess
import tempfile
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

REVISION = "cf17664d169116a2760741ea92737abf9b9ad212"
SOURCE = "pipelines/indicators/built-form/src/built_form/__init__.py"
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    args = parser.parse_args()
    source = subprocess.check_output(
        ["git", "-C", str(args.upstream), "show", f"{REVISION}:{SOURCE}"], text=True
    )
    tree = ast.parse(source)
    # Execute only the original pure functions and their literal constants.
    selected: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        or isinstance(node, ast.FunctionDef)
        and node.name in {"_number", "lot_indicators", "block_indicators"}
    ]
    with tempfile.TemporaryDirectory() as directory:
        reference = Path(directory) / "reference.py"
        reference.write_text(
            "from typing import Any\n"
            + ast.unparse(ast.Module(body=selected, type_ignores=[]))
        )
        namespace = runpy.run_path(str(reference))
    house = {
        "land_use": "single_family",
        "cos": 0.5,
        "num_levels": 2,
        "area": 200,
        "people_per_household": 3.4,
    }
    flat = {
        "land_use": "multifamily",
        "cos": 0.7,
        "num_levels": 4,
        "area": 800,
        "unit_size_m2": 70,
        "people_per_household": 2.8,
    }
    cases = [
        ("house", house),
        (
            "shed",
            {"land_use": "single_family", "cos": 0.05, "num_levels": 1, "area": 300},
        ),
        ("flats", flat),
        ("half_empty", {**flat, "cos": 0.8, "area": 700, "occupancy_rate": 0.5}),
        ("shop", {"land_use": "commerce", "cos": 1, "num_levels": 1, "area": 100}),
        ("unknown", {"cos": 0, "num_levels": 0, "area": 100}),
        ("threshold", {**house, "area": 20}),
        ("zero_occupancy", {**flat, "occupancy_rate": 0}),
        ("zero_area", {**house, "area": 0}),
        ("missing", {}),
        ("ties", {**flat, "cos": 1, "num_levels": 1, "area": 175}),
        ("mixed", {**flat, "land_use": "mixed_use"}),
        ("orphan", house),
        ("no_parent", house),
    ]
    rows, expected_lots, groups = (
        [],
        {},
        {"block-z": [], "block-a": [], "block-empty": []},
    )
    for i, (name, attributes) in enumerate(cases):
        parent = (
            ("block-z" if i % 2 else "block-a")
            if i < 12
            else ("absent-block" if i == 12 else None)
        )
        identifier = f"lot-{name}"
        rows.append(
            {
                "id": identifier,
                "parent_id": parent,
                "source_record_ref": f"synthetic:{name}",
                **attributes,
            }
        )
        values = namespace["lot_indicators"](attributes)
        expected_lots[identifier] = values
        if parent in groups:
            groups[parent].append(values)
    expected_blocks = {
        key: namespace["block_indicators"](values) for key, values in groups.items()
    }
    folder = ROOT / "src/reimagina_test_port/data/synthetic"
    folder.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        rows,
        geometry=[
            box(640000 + i * 25, 3610000, 640020 + i * 25, 3610020)
            for i in range(len(rows))
        ],
        crs=32611,
    ).to_parquet(folder / "lot.parquet")
    gpd.GeoDataFrame(
        {"id": list(groups), "parent_id": [None] * 3, "source_record_ref": [None] * 3},
        geometry=[
            box(640000, 3610000 + i * 100, 640400, 3610090 + i * 100) for i in range(3)
        ],
        crs=32611,
    ).to_parquet(folder / "block.parquet")
    output = {
        "revision": REVISION,
        "source": SOURCE,
        "lots": expected_lots,
        "blocks": expected_blocks,
    }
    (ROOT / "tests/expected.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
