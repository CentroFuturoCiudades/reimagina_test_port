"""Prepared snapshots (GeoParquet tables beneath ``REIMAGINA_SNAPSHOT_ROOT``, by
default the bundled ``data/``): the offline sources of create_snapshot_plugin."""

import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
from lyra.sdk import MetricParameters, dataset
from lyra.sdk.errors import MetricInputError
from pydantic import Field

from reimagina_test_port.common.schema import ATTRIBUTES, IDENTITY, source_declaration
from reimagina_test_port.indicators.accessibility.nodes import (
    EDGES_DECLARATION,
    NODES_DECLARATION,
    POIS_DECLARATION,
)


class SnapshotParameters(MetricParameters):
    """Select an explicit prepared snapshot beneath the configured root."""

    snapshot: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
        description="Prepared snapshot identifier; synthetic selects the bundled example.",
    )


def snapshot_error(message: str) -> MetricInputError:
    """Attach source-node and parameter context to acquisition failures."""
    return MetricInputError(
        "reimagina_prepared_snapshot", "parameters.snapshot", message
    )


def snapshot_folder(name: str) -> Path:
    """The snapshot's folder, directly beneath the configured root."""
    root = Path(
        os.environ.get("REIMAGINA_SNAPSHOT_ROOT", str(Path(__file__).parent / "data"))
    ).resolve()
    folder = (root / name).resolve()
    if folder.parent != root:
        raise snapshot_error("Snapshot must be directly beneath the configured root")
    return folder


def read_snapshot(name: str) -> dict[str, gpd.GeoDataFrame]:
    """Read prepared tables without changing source identity or area values."""
    folder = snapshot_folder(name)
    result = {}
    for kind in ("lot", "block"):
        path = folder / f"{kind}.parquet"
        try:
            frame = gpd.read_parquet(path)
        except (OSError, ValueError) as exc:
            raise snapshot_error(
                f"Cannot read {kind} table for snapshot {name!r}: {exc}"
            ) from exc
        if frame.crs is None or frame.geometry.name != "geometry":
            raise snapshot_error(f"{kind}: geometry column with declared CRS required")
        if "id" not in frame or "parent_id" not in frame:
            raise snapshot_error(f"{kind}: id and parent_id columns required")
        if (
            frame.id.isna().any()
            or frame.id.duplicated().any()
            or not frame.id.map(
                lambda value: isinstance(value, str) and bool(value.strip())
            ).all()
        ):
            raise snapshot_error(f"{kind}: IDs must be unique nonempty strings")
        for key in IDENTITY + ([] if kind == "block" else ATTRIBUTES):
            if key not in frame:
                frame[key] = None
        result[f"reimagina_prepared_{kind}s"] = frame[
            IDENTITY + ([] if kind == "block" else ATTRIBUTES) + ["geometry"]
        ].copy()
    return result


@dataset(
    name="reimagina_prepared_snapshot",
    description="Acquire prepared lot and block GeoParquet tables.",
    outputs={
        "reimagina_prepared_lots": source_declaration(),
        "reimagina_prepared_blocks": source_declaration(blocks=True),
    },
)
def prepared_snapshot(parameters: SnapshotParameters) -> dict[str, gpd.GeoDataFrame]:
    """Acquire both files once per graph invocation."""
    return read_snapshot(parameters.snapshot)


def read_snapshot_accessibility(name: str) -> dict[str, pd.DataFrame]:
    """POIs and road network tables of a prepared snapshot."""
    folder = snapshot_folder(name)
    try:
        pois = gpd.read_parquet(folder / "poi.parquet")
        nodes = pd.read_parquet(folder / "network_nodes.parquet")
        edges = pd.read_parquet(folder / "network_edges.parquet")
    except (OSError, ValueError) as exc:
        raise MetricInputError(
            "reimagina_prepared_accessibility",
            "parameters.snapshot",
            f"Cannot read accessibility inputs for snapshot {name!r}: {exc}",
        ) from exc
    return {
        "reimagina_prepared_pois": pois,
        "reimagina_prepared_network_nodes": nodes,
        "reimagina_prepared_network_edges": edges,
    }


@dataset(
    name="reimagina_prepared_accessibility",
    description="Acquire a prepared snapshot's POIs and road network tables.",
    outputs={
        "reimagina_prepared_pois": POIS_DECLARATION,
        "reimagina_prepared_network_nodes": NODES_DECLARATION,
        "reimagina_prepared_network_edges": EDGES_DECLARATION,
    },
)
def prepared_accessibility(parameters: SnapshotParameters) -> dict[str, pd.DataFrame]:
    """Acquire the three files once per graph invocation."""
    return read_snapshot_accessibility(parameters.snapshot)
