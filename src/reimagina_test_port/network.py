"""Road network H5 (pandana) as node and edge tables.

``projects.road_network_uri`` is an ``az://container/path`` in the CFC data lake or
a local path. Lake access uses ``DefaultAzureCredential``: the worker's managed
identity, or ``az login`` locally.
"""

import shutil
from pathlib import Path

import pandas as pd
from pandana.loaders.pandash5 import open_hdf_store

ACCOUNT_NAME = "cfcetlsadls"


def download(uri: str, directory: Path) -> Path:
    """Copy an ``az://`` or local file into ``directory``."""
    target = directory / uri.rsplit("/", 1)[-1]
    directory.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.part")
    if uri.startswith("az://"):
        import fsspec
        from azure.identity import DefaultAzureCredential

        filesystem = fsspec.filesystem(
            "az", account_name=ACCOUNT_NAME, credential=DefaultAzureCredential()
        )
        filesystem.get(uri, str(partial))
    else:
        shutil.copyfile(uri, partial)
    partial.replace(target)
    return target


def read_network(path: Path) -> dict[str, pd.DataFrame]:
    """Nodes (id, x, y) and edges (from, to, distance, two_way) of a pandana H5.

    ``distance`` is the file's first impedance, the one pandana queries by default.
    """
    with open_hdf_store(str(path), migrate_legacy=True) as store:
        nodes = store["nodes"]
        edges = store["edges"]
        two_way = bool(store["two_way"].iloc[0])
        impedance = store["impedance_names"].tolist()[0]
    return {
        "nodes": pd.DataFrame(
            {
                "id": nodes.index.astype("int64"),
                "x": nodes["x"].astype("float64").to_numpy(),
                "y": nodes["y"].astype("float64").to_numpy(),
            }
        ),
        "edges": pd.DataFrame(
            {
                "from": edges["from"].astype("int64").to_numpy(),
                "to": edges["to"].astype("int64").to_numpy(),
                "distance": edges[impedance].astype("float64").to_numpy(),
                "two_way": two_way,
            }
        ),
    }
