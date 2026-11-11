"""Access to the CFC data lake (``az://container/path`` URIs).

From reimagina-lyra ``ingestion/lake.py``. Authenticates with
``DefaultAzureCredential``: the worker's managed identity, or ``az login``
locally. Local paths are accepted too. Everything reads, except ``upload`` (the
street network's H5), which never overwrites a file.
"""

import shutil
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd

ACCOUNT_NAME = "cfcetlsadls"


@cache
def storage_options() -> dict[str, Any]:
    from azure.identity import DefaultAzureCredential

    return {"account_name": ACCOUNT_NAME, "credential": DefaultAzureCredential()}


def filesystem():
    import fsspec

    return fsspec.filesystem("az", **storage_options())


def download(uri: str, directory: Path) -> Path:
    """Copy an ``az://`` or local file into ``directory`` unless it is already there."""
    target = directory / uri.rsplit("/", 1)[-1]
    if not target.exists():
        directory.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f".{target.name}.part")
        if uri.startswith("az://"):
            filesystem().get(uri, str(partial))
        else:
            shutil.copyfile(uri, partial)
        partial.replace(target)
    return target


def download_shapefile(uri: str, directory: Path) -> Path:
    """Copy a ``.shp`` and its sidecar files (same stem) into ``directory``."""
    stem = uri.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    parent = uri.rsplit("/", 1)[0]
    if uri.startswith("az://"):
        names = [f"az://{p}" for p in filesystem().ls(parent, detail=False)]
    else:
        names = [str(p) for p in Path(parent).iterdir()]
    for name in names:
        if name.rsplit("/", 1)[-1].rsplit(".", 1)[0] == stem:
            download(name, directory)
    return directory / f"{stem}.shp"


def list_names(uri: str) -> list[str]:
    if uri.startswith("az://"):
        return [p.rsplit("/", 1)[-1] for p in filesystem().ls(uri, detail=False)]
    return sorted(p.name for p in Path(uri).iterdir())


def upload(path: Path, uri: str) -> str:
    """Copy a local file to ``uri`` (``az://`` or a local path); an existing file
    there fails with ``FileExistsError``."""
    if uri.startswith("az://"):
        fs = filesystem()
        if fs.exists(uri):
            raise FileExistsError(uri)
        fs.put_file(str(path), uri)
    else:
        target = Path(uri)
        if target.exists():
            raise FileExistsError(uri)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return uri


def read_parquet(uri: str, columns: list[str]) -> pd.DataFrame:
    """A (partitioned) parquet dataset from the lake or a local path."""
    options = storage_options() if uri.startswith("az://") else None
    return pd.read_parquet(uri, columns=columns, storage_options=options)
