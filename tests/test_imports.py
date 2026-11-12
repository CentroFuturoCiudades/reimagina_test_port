"""Every module imports, including the ones the Dockerfile imports by path, so a
module moved without updating its importers fails here and not at deploy."""

import importlib
import pkgutil
import re
from pathlib import Path

import pytest

import reimagina_test_port

ROOT = Path(__file__).resolve().parents[1]
MODULES = sorted(
    info.name
    for info in pkgutil.walk_packages(
        reimagina_test_port.__path__, f"{reimagina_test_port.__name__}."
    )
)


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_dockerfile_imports_exist():
    dockerfile = (ROOT / "Dockerfile").read_text()
    imported = set(re.findall(r"\breimagina_test_port(?:\.\w+)+", dockerfile))
    assert imported
    for name in sorted(imported):
        importlib.import_module(name)
