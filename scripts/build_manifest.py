"""Generate and stage the same manifest for discovery and distribution."""

import shutil
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
subprocess.run(["uv", "run", "lyra-plugin", "build-manifest"], cwd=root, check=True)
destination = root / "data/share/lyra/plugins/reimagina-test-port/lyra.plugin.json"
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(root / "lyra.plugin.json", destination)
