"""Write the synthetic snapshot's POIs and road network (a 50 m grid).

Invented, like the rest of the synthetic snapshot: three blocks in EPSG:32611 and a
few amenities around them. Run only when deliberately changing the fixture.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

FOLDER = Path(__file__).resolve().parents[1] / "src/reimagina_test_port/data/synthetic"
CRS = "EPSG:32611"
X0, Y0, STEP, COLUMNS, ROWS = 639_900, 3_609_900, 50, 13, 9

ids = np.arange(COLUMNS * ROWS)
nodes = pd.DataFrame(
    {"id": ids, "x": X0 + (ids % COLUMNS) * STEP, "y": Y0 + (ids // COLUMNS) * STEP}
).astype({"x": "float64", "y": "float64"})
right = ids[(ids % COLUMNS) < COLUMNS - 1]
up = ids[ids < COLUMNS * (ROWS - 1)]
edges = pd.DataFrame(
    {
        "from": np.concatenate([right, up]),
        "to": np.concatenate([right + 1, up + COLUMNS]),
        "distance": float(STEP),
        "two_way": True,
    }
)
# Small parcels, as elements are polygons; one unknown type is ignored.
pois = gpd.GeoDataFrame(
    {
        "id": ["poi-park", "poi-clinic", "poi-school", "poi-pharmacy", "poi-unknown"],
        "amenity_type": ["park", "clinic", "primary_school", "pharmacy", "casino"],
        "capacity": [None, 2.0, None, None, None],
    },
    geometry=[
        box(x, y, x + 10, y + 10)
        for x, y in [
            (639_950, 3_609_950),
            (640_300, 3_610_150),
            (640_120, 3_610_330),
            (640_450, 3_610_000),
            (640_200, 3_610_200),
        ]
    ],
    crs=CRS,
)
nodes.to_parquet(FOLDER / "network_nodes.parquet", index=False)
edges.to_parquet(FOLDER / "network_edges.parquet", index=False)
pois.to_parquet(FOLDER / "poi.parquet", index=False)
