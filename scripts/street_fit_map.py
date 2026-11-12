"""``street_fit`` on a real map: how each right of way is shared out, over a whole city if you like.

    uv run python scripts/street_fit_map.py                          # tests/data/test_area.gpkg
    uv run python scripts/street_fit_map.py lazaro_cardenas
    uv run python scripts/street_fit_map.py path/to/city.gpkg --out city.html

Adapted from reimagina_pipelines_lab's ``experiments/junctions/idea 3/city.py``. The area is a gpkg with
layers ``roads`` and ``lots`` (see ``docs/street_fit_test_data.md``), in a metric CRS.

``street_fit.fit`` returns an axis and its bands (``bands.use`` / ``bands.share``), not polygons, so each
band is drawn here as a strip across the right of way: sidewalks, lanes and median, from lot front to lot
front, centred on the axis. Junctions aren't built, so rights of way just meet at the nodes.

The page is deck.gl on MapLibre (streets or satellite under the shapes; it needs the network for the
libraries and tiles). Click a right of way for its attributes (Esc closes); the side list has the widest
ones and the "color by" menu paints them by class, width, lanes or median. Shapes are simplified to 10 cm
(lots to 25 cm) and stored as integer micro-degrees from the centre.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from pyproj import Transformer

from reimagina_test_port.ingestion import street_fit

DATA = Path(__file__).parents[1] / "tests/data"
KINDS = ("sidewalk", "lanes", "median")


def value(v):
    if isinstance(v, (list, tuple, set)):
        return ", ".join(map(str, sorted(v) if isinstance(v, set) else v))
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(v) else round(float(v), 2)
    if isinstance(v, np.integer):
        return int(v)
    return v


def strip(axis, lo: float, hi: float):
    """The part of the right of way between offsets ``lo`` < ``hi`` from its axis (left is positive)."""

    def side(d: float):
        return shapely.buffer(
            axis,
            d,
            single_sided=True,
            cap_style="flat",
            join_style="mitre",
            mitre_limit=2,
        )

    if lo >= 0:
        shape = side(hi).difference(side(lo)) if lo > 0 else side(hi)
    elif hi <= 0:
        shape = side(lo).difference(side(hi)) if hi < 0 else side(lo)
    else:
        shape = side(hi).union(side(lo))
    return shape


def band_shapes(fitted: gpd.GeoDataFrame) -> dict[str, list[tuple[int, object]]]:
    """Per band kind, (row index, strip) for every band of every right of way."""
    out: dict[str, list] = {k: [] for k in KINDS}
    for i, (axis, width, uses, shares) in enumerate(
        zip(
            fitted.geometry,
            fitted["width_m"],
            fitted["bands.use"],
            fitted["bands.share"],
            strict=True,
        )
    ):
        edge = width / 2
        for use, share in zip(uses, shares, strict=True):
            lo, hi = edge - width * share / 100, edge
            edge = lo
            if share > 0:
                shape = strip(axis, lo, hi)
                if not shape.is_empty:
                    out[use].append((i, shape))
    return out


class Pack:
    """Shapes → rings of integer micro-degrees from an origin, all of a layer's points projected at once."""

    def __init__(self, crs, origin: tuple[float, float]):
        self.to_lonlat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        self.ox, self.oy = self.to_lonlat.transform(*origin)

    def flat(self, xy: np.ndarray) -> np.ndarray:
        lon, lat = self.to_lonlat.transform(xy[:, 0], xy[:, 1])
        return np.column_stack(
            [np.round((lon - self.ox) * 1e6), np.round((lat - self.oy) * 1e6)]
        ).astype(int)

    def polygons(self, shapes, tolerance: float, owners_of=None) -> list[list]:
        """[[i, [ring, …]]] for every polygon of every shape (outer ring, then holes); a ring is [x, y, x, y, …]."""
        rings, owners = [], []
        for n, g in enumerate(shapes):
            i = owners_of[n] if owners_of is not None else n
            for p in shapely.get_parts(shapely.simplify(g, tolerance)):
                if p.geom_type == "Polygon" and not p.is_empty:
                    owners.append((i, len(p.interiors) + 1))
                    rings += [
                        shapely.get_coordinates(r) for r in (p.exterior, *p.interiors)
                    ]
        if not rings:
            return []
        xy = self.flat(np.concatenate(rings))
        cuts = np.cumsum([len(r) for r in rings])[:-1]
        flat = [r.ravel().tolist() for r in np.split(xy, cuts)]
        out, k = [], 0
        for i, n in owners:
            out.append([i, flat[k : k + n]])
            k += n
        return out

    def lines(self, shapes) -> list[list]:
        parts = [shapely.get_coordinates(g) for g in shapes if not g.is_empty]
        if not parts:
            return []
        xy = self.flat(np.concatenate(parts))
        return [
            r.ravel().tolist()
            for r in np.split(xy, np.cumsum([len(p) for p in parts])[:-1])
        ]


def page(
    roads: gpd.GeoDataFrame, lots, fitted: gpd.GeoDataFrame, title: str, seconds: float
) -> str:
    x0, y0, x1, y1 = roads.total_bounds
    pack = Pack(roads.crs, ((x0 + x1) / 2, (y0 + y1) / 2))
    columns = [c for c in fitted.columns if c not in ("geometry", "road_keys")]
    rows = [
        [value(v) for v in r] + [round(g.length, 1)]
        for r, g in zip(
            fitted[columns].itertuples(index=False), fitted.geometry, strict=True
        )
    ]
    bands = band_shapes(fitted)
    widest = (
        fitted.assign(i=range(len(fitted)))
        .sort_values("width_m", ascending=False)
        .head(25)
    )
    by_class = fitted.groupby("road_class")["width_m"].agg(["count", "median"])
    report = {
        "segments": len(roads),
        "rights of way": len(fitted),
        "dual (2 carriageways)": int((fitted["carriageways"] == 2).sum()),
        "both sides measured": int((fitted["measured_sides"] == 2).sum()),
        "median width m": round(float(fitted["width_m"].median()), 1),
        "lots": len(lots),
        "street_fit seconds": round(seconds),
        **{
            f"{k}: n · median m": f"{int(r['count'])} · {r['median']:.1f}"
            for k, r in by_class.iterrows()
        },
    }
    payload = {
        "ox": pack.ox,
        "oy": pack.oy,
        "title": title,
        "cols": [*columns, "length_m"],
        "rows": rows,
        "bands": {
            k: pack.polygons([s for _, s in v], 0.1, [i for i, _ in v])
            for k, v in bands.items()
        },
        "lots": pack.polygons(shapely.make_valid(lots.geometry.to_numpy()), 0.25),
        "axes": pack.lines(fitted.geometry.to_numpy()),
        "axes_of": list(range(len(fitted))),
        "network": pack.lines(roads.geometry.to_numpy()),
        "widest": [
            [
                int(r.i),
                r.width_m,
                *pack.flat(
                    np.array([r.geometry.interpolate(0.5, normalized=True).coords[0]])
                )[0].tolist(),
            ]
            for r in widest.itertuples()
        ],
        "report": report,
    }
    return PAGE.replace("__TITLE__", title).replace(
        "__DATA__", json.dumps(payload, separators=(",", ":"), default=value)
    )


PAGE = """<!doctype html><html lang="es"><head><meta charset="utf-8"><title>street_fit · __TITLE__</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<script src="https://unpkg.com/deck.gl@9.1.14/dist.min.js"></script>
<style>
:root { --bg:#f5f4f0; --ink:#27272a; --muted:#6b7280; --line:#dfe1e6; }
html, body { margin:0; height:100%; background:var(--bg); color:var(--ink); font:13px system-ui, sans-serif; }
#map { position:absolute; inset:0 300px 0 0; }
aside { position:absolute; top:0; right:0; bottom:0; width:300px; overflow:auto; background:#fff; border-left:1px solid var(--line); box-sizing:border-box; padding:12px 14px; }
h1 { font-size:15px; margin:0 0 8px; } h2 { font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); margin:16px 0 6px; }
label { display:block; margin:4px 0; } input[type=range], select { width:100%; }
table { border-collapse:collapse; width:100%; } td { padding:2px 4px; border-bottom:1px solid #f0f0f0; vertical-align:top; }
td:first-child { color:var(--muted); white-space:nowrap; } td:last-child { word-break:break-all; }
#widest tr { cursor:pointer; } #widest tr:hover { background:#fef3c7; }
#card { font:12px/1.45 ui-monospace, monospace; white-space:pre-wrap; background:#fafaf8; border:1px solid var(--line); border-radius:6px; padding:8px; display:none; }
.hint { color:var(--muted); } .sw { display:inline-block; width:10px; height:10px; margin-right:4px; border:1px solid #0002; vertical-align:-1px; }
@media (max-width:700px) { #map { inset:0 0 45% 0; } aside { top:55%; width:100%; border-left:0; border-top:1px solid var(--line); } }
</style></head><body>
<div id="map"></div>
<aside>
<h1>street_fit · __TITLE__</h1>
<div class="hint">click a right of way for its attributes (Esc closes)</div>
<h2>Selected</h2><div id="card"></div><div id="none" class="hint">nothing selected</div>
<h2>Show</h2>
<label><input type="checkbox" id="sat"> satellite</label>
<label><input type="checkbox" id="lt" checked> lots</label>
<label><input type="checkbox" id="ax"> fitted axes</label>
<label><input type="checkbox" id="nw"> network segments</label>
<label>color by <select id="by"><option value="bands">bands (sidewalk · lanes · median)</option><option value="road_class">road class</option><option value="width_m">width</option><option value="lanes">lanes</option><option value="measured_sides">measured sides</option></select></label>
<label>opacity <input type="range" id="op" min="0" max="1" step="0.05" value="1"></label>
<div id="legend" class="hint"></div>
<h2>Widest rights of way</h2><table id="widest"></table>
<h2>Run</h2><table id="report"></table>
</aside>
<script>
const D = __DATA__;
const dec = f => { const out = []; for (let k = 0; k < f.length; k += 2) out.push([D.ox + f[k] / 1e6, D.oy + f[k + 1] / 1e6]); return out; };
const polys = list => list.map(([i, rings]) => ({i, polygon: rings.map(dec)}));
const bands = Object.fromEntries(Object.keys(D.bands).map(k => [k, polys(D.bands[k])]));
const lots = polys(D.lots), axes = D.axes.map(dec), network = D.network.map(dec);
const col = Object.fromEntries(D.cols.map((c, k) => [c, k]));
const st = {sat: false, lots: true, axes: false, net: false, op: 1, by: "bands", sel: null};
const fmt = v => v === null || v === undefined ? "—" : typeof v === "number" ? v.toLocaleString("en", {maximumFractionDigits: 2}) : String(v);
const BANDS = {sidewalk: [214, 211, 205], lanes: [63, 63, 70], median: [101, 163, 13]};
const CLASS = {trunk: [185, 28, 28], primary: [234, 88, 12], secondary: [202, 138, 4], tertiary: [101, 163, 13], residential: [37, 99, 235], service: [124, 58, 237], pedestrian: [107, 114, 128]};
const RAMP = [[254, 243, 199], [252, 211, 77], [245, 158, 11], [217, 70, 11], [153, 27, 27]];
function ramp(t) { const x = Math.max(0, Math.min(1, t)) * (RAMP.length - 1), k = Math.min(Math.floor(x), RAMP.length - 2), f = x - k;
  return RAMP[k].map((c, j) => Math.round(c + (RAMP[k + 1][j] - c) * f)); }
const range = c => { const v = D.rows.map(r => r[col[c]]).filter(v => typeof v === "number"); return [Math.min(...v), Math.max(...v)]; };
function colorOf(i) {
  const v = D.rows[i][col[st.by]];
  if (st.by === "road_class") return CLASS[v] || [107, 114, 128];
  if (typeof v !== "number") return [200, 200, 200];
  const [a, b] = range(st.by); return ramp(b > a ? (v - a) / (b - a) : 0);
}
function legend() {
  let h = "";
  if (st.by === "bands") h = Object.entries(BANDS).map(([k, c]) => `<span class="sw" style="background:rgb(${c})"></span>${k} `).join("") + `<span class="sw" style="background:rgb(124,155,230)"></span>lot`;
  else if (st.by === "road_class") h = Object.entries(CLASS).map(([k, c]) => `<span class="sw" style="background:rgb(${c})"></span>${k} `).join("");
  else { const [a, b] = range(st.by); h = `<span class="sw" style="background:rgb(${ramp(0)})"></span>${fmt(a)} → <span class="sw" style="background:rgb(${ramp(1)})"></span>${fmt(b)}`; }
  document.getElementById("legend").innerHTML = h;
}
function props(i) { return ["right of way", ...D.cols.map((c, k) => `${c}: ${fmt(D.rows[i][k])}`)].join("\\n"); }
function select(i) {
  st.sel = i;
  card.style.display = i === null ? "none" : "block"; none.style.display = i === null ? "block" : "none";
  if (i !== null) card.textContent = props(i);
  draw();
}
const solid = (id, data, color, extra = {}) => new deck.PolygonLayer({id, data, getPolygon: d => d.polygon, filled: true, stroked: false,
  getFillColor: color, opacity: st.op, updateTriggers: {getFillColor: st.by}, ...extra});
function layers() {
  const pick = {pickable: true};
  const out = [];
  if (st.lots) out.push(solid("lots", lots, [124, 155, 230], {opacity: st.sat ? 0.35 : 0.9}));
  if (st.by === "bands") for (const k of ["sidewalk", "median", "lanes"]) out.push(solid(k, bands[k], BANDS[k], pick));
  else for (const k of ["sidewalk", "median", "lanes"]) out.push(solid(k, bands[k], d => colorOf(d.i), pick));
  if (st.net) out.push(new deck.PathLayer({id: "network", data: network, getPath: d => d, getColor: [239, 68, 68], widthUnits: "pixels", getWidth: 1}));
  if (st.axes) out.push(new deck.PathLayer({id: "axes", data: axes, getPath: d => d, getColor: [255, 255, 255], widthUnits: "pixels", getWidth: 1.5}));
  if (st.sel !== null) out.push(new deck.PolygonLayer({id: "selected", data: Object.values(bands).flat().filter(d => d.i === st.sel), getPolygon: d => d.polygon,
    filled: false, stroked: true, getLineColor: [202, 138, 4], lineWidthUnits: "pixels", getLineWidth: 3}));
  return out;
}
const getTooltip = ({object}) => object && object.i !== undefined ? {text: `${D.rows[object.i][col.name] || "(no name)"} · ${D.rows[object.i][col.road_class]} · ${fmt(D.rows[object.i][col.width_m])} m`} : null;

const POSITRON = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";
const SATELLITE = {version: 8, sources: {sat: {type: "raster", tileSize: 256, maxzoom: 19,
  tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
  attribution: "Imagery © Esri, Maxar, Earthstar Geographics"}}, layers: [{id: "sat", type: "raster", source: "sat"}]};
const map = new maplibregl.Map({container: "map", style: POSITRON, center: [D.ox, D.oy], zoom: 15, attributionControl: {compact: true}});
map.addControl(new maplibregl.NavigationControl(), "top-right");
const overlay = new deck.MapboxOverlay({interleaved: false, layers: [], getTooltip,
  onClick: info => select(info.object && info.object.i !== undefined && info.layer.id !== "lots" ? info.object.i : null)});
map.addControl(overlay);
function draw() { overlay.setProps({layers: layers()}); }

sat.onchange = () => { st.sat = sat.checked; map.setStyle(st.sat ? SATELLITE : POSITRON); draw(); };
lt.onchange = () => { st.lots = lt.checked; draw(); };
ax.onchange = () => { st.axes = ax.checked; draw(); };
nw.onchange = () => { st.net = nw.checked; draw(); };
by.onchange = () => { st.by = by.value; legend(); draw(); };
op.oninput = () => { st.op = +op.value; draw(); };
addEventListener("keydown", e => e.key === "Escape" && select(null));
document.getElementById("widest").innerHTML = D.widest.map(([i, w], k) => `<tr data-k="${k}"><td>${k + 1}</td><td>${fmt(w)} m · ${D.rows[i][col.name] || "(no name)"}</td></tr>`).join("");
document.querySelectorAll("#widest tr").forEach(tr => tr.onclick = () => {
  const [i, , x, y] = D.widest[+tr.dataset.k];
  map.flyTo({center: [D.ox + x / 1e6, D.oy + y / 1e6], zoom: 18, duration: 800});
  select(i);
});
document.getElementById("report").innerHTML = Object.entries(D.report).map(([k, v]) => `<tr><td>${k}</td><td>${fmt(v)}</td></tr>`).join("");
legend(); draw();
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "area",
        nargs="?",
        default="test_area",
        help="a name in tests/data, or a .gpkg path",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    path = Path(args.area)
    if path.suffix != ".gpkg":
        path = DATA / f"{args.area}.gpkg"
    roads = gpd.read_file(path, layer="roads")
    lots = gpd.read_file(path, layer="lots")
    for column in ("road_key", "from_node", "to_node", "road_type", "name"):
        roads[column] = roads[column].astype(object)
    # As scripts/street_fit_preview.py: footways aren't part of the vehicular (OSM drive) network.
    roads = roads[roads["road_type"] != "pedestrian"].reset_index(drop=True)
    start = time.time()
    fitted = street_fit.fit(roads, lots.geometry)
    seconds = time.time() - start
    out = args.out or Path(f"street_fit_{path.stem}_map.html")
    out.write_text(page(roads, lots, fitted, path.stem, seconds), encoding="utf-8")
    print(
        f"{out}: {len(roads)} segments -> {len(fitted)} rights of way in {seconds:.0f} s, {len(lots)} lots, {out.stat().st_size / 1e6:.1f} MB"
    )


if __name__ == "__main__":
    main()
