"""One self-contained HTML page to look at ``street_fit`` on a small map.

    uv run python scripts/street_fit_preview.py                       # tests/data/test_area.gpkg
    uv run python scripts/street_fit_preview.py lazaro_cardenas
    uv run python scripts/street_fit_preview.py path/to/area.gpkg --out /tmp/x.html

The area is a gpkg with layers ``roads`` (``street_fit.ROAD_COLUMNS`` + road_type) and
``lots``, as in ``tests/data``. The page (no network needed) draws the lots, the network's
segments and the fitted rights of way: a ribbon as wide as ``width_m`` and its axis.
Wheel zooms, drag pans, layers toggle, a click on a right of way shows its attributes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import shapely

from reimagina_test_port.ingestion import street_fit

DATA = Path(__file__).parents[1] / "tests/data"


def rings(geometry) -> list[list[list[float]]]:
    """Exterior rings of a (multi)polygon, as lists of [x, y]."""
    parts = getattr(geometry, "geoms", [geometry])
    return [[list(c) for c in p.exterior.coords] for p in parts if not p.is_empty]


def lines(geometry) -> list[list[list[float]]]:
    parts = getattr(geometry, "geoms", [geometry])
    return [[list(c) for c in p.coords] for p in parts if not p.is_empty]


def page(roads: gpd.GeoDataFrame, lots, fitted: gpd.GeoDataFrame, title: str) -> str:
    x0, y0, x1, y1 = roads.total_bounds
    origin = ((x0 + x1) / 2, (y0 + y1) / 2)

    def local(shapes):
        return shapely.transform(shapes, lambda c: c - origin)

    pedestrian = roads["road_type"] == "pedestrian"
    keep = ["key", "name", "road_class", "width_m", "measured_sides", "carriageways", "lanes",
            "median_m", "width_left_m", "width_right_m"]
    rows = []
    for row, geometry in zip(fitted[keep + ["bands.use", "bands.share", "road_keys"]].itertuples(index=False),
                             local(fitted.geometry.to_numpy()), strict=True):
        info = dict(zip(keep, row[: len(keep)], strict=True))
        info["bands"] = ", ".join(f"{u} {s:.0f}%" for u, s in zip(row[len(keep)], row[len(keep) + 1], strict=True))
        info["segments"] = len(row[len(keep) + 2])
        rows.append({"p": lines(geometry), "w": float(row[3]), "dual": row[5] == 2, "i": info})
    data = {
        "lots": [r for g in local(lots.geometry.to_numpy()) for r in rings(g)],
        "roads": [{"p": lines(g), "ped": bool(p), "c": c} for g, p, c in
                  zip(local(roads.geometry.to_numpy()), pedestrian, roads["road_type"], strict=True)],
        "fitted": rows,
        "extent": [x0 - origin[0], y0 - origin[1], x1 - origin[0], y1 - origin[1]],
    }
    return (TEMPLATE.replace("__TITLE__", title)
            .replace("__DATA__", json.dumps(data, default=float, separators=(",", ":"))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("area", nargs="?", default="test_area", help="a name in tests/data, or a .gpkg path")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    path = Path(args.area) if args.area.endswith(".gpkg") else DATA / f"{args.area}.gpkg"
    roads = gpd.read_file(path, layer="roads")
    lots = gpd.read_file(path, layer="lots")
    vehicular = roads[roads["road_type"] != "pedestrian"].reset_index(drop=True)
    fitted = street_fit.fit(vehicular, lots.geometry)
    street_fit.check(fitted)
    out = args.out or Path(f"street_fit_{path.stem}.html")
    out.write_text(page(roads, lots, fitted, path.stem), encoding="utf-8")
    print(f"{out}: {len(roads)} segments -> {len(fitted)} rights of way "
          f"({int((fitted['carriageways'] == 2).sum())} dual), {len(lots)} lots")


TEMPLATE = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>street_fit: __TITLE__</title>
<style>
 html,body{margin:0;height:100%;font:13px system-ui,sans-serif;background:#f4f2ee;color:#222}
 canvas{display:block;width:100vw;height:100vh;cursor:grab}
 #panel{position:fixed;top:10px;left:10px;background:#fffd;padding:8px 10px;border-radius:6px;box-shadow:0 1px 4px #0003}
 #panel label{display:block;margin:2px 0;cursor:pointer}
 #info{position:fixed;top:10px;right:10px;max-width:320px;background:#fffe;padding:8px 10px;border-radius:6px;
  box-shadow:0 1px 4px #0003;display:none;line-height:1.5}
 #info b{display:inline-block;min-width:105px;font-weight:600}
 .k{display:inline-block;width:12px;height:12px;vertical-align:-2px;margin-right:4px;border-radius:2px}
</style></head><body>
<canvas id="c"></canvas>
<div id="panel"><b>__TITLE__</b>
 <label><input type="checkbox" id="lots" checked> lotes</label>
 <label><input type="checkbox" id="roads" checked> red original</label>
 <label><input type="checkbox" id="ribbon" checked> ancho ajustado <span class="k" style="background:#4a90d9aa"></span>calle
  <span class="k" style="background:#e8833aaa"></span>con camellón</label>
 <label><input type="checkbox" id="axis" checked> eje ajustado</label>
 <span style="color:#666">rueda: zoom · arrastrar: mover · clic: atributos</span></div>
<div id="info"></div>
<script>
const D=__DATA__;
const cv=document.getElementById('c'),ctx=cv.getContext('2d'),info=document.getElementById('info');
let s=1,tx=0,ty=0;
const dpr=window.devicePixelRatio||1;
function fit(){cv.width=innerWidth*dpr;cv.height=innerHeight*dpr;
 const [a,b,c,d]=D.extent;s=Math.min(innerWidth/(c-a),innerHeight/(d-b))*.9;tx=innerWidth/2;ty=innerHeight/2;draw()}
const X=x=>tx+x*s,Y=y=>ty-y*s;
function path(p,close){ctx.beginPath();for(const l of p){l.forEach((q,i)=>i?ctx.lineTo(X(q[0]),Y(q[1])):ctx.moveTo(X(q[0]),Y(q[1])));if(close)ctx.closePath()}}
const COLORS={trunk:'#b33',primary:'#c60',secondary:'#c80',tertiary:'#777',residential:'#999',service:'#bbb',pedestrian:'#7b7'};
let picked=null;
function draw(){
 ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,innerWidth,innerHeight);ctx.lineJoin='round';ctx.lineCap='butt';
 if(lots.checked){ctx.fillStyle='#e3dfd5';ctx.strokeStyle='#cfc9ba';ctx.lineWidth=.5;
  for(const r of D.lots){path([r],true);ctx.fill();ctx.stroke()}}
 if(ribbon.checked)for(const f of D.fitted){path(f.p);ctx.strokeStyle=f.dual?'#e8833a88':'#4a90d988';
  ctx.lineWidth=Math.max(f.w*s,1);ctx.stroke()}
 if(roads.checked)for(const r of D.roads){path(r.p);ctx.strokeStyle=COLORS[r.c]||'#999';
  ctx.lineWidth=1;ctx.setLineDash(r.ped?[3,3]:[]);ctx.stroke()}
 ctx.setLineDash([]);
 if(axis.checked)for(const f of D.fitted){path(f.p);ctx.strokeStyle=f===picked?'#d00':'#123';ctx.lineWidth=f===picked?3:1.4;ctx.stroke()}
 if(picked){path(picked.p);ctx.strokeStyle='#d004';ctx.lineWidth=picked.w*s;ctx.stroke()}
}
function dist(p,x,y){let m=1e9;for(const l of p)for(let i=1;i<l.length;i++){
 const[ax,ay]=l[i-1],[bx,by]=l[i],dx=bx-ax,dy=by-ay,t=Math.max(0,Math.min(1,((x-ax)*dx+(y-ay)*dy)/(dx*dx+dy*dy||1)));
 m=Math.min(m,Math.hypot(x-ax-t*dx,y-ay-t*dy))}return m}
let drag=null,moved=false;
cv.onmousedown=e=>{drag=[e.clientX,e.clientY,tx,ty];moved=false;cv.style.cursor='grabbing'};
addEventListener('mouseup',e=>{cv.style.cursor='grab';if(drag&&!moved)pick(e.clientX,e.clientY);drag=null});
addEventListener('mousemove',e=>{if(!drag)return;if(Math.abs(e.clientX-drag[0])+Math.abs(e.clientY-drag[1])>3)moved=true;
 tx=drag[2]+e.clientX-drag[0];ty=drag[3]+e.clientY-drag[1];draw()});
cv.onwheel=e=>{e.preventDefault();const k=Math.exp(-e.deltaY*.0015),x=(e.clientX-tx)/s,y=(ty-e.clientY)/s;
 s*=k;tx=e.clientX-x*s;ty=e.clientY+y*s;draw()};
function pick(px,py){const x=(px-tx)/s,y=(ty-py)/s;let best=null,m=1e9;
 for(const f of D.fitted){const d=dist(f.p,x,y);if(d<m){m=d;best=f}}
 picked=m*s<20?best:null;draw();
 if(!picked){info.style.display='none';return}
 const i=picked.i;info.style.display='block';
 info.innerHTML=Object.entries(i).map(([k,v])=>`<b>${k}</b>${typeof v==='number'?+v.toFixed(2):v??'—'}`).join('<br>')}
for(const el of document.querySelectorAll('#panel input'))el.onchange=draw;
addEventListener('resize',fit);fit();
</script></body></html>
"""

if __name__ == "__main__":
    main()
