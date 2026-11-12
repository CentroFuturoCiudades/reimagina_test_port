"""A city's street network, built from scratch, as the pandana H5 the indicators read.

From exe_reimagina: ``src/data_prep/osm_network.py`` (OSM to pandana) and
``scripts/escenarios/condesa/build_network.py`` (a redesigned street layout sewn
into the city's network). Two steps:

1. ``osm_network`` (always): OpenStreetMap's streets over the convex hull of the
   city's blocks plus a margin, within the municipality, with the city's ``custom_filter`` (Mexicali: the
   ULI's pedestrian definition, which keeps private and gated streets), in the
   city's metric CRS.
2. ``merge_streets`` (optional): a file of street lines replaces the network
   inside its zone (the lines' convex hull, or a polygon given apart). Network
   edges whose midpoint lies in the zone are removed; the new lines are noded
   where they cross, and their ends join the network: on a network node within
   ``snap_m``, else by splitting the nearest network edge within ``split_max_m``.
   Pieces of the new streets still apart are bridged to the rest when the gap is
   at most ``connector_max_m``.

``save_h5`` writes the result with pandana; ``indicators/accessibility/network.py``
reads it back. The H5 stores straight edges between nodes with their length in
metres (``length``, the first impedance) and is two-way.

Edges also carry their shape (``geometry``), for the streets built over the network
(``ingestion/streets.py``): OSM's way, or the new street's line. Edges that step 2
splits or adds as connectors have none (``shapes`` gives them the straight line).

The OSM download is cached in the lake beside the H5s (``osm_cache_uris``), so a
city's network is fetched from Overpass once.
"""

import hashlib
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from reimagina_test_port.ingestion.cities import NetworkParams

logger = logging.getLogger(__name__)

OSM_CRS = "EPSG:4326"
# Ids of new nodes start above OSM's (which already reach 13 billion) and above
# the network's own maximum.
NODE_ID_FLOOR = 9_000_000_000


# ----------------------------------------------------------------------- step 1


def study_area(
    blocks: gpd.GeoSeries,
    crs: str,
    margin_m: float,
    boundary: gpd.GeoSeries | None = None,
) -> shapely.Geometry:
    """The blocks' convex hull plus ``margin_m``, within ``boundary`` when given,
    in EPSG:4326, to fetch OSM over.

    The hull, not the blocks alone: it holds the roads between the city and its
    outlying localities, which would otherwise be pieces apart from the rest
    (exe_reimagina's ``roads.py`` fetched the blocks' bounding box). The
    boundary, the municipality, keeps out what lies across it: Calexico's
    streets, which OSM joins to Mexicali's at the border crossings (the ULI's
    issue #610; exe_reimagina fetched the municipality).
    """
    metric = blocks.to_crs(crs)
    hull = shapely.convex_hull(
        shapely.multipoints(shapely.get_coordinates(metric.to_numpy()))
    )
    area = shapely.buffer(hull, margin_m)
    if boundary is not None:
        area = shapely.intersection(
            area, shapely.union_all(boundary.to_crs(crs).to_numpy())
        )
    return gpd.GeoSeries([area], crs=crs).to_crs(OSM_CRS).iloc[0]


def osm_graph(area: shapely.Geometry, params: NetworkParams):
    """OpenStreetMap's ways in ``area`` (EPSG:4326) as an osmnx graph.

    ``network_type="walk"`` keeps every way two-way; ``custom_filter`` replaces
    the walk filter when given.
    """
    import osmnx as ox  # only where the network is built

    return ox.graph_from_polygon(
        area,
        network_type="walk",
        retain_all=params.retain_all,
        custom_filter=params.custom_filter,
    )


def osm_tables(graph) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nodes (``id`` index, ``x``, ``y`` in EPSG:4326) and edges (``from``, ``to``,
    ``length`` in metres, ``geometry``: the way's shape, EPSG:4326) of an osmnx
    graph; parallel edges keep the shortest."""
    import osmnx as ox

    nodes, edges = ox.graph_to_gdfs(graph)
    nodes = nodes.to_crs(OSM_CRS)
    nodes = pd.DataFrame(
        {"x": nodes.geometry.x, "y": nodes.geometry.y},
        index=pd.Index(nodes.index.astype("int64"), name="id"),
    )
    edges = edges.to_crs(OSM_CRS).reset_index()
    edges = (
        pd.DataFrame(
            {
                "from": edges["u"].astype("int64"),
                "to": edges["v"].astype("int64"),
                "length": edges["length"].astype("float64"),
                "geometry": edges.geometry.to_numpy(),
            }
        )
        .query("`from` != `to`")
        .sort_values("length")
        .drop_duplicates(["from", "to"])
        .sort_index()
        .reset_index(drop=True)
    )
    return nodes, edges


def project(
    nodes: pd.DataFrame, edges: pd.DataFrame, crs: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``osm_tables``' nodes and edge shapes from EPSG:4326 into ``crs``."""
    points = gpd.GeoSeries.from_xy(nodes["x"], nodes["y"], crs=OSM_CRS).to_crs(crs)
    nodes = pd.DataFrame(
        {"x": points.x.to_numpy(), "y": points.y.to_numpy()}, index=nodes.index
    )
    shapes = gpd.GeoSeries(edges["geometry"].to_numpy(), crs=OSM_CRS).to_crs(crs)
    return nodes, edges.assign(geometry=shapes.to_numpy())


def graph_tables(graph, crs: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``osm_tables`` in ``crs``."""
    return project(*osm_tables(graph), crs)


def osm_cache_uris(
    prefix: str, cve_code: str, area: shapely.Geometry, params: NetworkParams
) -> tuple[str, str]:
    """Nodes' and edges' parquet of the OSM download for ``area`` and the city's
    filter, under ``prefix``: ``osm_<cve>_<hash>_{nodes,edges}.parquet``. Another
    area, filter or ``retain_all`` is another file."""
    digest = hashlib.sha256()
    for part in (
        shapely.to_wkb(area),
        str(params.custom_filter).encode(),
        str(params.retain_all).encode(),
    ):
        digest.update(part)
    stem = f"{prefix.rstrip('/')}/osm_{cve_code}_{digest.hexdigest()[:10]}"
    return f"{stem}_nodes.parquet", f"{stem}_edges.parquet"


def read_osm_cache(uris: tuple[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``osm_tables`` back from ``write_osm_cache``'s files."""
    from reimagina_test_port.common import lake

    nodes = lake.read_parquet(uris[0], ["id", "x", "y"]).set_index("id")
    edges = lake.read_parquet(uris[1], ["from", "to", "length", "geometry"])
    return nodes, edges.assign(geometry=shapely.from_wkb(edges["geometry"]))


def write_osm_cache(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    uris: tuple[str, str],
    workdir: Path,
    *,
    overwrite: bool,
) -> None:
    """``osm_tables``' nodes and edges (shapes as WKB) to ``uris``."""
    from reimagina_test_port.common import lake

    workdir.mkdir(parents=True, exist_ok=True)
    tables = (
        nodes.reset_index(),
        edges.assign(geometry=shapely.to_wkb(edges["geometry"].to_numpy())),
    )
    for table, uri in zip(tables, uris, strict=True):
        path = workdir / uri.rsplit("/", 1)[-1]
        table.to_parquet(path, index=False)
        lake.upload(path, uri, overwrite=overwrite)


def osm_network(
    blocks: gpd.GeoSeries,
    crs: str,
    params: NetworkParams,
    boundary: gpd.GeoSeries | None = None,
    *,
    cve_code: str | None = None,
    workdir: Path | None = None,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 1: the city's network from OpenStreetMap.

    With ``cve_code``, the download is cached under ``params.output_prefix``
    (``osm_cache_uris``): read from there when it exists, else saved there (written
    first in ``workdir``). ``refresh`` downloads again and replaces it.
    """
    from reimagina_test_port.common import lake

    area = study_area(blocks, crs, params.margin_m, boundary)
    cache = (
        None
        if cve_code is None
        else osm_cache_uris(params.output_prefix, cve_code, area, params)
    )
    if cache is not None and not refresh and all(map(lake.exists, cache)):
        logger.info("OSM: from %s", cache[1])
        nodes, edges = read_osm_cache(cache)
    else:
        nodes, edges = osm_tables(osm_graph(area, params))
        if cache is not None:
            write_osm_cache(
                nodes, edges, cache, workdir or Path("."), overwrite=refresh
            )
            logger.info("OSM: saved to %s", cache[1])
    logger.info("OSM: %s nodes / %s edges", f"{len(nodes):,}", f"{len(edges):,}")
    return project(nodes, edges, crs)


# ----------------------------------------------------------------------- step 2


def _segments(edges: pd.DataFrame, nodes: pd.DataFrame) -> np.ndarray:
    """Each edge as the straight line between its nodes (the H5 keeps no shape)."""
    xy = nodes[["x", "y"]]
    a = xy.reindex(edges["from"].to_numpy()).to_numpy()
    b = xy.reindex(edges["to"].to_numpy()).to_numpy()
    return shapely.linestrings(
        np.stack([a, b], axis=1).reshape(-1, 2),
        indices=np.repeat(np.arange(len(edges)), 2),
    )


def _components(nodes: pd.DataFrame, edges: pd.DataFrame) -> np.ndarray:
    """Connected component of each node."""
    position = pd.Series(np.arange(len(nodes)), index=nodes.index)
    graph = coo_matrix(
        (
            np.ones(len(edges)),
            (
                position.reindex(edges["from"]).to_numpy(),
                position.reindex(edges["to"]).to_numpy(),
            ),
        ),
        shape=(len(nodes), len(nodes)),
    )
    return connected_components(graph, directed=False)[1]


def cut_zone(
    nodes: pd.DataFrame, edges: pd.DataFrame, zone: shapely.Geometry
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop the edges whose midpoint lies in ``zone``, and the nodes left alone.

    The midpoint, not any contact: an avenue that only grazes the zone stays, as it
    is what joins the zone to the rest of the city.
    """
    xy = nodes[["x", "y"]]
    a = xy.reindex(edges["from"].to_numpy()).to_numpy()
    b = xy.reindex(edges["to"].to_numpy()).to_numpy()
    shapely.prepare(zone)
    inside = shapely.contains(zone, shapely.points((a + b) / 2))
    kept = edges.loc[~inside]
    used = set(kept["from"]) | set(kept["to"])
    return nodes.loc[nodes.index.isin(used)], kept.reset_index(drop=True)


def node_lines(lines: gpd.GeoSeries, precision: float) -> pd.DataFrame:
    """Lines split where they cross: ``start`` and ``end`` (x, y rounded to
    ``precision``), ``length`` and ``geometry`` of each piece between crossings."""
    exploded = [g for g in lines.explode(index_parts=False) if g is not None]
    noded = shapely.union_all([g for g in exploded if not g.is_empty])
    pieces = [
        g
        for g in shapely.get_parts(noded)
        if g.geom_type == "LineString" and g.length > precision
    ]
    coords = [np.asarray(g.coords)[:, :2] for g in pieces]

    def snap(points: np.ndarray) -> np.ndarray:
        return np.round(points / precision) * precision

    return pd.DataFrame(
        {
            "start": list(snap(np.array([c[0] for c in coords]).reshape(-1, 2))),
            "end": list(snap(np.array([c[-1] for c in coords]).reshape(-1, 2))),
            "length": [g.length for g in pieces],
            "geometry": pieces,
        }
    )


def sew(
    nodes: pd.DataFrame, edges: pd.DataFrame, pieces: pd.DataFrame, snap_m: float
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Add ``pieces`` to the network; an end within ``snap_m`` of a network node is
    that node. Returns the nodes, the edges and the first id of a new node."""
    ends = np.vstack([np.stack(pieces["start"]), np.stack(pieces["end"])])
    coords, inverse = np.unique(ends, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    offset = max(NODE_ID_FLOOR, int(nodes.index.max()) + 1 if len(nodes) else 0)
    ids = np.arange(len(coords), dtype="int64") + offset
    if len(nodes):
        distance, index = cKDTree(nodes[["x", "y"]].to_numpy()).query(
            coords, distance_upper_bound=snap_m
        )
        hit = np.isfinite(distance)
        ids[hit] = nodes.index.to_numpy()[index[hit]]
    new = ids >= offset
    added = pd.DataFrame(
        {"x": coords[new, 0], "y": coords[new, 1]},
        index=pd.Index(ids[new], name="id"),
    )
    n = len(pieces)
    new_edges = pd.DataFrame(
        {
            "from": ids[inverse[:n]],
            "to": ids[inverse[n:]],
            "length": pieces["length"],
            "geometry": pieces["geometry"],
        }
    )
    new_edges = new_edges[new_edges["from"] != new_edges["to"]]
    # One edge per pair of nodes, the shortest, and none the network already has.
    pair = np.sort(new_edges[["from", "to"]].to_numpy(), axis=1)
    new_edges = (
        new_edges.assign(_a=pair[:, 0], _b=pair[:, 1])
        .sort_values("length")
        .drop_duplicates(["_a", "_b"])
    )
    existing = np.sort(edges[["from", "to"]].to_numpy(), axis=1)
    known = pd.MultiIndex.from_arrays([new_edges["_a"], new_edges["_b"]]).isin(
        pd.MultiIndex.from_arrays([existing[:, 0], existing[:, 1]])
    )
    new_edges = new_edges.loc[~known].drop(columns=["_a", "_b"])
    return (
        pd.concat([nodes, added]),
        pd.concat([edges, new_edges], ignore_index=True),
        offset,
    )


def _split(
    nodes: pd.DataFrame,
    edge: pd.Series,
    at: list[tuple[float, int, float]],
    next_id: int,
) -> tuple[list[dict], int]:
    """Edges replacing ``edge`` split at ``(position along it, node to hang, its
    distance)``, in order; each split point gets a new node, appended to ``nodes``."""
    start, end = nodes.loc[edge["from"], ["x", "y"]], nodes.loc[edge["to"], ["x", "y"]]
    line = shapely.LineString([start.to_numpy(), end.to_numpy()])
    rows, previous, previous_at = [], edge["from"], 0.0
    for position, node, distance in sorted(at):
        point = line.interpolate(position)
        nodes.loc[next_id] = [point.x, point.y]
        rows.append({"from": previous, "to": next_id, "length": position - previous_at})
        rows.append({"from": next_id, "to": node, "length": max(distance, 0.1)})
        previous, previous_at = next_id, position
        next_id += 1
    rows.append(
        {"from": previous, "to": edge["to"], "length": line.length - previous_at}
    )
    return rows, next_id


def hang_loose_ends(
    nodes: pd.DataFrame, edges: pd.DataFrame, offset: int, split_max_m: float
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Hang each loose end of the new streets on the nearest network edge, split
    where the end meets it, within ``split_max_m``.

    Not on the nearest node: network nodes are at crossings, so a street reaching
    the middle of a block would be joined 100 m away and distances would grow.
    """
    degree = pd.concat([edges["from"], edges["to"]]).value_counts()
    loose = [n for n in nodes.index[nodes.index >= offset] if degree.get(n, 0) == 1]
    network = edges.loc[edges["from"].lt(offset) & edges["to"].lt(offset)]
    if not loose or network.empty:
        return nodes, edges, 0
    segments = _segments(network, nodes)
    points = shapely.points(nodes.loc[loose, ["x", "y"]].to_numpy())
    nearest = shapely.STRtree(segments).nearest(points)
    distance = shapely.distance(points, segments[nearest])
    splits: dict[int, list[tuple[float, int, float]]] = {}
    for node, i, d, point in zip(loose, nearest, distance, points, strict=True):
        if d <= split_max_m:
            at = float(shapely.line_locate_point(segments[i], point))
            splits.setdefault(int(i), []).append((at, node, float(d)))
    nodes = nodes.copy()
    next_id = int(nodes.index.max()) + 1
    rows, dropped = [], []
    for i, at in splits.items():
        edge = network.iloc[i]
        new_rows, next_id = _split(nodes, edge, at, next_id)
        rows += new_rows
        dropped.append(network.index[i])
    edges = pd.concat(
        [edges.drop(index=dropped), pd.DataFrame(rows)], ignore_index=True
    )
    return nodes, edges, sum(len(at) for at in splits.values())


def bridge_pieces(
    nodes: pd.DataFrame, edges: pd.DataFrame, offset: int, connector_max_m: float
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Join each piece holding new nodes that is apart from the largest one, by the
    shortest line to another piece within ``connector_max_m``, splitting the edges
    it lands on. Pieces of the network alone (gated communities kept by
    ``retain_all``) are left as they are."""
    labels = _components(nodes, edges)
    sizes = pd.Series(labels).value_counts()
    is_new = nodes.index.to_numpy() >= offset
    apart = [c for c in sizes.index[1:] if is_new[labels == c].any()]
    if not apart or connector_max_m <= 0:
        return nodes, edges, 0
    members = [set(nodes.index[labels == c]) for c in apart]
    nodes, edges = nodes.copy(), edges.reset_index(drop=True)
    next_id = int(nodes.index.max()) + 1
    bridged = 0
    for ids in members:
        own = edges["from"].isin(ids) & edges["to"].isin(ids)
        others = edges.loc[~(edges["from"].isin(ids) | edges["to"].isin(ids))]
        own_edges = edges.loc[own]
        own_segments = _segments(own_edges, nodes)
        piece = shapely.union_all(own_segments)
        other_segments = _segments(others, nodes)
        near = shapely.intersects(
            shapely.buffer(piece, connector_max_m), other_segments
        )
        if not near.any():
            logger.warning(
                "Network: a piece of %s nodes has nothing within %s m",
                len(ids),
                connector_max_m,
            )
            continue
        candidates, candidate_segments = others.loc[near], other_segments[near]
        bridge = shapely.shortest_line(piece, shapely.union_all(candidate_segments))
        if bridge.length > connector_max_m:
            continue
        here, there = shapely.points(shapely.get_coordinates(bridge))
        ends, rows, dropped = [], [], []
        for point, frame, segs in (
            (here, own_edges, own_segments),
            (there, candidates, candidate_segments),
        ):
            k = int(np.argmin(shapely.distance(point, segs)))
            edge = frame.iloc[k]
            xy = shapely.get_coordinates(point)[0]
            corners = nodes.loc[[edge["from"], edge["to"]], ["x", "y"]].to_numpy()
            reach = np.hypot(*(corners - xy).T)
            if reach.min() <= 0.1:
                ends.append([edge["from"], edge["to"]][int(np.argmin(reach))])
                continue
            at = float(shapely.line_locate_point(segs[k], point))
            nodes.loc[next_id] = xy
            rows += [
                {"from": edge["from"], "to": next_id, "length": at},
                {"from": next_id, "to": edge["to"], "length": segs[k].length - at},
            ]
            ends.append(next_id)
            dropped.append(frame.index[k])
            next_id += 1
        rows.append(
            {"from": ends[1], "to": ends[0], "length": max(bridge.length, 0.01)}
        )
        edges = pd.concat(
            [edges.drop(index=dropped), pd.DataFrame(rows)], ignore_index=True
        )
        bridged += 1
    labels = _components(nodes, edges)
    main = pd.Series(labels).value_counts().index[0]
    if left := int((nodes.index.to_numpy() >= offset)[labels != main].sum()):
        logger.warning("Network: %s new nodes are not joined to the network", left)
    return nodes, edges, bridged


def merge_streets(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    streets: gpd.GeoSeries,
    params: NetworkParams,
    zone: shapely.Geometry | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Step 2: ``streets`` (lines in the network's CRS) replace the network inside
    ``zone``, by default their convex hull."""
    zone = (
        shapely.convex_hull(shapely.union_all(streets.to_numpy()))
        if zone is None
        else zone
    )
    kept_nodes, kept_edges = cut_zone(nodes, edges, zone)
    pieces = node_lines(streets, params.precision_m)
    merged_nodes, merged_edges, offset = sew(
        kept_nodes, kept_edges, pieces, params.snap_m
    )
    merged_nodes, merged_edges, hung = hang_loose_ends(
        merged_nodes, merged_edges, offset, params.split_max_m
    )
    merged_nodes, merged_edges, bridged = bridge_pieces(
        merged_nodes, merged_edges, offset, params.connector_max_m
    )
    report = {
        "removed_edges": len(edges) - len(kept_edges),
        "new_pieces": len(pieces),
        "new_nodes": int((merged_nodes.index >= offset).sum()),
        "loose_ends_hung": hung,
        "pieces_bridged": bridged,
    }
    logger.info("Streets merged: %s", report)
    return merged_nodes, merged_edges.astype({"from": "int64", "to": "int64"}), report


# ----------------------------------------------------------------------- output


def shapes(nodes: pd.DataFrame, edges: pd.DataFrame) -> np.ndarray:
    """Each edge's ``geometry``, or the straight line between its nodes without
    one."""
    lines = _segments(edges, nodes)
    if "geometry" not in edges:
        return lines
    own = edges["geometry"].to_numpy()
    present = ~pd.isna(own)
    lines[present] = own[present]
    return lines


def save_h5(nodes: pd.DataFrame, edges: pd.DataFrame, path: Path) -> Path:
    """The network as pandana's H5, two-way, ``length`` its impedance."""
    import pandana as pdna

    network = pdna.Network(
        node_x=nodes["x"],
        node_y=nodes["y"],
        edge_from=edges["from"],
        edge_to=edges["to"],
        edge_weights=edges[["length"]],
        twoway=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    network.save_hdf5(str(path))
    return path


def h5_name(cve_code: str, nodes: pd.DataFrame, edges: pd.DataFrame) -> str:
    """``vialidades_<cve>_<hash>.h5``: a new network never overwrites another."""
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(nodes[["x", "y"]]).to_numpy().tobytes())
    digest.update(
        pd.util.hash_pandas_object(edges[["from", "to", "length"]], index=False)
        .to_numpy()
        .tobytes()
    )
    return f"vialidades_{cve_code}_{digest.hexdigest()[:10]}.h5"


class ProjectNotFoundError(LookupError):
    """The project does not exist."""


def set_road_network_uri(project_id: str, uri: str, *, dry_run: bool) -> str | None:
    """Point ``projects.road_network_uri`` at ``uri``; returns the previous URI.

    With ``dry_run`` the change is rolled back.
    """
    import psycopg
    from psycopg import sql

    from reimagina_test_port.scenario.source import SCHEMA, resolve_dsn

    projects = sql.Identifier(SCHEMA, "projects")
    with psycopg.connect(resolve_dsn()) as conn, conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT NULLIF(road_network_uri, '') FROM {} WHERE id = %s::uuid FOR UPDATE"
            ).format(projects),
            (project_id,),
        )
        row = cur.fetchone()
        if row is None:
            msg = f"Project {project_id} does not exist."
            raise ProjectNotFoundError(msg)
        cur.execute(
            sql.SQL("UPDATE {} SET road_network_uri = %s WHERE id = %s::uuid").format(
                projects
            ),
            (uri, project_id),
        )
        if dry_run:
            conn.rollback()
    return row[0]


def read_geometries(path: Path, crs: str, kinds: tuple[str, ...]) -> gpd.GeoSeries:
    """The ``kinds`` geometries of a (Geo)Parquet or any vector file, in ``crs``."""
    if path.suffix.lower() == ".parquet":
        frame = gpd.read_parquet(path)
    else:
        frame = gpd.read_file(path)
    geometries = frame.geometry[frame.geom_type.isin(kinds)]
    if geometries.empty:
        msg = f"{path.name} has no {' or '.join(kinds)} geometries"
        raise ValueError(msg)
    return geometries.to_crs(crs)
