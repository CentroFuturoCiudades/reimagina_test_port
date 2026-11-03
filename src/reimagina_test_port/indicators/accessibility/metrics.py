import geopandas as gpd
import networkx as nx
import numpy as np
import pandana as pdna
import pandas as pd
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
)
from shapely.ops import unary_union


NETWORK_ORIGIN_COL = "network_origin_id"
EGRESS_DISTANCE_COL = "egress_distance"
IS_PRIVATE_ORIGIN_COL = "is_private_origin"
PRIVATE_AREA_COL = "private_area_id"
POI_NETWORK_COL = "destination_network_id"
POI_SNAP_DISTANCE_COL = "destination_snap_distance"


def _iter_line_parts(geometry):
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString | GeometryCollection):
        for part in geometry.geoms:
            yield from _iter_line_parts(part)


def _iter_point_parts(geometry):
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, Point):
        yield geometry
    elif isinstance(geometry, MultiPoint | GeometryCollection):
        for part in geometry.geoms:
            yield from _iter_point_parts(part)
    elif isinstance(geometry, LineString):
        yield geometry.interpolate(0.5, normalized=True)
        yield Point(geometry.coords[0])
        yield Point(geometry.coords[-1])


def _dedupe_points(points: list[Point]) -> list[Point]:
    seen = set()
    deduped = []
    for point in points:
        key = (round(point.x, 6), round(point.y, 6))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(point)
    return deduped


def _build_line_graph(line_geometries) -> tuple[nx.Graph, list[LineString]]:
    lines = [line for geom in line_geometries for line in _iter_line_parts(geom)]
    if not lines:
        return nx.Graph(), []

    noded_lines = unary_union(lines)
    segments = []
    graph = nx.Graph()
    for line in _iter_line_parts(noded_lines):
        coords = list(line.coords)
        for start, end in zip(coords[:-1], coords[1:], strict=False):
            segment = LineString([start, end])
            if segment.length == 0:
                continue
            start_key = (start[0], start[1])
            end_key = (end[0], end[1])
            segments.append(segment)
            graph.add_edge(start_key, end_key, weight=segment.length)

    return graph, segments


def _nearest_segment(
    point: Point,
    segments: list[LineString],
) -> tuple[LineString, Point, float]:
    segment = min(segments, key=lambda item: item.distance(point))
    projected = segment.interpolate(segment.project(point))
    return segment, projected, point.distance(projected)


def _connect_projected_point(
    graph: nx.Graph,
    segments: list[LineString],
    point: Point,
) -> tuple[tuple[float, float], float]:
    segment, projected, snap_distance = _nearest_segment(point, segments)
    projected_key = (projected.x, projected.y)
    start = tuple(segment.coords[0])
    end = tuple(segment.coords[-1])

    if projected_key == start or projected_key == end:
        return projected_key, snap_distance

    distance_from_start = segment.project(projected)
    distance_to_end = segment.length - distance_from_start
    graph.add_edge(projected_key, start, weight=distance_from_start)
    graph.add_edge(projected_key, end, weight=distance_to_end)
    return projected_key, snap_distance


def _internal_network_distance(
    graph: nx.Graph,
    segments: list[LineString],
    origin_point: Point,
    exit_point: Point,
) -> float:
    if not segments:
        return origin_point.distance(exit_point)

    temp_graph = graph.copy()
    origin_node, origin_snap = _connect_projected_point(temp_graph, segments, origin_point)
    exit_node, exit_snap = _connect_projected_point(temp_graph, segments, exit_point)
    try:
        path_distance = nx.shortest_path_length(
            temp_graph,
            source=origin_node,
            target=exit_node,
            weight="weight",
        )
    except nx.NetworkXNoPath:
        path_distance = origin_point.distance(exit_point)
        origin_snap = 0
        exit_snap = 0

    return origin_snap + path_distance + exit_snap


def _infer_exit_points(area_geometry, street_geometries) -> list[Point]:
    lines = [line for geom in street_geometries for line in _iter_line_parts(geom)]
    if not lines:
        return []

    street_union = unary_union(lines)
    boundary_intersections = street_union.intersection(area_geometry.boundary)
    exit_points = list(_iter_point_parts(boundary_intersections))

    if not exit_points:
        endpoints = []
        for line in lines:
            endpoints.extend([Point(line.coords[0]), Point(line.coords[-1])])
        if endpoints:
            endpoint_distances = pd.Series(
                [point.distance(area_geometry.boundary) for point in endpoints],
                index=range(len(endpoints)),
            )
            min_distance = endpoint_distances.min()
            exit_points = [
                endpoints[index]
                for index, distance in endpoint_distances.items()
                if np.isclose(distance, min_distance)
            ]

    return _dedupe_points(exit_points)


def _nearest_network_node_distance(net: pdna.Network, point: Point) -> tuple[int, float]:
    point_index = pd.Index([0])
    node_id = net.get_node_ids(
        pd.Series([point.x], index=point_index),
        pd.Series([point.y], index=point_index),
    ).iloc[0]
    node = net.nodes_df.loc[node_id]
    node_distance = point.distance(Point(node["x"], node["y"]))
    return node_id, node_distance


def _poi_network_assignments(net: pdna.Network, pois_coords: pd.DataFrame) -> pd.DataFrame:
    poi_network_ids = net.get_node_ids(pois_coords["x"], pois_coords["y"])
    network_nodes = net.nodes_df.loc[poi_network_ids.values]
    poi_snap_distances = np.sqrt(
        (pois_coords["x"].to_numpy() - network_nodes["x"].to_numpy()) ** 2
        + (pois_coords["y"].to_numpy() - network_nodes["y"].to_numpy()) ** 2,
    )
    return pd.DataFrame(
        {
            POI_NETWORK_COL: poi_network_ids.values,
            POI_SNAP_DISTANCE_COL: poi_snap_distances,
        },
        index=pois_coords.index,
    )


def _synthetic_origin_ids(size: int, existing_ids) -> list[int]:
    if size == 0:
        return []
    numeric_existing_ids = pd.to_numeric(pd.Index(existing_ids), errors="coerce")
    min_existing_id = np.nanmin(numeric_existing_ids) if len(numeric_existing_ids) else 0
    start = int(min(min_existing_id, 0)) - 1
    return list(range(start, start - size, -1))


def _private_origin_assignments(
    net: pdna.Network,
    block_centroids: gpd.GeoDataFrame,
    private_areas: gpd.GeoDataFrame,
    private_streets: gpd.GeoDataFrame,
    private_id_col: str,
) -> pd.DataFrame:
    if private_areas is None and private_streets is None:
        return pd.DataFrame(index=block_centroids.index)
    if private_areas is None or private_streets is None:
        raise ValueError("private_areas and private_streets must be provided together.")
    if private_id_col not in private_areas.columns:
        raise ValueError(f"private_areas must include '{private_id_col}'.")
    if private_id_col not in private_streets.columns:
        raise ValueError(f"private_streets must include '{private_id_col}'.")

    private_areas = private_areas[[private_id_col, "geometry"]].copy()
    private_streets = private_streets[[private_id_col, "geometry"]].copy()
    if private_areas.crs != block_centroids.crs:
        private_areas = private_areas.to_crs(block_centroids.crs)
    if private_streets.crs != block_centroids.crs:
        private_streets = private_streets.to_crs(block_centroids.crs)

    joined = gpd.sjoin(block_centroids, private_areas, how="inner", predicate="within")
    if joined.empty:
        return pd.DataFrame(index=block_centroids.index)

    joined = joined[~joined.index.duplicated(keep="first")]
    assignments = []
    streets_by_private_id = dict(tuple(private_streets.groupby(private_id_col)))
    areas_by_private_id = private_areas.drop_duplicates(private_id_col).set_index(private_id_col)

    for private_id, blocks_in_private in joined.groupby(private_id_col):
        if private_id not in streets_by_private_id or private_id not in areas_by_private_id.index:
            continue

        streets = streets_by_private_id[private_id]
        graph, segments = _build_line_graph(streets.geometry)
        if not segments:
            continue

        area_geometry = areas_by_private_id.loc[private_id].geometry
        exit_points = _infer_exit_points(area_geometry, streets.geometry)
        if not exit_points:
            continue

        exit_network_nodes = [
            (*_nearest_network_node_distance(net, exit_point), exit_point)
            for exit_point in exit_points
        ]

        for block_id, block in blocks_in_private.iterrows():
            origin_point = block.geometry
            candidates = []
            for network_node_id, network_snap_distance, exit_point in exit_network_nodes:
                egress_distance = (
                    _internal_network_distance(graph, segments, origin_point, exit_point)
                    + network_snap_distance
                )
                candidates.append((egress_distance, network_node_id))

            egress_distance, network_node_id = min(candidates, key=lambda item: item[0])
            assignments.append(
                {
                    "block_id": block_id,
                    NETWORK_ORIGIN_COL: network_node_id,
                    EGRESS_DISTANCE_COL: egress_distance,
                    PRIVATE_AREA_COL: private_id,
                    IS_PRIVATE_ORIGIN_COL: True,
                },
            )

    if not assignments:
        return pd.DataFrame(index=block_centroids.index)

    return pd.DataFrame(assignments).set_index("block_id")


def create_origin_destination_mappings(
    net: pdna.Network,
    df_blocks: gpd.GeoDataFrame,
    numerical_cols: list = None,
    id_col: str = "id",
    private_areas: gpd.GeoDataFrame = None,
    private_streets: gpd.GeoDataFrame = None,
    private_id_col: str = "private_id",
) -> tuple:
    
    if numerical_cols is None:
        numerical_cols = df_blocks.select_dtypes(include=[np.number]).columns.tolist()
        if id_col in numerical_cols:
            numerical_cols.remove(id_col)

    block_centroids = gpd.GeoDataFrame(
        df_blocks.set_index(id_col).drop(columns=["geometry"], errors="ignore"),
        geometry=df_blocks.set_index(id_col).centroid,
        crs=df_blocks.crs,
    )
    coords = block_centroids.get_coordinates()

    # Get the nearest network node for each CVEGEO
    id_to_network_origin_map = net.get_node_ids(coords["x"], coords["y"])
    origin_metadata = pd.DataFrame(
        {
            "origin_id": id_to_network_origin_map,
            NETWORK_ORIGIN_COL: id_to_network_origin_map,
            EGRESS_DISTANCE_COL: 0.0,
            PRIVATE_AREA_COL: pd.NA,
            IS_PRIVATE_ORIGIN_COL: False,
        },
        index=id_to_network_origin_map.index,
    )

    use_private_inputs = private_areas is not None or private_streets is not None
    private_assignments = (
        _private_origin_assignments(
            net=net,
            block_centroids=block_centroids.loc[id_to_network_origin_map.index],
            private_areas=private_areas,
            private_streets=private_streets,
            private_id_col=private_id_col,
        )
        if use_private_inputs
        else pd.DataFrame(index=block_centroids.index)
    )
    if NETWORK_ORIGIN_COL in private_assignments.columns:
        private_assignments = private_assignments.dropna(
            subset=[NETWORK_ORIGIN_COL],
            how="any",
        )
    if not private_assignments.empty:
        synthetic_ids = _synthetic_origin_ids(
            len(private_assignments),
            existing_ids=net.node_ids,
        )
        origin_metadata.loc[private_assignments.index, "origin_id"] = synthetic_ids
        origin_metadata.loc[private_assignments.index, NETWORK_ORIGIN_COL] = (
            private_assignments[NETWORK_ORIGIN_COL]
        )
        origin_metadata.loc[private_assignments.index, EGRESS_DISTANCE_COL] = (
            private_assignments[EGRESS_DISTANCE_COL]
        )
        origin_metadata.loc[private_assignments.index, PRIVATE_AREA_COL] = (
            private_assignments[PRIVATE_AREA_COL]
        )
        origin_metadata.loc[private_assignments.index, IS_PRIVATE_ORIGIN_COL] = True

    id_to_origin_map = origin_metadata["origin_id"].astype(id_to_network_origin_map.dtype)
    origin_to_id_map = pd.Series(id_to_origin_map.index, index=id_to_origin_map.values)
    origin_to_id_map.name = id_col

    metadata_by_origin = (
        origin_metadata.reset_index(names=id_col)
        .groupby("origin_id", observed=True)
        .agg(
            {
                NETWORK_ORIGIN_COL: "first",
                EGRESS_DISTANCE_COL: "first",
                PRIVATE_AREA_COL: "first",
                IS_PRIVATE_ORIGIN_COL: "any",
            },
        )
    )

    df_accessibility = (
        df_blocks.assign(origin_id=lambda df: df[id_col].map(id_to_origin_map))
        .groupby("origin_id")[numerical_cols]
        .sum()
    )
    if use_private_inputs:
        df_accessibility = df_accessibility.join(metadata_by_origin)

    return df_accessibility, origin_to_id_map, id_to_origin_map

def get_proximities(
    net: pdna.Network,
    pois_coords: pd.DataFrame,
    pois_category: str,
    max_distance: float,
    num_pois: int = 1,
    origin_network_map: pd.DataFrame = None,
    include_poi_snap_distance: bool = False,
) -> pd.DataFrame:
    
    net.set_pois(pois_category, max_distance, num_pois, pois_coords["x"], pois_coords["y"])
    origin_col = NETWORK_ORIGIN_COL if origin_network_map is not None else "origin_id"
    res = (
        net.nearest_pois(
            distance=max_distance,
            category=pois_category,
            num_pois=num_pois,
            include_poi_ids=True,
        )
        .rename(columns={x: f"distance{x}" for x in range(1, num_pois + 1)})
        .reset_index(names=origin_col)
    )
    final_df = pd.wide_to_long(
        res,
        stubnames=["distance", "poi"],
        i=origin_col,
        j="num_poi",
        sep="",
    ).rename(columns={"poi": "destination_id"}).reset_index()

    poi_network_assignments = _poi_network_assignments(net, pois_coords)
    final_df = final_df.merge(
        poi_network_assignments,
        left_on="destination_id",
        right_index=True,
        how="left",
    )
    if include_poi_snap_distance:
        final_df["distance"] = (
            final_df["distance"] + final_df[POI_SNAP_DISTANCE_COL].fillna(0)
        )

    if origin_network_map is not None:
        origins = (
            origin_network_map[[NETWORK_ORIGIN_COL, EGRESS_DISTANCE_COL]]
            .reset_index(names="origin_id")
            .assign(**{EGRESS_DISTANCE_COL: lambda df: df[EGRESS_DISTANCE_COL].fillna(0)})
        )
        final_df = origins.merge(final_df, on=NETWORK_ORIGIN_COL, how="left")
        final_df["distance"] = final_df["distance"] + final_df[EGRESS_DISTANCE_COL]
        beyond_max_distance = final_df["distance"] > max_distance
        final_df.loc[beyond_max_distance, ["distance", "destination_id"]] = np.nan

    # TODO: validate if filtering here is correct
    #final_df = final_df[final_df.distance<max_distance]
    #final_df['destination_id'] = final_df['destination_id'].astype(int)
    return final_df


def calculate_time(distance_col: pd.Series, travel_speed: float) -> pd.Series:
    return (distance_col / 1000) * 60 / travel_speed


def get_accessibility_metrics(
    proximities: pd.DataFrame,
    destinations: gpd.GeoDataFrame, #df equipamiento
    weights: pd.Series, # series de poblacion/viviendas por origin_id
    radius: float,
    adjustment_factor: float,
) -> pd.DataFrame:
    proximities = proximities.join(weights, on="origin_id")
    proximities = proximities.join(destinations[["capacity"]], on="destination_id")
    proximities["gravity"] = 1 / np.exp(
        1 / (radius / adjustment_factor) * proximities["distance"],
    )
    proximities["reach"] = proximities["gravity"] * proximities[weights.name]
    
    # Calcular pob_reach por destino (suma de reach de todos los origenes hacia ese destino)
    dest_pob_reach = proximities.groupby("destination_id")["reach"].sum().rename("pob_reach")
    
    destinations_agg = gpd.GeoDataFrame(
        proximities.groupby("destination_id")
        .agg({"reach": "sum", "capacity": "first"})
        .assign(
            opportunities_ratio=lambda df: df.apply(
                lambda x: x["capacity"] / x["reach"] if x["reach"] > 0 else 0,
                axis=1,
            ),
        )
        .query("opportunities_ratio > 0"),
    )
    proximities["accessibility_score"] = proximities.apply(
        lambda x: destinations_agg["opportunities_ratio"].loc[x["destination_id"]]
        * x["gravity"]
        if x["destination_id"] in destinations_agg["opportunities_ratio"].index
        else 0,
        axis=1,
    )
    
    # Unir pob_reach y attraction (capacity) a proximities para exportar por par origen-destino
    proximities = proximities.join(dest_pob_reach, on="destination_id")
    proximities["attraction"] = proximities["capacity"]

    # Agregar por origin_id: solo accessibility_score y time (min)
    temp_proximities = (proximities.reset_index()
        .groupby("origin_id")
        .agg({
            "accessibility_score": "sum",
            "time": "min",
        })
        .assign(
            accessibility_score=lambda df: (
                df["accessibility_score"] - df["accessibility_score"].min()
            )
            / (df["accessibility_score"].max() - df["accessibility_score"].min()),
        )
    )

    # Incluir columnas del POI más cercano (num_poi==1): distance, destination_id, gravity, pob_reach, attraction
    detail_col_names = ["distance", "destination_id", "gravity", "pob_reach", "attraction"]
    detail_col_names.extend(
        col for col in [POI_NETWORK_COL, POI_SNAP_DISTANCE_COL] if col in proximities.columns
    )
    detail_cols = proximities.query("num_poi==1").set_index("origin_id")[detail_col_names]
    return temp_proximities.join(detail_cols)


def compute_pois_accessibility_scores(
    POIS_CONFIG: dict,
    df_pois: gpd.GeoDataFrame,
    net: pdna.Network,
    df_accessibility: pd.DataFrame,
    weight_col: str,
    max_distance: float,
    num_pois: int,
    adjustment_factor: float,
    travel_speed: float,
    include_poi_snap_distance: bool = False,
) -> pd.DataFrame:
    scores = []
    origin_network_map = None
    has_origin_network_map = {
        NETWORK_ORIGIN_COL,
        EGRESS_DISTANCE_COL,
        IS_PRIVATE_ORIGIN_COL,
    }.issubset(
        df_accessibility.columns,
    )
    if has_origin_network_map and df_accessibility[IS_PRIVATE_ORIGIN_COL].any():
        origin_network_map = df_accessibility[[NETWORK_ORIGIN_COL, EGRESS_DISTANCE_COL]]

    for category_name, item in POIS_CONFIG.items():
        df_temp = df_pois.query("category == @category_name")
        proximities = get_proximities(
            net=net,
            pois_coords=df_temp.get_coordinates(),
            pois_category=category_name,
            max_distance=max_distance,
            num_pois=num_pois,
            origin_network_map=origin_network_map,
            include_poi_snap_distance=include_poi_snap_distance,
        )
        proximities['time'] = calculate_time(proximities['distance'], travel_speed)
        accessibility_scores = get_accessibility_metrics(
            proximities=proximities,
            destinations=df_temp,
            weights=df_accessibility[item[weight_col]],
            radius=item["radius"],
            adjustment_factor=adjustment_factor,
        ).assign(poi_category=category_name)
        scores.append(accessibility_scores)

    scores = pd.concat(scores)
    scores.dropna(subset='destination_id',inplace=True)

    return scores

def compute_block_accessibility(
    df_blocks: gpd.GeoDataFrame,
    df_accessibility: pd.DataFrame,
    scores: pd.DataFrame,
    origin_to_id_map: pd.DataFrame,
    id_to_origin_map: pd.Series,
    poi_weights_map: dict,
    id_col: str="id",
) -> gpd.GeoDataFrame:
    
    final_score = scores.assign(
        poi_weight=scores['poi_category'].map(poi_weights_map).fillna(0),
        accessibility_score=lambda df: df['accessibility_score'] * df['poi_weight']
    ).groupby("origin_id", observed=True)["accessibility_score"].sum()


    block_accessibility = (
        df_accessibility.join(final_score)
        .join(origin_to_id_map)
        .explode(id_col)
        .set_index(id_col)["accessibility_score"]
    )
    df_blocks_accessibility = df_blocks.set_index(id_col).assign(
        accessibility_score=block_accessibility,
        origin_id=lambda df: df.index.map(id_to_origin_map),
    )

       
    # Normalizacion por bloques (0-1)
    # TODO: validar el uso del logaritmo para suavizar la distribucion => reducir outliers
    df_blocks_accessibility = (df_blocks_accessibility.assign(
        accessibility_score=lambda df: np.log1p(df["accessibility_score"])
        ).assign(
            accessibility_score=lambda df: (
                df["accessibility_score"] - df["accessibility_score"].min()
            )
            / (df["accessibility_score"].max() - df["accessibility_score"].min()),
        ))
    
    
    return df_blocks_accessibility


def compute_accessibility_indicators(
    df_blocks_accessibility: gpd.GeoDataFrame,
    scores: pd.DataFrame,
    pois_config: dict,
) -> gpd.GeoDataFrame:
    temp = df_blocks_accessibility.join(
        scores[["time", "poi_category", "distance", "destination_id"]], on="origin_id"
    )

    for c in temp.poi_category.unique():
        if pd.isna(c):
            continue
        n = pois_config[c]["time"]
        viv_col = pois_config[c]["VIV_COL"]
        df_blocks_accessibility[f"tiempo_{c.replace(' ', '_').lower()}"] = temp[
            temp["poi_category"] == c
        ]["time"]
        df_blocks_accessibility[f"distancia_{c.replace(' ', '_').lower()}"] = temp[
            temp["poi_category"] == c
        ]["distance"]
        df_blocks_accessibility[f"id_destino_{c.replace(' ', '_').lower()}"] = temp[
            temp["poi_category"] == c
        ]["destination_id"]
        df_blocks_accessibility[
            f"hogares_{c.replace(' ', '_').lower()}_{n}mi"
        ] = temp[temp["poi_category"] == c].apply(
            lambda row: (row[viv_col] if row["time"] <= n else 0)
            / row[viv_col],
            axis=1,
        ).fillna(0)

    #df_blocks_accessibility.drop(columns=["origin_id"], inplace=True)

    return df_blocks_accessibility
