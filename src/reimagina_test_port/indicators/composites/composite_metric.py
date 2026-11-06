"""Lyra metrics for composite indicators: one node per composite of the tree in
``composite_config.yaml``, each depending on its children, the indicator tables its
leaves read and the outputs of its child composites. Lyra then runs the tree
bottom-up: a composite runs once its children have. With a ``boundary`` (a
GeoParquet URL) in the config, every composite also depends on
reimagina_composite_boundary and keeps only the elements inside it."""

import inspect
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import yaml
from lyra.sdk import (
    Dependency,
    MetricParameters,
    RunContext,
    TableOutput,
    Unit,
    dataset,
    metric,
)
from lyra.sdk.errors import MetricInputError

from reimagina_test_port.common import lake
from reimagina_test_port.common.nodes import column, timed
from reimagina_test_port.common.schema import IDENTITY, UNITS, identity_columns
from reimagina_test_port.indicators.accessibility.nodes import ACCESSIBILITY_KEYS
from reimagina_test_port.indicators.composites.composite import (
    build_composite,
    normalize_indicators,
)

CONFIG = Path(__file__).with_name("composite_config.yaml")
# A leaf's source -> its Lyra output prefix and columns; the output is
# <prefix>_<element>.
SOURCES = {
    "accessibility": ("reimagina_accessibility", ACCESSIBILITY_KEYS),
    "built_form": ("reimagina_built_form", list(UNITS)),
}
ELEMENTS = ("blocks", "lots")
LEAF_KEYS = {
    "name", "source", "variable", "polarity", "method", "transform", "reference",
    "goalposts", "mean", "sd", "compress", "weight",
}  # fmt: skip
COMPOSITE_KEYS = {"name", "phenomenon", "min_valid", "weight", "children"}
RESULT = ["index", "mean", "penalty", "n"]
# Child names become output columns and handler arguments: none of these.
RESERVED = {*RESULT, *IDENTITY, "geometry", *SOURCES, "parameters", "context",
            "location", "bounds", "boundary"}  # fmt: skip
NAME = re.compile(r"^[a-z][a-z0-9_]*$")
BOUNDARY = "reimagina_composite_boundary"


class CompositeConfigError(ValueError):
    """The composite tree has a missing, unknown or inconsistent value."""


def output_name(name: str, element: str) -> str:
    return f"reimagina_composite_{name}_{element}"


def is_composite(node: dict) -> bool:
    return "children" in node


def is_ref(node: dict) -> bool:
    """A child that reuses a composite defined elsewhere: ``{ref: <name>}``."""
    return "ref" in node


def child_name(child: dict) -> str:
    """Its column in its parent's output; a ref's is the composite it reuses."""
    return child["ref"] if is_ref(child) else child["name"]


def reads_composite(child: dict) -> bool:
    """A child whose score is another composite's index."""
    return is_composite(child) or is_ref(child)


def leaf_arguments(leaf: dict) -> dict[str, Any]:
    """``normalize_indicators``'s arguments from a leaf."""
    return {
        "polarity": leaf.get("polarity", "positive"),
        "method": leaf.get("method", "ampi"),
        "transform": leaf.get("transform"),
        "reference": leaf.get("reference"),
        "goalposts": leaf.get("goalposts"),
        "mean": leaf.get("mean"),
        "sd": leaf.get("sd"),
        "compress": bool(leaf.get("compress", False)),
    }


def min_valid(node: dict) -> int | None:
    value = node.get("min_valid", "all")
    return None if value == "all" else int(value)


def check_tree(config: dict) -> tuple[str, list[dict], str | None]:
    """The config's element, its composites (every tree's, each once, children
    before parents) and its boundary URL, if any.

    Raises:
        CompositeConfigError: a node that can't be computed, named by its path.
    """
    root = dict(config)
    element = root.pop("element", None)
    boundary = root.pop("boundary", None)
    trees = root.pop("trees", None)
    if unknown := sorted(root):
        msg = f"unknown keys {unknown}; the composites go under trees"
        raise CompositeConfigError(msg)
    if boundary is not None and not isinstance(boundary, str):
        msg = "boundary must be the URL of a GeoParquet"
        raise CompositeConfigError(msg)
    if element not in ELEMENTS:
        msg = f"element must be one of {list(ELEMENTS)}, not {element!r}"
        raise CompositeConfigError(msg)
    if not trees or not isinstance(trees, list):
        msg = "trees must list at least one composite"
        raise CompositeConfigError(msg)
    names: set[str] = set()
    defined: dict[str, dict] = {}  # composite name -> node
    refs: list[tuple[str, str]] = []  # (where, composite it reuses)

    def fail(where: str, detail: str) -> None:
        msg = f"{where}: {detail}"
        raise CompositeConfigError(msg)

    def visit(node: dict, path: list[str]) -> None:
        if isinstance(node, dict) and is_ref(node):
            where = " > ".join([*path, f"ref {node['ref']}"])
            if unknown := sorted(set(node) - {"ref", "weight"}):
                fail(where, f"unknown keys {unknown}; a ref only takes a weight")
            if not float(node.get("weight", 1)) > 0:
                fail(where, "weight must be positive")
            refs.append((where, node["ref"]))
            return
        name = node.get("name") if isinstance(node, dict) else None
        where = " > ".join([*path, str(name)])
        if not isinstance(name, str) or not NAME.match(name):
            fail(where, "name must be lower_snake_case")
        if name in names:
            fail(where, "name repeated; reuse a composite with {ref: name}")
        if path and name in RESERVED:  # a root is no one's column
            fail(where, f"name reserved, not one of {sorted(RESERVED)}")
        names.add(name)
        if not float(node.get("weight", 1)) > 0:
            fail(where, "weight must be positive")
        if is_composite(node):
            if not path and "weight" in node:
                fail(where, "a tree's root has no parent to weigh it")
            if unknown := sorted(set(node) - COMPOSITE_KEYS):
                fail(where, f"unknown keys {unknown}")
            children = node["children"] or []
            if not children:
                fail(where, "a composite needs children")
            if node.get("phenomenon", "positive") not in ("positive", "negative"):
                fail(where, "phenomenon must be positive or negative")
            required = min_valid(node)
            if required is not None and not 1 <= required <= len(children):
                fail(where, f"min_valid must be 'all' or 1-{len(children)}")
            for child in children:
                visit(child, [*path, name])
            columns = [child_name(c) for c in children if isinstance(c, dict)]
            if len(columns) != len(set(columns)):
                fail(where, "a child appears twice")
            defined[name] = node
            return
        if not path:
            fail(where, "a tree's root must be a composite (have children)")
        if unknown := sorted(set(node) - LEAF_KEYS):
            fail(where, f"unknown keys {unknown}")
        source = node.get("source")
        if source not in SOURCES:
            fail(where, f"source must be one of {sorted(SOURCES)}")
        if node.get("variable") not in SOURCES[source][1]:
            fail(where, f"{source} has no column {node.get('variable')!r}")
        try:  # the normalisation's own checks, on one value
            normalize_indicators(pd.Series([0.0]), **leaf_arguments(node))
        except (ValueError, TypeError, KeyError) as exc:
            fail(where, str(exc))

    for tree in trees:
        visit(tree, [])
    for where, target in refs:
        if target not in defined:
            fail(where, f"no composite named {target!r} to reuse")
        if target in RESERVED:
            fail(where, f"{target!r} can't be a column; rename that composite")

    # Children before parents; a composite that ends up needing itself fails.
    order: list[dict] = []
    state: dict[str, str] = {}

    def place(name: str, chain: list[str]) -> None:
        if state.get(name) == "done":
            return
        if state.get(name) == "open":
            fail(" > ".join([*chain, name]), "a cycle: a composite needs itself")
        state[name] = "open"
        for child in defined[name]["children"]:
            if reads_composite(child):
                place(child_name(child), [*chain, name])
        state[name] = "done"
        order.append(defined[name])

    for name in defined:
        place(name, [])
    return element, order, boundary


def composite_inputs(node: dict, element: str) -> dict[str, Dependency]:
    """One dependency per indicator source its leaves read, one per child
    composite."""
    inputs = {}
    for child in node["children"]:
        if reads_composite(child):
            name = child_name(child)
            inputs[name] = Dependency(output_name(name, element))
        else:
            prefix = SOURCES[child["source"]][0]
            inputs[child["source"]] = Dependency(f"{prefix}_{element}")
    return inputs


def by_id(table: pd.DataFrame, key: str, ids: pd.Series) -> pd.Series:
    """``table[key]`` in the order of ``ids``; ids it lacks are missing."""
    values = pd.Series(table[key].to_numpy(), index=table["id"].astype(str))
    return pd.Series(values.reindex(ids).to_numpy(), dtype="float64")


def read_boundary(uri: str) -> gpd.GeoDataFrame:
    """The GeoParquet at ``uri`` (``az://`` or a local path), its polygons merged
    into one row, EPSG:4326; its CRS comes from the file."""
    options = lake.storage_options() if uri.startswith("az://") else None
    frame = gpd.read_parquet(uri, storage_options=options)
    if frame.empty or frame.crs is None:
        msg = f"Boundary {uri}: no polygons, or no CRS"
        raise ValueError(msg)
    union = frame.to_crs(4326).geometry.union_all()
    return gpd.GeoDataFrame({"name": [uri]}, geometry=[union], crs=4326)


def composite_table(
    node: dict,
    tables: dict[str, gpd.GeoDataFrame],
    boundary: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Each child's score (leaves normalised, child composites' index), then
    the Mazziotta-Pareto composite, per element of the first input. With a
    boundary, only the elements whose representative point lies inside it, so
    the others are out of any normalisation statistics (compress) too."""
    grain = next(iter(tables.values()))
    if boundary is not None:
        area = boundary.to_crs(grain.crs).geometry.union_all()
        grain = grain[grain.geometry.representative_point().within(area)]
    result = grain[IDENTITY + ["geometry"]].reset_index(drop=True)
    ids = result["id"].astype(str)
    scores = []
    for child in node["children"]:
        name = child_name(child)
        if reads_composite(child):
            score = by_id(tables[name], "index", ids)
        else:
            values = by_id(tables[child["source"]], child["variable"], ids)
            score = normalize_indicators(values, **leaf_arguments(child))
        result[name] = score.to_numpy()
        scores.append(score.rename(name))
    combined = build_composite(
        scores,
        weights=[float(child.get("weight", 1)) for child in node["children"]],
        phenomenon=node.get("phenomenon", "positive"),
        min_valid=min_valid(node),
    )
    for key in RESULT:
        result[key] = combined[key].to_numpy()
    return result


def composite_declaration(node: dict) -> TableOutput:
    """Identity, each child's score, the composite and geometry."""
    children = [
        column(
            child_name(child),
            "number",
            Unit.SCORE,
            f"Composite index of {child_name(child)}, around 100."
            if reads_composite(child)
            else f"{child['variable']} from {child['source']}, normalised "
            f"({child.get('method', 'ampi')}) around 100.",
        )
        for child in node["children"]
    ]
    return TableOutput(
        columns=[
            *identity_columns(),
            *children,
            column(
                "index",
                "number",
                Unit.SCORE,
                "Mazziotta-Pareto composite: the "
                "mean less its imbalance penalty (plus, if negative); null when "
                "fewer children than min_valid have a value.",
            ),
            column("mean", "number", Unit.SCORE, "Weighted mean of the children."),
            column(
                "penalty", "number", Unit.SCORE, "Imbalance penalty: SD × SD / mean."
            ),
            column("n", "integer", Unit.COUNT, "Children with a value."),
        ],
        geometry="geometry",
    )


def composite_metric(
    node: dict,
    element: str,
    parameters_type: type[MetricParameters],
    boundary: bool,
):
    """The Lyra metric of one composite."""
    children = composite_inputs(node, element)
    inputs = children | ({"boundary": Dependency(BOUNDARY)} if boundary else {})
    output = output_name(node["name"], element)

    def handler(**arguments: Any) -> gpd.GeoDataFrame:
        tables = {key: arguments[key] for key in children}
        with timed(f"composite {node['name']}", arguments["context"].logger):
            return composite_table(node, tables, arguments.get("boundary"))

    # Lyra binds dependencies by argument name: one per input of this composite.
    handler.__signature__ = inspect.Signature(
        [
            inspect.Parameter(key, inspect.Parameter.KEYWORD_ONLY)
            for key in ["parameters", *inputs, "context"]
        ]
    )
    handler.__annotations__ = {"parameters": parameters_type, "context": RunContext}
    names = ", ".join(child_name(child) for child in node["children"])
    return metric(
        name=f"reimagina_composite_{node['name']}",
        description=(
            f"Composite indicator {node['name']} per {element[:-1]}: the weighted "
            f"Mazziotta-Pareto index ({node.get('phenomenon', 'positive')}) of "
            f"{names}, each normalised (AMPI or MPI) as composite_config.yaml "
            f"says{', inside its boundary' if boundary else ''}."
        ),
        inputs=inputs,
        outputs={output: composite_declaration(node)},
    )(handler)


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def boundary_dataset(uri: str, parameters_type: type[MetricParameters]):
    """reimagina_composite_boundary: the config's boundary, read once a run."""

    def handler(parameters) -> gpd.GeoDataFrame:
        try:
            return read_boundary(uri)
        except (OSError, ValueError) as exc:
            raise MetricInputError(BOUNDARY, "boundary", str(exc)) from exc

    handler.__annotations__ = {"parameters": parameters_type}
    return dataset(
        name=BOUNDARY,
        description=f"The area composite indicators are read over: {uri}.",
        outputs={BOUNDARY: TableOutput(
            columns=[column("name", "string", None, "The boundary's URL.")],
            geometry="geometry",
        )},
    )(handler)  # fmt: skip


def composite_metrics(
    parameters_type: type[MetricParameters], config: dict | None = None
) -> list:
    """One metric per composite of ``config`` (by default composite_config.yaml),
    and the boundary dataset when it names one."""
    element, composites, boundary = check_tree(
        load_config() if config is None else config
    )
    nodes = [boundary_dataset(boundary, parameters_type)] if boundary else []
    return nodes + [
        composite_metric(node, element, parameters_type, boundary is not None)
        for node in composites
    ]


# Each composite's index is stored as composite_<name>, on the config's elements.
ELEMENT_TYPES = {"blocks": "block", "lots": "lot"}


def composite_key(name: str) -> str:
    return f"composite_{name}"


def composite_outputs(config: dict | None = None) -> tuple[str, list[str]]:
    """The config's element and its composites' names, children first."""
    element, composites, _ = check_tree(load_config() if config is None else config)
    return element, [node["name"] for node in composites]
