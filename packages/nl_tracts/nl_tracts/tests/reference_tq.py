"""tract_querier's query evaluator, as the oracle of the WMQL transpiler.

``query_processor.py`` is loaded **by path**, with ``code_util`` stubbed: it
needs only the standard library, and the package's ``__init__`` imports vtk
and the tractography readers. Where it is found: ``TRACT_QUERIER_DIR`` (the
directory of the ``tract_querier`` package, with ``query_processor.py`` and
``data``), else an installed ``tract_querier``. Not found: the tests that
need it are skipped. It is never a dependency of nl_tracts.
"""

import ast
import importlib.util
import os
import sys
import types

import numpy as np
import pytest


def reference_dir():
    path = os.environ.get("TRACT_QUERIER_DIR")
    if not path:
        spec = importlib.util.find_spec("tract_querier")  # not imported
        path = spec and list(spec.submodule_search_locations or [None])[0]
    if path and os.path.isfile(os.path.join(path, "query_processor.py")):
        return path
    return None


def load_reference():
    """The module ``query_processor`` of tract_querier, or skip."""
    path = reference_dir()
    if path is None:
        pytest.skip("tract_querier not found (set TRACT_QUERIER_DIR)")
    name = "_tq_reference"
    package = types.ModuleType(name)
    package.__path__ = []
    code_util = types.ModuleType(f"{name}.code_util")
    code_util.DocStringInheritor = type  # a Python 2 metaclass, unused
    spec = importlib.util.spec_from_file_location(
        f"{name}.query_processor", os.path.join(path, "query_processor.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.update(
        {name: package, f"{name}.code_util": code_util, spec.name: module}
    )
    try:
        spec.loader.exec_module(module)
    finally:
        for key in (name, f"{name}.code_util", spec.name):
            sys.modules.pop(key, None)
    return module


def load_indexing():
    """tract_querier's ``tract_label_indices`` (numpy and scipy), loaded by
    path; it imports ``.aabb``, which is registered beside it."""
    path = reference_dir()
    if path is None:
        pytest.skip("tract_querier not found (set TRACT_QUERIER_DIR)")
    name = "_tq_reference_idx"
    package = types.ModuleType(name)
    package.__path__ = []
    modules = {name: package}
    for part in ("aabb", "tract_label_indices"):
        spec = importlib.util.spec_from_file_location(
            f"{name}.{part}", os.path.join(path, f"{part}.py")
        )
        module = importlib.util.module_from_spec(spec)
        modules[spec.name] = module
        sys.modules.update(modules)
        spec.loader.exec_module(module)
    for key in modules:
        sys.modules.pop(key, None)
    return modules[f"{name}.tract_label_indices"]


def load_aabb():
    """tract_querier's ``aabb.BoundingBox`` (numpy only), loaded by path."""
    path = reference_dir()
    if path is None:
        pytest.skip("tract_querier not found (set TRACT_QUERIER_DIR)")
    spec = importlib.util.spec_from_file_location(
        "_tq_reference_aabb", os.path.join(path, "aabb.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def data_file(name):
    path = reference_dir()
    full = path and os.path.join(path, "data", name)
    if not full or not os.path.isfile(full):
        pytest.skip(f"tract_querier data file {name} not found")
    return full


BOX_FIELDS = ("left", "posterior", "inferior", "right", "anterior", "superior")


class Indexing:
    """What ``EvaluateQueries`` reads of a ``TractographySpatialIndexing``:
    the four dictionaries, and for the directional terms the boxes of the
    labels and of the streamlines and the positions of their endpoints."""

    def __init__(self, crossing, endings, label_boxes=None, tract_boxes=None,
                 endpoints_pos=None):
        """crossing: {tract: set of labels}; endings: ({tract: label} for
        the first point, {tract: label} for the last); label_boxes: {label:
        (low xyz, high xyz)}; tract_boxes: (n, 2, 3) low and high xyz of
        each streamline; endpoints_pos: (n, 2, 3) first and last point."""
        self.crossing_tracts_labels = {k: set(v) for k, v in crossing.items()}
        self.crossing_labels_tracts = {}
        for tract, labels in crossing.items():
            for label in labels:
                self.crossing_labels_tracts.setdefault(label, set()).add(tract)
        self.ending_tracts_labels = tuple(dict(e) for e in endings)
        self.ending_labels_tracts = tuple({} for _ in endings)
        for i, e in enumerate(endings):
            for tract, label in e.items():
                self.ending_labels_tracts[i].setdefault(label, set()).add(
                    tract
                )
        box = load_aabb().BoundingBox if label_boxes else None
        self.label_bounding_boxes = {
            label: box(np.r_[lo, hi]) for label, (lo, hi) in
            (label_boxes or {}).items()
        }
        n = len(crossing)
        boxes = np.empty(n, dtype=[(name, float) for name in BOX_FIELDS])
        if tract_boxes is not None:
            flat = np.asarray(tract_boxes, float).reshape(n, 6)
            for i, name in enumerate(BOX_FIELDS):
                boxes[name] = flat[:, i]
        self.tract_bounding_boxes = boxes
        self.tract_endpoints_pos = (
            np.zeros((n, 2, 3)) if endpoints_pos is None
            else np.asarray(endpoints_pos, float)
        )


def evaluate(reference, source, indexing, include_folders=()):
    """{name: (tracts, labels, end0, end1)} of every query of ``source``,
    by the reference."""
    body = reference.queries_preprocess(
        source, include_folders=list(include_folders)
    )
    evaluator = reference.EvaluateQueries(indexing)
    evaluator.visit(ast.Module(body, type_ignores=[]))
    return {
        name: (
            set(info.tracts), set(info.labels),
            set(info.tracts_endpoints[0]), set(info.tracts_endpoints[1]),
        )
        for name, info in evaluator.evaluated_queries_info.items()
    }
