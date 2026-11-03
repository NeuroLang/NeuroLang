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


def data_file(name):
    path = reference_dir()
    full = path and os.path.join(path, "data", name)
    if not full or not os.path.isfile(full):
        pytest.skip(f"tract_querier data file {name} not found")
    return full


class Indexing:
    """What ``EvaluateQueries`` reads of a ``TractographySpatialIndexing``:
    the four dictionaries."""

    def __init__(self, crossing, endings):
        """crossing: {tract: set of labels}; endings: ({tract: label} for
        the first point, {tract: label} for the last)."""
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
