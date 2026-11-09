"""tract_querier's indexing, as the oracle of the indexing of nl_tracts.

``tract_label_indices.py`` is loaded **by path**: the package's
``__init__`` imports vtk and the tractography readers. Where it is found:
``TRACT_QUERIER_DIR`` (the directory of the ``tract_querier`` package, with
``tract_label_indices.py``), else an installed ``tract_querier``. Not
found: the tests that need it are skipped. It is never a dependency of
nl_tracts.
"""

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
    if path and os.path.isfile(os.path.join(path, "tract_label_indices.py")):
        return path
    return None


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
