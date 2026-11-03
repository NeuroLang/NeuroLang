"""
The modules of NeuroLang under the names they had before the split into
packages: ``neurolang.datalog`` is :mod:`nl_core.datalog`,
``neurolang.expressions`` is :mod:`nl_expr.expressions`,
``neurolang.utils.cli`` is :mod:`neurolang_apps.utils.cli`, and so on.

An old name is the *same* module as the new one, not a copy: its classes are
the same classes, ``mock.patch`` on either name patches both, and objects
pickled under an old name load. ``python -m neurolang.utils.cli`` runs the
module of the new name.
"""

import importlib
import importlib.abc
import importlib.util
import sys

# Where the modules of NeuroLang went, by the longest prefix of their name
# relative to ``neurolang``. The other modules stayed in ``neurolang``.
MOVED = {
    "expressions": "nl_expr",
    "expression_walker": "nl_expr",
    "expression_pattern_matching": "nl_expr",
    "type_system": "nl_expr",
    "typed_symbol_table": "nl_expr",
    "exceptions": "nl_expr",
    "logic": "nl_expr",
    "config": "nl_expr",
    "solver": "nl_expr",
    "neurolang_compiler": "nl_expr",
    "utils.orderedset": "nl_expr",
    "utils.various": "nl_expr",
    "utils.testing": "nl_expr",
    "relational_algebra": "nl_core",
    "relational_algebra_provenance": "nl_core",
    "datalog": "nl_core",
    "datalog_chase": "nl_core",
    "probabilistic": "nl_core",
    "utils.relational_algebra_set": "nl_core",
    "utils.cli": "neurolang_apps",
    "utils.tui": "neurolang_apps",
    "utils.llm": "neurolang_apps",
    "utils.engine_registry": "neurolang_apps",
    "utils.engines": "neurolang_apps",
    "utils.text_embeddings": "neurolang_apps",
    "utils.sort_hints": "neurolang_apps",
    "utils.predicate_catalog": "neurolang_apps",
    "utils.query_highlight": "neurolang_apps",
    "utils.squall_completion": "neurolang_apps",
    "utils.squall_sorts": "neurolang_apps",
}

PREFIX = "neurolang."


def new_name(name):
    """The name of the module ``name`` had before the split, or None if it
    did not move."""
    if not name.startswith(PREFIX):
        return None
    relative = name[len(PREFIX):]
    parts = relative.split(".")
    for i in range(len(parts), 0, -1):
        package = MOVED.get(".".join(parts[:i]))
        if package is not None:
            return f"{package}.{relative}"
    return None


class AliasLoader(importlib.abc.Loader):
    """Loads an old name as the module of the new one."""

    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        return importlib.import_module(self.target)

    def exec_module(self, module):
        pass

    def get_code(self, fullname):
        """The code of the new module, for ``python -m`` with an old name."""
        spec = importlib.util.find_spec(self.target)
        return spec.loader.get_code(self.target)


class AliasFinder(importlib.abc.MetaPathFinder):
    """Finds the modules of NeuroLang under the names they had."""

    def find_spec(self, fullname, path=None, target=None):
        target_name = new_name(fullname)
        if target_name is None:
            return None
        try:
            target_spec = importlib.util.find_spec(target_name)
        except ModuleNotFoundError:
            return None
        if target_spec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname, AliasLoader(target_name),
            is_package=target_spec.submodule_search_locations is not None,
        )


def install():
    """Put the finder first in ``sys.meta_path``, once."""
    if not any(isinstance(f, AliasFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, AliasFinder())
