"""
Check the packages of NeuroLang import only the packages below them: the
expression engine (nl_expr) needs neither the relational algebra sets nor
pandas, and the Datalog, relational algebra and probabilistic core
(nl_core) needs neither the frontends nor the neuroimaging and parsing
libraries. ``.importlinter`` checks the imports between the modules.
"""

import subprocess
import sys
import textwrap

import pytest


def _run_blocking(modules, code):
    """Run ``code`` in a new interpreter in which ``modules`` can't be
    imported."""
    block = "".join(f"sys.modules[{m!r}] = None; " for m in modules)
    return subprocess.run(
        [sys.executable, "-c", "import sys; " + block + textwrap.dedent(code)],
        capture_output=True, text=True,
    )


ENGINE_MODULES = (
    "nl_expr.expressions",
    "nl_expr.expression_walker",
    "nl_expr.expression_pattern_matching",
    "nl_expr.type_system",
    "nl_expr.typed_symbol_table",
    "nl_expr.logic",
    "nl_expr.logic.transformations",
    "nl_expr.logic.expression_processing",
)

CORE_MODULES = (
    "nl_core.relational_algebra",
    "nl_core.relational_algebra_provenance",
    "nl_core.datalog",
    "nl_core.datalog.chase",
    "nl_core.datalog.range_restriction",
    "nl_core.probabilistic",
    "nl_core.probabilistic.cplogic.program",
    "nl_core.probabilistic.dalvi_suciu_lift",
)


@pytest.mark.parametrize("module", ENGINE_MODULES)
def test_expression_engine_imports_without_the_core(module):
    result = _run_blocking(
        ("pandas", "polars", "nibabel", "lark", "nilearn"),
        f"""
        import {module}
        loaded = sorted(
            m for m in sys.modules
            if m.startswith(("nl_core", "neurolang"))
        )
        print(loaded)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_imports_without_the_frontends(module):
    result = _run_blocking(
        ("nibabel", "lark", "nilearn"),
        f"""
        import {module}
        print(sorted(m for m in sys.modules if m.startswith("neurolang")))
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_package_exports_the_frontends_lazily():
    result = _run_blocking(
        (),
        """
        import neurolang
        assert "neurolang.frontend" not in sys.modules
        from neurolang import NeurolangPDL, ExplicitVBR, config, exceptions
        from neurolang.frontend import NeurolangPDL as PDL
        assert NeurolangPDL is PDL
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_the_moved_modules_keep_their_names_under_neurolang():
    result = _run_blocking(
        (),
        """
        from neurolang.datalog.expressions import Fact
        from neurolang.expressions import Symbol
        from neurolang.utils import OrderedSet, RelationalAlgebraSet
        import neurolang.datalog.chase
        import nl_core.datalog.chase
        from nl_core.datalog.expressions import Fact as CoreFact
        from nl_expr.expressions import Symbol as ExprSymbol
        assert Fact is CoreFact and Symbol is ExprSymbol
        assert neurolang.datalog.chase is nl_core.datalog.chase
        modules = sys.modules
        assert modules["neurolang.datalog"] is modules["nl_core.datalog"]
        import neurolang
        from neurolang import config
        import neurolang.config
        assert neurolang.config is config
        assert type(config).__name__ == "NeurolangConfigParser"
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
