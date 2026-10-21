"""
Check the layers of NeuroLang import only the layers below them: the
expression engine (expressions, walkers, types, logic) needs neither the
relational algebra sets nor pandas, and the Datalog, relational algebra and
probabilistic core needs neither the frontends nor the neuroimaging and
parsing libraries. ``.importlinter`` checks the imports between the modules.
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
    "neurolang.expressions",
    "neurolang.expression_walker",
    "neurolang.expression_pattern_matching",
    "neurolang.type_system",
    "neurolang.typed_symbol_table",
    "neurolang.logic",
    "neurolang.logic.transformations",
    "neurolang.logic.expression_processing",
)

CORE_MODULES = (
    "neurolang.relational_algebra",
    "neurolang.relational_algebra_provenance",
    "neurolang.datalog",
    "neurolang.datalog.chase",
    "neurolang.datalog.range_restriction",
    "neurolang.probabilistic",
    "neurolang.probabilistic.cplogic.program",
    "neurolang.probabilistic.dalvi_suciu_lift",
)


@pytest.mark.parametrize("module", ENGINE_MODULES)
def test_expression_engine_imports_without_the_core(module):
    result = _run_blocking(
        ("pandas", "polars", "nibabel", "lark", "nilearn"),
        f"""
        import {module}
        loaded = sorted(
            m for m in sys.modules
            if m.startswith(("neurolang.datalog", "neurolang.probabilistic",
                             "neurolang.relational_algebra",
                             "neurolang.utils.relational_algebra_set",
                             "neurolang.frontend"))
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
        print(sorted(m for m in sys.modules if m.startswith(
            ("neurolang.frontend", "neurolang.regions", "neurolang.commands")
        )))
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
