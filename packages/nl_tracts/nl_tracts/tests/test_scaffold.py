"""The scaffold imports, and its core does not need NeuroLang."""

import subprocess
import sys


def test_core_imports_without_neurolang():
    """But its atlases, neurolang.atlases, with what the package neurolang
    imports first (its configuration)."""
    code = (
        "import sys, nl_tracts, nl_tracts.wmql_vocab;"
        "assert nl_tracts.vocabulary_predicates();"
        "bad = [m for m in sys.modules if m.split('.')[0] in "
        "('nl_core', 'lark', 'nilearn') or m.startswith('neurolang.') "
        "and m not in ('neurolang.atlases', 'neurolang._aliases', "
        "'neurolang.config', 'neurolang.neurolang')];"
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_frontend_imports():
    from nl_tracts import frontend

    assert "traverses" in frontend.VOCABULARY_PREDICATES
