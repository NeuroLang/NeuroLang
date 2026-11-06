"""The scaffold imports, and its core does not need NeuroLang."""

import subprocess
import sys


def test_core_imports_without_neurolang():
    code = (
        "import sys, nl_tracts, nl_tracts.vocabulary;"
        "assert nl_tracts.vocabulary.prelude();"
        "bad = [m for m in sys.modules if m.split('.')[0] in "
        "('neurolang', 'nl_core', 'nl_expr', 'lark', 'nilearn')];"
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_frontend_imports():
    from nl_tracts import frontend

    assert "streamline_label" in frontend.RELATION_NAMES
    # the names of a vocabulary of rules, not of stored relations
    assert "traverses" not in frontend.RELATION_NAMES
    assert "endpoint" not in frontend.RELATION_NAMES
