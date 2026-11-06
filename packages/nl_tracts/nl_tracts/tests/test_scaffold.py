"""The scaffold imports, and its core does not need NeuroLang."""

import subprocess
import sys


def test_core_imports_without_neurolang():
    code = (
        "import sys, nl_tracts, nl_tracts.cache;"
        "bad = [m for m in sys.modules if m.split('.')[0] in "
        "('neurolang', 'nl_core', 'nl_expr', 'lark', 'nilearn')];"
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
