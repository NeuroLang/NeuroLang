"""Run the tests of the model of the REPL page in the browser with Node."""

import shutil
import subprocess
from pathlib import Path

import pytest

JS_TESTS = sorted((Path(__file__).parent / "js").glob("*.test.mjs"))


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_the_table_and_history_models_of_the_page_pass_their_tests():
    result = subprocess.run(
        ["node", "--test", *map(str, JS_TESTS)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr
