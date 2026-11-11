"""The documentation's REPL examples and its page on the REPL stay true."""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

from neurolang.frontend import NeurolangPDL  # noqa: E402
from neurolang.frontend.datalog.squall_syntax_lark import (  # noqa: E402
    parser as squall_parser,
)
from neurolang_apps.utils.tui.commands import COMMANDS  # noqa: E402
from neurolang_apps.utils.tui.statements import (  # noqa: E402
    split_statements,
)

ROOT = Path(__file__).resolve().parents[6]
DOC = ROOT / "doc"
SHARE_JS = DOC / "_static" / "repl" / "share.js"


def _module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pytestmark = pytest.mark.skipif(
    not (DOC / "repl.rst").exists(), reason="needs the documentation"
)


@pytest.fixture(scope="module")
def repl_links():
    return _module(DOC / "sphinxext" / "repl_links.py")


@pytest.fixture(scope="module")
def blocks():
    runner = _module(DOC / "tools" / "run_repl_blocks.py")
    return [
        (page.name, block)
        for page in sorted(DOC.glob("*.rst"))
        for block in runner.repl_blocks(page.read_text())
    ]


def test_the_repl_page_names_every_command():
    page = (DOC / "repl.rst").read_text()
    missing = [name for name in COMMANDS if f"``{name}" not in page]
    assert not missing, f"doc/repl.rst does not name {missing}"


def test_the_examples_of_the_documentation_parse(blocks):
    assert blocks
    nl = NeurolangPDL()
    for page, (code, mode, _) in blocks:
        # the statements as the REPL splits them, dot-commands apart
        for part in split_statements(code.splitlines(), lambda: mode == "squall"):
            if part.startswith("."):
                continue
            try:
                if mode == "squall":
                    squall_parser(part)
                else:
                    nl.datalog_parser(part)
            except Exception as error:
                pytest.fail(f"{page}: {part!r} does not parse: {error}")


def test_the_lexer_keeps_the_text():
    lexer = _module(DOC / "sphinxext" / "neurolang_lexer.py").NeuroLangLexer()
    text = 'ans(s) :- mentions(s, "pain") % a comment\n\nq(x) :- ~r(x)\n'
    tokens = lexer.get_tokens_unprocessed(text)
    assert "".join(fragment for _, _, fragment in tokens) == text


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_a_link_opens_the_page_with_the_example(repl_links, tmp_path):
    examples = [
        ("ans(s) :- mentions(s, \"pain\")", "datalog"),
        ("q(x) :- r(x) % é, ∧\nans(x) :- q(x)", "datalog"),
        ("obtain every Study that mentions 'pain'.", "squall"),
    ]
    links = [
        repl_links.repl_link(code, mode, "neurosynth")
        for code, mode in examples
    ]
    script = tmp_path / "decode.mjs"
    script.write_text(
        f"import {{ decodeShare }} from {json.dumps(SHARE_JS.as_uri())};\n"
        f"const links = {json.dumps(links)};\n"
        "console.log(JSON.stringify(links.map((link) => decodeShare(link.split('#')[1]))));\n"
    )
    result = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout) == [
        {"code": code, "mode": mode} for code, mode in examples
    ]
    assert all("?engine=neurosynth#" in link for link in links)
