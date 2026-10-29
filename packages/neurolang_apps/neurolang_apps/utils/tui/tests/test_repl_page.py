"""Run the tests of the model of the REPL page in the browser with Node."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from neurolang_apps.utils import query_highlight

JS_TESTS = sorted((Path(__file__).parent / "js").glob("*.test.mjs"))

# Lines of queries whose fragments the page's highlighting must give as the
# terminal's does.
HIGHLIGHTED = [
    ("ans(s, p) :- PROB[a(s)] = p, (p > 0.5), ~b(s, 'x')", False),
    ("% the studies (all) of 'pain'", False),
    ("a(s, '50%') % note", False),
    ("q(x) \u2190 r(x) \u2227 \u00acs(x)", False),
    ("naïve(é) :- x == 1.5e-3, 'open", False),
    ("p(x) :- AGGREGATE[x](q(x, y) @ count(y)) = n // r(n)", False),
    ("obtain every Study that mentions 'pain'.", True),
    ("define as Popular every Study that mentions pain.", True),
    ("", False),
]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_the_table_and_history_models_of_the_page_pass_their_tests(tmp_path):
    fixtures = tmp_path / "highlight.json"
    fixtures.write_text(json.dumps({
        "words": query_highlight.word_sets(),
        "lines": [
            {"line": line, "squall": squall,
             "tokens": [list(t) for t in query_highlight.tokens(line, squall)]}
            for line, squall in HIGHLIGHTED
        ],
    }))
    result = subprocess.run(
        ["node", "--test", *map(str, JS_TESTS)], capture_output=True, text=True,
        env={**os.environ, "HIGHLIGHT_FIXTURES": str(fixtures)},
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr


ROOT = Path(__file__).resolve().parents[6]


def test_the_links_of_the_page_point_at_headings_of_the_tutorial():
    """
    The page links its keywords to the sections of the Datalog tutorial; the
    anchors are the ids docutils makes of the headings, so a renamed heading
    shows here rather than in a link that goes nowhere.
    """
    nodes = pytest.importorskip("docutils.nodes")
    lines = (ROOT / "doc" / "tutorial_datalog.rst").read_text().splitlines()
    headings = [
        above for above, line in zip(lines, lines[1:])
        if above.strip() and line and set(line) <= set("=-~^\"'`#*+")
        and len(line) >= len(above.rstrip())
    ]
    ids = {nodes.make_id(heading) for heading in headings}
    docs = (ROOT / "doc" / "_static" / "repl" / "docs.js").read_text()
    anchors = re.findall(r"\$\{DATALOG\}#([\w-]+)", docs)
    assert len(anchors) >= 8
    assert set(anchors) <= ids, sorted(set(anchors) - ids)
