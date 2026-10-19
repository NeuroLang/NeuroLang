"""Tests for the REPL driven from a web page."""

from pathlib import Path

import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

from neurolang.frontend import NeurolangPDL  # noqa: E402
from neurolang.utils.tui import nifti  # noqa: E402
from neurolang.utils.tui.web import WebRepl  # noqa: E402


@pytest.fixture
def repl():
    nl = NeurolangPDL()
    nl.add_tuple_set([("alice", 3), ("bob", 5)], name="person")
    return WebRepl(nl, engine_name="neurosynth")


def test_a_query_comes_back_as_html_with_its_rows(repl):
    reply = repl.submit("ans(x, n) :- person(x, n)")
    assert reply.tables[0]["rows"]
    assert "Query completed" in reply.html
    assert "<span" in reply.html  # styled, as on a terminal
    assert not reply.failed


def test_the_session_keeps_its_rules_between_submissions(repl):
    repl.submit("big(x) :- person(x, n) & (n > 4)")
    [table] = repl.submit("ans(x) :- big(x)").tables
    assert table["rows"] == [["bob"]]


def test_a_pasted_block_runs_as_in_the_repl(repl):
    reply = repl.submit(
        "% the people with more than 4\n"
        "big(x) :- person(x, n) & (n > 4)\n"
        "ans(x) :- big(x)\n"
        ".head 1\n"
    )
    assert any(["bob"] in table["rows"] for table in reply.tables)
    assert not reply.failed


def test_an_error_is_reported_and_marks_the_reply_failed(repl):
    reply = repl.submit("ans(x) :- person(x, n")
    assert reply.failed
    assert "error" in reply.html.lower()
    assert not repl.submit("ans(x) :- person(x, n)").failed


def test_replies_do_not_repeat_earlier_output(repl):
    repl.submit("ans(x) :- person(x, n)")
    assert "alice" not in repl.submit(".help").html


def test_the_banner_names_the_engine_and_the_session_command_line():
    nl = NeurolangPDL()
    banner = WebRepl(nl, engine_name="neurosynth", resolution=2).banner()
    assert "neurosynth" in banner
    assert "neurolang-query -i --engine neurosynth --resolution 2" in banner


def test_quit_does_not_end_the_page(repl):
    reply = repl.submit(".quit")
    assert "reload" in reply.html.lower()


def test_view_pages_are_returned_not_opened(repl, monkeypatch, tmp_path):
    monkeypatch.setattr(nifti.tempfile, "tempdir", str(tmp_path))
    repl.submit("ans(x) :- person(x, n)")
    monkeypatch.setattr(
        nifti, "last_result_images", lambda app, save: [("layer", Path("a"))]
    )
    monkeypatch.setattr(
        nifti, "_viewer_page", lambda *args, **kwargs: "<html>page</html>"
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("opened a browser")

    monkeypatch.setattr(nifti.webbrowser, "open", forbidden)
    reply = repl.submit(".view")
    assert reply.viewers == ["<html>page</html>"]
    assert "Open it in a browser" not in reply.html
    assert not list(tmp_path.glob("neurolang_view_*.html"))


class TestTables:
    """Check results come back as data for the page's table viewer."""

    def test_a_result_is_a_table_not_text(self, repl):
        reply = repl.submit("ans(x, n) :- person(x, n)")
        [table] = reply.tables
        assert table["columns"] == ["x", "n"]
        assert sorted(table["rows"]) == [["alice", 3], ["bob", 5]]
        assert table["kinds"] == ["str", "int"]
        assert table["total"] == 2 and not table["truncated"]
        assert "alice" not in reply.html
        assert "2 rows" in reply.html

    def test_the_tables_are_json(self, repl):
        import json

        reply = repl.submit("ans(x, n) :- person(x, n)")
        assert json.loads(json.dumps(reply.tables)) == reply.tables

    def test_missing_values_are_null_and_infinities_are_text(self):
        import json

        import pandas as pd

        from neurolang.utils.tui.web import table_payload

        frame = pd.DataFrame(
            {"f": [1.5, float("nan"), float("inf")], "s": ["a", None, "b"]}
        )
        table = table_payload(frame)
        assert table["kinds"] == ["float", "str"]
        assert table["rows"] == [[1.5, "a"], [None, None], ["Infinity", "b"]]
        json.loads(json.dumps(table))  # a page can parse it

    def test_regions_are_shown_as_in_the_terminal(self):
        import numpy as np

        from neurolang.regions import ExplicitVBR

        region = ExplicitVBR(np.array([[0, 0, 0]]), np.eye(4))
        nl = NeurolangPDL()
        nl.add_tuple_set([("a", region)], name="area")
        [table] = WebRepl(nl).submit("ans(n, r) :- area(n, r)").tables
        assert table["kinds"] == ["str", "other"]
        assert table["rows"] == [["a", "VBR(n_voxels=1, centroid=(0.0, 0.0, 0.0) mm)"]]

    def test_a_long_result_is_cut_and_says_so(self):
        nl = NeurolangPDL()
        nl.add_tuple_set([(i,) for i in range(10)], name="n")
        reply = WebRepl(nl, max_rows=4).submit("ans(i) :- n(i)")
        [table] = reply.tables
        assert len(table["rows"]) == 4
        assert table["total"] == 10 and table["truncated"]
        assert "10 rows" in reply.html and "4" in reply.html

    def test_an_empty_result_has_no_rows(self, repl):
        [table] = repl.submit("ans(x) :- person(x, 99)").tables
        assert table["rows"] == [] and table["total"] == 0

    def test_a_boolean_or_a_rule_has_no_table(self, repl):
        assert repl.submit("ans() :- person(x, n)").tables == []
        assert repl.submit("big(x) :- person(x, n)").tables == []

    def test_sorting_again_gives_the_table_again(self, repl):
        repl.submit("ans(x, n) :- person(x, n)")
        [table] = repl.submit(".sort n:desc").tables
        assert [row[0] for row in table["rows"]] == ["bob", "alice"]

    def test_no_rows_are_left_for_more(self, repl):
        repl.submit("ans(x, n) :- person(x, n)")
        assert repl.submit(".more").tables == []
