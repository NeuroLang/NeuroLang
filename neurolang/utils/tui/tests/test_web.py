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
    assert "alice" in reply.html and "bob" in reply.html
    assert "<span" in reply.html  # styled, as on a terminal
    assert not reply.failed


def test_the_session_keeps_its_rules_between_submissions(repl):
    repl.submit("big(x) :- person(x, n) & (n > 4)")
    reply = repl.submit("ans(x) :- big(x)")
    assert "bob" in reply.html and "alice" not in reply.html


def test_a_pasted_block_runs_as_in_the_repl(repl):
    reply = repl.submit(
        "% the people with more than 4\n"
        "big(x) :- person(x, n) & (n > 4)\n"
        "ans(x) :- big(x)\n"
        ".head 1\n"
    )
    assert "bob" in reply.html
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
