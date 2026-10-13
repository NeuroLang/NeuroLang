"""Tests for the session commands of the interactive REPL."""

import io

import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

from rich.console import Console  # noqa: E402

from neurolang.frontend import NeurolangPDL  # noqa: E402
from neurolang.utils.tui import InteractiveTuiApp, TuiSettings  # noqa: E402


def _output(app):
    return app.console.file.getvalue()


@pytest.fixture
def app():
    """Return a REPL whose session defined ``active``."""
    nl = NeurolangPDL()
    nl.add_tuple_set([("alice",), ("bob",)], name="person")
    nl.add_tuple_set([("alice",)], name="plays")
    console = Console(file=io.StringIO(), width=100)
    app = InteractiveTuiApp(
        nl, TuiSettings(engine_name="neurosynth"), console=console,
        err_console=console,
    )
    app._execute_and_display("active(x) :- person(x), plays(x)")
    return app


def _idb_names(app):
    return [s.name for s in app._nl.program_ir.intensional_database()]


def test_idb_lists_session_predicate(app):
    app._handle_dot_command(".idb")
    assert "active" in _output(app)


def test_clear_all_removes_session_idb(app):
    app._handle_dot_command(".clear")
    assert "cleared" in _output(app)
    assert "active" not in _idb_names(app)
    app._execute_and_display("again(x) :- person(x)")
    assert "again" in _idb_names(app)


def test_clear_specific_removes_predicate(app):
    app._handle_dot_command(".clear active")
    assert "IDB predicate 'active' cleared" in _output(app)
    assert "active" not in _idb_names(app)


def test_clear_rejects_engine_predicate(app):
    app._handle_dot_command(".clear person")
    assert "is part of the engine" in _output(app)
    assert "person" in [s.name for s in app._nl.program_ir.symbol_table]


def test_sets_reuses_cli_listing(app, capsys):
    app._handle_dot_command(".sets")
    out = capsys.readouterr().out
    assert "person" in out and "Total RA sets" in out


def test_predicates_reuses_cli_listing(app, capsys):
    app._handle_dot_command(".predicates")
    assert "peak_reported(i, j, k, study_id)" in capsys.readouterr().out


def test_engines(app, capsys):
    app._handle_dot_command(".engines")
    assert "neurosynth" in capsys.readouterr().out


def test_rewritten_shows_last_query(app, capsys):
    app._execute_and_display("ans(x) :- active(x)")
    capsys.readouterr()
    app._handle_dot_command(".rewritten")
    assert "── rewritten program ──" in capsys.readouterr().out
    assert _idb_names(app).count("ans") <= 1


def test_rewritten_without_query(capsys):
    console = Console(file=io.StringIO(), width=100)
    app = InteractiveTuiApp(NeurolangPDL(), console=console)
    app._handle_dot_command(".rewritten")
    assert "No query has been run yet." in _output(app)


def test_datalog_shows_squall_translation(app, capsys):
    app._handle_dot_command(".mode squall")
    app._execute_and_display("obtain every person that plays.")
    capsys.readouterr()
    app._handle_dot_command(".datalog")
    assert "── query (obtain_0) ──" in capsys.readouterr().out


def test_datalog_needs_squall_query(app):
    app._handle_dot_command(".datalog")
    assert "use .rewritten for a Datalog query" in _output(app)
