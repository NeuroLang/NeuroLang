"""Tests for the core of the interactive REPL."""

import pandas as pd
import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

from neurolang.utils.tui import InteractiveTuiApp, TuiSettings  # noqa: E402
from neurolang.utils.tui.results import rows_table  # noqa: E402


class TestDfToRichTable:
    def test_basic_table(self):
        df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        table = rows_table(df)
        assert [c.header for c in table.columns] == ["x", "y"]

    def test_empty_dataframe(self):
        df = pd.DataFrame({"x": pd.Series([], dtype=int)})
        table = rows_table(df)
        assert [c.header for c in table.columns] == ["x"]

    def test_multiple_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2.5], "c": ["x"]})
        table = rows_table(df)
        assert [c.header for c in table.columns] == ["a", "b", "c"]


def _console():
    import io

    from rich.console import Console

    return Console(file=io.StringIO(), width=100)


def _output(app):
    return app.console.file.getvalue()


@pytest.fixture
def engine():
    """Return a small engine holding the relation ``P``."""
    from neurolang.frontend import NeurolangPDL

    nl = NeurolangPDL()
    nl.add_tuple_set([(1,), (2,)], name="P")
    return nl


def _app(engine, **settings):
    """Return a REPL on *engine* printing everything to one buffer."""
    console = _console()
    return InteractiveTuiApp(
        engine, TuiSettings(**settings), console=console,
        err_console=console,
    )


@pytest.fixture
def small_app(engine):
    """Return a REPL on a small engine, printing to a string buffer."""
    return _app(engine)


class TestInteractiveTuiApp:
    def test_datalog_prompt(self, engine):
        app = InteractiveTuiApp(engine, TuiSettings(engine_name="neurosynth"))
        assert app._get_prompt() == "nl(neurosynth:DL)> "

    def test_squall_prompt(self, engine):
        app = InteractiveTuiApp(
            engine, TuiSettings(engine_name="destrieux", squall=True)
        )
        assert app._get_prompt() == "nl(destrieux:SQ)> "

    def test_mode_switch(self, small_app):
        small_app._handle_dot_command(".mode squall")
        assert small_app.settings.squall
        small_app._handle_dot_command(".mode datalog")
        assert not small_app.settings.squall


def test_query_result_sorted_with_cli_helpers(engine):
    engine.add_tuple_set([(1,), (3,), (2,)], name="Q")
    app = InteractiveTuiApp(
        engine, TuiSettings(sort_by=[("x", False)]), console=_console()
    )
    app._execute_and_display("ans(x) :- Q(x)")
    assert list(app.results[0].frame["x"]) == [3, 2, 1]
    text = _output(app)
    assert text.index("3") < text.index("2") < text.index("1 ")


@pytest.fixture
def pipe_session(small_app):
    """Yield a function making the REPL session fed with the given keys."""
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    with create_pipe_input() as pipe:
        def make(keys):
            pipe.send_text(keys)
            return small_app._make_session(input=pipe, output=DummyOutput())
        yield make


def test_ctrl_c_at_prompt_keeps_session(small_app, pipe_session):
    session = pipe_session("ans(\x03ans(x) :- P(x)\r\x04")
    small_app.run(session)
    out = _output(small_app)
    assert "Query completed" in out
    assert out.count("Query completed") == 1
    assert "Goodbye" in out


def test_ctrl_c_during_query_keeps_session(
    small_app, pipe_session, monkeypatch
):
    calls = []
    original = small_app._execute

    def interrupted_once(program):
        calls.append(program)
        if len(calls) == 1:
            raise KeyboardInterrupt()
        return original(program)

    monkeypatch.setattr(small_app, "_execute", interrupted_once)
    session = pipe_session("ans(x) :- P(x)\rans(x) :- P(x)\r\x04")
    small_app.run(session)
    out = _output(small_app)
    assert "Query interrupted" in out
    assert len(calls) == 2
    assert "Query completed" in out


@pytest.mark.parametrize("quit_command", [".quit", ".exit"])
def test_quit_commands_end_session(small_app, pipe_session, quit_command):
    session = pipe_session(f"{quit_command}\rans(x) :- P(x)\r")
    small_app.run(session)
    assert "Query completed" not in _output(small_app)
    assert "Goodbye" in _output(small_app)


def test_ctrl_d_ends_session(small_app, pipe_session):
    small_app.run(pipe_session("\x04"))
    assert "Goodbye" in _output(small_app)


def test_parse_error_is_formatted(small_app):
    small_app._execute_and_display("ans(x) :- P(x")
    out = _output(small_app)
    assert "Datalog parse error" in out
    assert "^" in out
    assert "Location: line 1, column 14" in out
    assert "Traceback" not in out


def test_squall_error_is_formatted(engine):
    engine.add_tuple_set([(1,)], name="person")
    app = _app(engine, squall=True)
    app._execute_and_display("obtain every persn.")
    out = _output(app)
    assert "Undefined symbol" in out
    assert "Did you mean: person" in out


def test_debug_shows_traceback(engine):
    app = _app(engine, debug=True)
    app._execute_and_display("ans(x) :- P(x")
    out = _output(app)
    assert "Traceback (most recent call last)" in out
    assert "Datalog parse error" in out


def test_undefined_predicate_warning_is_a_short_note(engine):
    engine.add_tuple_set([(1,)], name="person")
    app = _app(engine)
    app._execute_and_display("ans(x) :- persn(x)")
    out = _output(app)
    assert (
        "note: predicate persn has no facts or rules — "
        "did you mean person?"
    ) in out
    assert ".py" not in out
    assert "warnings.warn" not in out


def _lines(app):
    return [line.strip() for line in _output(app).splitlines()]


def test_sort_command_resorts_last_result(small_app, capsys):
    small_app._execute_and_display("ans(x) :- P(x)")
    small_app._handle_dot_command(".sort x:desc")
    assert small_app.settings.sort_by == [("x", False)]
    assert list(small_app.results[0].frame["x"]) == [2, 1]
    small_app._handle_dot_command(".sort")
    assert "Sort: x:desc" in _output(small_app)
    small_app._handle_dot_command(".sort x:sideways")
    assert "invalid sort direction 'sideways'" in capsys.readouterr().err
    small_app._handle_dot_command(".sort none")
    assert small_app.settings.sort_by == []


def test_format_command(small_app):
    small_app._handle_dot_command(".format csv")
    small_app._execute_and_display("ans(x) :- P(x)")
    lines = _lines(small_app)
    assert lines[lines.index("x"):lines.index("x") + 3] == ["x", "1", "2"]
    small_app._handle_dot_command(".format xml")
    assert "Unknown format 'xml'" in _output(small_app)
    assert small_app.settings.output_format == "csv"


def test_ra_toggle_prints_plans_instead_of_results(small_app, capsys):
    small_app._handle_dot_command(".ra on")
    small_app._execute_and_display("ans(x) :- P(x)")
    assert "── deterministic stratum ──" in capsys.readouterr().out
    assert small_app.results == []
    small_app._handle_dot_command(".ra off")
    small_app._execute_and_display("ans(x) :- P(x)")
    assert "deterministic stratum" not in capsys.readouterr().out
    assert list(small_app.results[0].frame["x"]) == [1, 2]


def test_rewritten_toggle(small_app, capsys):
    small_app._handle_dot_command(".rewritten on")
    small_app._execute_and_display("ans(x) :- P(x)")
    assert "── rewritten program ──" in capsys.readouterr().out
    assert list(small_app.results[0].frame["x"]) == [1, 2]
    small_app._handle_dot_command(".rewritten off")
    assert "Rewritten program: off" in _output(small_app)


@pytest.fixture
def long_app(engine):
    """Return a REPL whose last result has the 45 rows 0..44."""
    engine.add_tuple_set([(i,) for i in range(45)], name="R")
    app = _app(engine)
    app._execute_and_display("ans(x) :- R(x)")
    return app


def _shown_numbers(text):
    import re

    return [int(n) for n in re.findall(r"│\s*(\d+)\s*│", text)]


def test_result_shows_first_page_with_caption(long_app):
    out = _output(long_app)
    assert _shown_numbers(out) == list(range(20))
    assert "45 rows · showing 1–20 · .more for next" in out


def test_more_shows_next_pages(long_app):
    console = long_app.console
    console.file.truncate(0)
    console.file.seek(0)
    long_app._handle_dot_command(".more")
    out = _output(long_app)
    assert _shown_numbers(out) == list(range(20, 40))
    assert "45 rows · showing 21–40 · .more for next" in out
    long_app._handle_dot_command(".more")
    assert "45 rows · showing 41–45 · end" in _output(long_app)
    long_app._handle_dot_command(".more")
    assert "No more rows." in _output(long_app)


def test_head_sets_page_size(long_app):
    long_app._handle_dot_command(".head 5")
    assert long_app.settings.page_size == 5
    assert "45 rows · showing 1–5 · .more for next" in _output(long_app)
    long_app._handle_dot_command(".head many")
    assert "Page size must be a positive integer" in _output(long_app)
    assert long_app.settings.page_size == 5


def test_all_shows_every_row(long_app):
    long_app._handle_dot_command(".all")
    assert set(_shown_numbers(_output(long_app))) == set(range(45))
    assert long_app.results[0].remaining == 0


def test_caption_is_one_line_on_narrow_tables(long_app):
    caption = [
        line for line in _output(long_app).splitlines() if "rows" in line
    ]
    assert caption == ["45 rows · showing 1–20 · .more for next"]


def test_small_result_caption(small_app):
    small_app._execute_and_display("ans(x) :- P(x)")
    assert "2 rows" in _lines(small_app)


@pytest.mark.parametrize("typed, suggestion", [
    (".hlep", ".help"), (".mroe", ".more"), (".quti", ".quit"),
])
def test_unknown_command_suggests_closest(small_app, typed, suggestion):
    small_app._handle_dot_command(typed)
    out = " ".join(_output(small_app).split())
    assert f"Unknown command {typed}. Did you mean {suggestion}?" in out


def test_unknown_command_without_close_match(small_app):
    small_app._handle_dot_command(".zzzzzz")
    out = _output(small_app)
    assert "Unknown command .zzzzzz." in out
    assert "Did you mean" not in out


@pytest.fixture
def peaks_engine(engine):
    """Return the small engine with a 4-column ``peak_reported``."""
    engine.add_tuple_set(
        [(1, 2, 3, 10), (4, 5, 6, 11)], name="peak_reported"
    )
    engine.add_tuple_set([(10,), (11,)], name="study")
    return engine


def test_banner_lists_relations_and_example(peaks_engine):
    app = _app(peaks_engine, engine_name="neurosynth")
    app._print_banner()
    out = _output(app)
    assert "Engine: neurosynth" in out
    assert (
        "Mode: Datalog · .help for help · Tab completes · "
        "Ctrl+D or .quit to exit"
    ) in out
    assert "term_in_study_tfidf(study_id, term, tfidf)" in out
    example = "ans(i, j, k, study_id) :- peak_reported(i, j, k, study_id)"
    assert f"Try: {example}" in out
    app._execute_and_display(example)
    assert list(app.results[0].frame.columns) == ["i", "j", "k", "study_id"]
    assert len(app.results[0].frame) == 2


def test_banner_squall_example(peaks_engine):
    app = _app(peaks_engine, engine_name="neurosynth", squall=True)
    app._print_banner()
    assert "Try: obtain every peak_reported." in _output(app)


def test_help_examples_match_engine_columns(small_app):
    small_app.settings.engine_name = "neurosynth"
    small_app._handle_dot_command(".help")
    out = _output(small_app)
    assert "term_in_study_tfidf(t, w, s)" not in out
    assert "obtain every peak_reported." in out
    assert ".more" in out


def test_squall_obtain_shows_declared_column_names(peaks_engine):
    app = _app(peaks_engine, engine_name="neurosynth", squall=True)
    app._execute_and_display("obtain every peak_reported.")
    assert list(app.results[0].frame.columns) == ["i", "j", "k", "study_id"]
    assert "study_id" in _output(app)


def test_several_obtain_blocks_are_all_kept(peaks_engine):
    peaks_engine.add_tuple_set([(i,) for i in range(30)], name="many")
    app = _app(peaks_engine, engine_name="neurosynth", squall=True)
    app._execute_and_display(
        "obtain every peak_reported as Peaks. obtain every many."
    )
    titles = [block.title for block in app.results]
    assert titles == ["peaks", "obtain_1"]
    assert list(app.results[0].frame.columns) == [
        "i", "j", "k", "study_id"
    ]
    out = _output(app)
    assert "── peaks ──" in out and "── obtain_1 ──" in out
    assert "30 rows · showing 1–20 · .more for next" in out
    app._handle_dot_command(".more")
    assert "30 rows · showing 21–30 · end" in _output(app)
    assert app.results[0].remaining == 0


def test_enter_continues_incomplete_statement(small_app, pipe_session):
    calls = []
    original = small_app._execute

    def recording(program):
        calls.append(program)
        return original(program)

    small_app._execute = recording
    small_app.run(pipe_session("ans(x) :-\r  P(x)\r\x04"))
    assert calls == ["ans(x) :-\n  P(x)"]


def test_enter_on_empty_line_submits_incomplete_statement(
    small_app, pipe_session
):
    small_app.run(pipe_session("ans(x) :- P(x,\r\r\x04"))
    assert "Datalog parse error" in _output(small_app)


def test_batch_runs_statements_and_reports_status(small_app):
    lines = ["ans(x) :-\n", "  P(x)\n", ".format csv\n", "ans(x) :- P(x)\n"]
    assert small_app.run_batch(lines) == 0
    assert "2 rows" in _output(small_app)
    assert "x\n1\n2" in _output(small_app)


@pytest.mark.parametrize("statement", ["ans(x) :- P(x", ".nosuchcommand"])
def test_batch_failure_status(small_app, statement):
    lines = [statement + "\n", "\n", "ans(x) :- P(x)\n"]
    assert small_app.run_batch(lines) == 1
    assert "2 rows" in _output(small_app)


def test_batch_stops_at_quit(small_app):
    assert small_app.run_batch([".quit\n", "ans(x) :- P(\n"]) == 0


def test_banner_shows_relations_without_columns_by_name(engine):
    from neurolang.utils.tui.commands import engine_relations

    relations = engine_relations("neurosynth")
    assert "schaefer" in relations
    assert "schaefer()" not in relations


def test_no_sort_warning_on_empty_result(engine, capsys):
    app = _app(engine, sort_by=[("term", True)])
    app._execute_and_display("ans(x) :- P(x), Q(x)")
    assert "(empty)" in _output(app)
    assert "sort column" not in capsys.readouterr().err
