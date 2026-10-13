"""Tests for the core of the interactive REPL."""

import pandas as pd
import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

from neurolang.utils.tui import InteractiveTuiApp  # noqa: E402
from neurolang.utils.tui.results import _df_to_rich_table  # noqa: E402


class TestDfToRichTable:
    def test_basic_table(self):
        df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        table = _df_to_rich_table(df, title="Test")
        assert table.title == "Test"
        assert [c.header for c in table.columns] == ["x", "y"]

    def test_empty_dataframe(self):
        df = pd.DataFrame({"x": pd.Series([], dtype=int)})
        table = _df_to_rich_table(df)
        assert [c.header for c in table.columns] == ["x"]

    def test_multiple_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2.5], "c": ["x"]})
        table = _df_to_rich_table(df)
        assert [c.header for c in table.columns] == ["a", "b", "c"]


class TestInteractiveTuiApp:
    def test_datalog_prompt(self, tmp_path):
        app = InteractiveTuiApp(
            engine_name="neurosynth", history_path=str(tmp_path / "h")
        )
        assert app._get_prompt() == "nl(neurosynth:DL)> "

    def test_squall_prompt(self, tmp_path):
        app = InteractiveTuiApp(
            engine_name="destrieux", squall_mode=True,
            history_path=str(tmp_path / "h"),
        )
        assert app._get_prompt() == "nl(destrieux:SQ)> "

    def test_mode_switch(self, tmp_path):
        app = InteractiveTuiApp(history_path=str(tmp_path / "h"))
        app._handle_dot_command(".mode squall")
        assert app.squall_mode
        app._handle_dot_command(".mode datalog")
        assert not app.squall_mode


def test_query_result_sorted_with_cli_helpers(tmp_path):
    import io

    from rich.console import Console

    from neurolang.frontend import NeurolangPDL

    nl = NeurolangPDL()
    nl.add_tuple_set([(1,), (3,), (2,)], name="P")
    out = io.StringIO()
    app = InteractiveTuiApp(
        history_path=str(tmp_path / "h"), sort_specs=["x:desc"],
        console=Console(file=out, width=80),
    )
    app._nl = nl
    app._execute_and_display("ans(x) :- P(x)")
    assert list(app._last_df["x"]) == [3, 2, 1]
    text = out.getvalue()
    assert text.index("3") < text.index("2") < text.index("1 ")


@pytest.fixture
def pipe_session():
    """Yield a function making a prompt session fed with the given keys."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    with create_pipe_input() as pipe:
        def make(keys):
            pipe.send_text(keys)
            return PromptSession(input=pipe, output=DummyOutput())
        yield make


@pytest.fixture
def small_app(tmp_path):
    """Return a REPL on a small engine, printing to a string buffer."""
    import io

    from rich.console import Console

    from neurolang.frontend import NeurolangPDL

    nl = NeurolangPDL()
    nl.add_tuple_set([(1,), (2,)], name="P")
    app = InteractiveTuiApp(
        history_path=str(tmp_path / "h"),
        console=Console(file=io.StringIO(), width=100),
    )
    app._nl = nl
    return app


def _output(app):
    return app.console.file.getvalue()


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
