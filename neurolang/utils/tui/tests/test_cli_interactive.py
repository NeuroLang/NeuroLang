"""Tests for the ``--interactive`` option of ``neurolang-query``."""

import re

import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

from prompt_toolkit import PromptSession  # noqa: E402
from prompt_toolkit.input import create_pipe_input  # noqa: E402
from prompt_toolkit.output import DummyOutput  # noqa: E402

from neurolang.frontend import NeurolangPDL  # noqa: E402
from neurolang.utils import engine_registry  # noqa: E402
from neurolang.utils.cli import main  # noqa: E402
from neurolang.utils.tui import app as tui_app  # noqa: E402


@pytest.fixture
def built(monkeypatch):
    """Make the CLI build a small engine, recording each build."""
    calls = []
    nl = NeurolangPDL()
    nl.add_tuple_set([(1,), (2,)], name="P")

    def fake_build_engine(name, data_dir, resolution=None):
        calls.append(name)
        return nl

    monkeypatch.setattr(engine_registry, "build_engine", fake_build_engine)
    return calls


@pytest.fixture
def keys(monkeypatch):
    """Feed the REPL prompt with the keys given to the returned function."""
    apps = []
    with create_pipe_input() as pipe:
        def make_session(self):
            apps.append(self)
            return PromptSession(input=pipe, output=DummyOutput())

        monkeypatch.setattr(
            tui_app.InteractiveTuiApp, "_make_session", make_session
        )

        def send(text):
            pipe.send_text(text)
            return apps
        yield send


def test_interactive_builds_engine_once(built, keys, capsys):
    apps = keys("ans(x) :- P(x)\r\x04")
    main(["-i", "-e", "neurosynth", "-d", "/nonexistent"])
    assert built == ["neurosynth"]
    assert len(apps) == 1
    assert apps[0]._nl.symbol_table is not None
    assert "Query completed" in capsys.readouterr().out


def test_interactive_runs_query_argument_first(built, keys, capsys):
    keys("\x04")
    main(["-i", "ans(x) :- P(x)"])
    out = capsys.readouterr().out
    assert "Query completed" in out
    assert re.search(r"│\s*1\s*│", out)
    assert re.search(r"│\s*2\s*│", out)


def test_interactive_with_show_ra(built, keys, capsys):
    keys("\x04")
    main(["-i", "--show-ra", "ans(x) :- P(x)"])
    captured = capsys.readouterr()
    assert "no query provided" not in captured.err
    assert "── deterministic stratum ──" in captured.out


@pytest.mark.parametrize(
    "flag", ["--list-sets", "--list-predicates", "--list-engines"]
)
def test_interactive_rejects_listing_flags(built, flag, capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["-i", flag])
    assert exc_info.value.code == 1
    assert f"{flag} cannot be used with --interactive" in (
        capsys.readouterr().err
    )
    assert built == []
