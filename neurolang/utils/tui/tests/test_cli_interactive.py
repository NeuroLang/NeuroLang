"""Tests for the ``--interactive`` option of ``neurolang-query``."""

import io
import re
import sys

import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

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


class _Terminal(io.StringIO):
    """A standard input that claims to be a terminal."""

    def isatty(self):
        return True


@pytest.fixture
def keys(monkeypatch):
    """Feed the REPL prompt with the keys given to the returned function."""
    monkeypatch.setattr(sys, "stdin", _Terminal())
    apps = []
    original = tui_app.InteractiveTuiApp._make_session
    with create_pipe_input() as pipe:
        def make_session(self):
            apps.append(self)
            return original(self, input=pipe, output=DummyOutput())

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
    assert "Query completed" in capsys.readouterr().err


def test_interactive_runs_query_argument_first(built, keys, capsys):
    keys("\x04")
    main(["-i", "ans(x) :- P(x)"])
    captured = capsys.readouterr()
    assert "Query completed" in captured.err
    out = captured.out
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


def test_interactive_passes_format_and_sort(built, keys, capsys):
    keys("\x04")
    main(["-i", "--format", "csv", "--sort", "x:desc", "ans(x) :- P(x)"])
    out = capsys.readouterr().out.splitlines()
    assert out[out.index("x"):out.index("x") + 3] == ["x", "2", "1"]


def test_piped_stdin_runs_as_batch(built, monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "stdin", io.StringIO("ans(x) :- P(x)\n.quit\nans(x) :- P(\n")
    )
    main(["-i"])
    captured = capsys.readouterr()
    assert re.search(r"│\s*2\s*│", captured.out)
    assert "Query completed" in captured.err
    assert "parse error" not in captured.err


def test_piped_stdin_failure_exits_with_1(built, monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "stdin", io.StringIO("ans(x) :- Q(x\n\nans(x) :- P(x)\n")
    )
    with pytest.raises(SystemExit) as exc_info:
        main(["-i"])
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Datalog parse error" in captured.err
    assert re.search(r"│\s*2\s*│", captured.out)


def test_piped_stdin_writes_transcript(built, monkeypatch, tmp_path):
    path = tmp_path / "session.md"
    monkeypatch.setattr(
        sys, "stdin", io.StringIO(f"ans(x) :- P(x)\n.transcript {path}\n")
    )
    main(["-i", "-d", str(tmp_path), "--format", "csv"])
    text = path.read_text()
    assert f"- Data directory: {tmp_path.resolve()}\n" in text
    assert (
        f"- Command line: `neurolang-query -i -d {tmp_path} --format csv`"
    ) in text
    assert "```datalog\nans(x) :- P(x)\n```\n\n```text\n" in text
    assert "x\n1\n2\n" in text
