"""Tests for the transcript of the interactive REPL."""

import io
import re

import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

from rich.console import Console  # noqa: E402

from neurolang.frontend import NeurolangPDL  # noqa: E402
from neurolang_apps.utils.tui import (  # noqa: E402
    InteractiveTuiApp,
    TuiSettings,
)
from neurolang_apps.utils.tui.transcript import (  # noqa: E402
    Entry,
    _fence,
    initial_commands,
)


def _engine():
    """Return a small engine holding the relations ``P`` and ``R``."""
    nl = NeurolangPDL()
    nl.add_tuple_set([(1,), (2,)], name="P")
    nl.add_tuple_set([(i,) for i in range(45)], name="R")
    return nl


def _app(**settings):
    """Return a REPL on a small engine printing to one buffer."""
    console = Console(file=io.StringIO(), width=1000)
    return InteractiveTuiApp(
        _engine(), TuiSettings(**settings), console=console,
        err_console=console,
    )


def _output(app):
    return app.console.file.getvalue()


@pytest.fixture
def scores(tmp_path):
    """Return the path of a CSV file of two scores."""
    path = tmp_path / "scores.csv"
    path.write_text("name,score\nalice,3\nbob,5\n")
    return path


SESSION = [
    ".load {scores}\n",
    "ans(x, s) :-\n",
    "  scores(x, s)\n",
    "ans(x) :- P(x\n",
    ".sort x:desc\n",
    ".idb\n",
    ".help\n",
    ".nosuchcommand\n",
    "ans(x) :- R(x)\n",
    ".more\n",
    ".mode squall\n",
    "obtain every scores.\n",
]


@pytest.fixture
def session(scores):
    """Return a REPL that ran the statements of :data:`SESSION`."""
    app = _app(engine_name="neurosynth")
    lines = [line.format(scores=scores) for line in SESSION]
    assert app.run_batch(lines) == 1
    return app


def _write(app, tmp_path, arg, name="t.md"):
    """Run ``.transcript`` to *name* with *arg*; return the file text."""
    path = tmp_path / name
    app._run_line(f".transcript {path} {arg}")
    return path.read_text(encoding="utf-8")


def test_transcript_records_every_entry(session, scores, tmp_path):
    text = _write(session, tmp_path, "")
    assert text.startswith("# NeuroLang session transcript\n")
    assert "- Engine: neurosynth\n" in text
    assert "- Initial mode: Datalog\n" in text
    assert re.search(r"- Started: \d{4}-\d\d-\d\d \d\d:\d\d:\d\d\n", text)
    assert "- NeuroLang version: " in text
    assert len(re.findall(r"^## \d+ · ", text, re.MULTILINE)) == 11
    assert re.search(r"## 2 · \d\d:\d\d:\d\d · Datalog · \d+\.\d\d s\n", text)
    assert "```datalog\nans(x, s) :-\n  scores(x, s)\n```" in text
    assert "```squall\nobtain every scores.\n```" in text
    assert f"```text\n.load {scores}\n```" in text
    assert f"loaded scores(name, score): 2 rows from {scores}" in text
    assert re.search(r"│ alice\s*│ 3 │", text)
    assert "45 rows · showing 21–40 · .more for next" in text


def test_failed_entries_are_marked(session, tmp_path):
    text = _write(session, tmp_path, "")
    failed = re.findall(r"^## (\d+) · .*✗ failed$", text, re.MULTILINE)
    assert failed == ["3", "7"]
    error = text.split("## 3 · ")[1].split("## 4 · ")[0]
    assert "Datalog parse error" in error
    assert "|              ^" in error
    assert "Unknown command .nosuchcommand" in text


def test_entries_keep_status_and_session_status(session):
    entries = session.transcript.entries
    assert [entry.failed for entry in entries].count(True) == 2
    assert entries[-1].squall and not entries[0].squall
    assert session.failed


def test_output_has_no_ansi_codes(scores):
    console = Console(file=io.StringIO(), width=100, force_terminal=True)
    app = InteractiveTuiApp(_engine(), console=console, err_console=console)
    app._run_line("ans(x) :- P(x")
    app._run_line("ans(x) :- P(x)")
    assert "\x1b[" in _output(app)
    outputs = [entry.output for entry in app.transcript.entries]
    assert all("\x1b" not in output for output in outputs)
    assert "Datalog parse error" in outputs[0]


def test_stderr_and_plain_prints_are_captured(capsys):
    app = InteractiveTuiApp(
        _engine(), console=Console(file=io.StringIO(), width=100)
    )
    app._run_line("ans(x) :- P(x")
    app._run_line(".sort x:sideways")
    app._run_line(".engines")
    error, sort, engines = (e.output for e in app.transcript.entries)
    assert "Datalog parse error" in error and "^" in error
    assert "invalid sort direction 'sideways'" in sort
    assert "neurosynth" in engines
    captured = capsys.readouterr()
    assert "Datalog parse error" in captured.err
    assert "neurosynth" in captured.out


def test_long_output_is_cut(tmp_path):
    app = _app()
    app.transcript.max_lines = 10
    app._run_line("ans(x) :- R(x)")
    app._run_line(".all")
    output = app.transcript.entries[-1].output
    lines = output.splitlines()
    assert len(lines) == 11
    assert re.fullmatch(r"… \(\d+ more lines\)", lines[-1])
    assert "… (" in _write(app, tmp_path, "")


def test_transcript_is_not_recorded(tmp_path):
    app = _app()
    app._run_line("ans(x) :- P(x)")
    _write(app, tmp_path, "")
    text = _write(app, tmp_path, "--force")
    assert ".transcript" not in text
    assert len(app.transcript.entries) == 1
    assert "transcript of 1 entry written to" in _output(app)


def test_inputs_replay_the_session(session, tmp_path):
    inputs = _write(session, tmp_path, "--inputs", "inputs.nl")
    statements = [
        entry.text for entry in session.transcript.entries
        if entry.kind == "statement" and not entry.failed
    ]
    assert ".mode datalog\n.load " in inputs
    for text in statements + [".sort x:desc", ".mode squall"]:
        assert text in inputs
    for excluded in ("ans(x) :- P(x\n", ".idb", ".help", ".nosuch", ".more"):
        assert excluded not in inputs
    assert "inputs of 11 entries written to" in _output(session)

    replay = _app(engine_name="neurosynth")
    with open(tmp_path / "inputs.nl", encoding="utf-8") as lines:
        assert replay.run_batch(lines) == 0
    replayed = [
        entry for entry in replay.transcript.entries
        if entry.kind == "statement"
    ]
    assert [entry.text for entry in replayed] == statements
    original = [
        entry for entry in session.transcript.entries
        if entry.kind == "statement" and not entry.failed
    ]
    assert [_results(e.output) for e in replayed] == [
        _results(e.output) for e in original
    ]


def _results(output):
    """Return the lines of *output* but the elapsed time."""
    return [
        line for line in output.splitlines()
        if not line.startswith("Query completed in")
    ]


def test_inputs_restore_initial_settings():
    settings = TuiSettings(
        squall=True, output_format="csv", sort_by=[("x", False), ("y", True)],
        show_ra=True, show_rewritten=True, logic_notation=True,
    )
    assert initial_commands(settings) == [
        ".mode squall", ".format csv", ".sort x:desc y:asc", ".ra on",
        ".rewritten on", ".logic on",
    ]


@pytest.mark.parametrize("text, replayable", [
    (".head", False), (".head 5", True), (".rewritten", False),
    (".rewritten on", True), (".logic", False), (".logic on", True),
    (".clear", True), (".drop p", True),
    (".all", False), (".datalog", False), (".quit", False),
])
def test_replayable_commands(text, replayable):
    entry = Entry(text, "command", False, None)
    assert entry.replayable() == replayable


def test_initial_program_is_recorded(tmp_path):
    app = _app()
    assert app.run_batch([], initial_program="ans(x) :- P(x)") == 0
    assert app.transcript.entries[0].text == "ans(x) :- P(x)"
    assert "ans(x) :- P(x)\n" in _write(app, tmp_path, "--inputs")


def test_refuses_to_overwrite(tmp_path):
    app = _app()
    path = tmp_path / "t.md"
    path.write_text("keep")
    app._run_line(f".transcript {path}")
    assert f"{path} already exists; add --force to overwrite it." in (
        _output(app)
    )
    assert path.read_text() == "keep"
    assert app.failed
    app._run_line(f".transcript --force {path}")
    assert path.read_text().startswith("# NeuroLang session transcript")


def test_missing_directory(tmp_path):
    app = _app()
    app._run_line(f".transcript {tmp_path / 'nope' / 't.md'}")
    assert f"No such directory: {tmp_path / 'nope'}" in _output(app)
    assert not (tmp_path / "nope").exists()


def test_cannot_write(tmp_path):
    app = _app()
    app._run_line(f".transcript {tmp_path} --force")
    assert f"Cannot write {tmp_path}: " in _output(app)
    assert app.failed


def test_home_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    app = _app()
    app._run_line(".transcript ~/t.md")
    assert (tmp_path / "t.md").exists()
    assert f"written to {tmp_path / 't.md'}" in _output(app)


@pytest.mark.parametrize(
    "arg", ["", "a.md b.md", "--inputs", "'a.md", "--bogus"]
)
def test_usage(arg):
    app = _app()
    app._run_line(f".transcript {arg}")
    assert "Usage: .transcript PATH [--inputs] [--force]" in _output(app)
    assert app.failed


def test_help_lists_transcript():
    app = _app()
    app._run_line(".help")
    assert ".transcript PATH [--inputs] [--force]" in _output(app)


def test_fence_is_longer_than_backticks_in_output(tmp_path):
    assert _fence("a ``` b") == "````"
    app = _app()
    app._run_line(".mode ```")
    text = _write(app, tmp_path, "")
    assert "````text\nUnknown mode '```'" in text


def test_entry_without_output_and_blank_lines(tmp_path):
    app = _app()
    app._run_line("   ")
    app._run_line(".head 5")
    assert [entry.text for entry in app.transcript.entries] == [".head 5"]
    text = _write(app, tmp_path, "")
    assert text.endswith("```text\n.head 5\n```\n")
