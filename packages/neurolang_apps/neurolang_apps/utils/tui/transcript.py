"""
Transcript of a REPL session and the ``.transcript`` command.

Every entry of the session, a statement or a dot-command, is recorded
from the start with the plain text of the output it produced, so that
``.transcript PATH`` can write the session as a Markdown file and
``.transcript PATH --inputs`` the inputs that replay it as a batch
(``neurolang-query -i < PATH``).

The output is captured by teeing the streams it is written to while an
entry runs: the standard output and error, where the CLI helpers and the
error reporter print, and the files of the REPL's consoles. This keeps
the order of the results, notes and errors as they were shown; the ANSI
codes of the terminal are removed.
"""

import re
import shlex
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import neurolang

from .commands import command, sort_spec_text
from .session import _plural

MAX_OUTPUT_LINES = 200

_ANSI = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"
)

# Commands that change the session, replayed with --inputs; those of
# _SETTERS only when they set something (with an argument).
_STATE_COMMANDS = (".mode", ".load", ".drop", ".clear")
_SETTERS = (".sort", ".format", ".ra", ".rewritten", ".head")


def _mode_name(squall: bool) -> str:
    """Return the name of the language: ``SQUALL`` or ``Datalog``."""
    return "SQUALL" if squall else "Datalog"


@dataclass
class Entry:
    """
    A statement or a dot-command run in the session.

    *kind* is ``statement`` or ``command`` and *squall* the mode it ran
    in; *output* is the plain text it printed, cut after
    :data:`MAX_OUTPUT_LINES` lines.
    """

    text: str
    kind: str
    squall: bool
    started: datetime
    elapsed: float = 0.0
    failed: bool = False
    output: str = ""

    def replayable(self) -> bool:
        """Return whether ``--inputs`` writes this entry."""
        if self.failed:
            return False
        if self.kind == "statement":
            return True
        words = self.text.split()
        return words[0].lower() in _STATE_COMMANDS or (
            words[0].lower() in _SETTERS and len(words) > 1
        )


class _Capture:
    """The output of an entry, keeping at most *max_lines* lines."""

    def __init__(self, max_lines: int):
        """Start an empty capture of at most *max_lines* lines."""
        self.max_lines = max_lines
        self.parts = []
        self.lines = 0
        self.dropped = 0

    def write(self, text: str) -> None:
        """Add *text*, only counting its lines once the capture is full."""
        if self.lines < self.max_lines:
            self.parts.append(text)
            self.lines += text.count("\n")
        else:
            self.dropped += text.count("\n")

    def text(self) -> str:
        """Return the plain text captured, with a marker if it was cut."""
        plain = _ANSI.sub("", "".join(self.parts))
        lines = [line.rstrip() for line in plain.splitlines()]
        while lines and not lines[-1]:
            lines.pop()
        more = len(lines) + self.dropped - self.max_lines
        if more > 0:
            lines = lines[:self.max_lines]
            lines.append(f"… ({_plural(more, 'more line')})")
        return "\n".join(lines)


class _Tee:
    """A text stream writing to *stream* and to the current capture."""

    def __init__(self, stream, transcript: "Transcript"):
        """Tee the writes to *stream* into the capture of *transcript*."""
        self._stream = stream
        self._transcript = transcript

    def write(self, text: str) -> int:
        """Write *text* to the stream, capturing it during an entry."""
        capture = self._transcript.capture
        if capture is not None:
            capture.write(text)
        return self._stream.write(text)

    def __getattr__(self, name):
        """Return the attribute *name* of the stream."""
        return getattr(self._stream, name)


class Transcript:
    """
    The entries of a REPL session, recorded from its start.

    The header of the Markdown transcript is taken from the session's
    *settings* when it starts.
    """

    def __init__(self, settings, max_lines: int = MAX_OUTPUT_LINES):
        """Start an empty transcript of a session with *settings*."""
        self.started = datetime.now()
        self.engine_name = settings.engine_name
        self.data_dir = settings.data_dir
        self.command_line = settings.command_line
        self.squall = settings.squall
        self.initial_commands = initial_commands(settings)
        self.max_lines = max_lines
        self.entries: list = []
        self.capture: Optional[_Capture] = None

    def tee(self, stream) -> _Tee:
        """Return *stream* teed into the output of the entries."""
        return _Tee(stream, self)

    @contextmanager
    def record(self, text: str, kind: str, squall: bool):
        """
        Record the entry *text* of *kind* run in the block.

        The entry is yielded so that the block can mark it as failed.
        """
        entry = Entry(text, kind, squall, datetime.now())
        self.capture = _Capture(self.max_lines)
        t0 = time.perf_counter()
        try:
            with redirect_stdout(self.tee(sys.stdout)), \
                    redirect_stderr(self.tee(sys.stderr)):
                yield entry
        finally:
            entry.elapsed = time.perf_counter() - t0
            entry.output = self.capture.text()
            self.capture = None
            self.entries.append(entry)

    def markdown(self) -> str:
        """Return the transcript as a Markdown document."""
        parts = [self._header()]
        parts += [
            entry_markdown(number, entry)
            for number, entry in enumerate(self.entries, 1)
        ]
        return "\n\n".join(parts) + "\n"

    def _header(self) -> str:
        """Return the title and the description of the session."""
        items = [
            ("Engine", self.engine_name or "(none)"),
            ("Data directory", self.data_dir),
            ("Initial mode", _mode_name(self.squall)),
            ("Started", f"{self.started:%Y-%m-%d %H:%M:%S}"),
            ("NeuroLang version", neurolang.__version__),
            ("Command line", self.command_line and f"`{self.command_line}`"),
        ]
        lines = ["# NeuroLang session transcript", ""]
        lines += [f"- {label}: {value}" for label, value in items if value]
        return "\n".join(lines)

    def inputs(self) -> str:
        """
        Return the inputs replaying the session, as a batch script.

        The script starts with the commands restoring the initial
        settings, then has the statements that succeeded and the
        commands changing the session, a statement being followed by an
        empty line so that it ends there when the script is split.
        """
        lines = list(self.initial_commands)
        for entry in self.entries:
            if entry.replayable():
                lines.append(entry.text)
                if entry.kind == "statement":
                    lines.append("")
        return "\n".join(lines) + "\n"


def initial_commands(settings) -> list:
    """Return the dot-commands setting the session's *settings*."""
    commands = [f".mode {_mode_name(settings.squall).lower()}"]
    if settings.output_format != "table":
        commands.append(f".format {settings.output_format}")
    if settings.sort_by:
        commands.append(f".sort {sort_spec_text(settings.sort_by, ' ')}")
    if settings.show_ra:
        commands.append(".ra on")
    if settings.show_rewritten:
        commands.append(".rewritten on")
    return commands


def _fence(text: str) -> str:
    """Return a code fence longer than the backtick runs of *text*."""
    longest = max((len(run) for run in re.findall("`+", text)), default=0)
    return "`" * max(3, longest + 1)


def _code_block(text: str, language: str) -> str:
    """Return *text* as a fenced Markdown code block of *language*."""
    fence = _fence(text)
    return f"{fence}{language}\n{text}\n{fence}"


def entry_markdown(number: int, entry: Entry) -> str:
    """Return the Markdown section of the *number*-th *entry*."""
    title = [str(number), f"{entry.started:%H:%M:%S}"]
    if entry.kind == "statement":
        title += [_mode_name(entry.squall), f"{entry.elapsed:.2f} s"]
        language = "squall" if entry.squall else "datalog"
    else:
        title.append("command")
        language = "text"
    if entry.failed:
        title.append("✗ failed")
    parts = [f"## {' · '.join(title)}", _code_block(entry.text, language)]
    if entry.output:
        parts.append(_code_block(entry.output, "text"))
    return "\n\n".join(parts)


def is_transcript_command(text: str) -> bool:
    """Return whether *text* is the ``.transcript`` command."""
    words = text.split(maxsplit=1)
    return bool(words) and words[0].lower() == ".transcript"


TRANSCRIPT_USAGE = ".transcript PATH [--inputs] [--force]"


def _transcript_arguments(arg: str):
    """
    Return the path and the options of *arg*, or None if malformed.

    The options are the set of the flags ``--inputs`` and ``--force``.
    """
    try:
        words = shlex.split(arg)
    except ValueError:
        return None
    flags = {word for word in words if word in ("--inputs", "--force")}
    paths = [word for word in words if word not in flags]
    if len(paths) != 1 or paths[0].startswith("--"):
        return None
    return Path(paths[0]).expanduser(), flags


def _write_problem(path: Path, force: bool):
    """Return why the transcript cannot be written to *path*, or None."""
    if not path.parent.is_dir():
        return f"No such directory: {path.parent}"
    if path.exists() and not force:
        return f"{path} already exists; add --force to overwrite it."
    return None


def _transcript_failed(app, message: str) -> None:
    """Report that ``.transcript`` failed with *message*."""
    app.failed = True
    app.err_console.print(
        message, style="red", markup=False, soft_wrap=True
    )


@command(
    ".transcript", "Write the session as Markdown, or its inputs as a "
    "script to replay with neurolang-query -i < PATH", TRANSCRIPT_USAGE,
)
def cmd_transcript(app, arg: str) -> None:
    """Write the transcript of the session to the path in *arg*."""
    parsed = _transcript_arguments(arg)
    if parsed is None:
        _transcript_failed(app, f"Usage: {TRANSCRIPT_USAGE}")
        return
    path, flags = parsed
    problem = _write_problem(path, "--force" in flags)
    if problem:
        _transcript_failed(app, problem)
        return
    transcript = app.transcript
    count = len(transcript.entries)
    entries = f"{count} {'entry' if count == 1 else 'entries'}"
    if "--inputs" in flags:
        text = transcript.inputs()
        what = f"inputs of {entries}"
    else:
        text = transcript.markdown()
        what = f"transcript of {entries}"
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        _transcript_failed(app, f"Cannot write {path}: {exc.strerror}")
        return
    app.console.print(
        f"{what} written to {path}", style="green", markup=False,
        soft_wrap=True,
    )
