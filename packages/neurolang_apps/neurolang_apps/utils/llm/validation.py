"""
Deterministic checks of the queries an LLM writes.

A query is parsed and dry-run on the engine, without running the chase
(:func:`check_query`): the errors, formatted as for the user, are sent
back to the model so that it can correct its query. A query can also be
run to show the model a few rows of its result (:func:`run_preview`).
Both work in a scope of the engine that is discarded afterwards: the
rules of the query never stay in the session.
"""

import contextlib
import ctypes
import io
import os
import pickle
import re
import signal
import subprocess
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Optional

from neurolang.frontend.error_formatting import format_error
from neurolang_apps.utils.cli import (
    _known_symbol_names,
    _show_squall_datalog,
    result_to_dataframe,
    squall_column_names,
    warning_note,
)
from neurolang_apps.utils.predicate_catalog import predicate_catalog

LANGUAGE_NAMES = {"squall": "SQUALL", "datalog": "Datalog"}
CHILD_POLL_SECONDS = 0.25
_DATALOG_QUERY = re.compile(r"(^|[\s.])ans\s*\(")
_CALL = re.compile(
    r"(?<![\w.:'\"])([A-Za-z_][A-Za-z0-9_]*(?::[A-Za-z_][A-Za-z0-9_-]*)?)"
    r"\s*\("
)
# String literals and IRIs, whose characters are not syntax.
_OPAQUE = re.compile(
    r"'[^'\n]*'|\"[^\"\n]*\"|<[A-Za-z][A-Za-z0-9+.-]*:[^<>\s]*>"
)


class QueryCheckError(Exception):
    """A query that parses but cannot be run as it is."""


@dataclass
class CheckResult:
    """
    The outcome of checking a query.

    *message* is the formatted error when the query is not *ok*.
    *datalog* is the Datalog translation of a SQUALL query.
    """

    ok: bool
    message: str = ""
    datalog: str = ""


def _parse(text: str, language: str):
    """Parse *text* in *language*, returning the parsed program."""
    if language == "squall":
        from neurolang.frontend.datalog.squall_syntax_lark import (
            SquallProgram,
            parser,
        )
        parsed = parser(text)
        if not isinstance(parsed, SquallProgram) or not parsed.queries:
            raise QueryCheckError(
                "The program has no query: end it with a sentence "
                "starting with 'obtain'."
            )
        return parsed
    from neurolang.frontend.datalog.standard_syntax import parser
    parsed = parser(text)
    if not _DATALOG_QUERY.search(text):
        raise QueryCheckError(
            "The program has no query: add a rule whose head is ans(...)."
        )
    return parsed


def _execute(nl, text: str, language: str, dry_run: bool):
    """Execute *text* in a discarded scope, its output being captured."""
    execute = (
        nl.execute_squall_program if language == "squall"
        else nl.execute_datalog_program
    )
    with contextlib.redirect_stdout(io.StringIO()):
        with nl.scope:
            return execute(text, dry_run=dry_run)


class QueryTimeout(Exception):
    """Raised in a query that ran longer than its time limit."""


def _with_timeout(function, timeout):
    """
    Return ``function()``, interrupting it after *timeout* seconds.

    It runs in a worker thread in which :class:`QueryTimeout` is raised
    when the time is up; the query stops as soon as it runs Python code
    again. Without *timeout*, or where threads cannot be started, as in
    WebAssembly, *function* runs as it is.
    """
    if not timeout or sys.platform == "emscripten":
        return function()
    outcome = {}
    done = threading.Event()

    def target():
        try:
            outcome["value"] = function()
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            done.set()

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    if not done.wait(timeout):
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(worker.ident), ctypes.py_object(QueryTimeout)
        )
        done.wait()
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


_UNDEFINED = re.compile(r"Symbol (\S+) used in rule body has no entry")


def _undefined_predicates(caught, known, text: str) -> list:
    """
    Return the notes of the *caught* warnings on undefined predicates.

    Only the predicates written in the query *text* are reported: those
    the engine generates while rewriting it (the magic sets'
    ``magic_p^bf_0``, the rules SQUALL numbers) are defined when it runs.
    """
    written = {word.lower() for word in re.findall(r"[\w:]+", text)}
    notes = []
    for warning in caught:
        message = str(warning.message)
        match = _UNDEFINED.match(message)
        if match and match.group(1).lower() not in written:
            continue
        note = warning_note(message, known)
        if "has no facts or rules" in note:
            notes.append(note.removeprefix("note: "))
    return list(dict.fromkeys(notes))


def _argument_count(text: str, start: int) -> int:
    """Return the number of arguments of the call opened at *start*."""
    depth, count, empty, quote = 0, 1, True, None
    for char in text[start:]:
        if quote:
            quote = None if char == quote else quote
            empty = False
            continue
        if char in "'\"":
            quote = char
        elif char in "([{":
            depth += 1
            if depth == 1:
                continue
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return 0 if empty else count
        elif char == "," and depth == 1:
            count += 1
        if not char.isspace():
            empty = False
    return count


def arity_errors(nl, datalog: str) -> list:
    """
    Return the predicates of the Datalog text with a wrong arity.

    Each is described as ``name has N columns (cols), not M``, from the
    predicates the engine *nl* defines with a known arity.
    """
    catalog = predicate_catalog(nl)
    datalog = _OPAQUE.sub(lambda m: "_" * len(m.group(0)), datalog)
    errors = []
    for match in _CALL.finditer(datalog):
        info = catalog.get(match.group(1))
        if info is None or info.kind == "function" or info.arity is None:
            continue
        used = _argument_count(datalog, match.end() - 1)
        if used != info.arity:
            errors.append(
                f"{info.name} has {info.arity} columns, not {used}: "
                f"use it as {info.signature()}"
            )
    return list(dict.fromkeys(errors))


_AGGREGATE = re.compile(r"AGGREGATE\s*\[([^\]]*)\]\s*\(")


def aggregation_errors(text: str) -> list:
    """
    Return the misuses of AGGREGATE in the Datalog program *text*.

    The aggregated relation must be a single atom (a rule of its own
    otherwise), and the variable aggregated must not be a group key.
    """
    masked = _OPAQUE.sub(lambda m: "_" * len(m.group(0)), text)
    errors = []
    for match in _AGGREGATE.finditer(masked):
        groups = {g.strip() for g in match.group(1).split(",") if g.strip()}
        depth, start = 1, match.end()
        for end in range(start, len(masked)):
            depth += {"(": 1, ")": -1}.get(masked[end], 0)
            if depth == 0:
                break
        body, _, function = masked[start:end].rpartition("@")
        atoms = [a for a in re.split(r",(?![^()]*\))", body) if a.strip()]
        if len(atoms) > 1:
            errors.append(
                "AGGREGATE[...](... @ f(x)) takes a single atom: define the "
                "conjunction in a rule of its own, e.g. "
                "item(g, x) :- p(g, y), q(y, x), and aggregate it: "
                "AGGREGATE[g](item(g, x) @ count(x)) = n"
            )
        arguments = function.split("(", 1)[-1]
        aggregated = set(re.findall(r"[A-Za-z_]\w*", arguments))
        if groups & aggregated:
            errors.append(
                f"AGGREGATE groups by {', '.join(sorted(groups & aggregated))}"
                ", the variable it aggregates: group by the other variables "
                "of the atom, e.g. AGGREGATE[g](item(g, x) @ count(x)) = n"
            )
    return list(dict.fromkeys(errors))


def error_message(nl, exc: BaseException, text: str, language: str) -> str:
    """Return the error *exc* of the query *text* formatted as for users."""
    if isinstance(exc, QueryCheckError):
        return str(exc)
    return format_error(
        exc, text, language=LANGUAGE_NAMES[language],
        known_symbols=_known_symbol_names(nl), color=False,
    ).strip()


def squall_datalog(text: str) -> str:
    """Return the Datalog translation of the SQUALL program *text*."""
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        _show_squall_datalog(text)
    return output.getvalue().strip()


def check_query(nl, text: str, language: str) -> CheckResult:
    """
    Check that the query *text* in *language* can run on the engine *nl*.

    It is parsed, then dry-run: the program is rewritten for the
    engine, which finds the predicates that do not exist or are used
    with the wrong number of arguments, but the chase is not run.
    """
    text = text.strip()
    if not text:
        return CheckResult(False, "The query is empty.")
    try:
        _parse(text, language)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _execute(nl, text, language, dry_run=True)
    except Exception as exc:
        return CheckResult(False, error_message(nl, exc, text, language))
    problems = _undefined_predicates(caught, _known_symbol_names(nl), text)
    datalog = ""
    if language == "squall":
        try:
            datalog = squall_datalog(text)
        except Exception:
            datalog = ""
    problems += arity_errors(nl, datalog or text)
    if language == "datalog":
        problems += aggregation_errors(text)
    if problems:
        return CheckResult(False, "\n".join(problems), datalog)
    return CheckResult(True, datalog=datalog)


@dataclass
class Preview:
    """The rows of a query's result: their number and the first ones."""

    ok: bool
    text: str
    rows: int = 0
    frame: object = None


def _frames(result, column_names: dict) -> list:
    """Return ``(name, frame)`` for each relation of a query *result*."""
    if isinstance(result, dict):
        return [
            (key, result_to_dataframe(value, column_names.get(key)))
            for key, value in result.items()
        ]
    names = next(iter(column_names.values()), None)
    return [("", result_to_dataframe(result, names))]


class QueryTooLarge(Exception):
    """Raised for a query using more memory than its limit."""


def _memory_mb(pid: int) -> float:
    """Return the resident memory of the process *pid*, in MB."""
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return int(out) / 1024 if out else 0.0
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def _in_child(function, timeout, memory_mb):
    """
    Return ``function()``, run in a forked child process.

    The child shares the engine's memory with the parent until it
    writes, so forking is cheap. It is killed when it runs longer than
    *timeout* seconds (QueryTimeout) or uses more than *memory_mb* MB
    (QueryTooLarge), so that a runaway query cannot bring the parent
    down. Without fork, or when the system refuses it (a parent too
    large to copy where the memory is not overcommitted), *function* runs
    in a thread with *timeout* only.
    """
    if sys.platform == "emscripten" or not hasattr(os, "fork"):
        return _with_timeout(function, timeout)
    read, write = os.pipe()
    try:
        with warnings.catch_warnings():
            # A child deadlocked by a lock of another thread is killed at
            # the timeout like any other stuck query.
            warnings.filterwarnings(
                "ignore", message=".*multi-threaded, use of fork",
                category=DeprecationWarning,
            )
            pid = os.fork()
    except OSError:
        os.close(read)
        os.close(write)
        return _with_timeout(function, timeout)
    if pid == 0:  # the child
        os.close(read)
        try:
            payload = ("value", function())
        except BaseException as exc:
            payload = ("error", exc)
        try:
            data = pickle.dumps(payload)
        except Exception:
            data = pickle.dumps(("error", RuntimeError(repr(payload[1]))))
        with os.fdopen(write, "wb") as stream:
            stream.write(data)
        os._exit(0)
    os.close(write)
    received = {}
    reader = threading.Thread(
        target=lambda: received.setdefault(
            "data", os.fdopen(read, "rb").read()
        ),
        daemon=True,
    )
    reader.start()
    start, stop = time.monotonic(), None
    while reader.is_alive():
        reader.join(CHILD_POLL_SECONDS)
        if not reader.is_alive():
            break
        if timeout and time.monotonic() - start > timeout:
            stop = QueryTimeout
        elif memory_mb and _memory_mb(pid) > memory_mb:
            stop = QueryTooLarge
        if stop is not None:
            os.kill(pid, signal.SIGKILL)
            reader.join()
            break
    os.waitpid(pid, 0)
    if stop is not None:
        raise stop()
    if not received.get("data"):
        raise QueryTooLarge()
    kind, value = pickle.loads(received["data"])
    if kind == "error":
        raise value
    return value


def run_preview(
    nl, text: str, language: str, rows: int = 10, engine_name: str = "",
    timeout: Optional[float] = None, memory_mb: Optional[float] = None,
) -> Preview:
    """
    Run the query *text* and return the first *rows* rows of its result.

    The rules it defines are discarded with the scope it runs in. With a
    *timeout* (seconds) or a *memory_mb* limit, the query runs in a child
    process stopped when it exceeds them.
    """
    if not (timeout or memory_mb):
        return _preview(nl, text, language, rows, engine_name)
    try:
        return _in_child(
            lambda: _preview(nl, text, language, rows, engine_name),
            timeout, memory_mb,
        )
    except QueryTimeout:
        return Preview(
            False,
            f"The query ran for more than {timeout:.0f} s and was stopped: "
            "restrict it with constants or joins before running it again.",
        )
    except QueryTooLarge:
        return Preview(
            False,
            "The query was stopped: it used too much memory, e.g. joining "
            "large relations without a shared variable. Restrict it, e.g. "
            "with the class of its entities, before running it again.",
        )


def _preview(nl, text, language, rows, engine_name) -> Preview:
    """Run the query *text* and return its preview (see run_preview)."""
    try:
        result = _execute(nl, text.strip(), language, dry_run=False)
    except Exception as exc:
        return Preview(False, error_message(nl, exc, text, language))
    if result is None:
        return Preview(True, "(no result)")
    if isinstance(result, bool):
        return Preview(True, "true" if result else "false", int(result))
    names = (
        squall_column_names(engine_name, text) if language == "squall"
        else {}
    )
    parts, total, first = [], 0, None
    for key, frame in _frames(result, names):
        total += len(frame)
        first = frame if first is None else first
        title = f"{key}: " if key else ""
        if frame.empty:
            parts.append(f"{title}0 rows")
        else:
            parts.append(
                f"{title}{len(frame)} rows; columns "
                f"{', '.join(map(str, frame.columns))}\n"
                + frame.head(rows).to_string(index=False)
            )
    return Preview(True, "\n".join(parts), total, first)
