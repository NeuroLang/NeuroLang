"""
Error formatting and reporting for NeuroLang frontend.

Provides readable error messages for Datalog and SQUALL parse and semantic
errors, with source context, location pointers and actionable suggestions.
"""

import difflib
import os
import re
import sys
from functools import partial
from typing import Callable, Iterable, NamedTuple, Optional, Tuple

from lark.exceptions import (
    UnexpectedCharacters,
    UnexpectedInput,
    UnexpectedToken,
)

from ..exceptions import (
    NeuroLangException,
    SquallSemanticError,
    SymbolNotFoundError,
    WrongArgumentsInPredicateError,
)
from .datalog.squall_syntax_lark import compiled_grammar as _squall_grammar
from .datalog.standard_syntax import compiled_grammar as _datalog_grammar

# ── Terminal colour support ────────────────────────────────────────────────

_COLORS = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "cyan": "\033[96m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "reset": "\033[0m",
}


def use_color(stream=None) -> bool:
    """
    Return whether ANSI colours should be written to *stream*.

    Colours are used only if *stream* (default: ``sys.stderr``) is a
    terminal and the ``NO_COLOR`` environment variable is unset or empty.
    """
    if os.environ.get("NO_COLOR"):
        return False
    isatty = getattr(sys.stderr if stream is None else stream, "isatty", None)
    return bool(isatty and isatty())


def _plain(name: str, text: str) -> str:
    """Return *text* unchanged (colour disabled)."""
    return text


def _ansi(name: str, text: str) -> str:
    """Wrap *text* in the ANSI colour *name*."""
    return f"{_COLORS[name]}{text}{_COLORS['reset']}"


class _Context(NamedTuple):
    """Everything a formatter needs besides the exception itself."""

    source_lines: list
    language: str
    known_symbols: Tuple[str, ...]
    paint: Callable[[str, str], str]

    def label(self, colour: str, label: str, text: str) -> str:
        """Return an indented ``Label: text`` line."""
        return f"  {self.paint(colour, label)} {text}"


# ── Source context helpers ─────────────────────────────────────────────────


def _source_window(
    ctx: _Context, line: int, column: Optional[int], context_lines: int
) -> list:
    """Return numbered source lines around *line* with a caret at *column*."""
    lines = ctx.source_lines
    if not 1 <= line <= len(lines):
        return []
    window = []
    for lineno in range(
        max(1, line - context_lines), min(len(lines), line + context_lines) + 1
    ):
        text = lines[lineno - 1].rstrip()
        style = "bold" if lineno == line else "dim"
        window.append(f"{ctx.paint(style, f' {lineno:>4} |')} {text}")
        if lineno == line and column:
            caret = " " * min(max(column - 1, 0), len(text))
            window.append(f"      | {caret}{ctx.paint('green', '^~~~')}")
    return window


def _location_block(
    ctx: _Context,
    line: Optional[int],
    column: Optional[int],
    context_lines: int = 2,
) -> list:
    """Return the source window and ``Location:`` line of an error."""
    if line is None:
        return []
    where = f"line {line}" + (f", column {column}" if column else "")
    return _source_window(ctx, line, column, context_lines) + [
        ctx.label("red", "Location:", where)
    ]


def _end_position(source_lines: list) -> Tuple[Optional[int], Optional[int]]:
    """Return the (line, column) just after the last non-blank character."""
    for lineno in range(len(source_lines), 0, -1):
        text = source_lines[lineno - 1].rstrip()
        if text:
            return lineno, len(text) + 1
    return None, None


# ── Suggestion engine ──────────────────────────────────────────────────────

_HINTS = {
    "Datalog": (
        (re.compile(r"\)\s*:-\s*$"),
         "The rule body is empty after ':-'. Add at least one predicate, "
         "e.g.\n    ans(x) :- predicate(x)"),
        (re.compile(r"<-(?!>)"),
         "Did you mean ':-' (or '←') instead of '<-'?\n"
         "    Correct: ans(x) :- R(x)"),
        (re.compile(r"=>"),
         "Rules are written head first, with ':-' (or '←'), not "
         "'body => head'.\n    Correct: ans(x) :- R(x)"),
        (re.compile(r"\bselect\b", re.IGNORECASE),
         "NeuroLang uses Datalog syntax, not SQL. Use "
         "'ans(x) :- predicate(x)'\n"
         "    instead of 'SELECT x FROM predicate'."),
        (re.compile(r"\bwhere\b", re.IGNORECASE),
         "In Datalog, conditions go in the rule body separated by commas, "
         "not\n    with WHERE. Example: ans(x) :- R(x), (x > 0)"),
        (re.compile(r"\bnot\b", re.IGNORECASE),
         "Use '~' for negation in Datalog, not 'not'.\n"
         "    Correct: ans(x) :- R(x), ~S(x)"),
    ),
    "SQUALL": tuple(
        (re.compile(rf"\b{word}\b", re.IGNORECASE),
         f"Use 'obtain' instead of '{word}' in SQUALL.\n"
         "    Correct: obtain every X that ...")
        for word in ("select", "get", "find")
    ),
}


def _suggest_fixes(text: str, table: Iterable) -> list:
    """Return the hints of *table* whose pattern matches *text*."""
    return [hint for pattern, hint in table if pattern.search(text)]


def _suggestions_block(ctx: _Context, line: Optional[int]) -> list:
    """Return the ``Suggestions:`` lines for hints matching error *line*."""
    if line is None or not 1 <= line <= len(ctx.source_lines):
        return []
    hints = _suggest_fixes(
        ctx.source_lines[line - 1], _HINTS.get(ctx.language, ())
    )
    if not hints:
        return []
    return [ctx.paint("yellow", "  Suggestions:")] + [
        f"    {hint}" for hint in hints
    ]


def _tip_block(ctx: _Context, tip: Optional[str]) -> list:
    """Return a ``Tip:`` line for *tip*, if any."""
    return [ctx.label("yellow", "Tip:", tip)] if tip else []


# ── Lark parse errors ──────────────────────────────────────────────────────

_MAX_EXPECTED = 8

_QUOTED_WORD = re.compile(r"'\w+'")

_GRAMMARS = {"Datalog": _datalog_grammar, "SQUALL": _squall_grammar}


def _describe_terminal(grammar, name: str) -> str:
    """Return the literal text of terminal *name*, or *name* itself."""
    try:
        pattern = grammar.get_terminal(name).pattern
    except KeyError:
        return name
    return repr(pattern.value) if pattern.type == "str" else name


def _format_expected(ctx: _Context, names: Iterable[str]) -> str:
    """Render a capped, de-duplicated list of expected terminals."""
    grammar = _GRAMMARS.get(ctx.language, _datalog_grammar)()
    # Quoted keywords first: they are the most readable alternatives.
    shown = sorted(
        {_describe_terminal(grammar, n) for n in names},
        key=lambda e: (not _QUOTED_WORD.fullmatch(e), e),
    )
    if len(shown) > _MAX_EXPECTED:
        shown = shown[:_MAX_EXPECTED] + ["…"]
    return ", ".join(ctx.paint("cyan", e) for e in shown)


def _lark_error_details(exc: UnexpectedInput) -> tuple:
    """Return (kind, found text, expected terminal names) of a Lark error."""
    if isinstance(exc, UnexpectedToken) and exc.token.type != "$END":
        return "unexpected token", repr(str(exc.token)), exc.expected
    if isinstance(exc, UnexpectedCharacters):
        return "unexpected character", repr(exc.char), exc.allowed
    return "unexpected end of input", None, getattr(exc, "expected", None)


def _format_lark_error(exc: UnexpectedInput, ctx: _Context) -> str:
    """Format a Lark parse error with context, expectations and hints."""
    kind, found, expected = _lark_error_details(exc)
    line, column = exc.line, exc.column
    if found is None or line is None or line < 1:
        line, column = _end_position(ctx.source_lines)
    parts = [ctx.paint("bold", f"{ctx.language} parse error") + f" — {kind}"]
    parts += _location_block(ctx, line, column)
    if found is not None:
        parts.append(ctx.label("red", "Found:", ctx.paint("bold", found)))
    if expected:
        parts.append(ctx.label(
            "yellow", "Expected:", "one of " + _format_expected(ctx, expected)
        ))
    return "\n".join(parts + _suggestions_block(ctx, line))


# ── NeuroLang exceptions ───────────────────────────────────────────────────


def _did_you_mean(name: str, known_symbols: Iterable[str]) -> list:
    """Return up to three *known_symbols* close to *name* (case-insensitive)."""
    by_lower = {}
    for symbol in known_symbols:
        by_lower.setdefault(symbol.lower(), symbol)
    matches = difflib.get_close_matches(name.lower(), by_lower, n=3)
    return [by_lower[m] for m in matches]


_SYMBOL_NAME = re.compile(r"Symbol\s+(?:not found\s+)?(\S+)")


def _format_symbol_not_found(exc: SymbolNotFoundError, ctx: _Context) -> str:
    """Format a SymbolNotFoundError, suggesting similar engine symbols."""
    msg = str(exc)
    match = _SYMBOL_NAME.search(msg)
    similar = _did_you_mean(match.group(1), ctx.known_symbols) if match else []
    parts = [ctx.paint("bold", "Undefined symbol") + " — symbol not found"]
    parts += _location_block(
        ctx, getattr(exc, "line", None), getattr(exc, "column", None)
    )
    parts.append(ctx.label("red", "Error:", msg))
    if similar:
        parts.append(ctx.label(
            "yellow", "Did you mean:",
            ", ".join(ctx.paint("bold", s) for s in similar),
        ))
    else:
        parts += _tip_block(ctx, exc.suggestion)
    return "\n".join(parts)


def _format_squall_semantic_error(
    exc: SquallSemanticError, ctx: _Context
) -> str:
    """Format a SquallSemanticError with source context."""
    context_lines = 1
    if not ctx.source_lines and exc.source_line and exc.line:
        # Only the offending line is known: show it alone.
        padding = [""] * (exc.line - 1)
        ctx = ctx._replace(source_lines=padding + [exc.source_line])
        context_lines = 0
    parts = [ctx.paint("bold", "SQUALL semantic error")]
    parts += _location_block(ctx, exc.line, exc.column, context_lines)
    parts.append(ctx.label("red", "Error:", exc.message))
    return "\n".join(parts)


def _format_neurolang_error(
    exc: NeuroLangException, ctx: _Context, title: Optional[str] = None
) -> str:
    """Format a NeuroLangException with its location and suggestion."""
    header = title or f"Error — {type(exc).__name__}"
    parts = [ctx.paint("bold", header)]
    parts += _location_block(
        ctx, getattr(exc, "line", None), getattr(exc, "column", None)
    )
    parts.append(ctx.label("red", "Error:", str(exc)))
    return "\n".join(parts + _tip_block(ctx, exc.suggestion))


def _format_unexpected_error(exc: BaseException, ctx: _Context) -> str:
    """Format an exception that is not a NeuroLang or parse error."""
    return (
        ctx.paint("bold", "Internal error")
        + f" — {type(exc).__name__}: {exc}"
    )


# ── Main formatting entry point ────────────────────────────────────────────

#: Formatter of each exception type; the most specific class in the MRO
#: of the exception wins.
_FORMATTERS = {
    SquallSemanticError: _format_squall_semantic_error,
    SymbolNotFoundError: _format_symbol_not_found,
    WrongArgumentsInPredicateError: partial(
        _format_neurolang_error,
        title="Arity mismatch — wrong number of arguments",
    ),
    UnexpectedInput: _format_lark_error,
    NeuroLangException: _format_neurolang_error,
    BaseException: _format_unexpected_error,
}


def _unwrap(exc: BaseException) -> BaseException:
    """Return the Lark error behind a NeuroLang parser exception, if any."""
    if isinstance(exc, NeuroLangException) and isinstance(
        exc.__cause__, UnexpectedInput
    ):
        return exc.__cause__
    return exc


def format_error(
    exc: BaseException,
    source_text: str = "",
    source_lines: Optional[list] = None,
    *,
    language: str = "Datalog",
    known_symbols: Iterable[str] = (),
    color: bool = False,
) -> str:
    """
    Format any NeuroLang exception into a readable error message.

    Parameters
    ----------
    exc : BaseException
        The exception to format.
    source_text : str
        The original query/program text (used for context and hints).
    source_lines : list, optional
        Pre-split source lines. If not provided, derived from source_text.
    language : str
        "Datalog" or "SQUALL": the parser that produced the error. Used
        in error headers and to select the syntax hints.
    known_symbols : iterable of str
        Symbol names defined in the engine, used for "Did you mean"
        suggestions on undefined symbols.
    color : bool
        Whether to use ANSI colours, see :func:`use_color`.

    Returns
    -------
    str
        A formatted, human-readable error message.
    """
    if source_lines is None:
        source_lines = source_text.splitlines()
    ctx = _Context(
        list(source_lines), language, tuple(known_symbols),
        _ansi if color else _plain,
    )
    exc = _unwrap(exc)
    formatter = next(
        _FORMATTERS[cls] for cls in type(exc).__mro__ if cls in _FORMATTERS
    )
    return formatter(exc, ctx)


def print_formatted_error(
    exc: BaseException,
    source_text: str = "",
    source_lines: Optional[list] = None,
    *,
    language: str = "Datalog",
    known_symbols: Iterable[str] = (),
    file=None,
) -> None:
    """
    Print a formatted error message, coloured if *file* is a terminal.

    Parameters
    ----------
    exc : BaseException
        The exception to format and print.
    source_text : str
        The original query/program text.
    source_lines : list, optional
        Pre-split source lines.
    language : str
        "Datalog" or "SQUALL", see :func:`format_error`.
    known_symbols : iterable of str
        Symbol names defined in the engine, see :func:`format_error`.
    file : file-like, optional
        Stream to print to; defaults to ``sys.stderr``.
    """
    stream = sys.stderr if file is None else file
    msg = format_error(
        exc, source_text, source_lines, language=language,
        known_symbols=known_symbols, color=use_color(stream),
    )
    print(msg, file=stream, flush=True)
