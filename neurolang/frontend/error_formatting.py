"""
Error formatting and reporting for NeuroLang frontend.

Provides rich, readable error messages for Datalog and SQUALL parse/semantic
errors, with source context, location pointers, and actionable suggestions.
All modifications are confined to the frontend module.
"""

import difflib
import re
import sys
from typing import Optional, Tuple

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


# ── Terminal colour support ────────────────────────────────────────────────

_COLORS = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "reset": "\033[0m",
}

_USE_COLOR = sys.stderr.isatty()


def _c(name: str, text: str) -> str:
    """Wrap *text* in ANSI colour *name* if stderr is a TTY."""
    if not _USE_COLOR:
        return text
    return f"{_COLORS[name]}{text}{_COLORS['reset']}"


# ── Source context helpers ─────────────────────────────────────────────────


def _extract_source_context(
    source_lines: list,
    line: int,
    column: int,
    context_lines: int = 2,
) -> Tuple[list, int]:
    """Extract a window of source lines around *line* (1-based).

    Returns (window_lines, offset) where offset is the 0-based index of
    the error line within window_lines.
    """
    if not source_lines or line is None:
        return [], 0
    start = max(0, line - 1 - context_lines)
    end = min(len(source_lines), line + context_lines)
    window = source_lines[start:end]
    offset = line - 1 - start
    return window, offset


def _format_source_pointer(
    source_line: str,
    column: int,
    marker: str = "^~~~",
) -> str:
    """Build a caret pointer line under a source line at *column* (1-based)."""
    if column is None or column < 1:
        return ""
    col = max(0, column - 1)
    # Clamp to line length
    col = min(col, len(source_line.rstrip("\n")))
    return " " * col + _c("green", marker)


def _format_source_window(
    source_lines: list,
    error_line: int,
    error_column: int,
    context_lines: int = 2,
) -> str:
    """Format a source window with line numbers and a pointer."""
    window, offset = _extract_source_context(
        source_lines, error_line, error_column, context_lines
    )
    if not window:
        return ""

    lines = []
    for i, line_text in enumerate(window):
        lineno = error_line - offset + i
        prefix = _c("dim", f" {lineno:>4} |")
        if i == offset:
            # Error line — highlight it
            prefix = _c("bold", f" {lineno:>4} |")
            lines.append(f"{prefix} {line_text.rstrip()}")
            pointer = _format_source_pointer(line_text, error_column)
            if pointer:
                lines.append(f"      | {pointer}")
        else:
            lines.append(f"{prefix} {line_text.rstrip()}")
    return "\n".join(lines)


# ── Suggestion engine ──────────────────────────────────────────────────────


_COMMON_DATALOG_MISTAKES = [
    (re.compile(r"\)\s*:-\s*$"),
     "The rule body is empty after ':-'. Add at least one predicate, e.g.\n"
     "    ans(x) :- predicate(x)"),
    (re.compile(r"<-(?!>)"),
     "Did you mean ':-' (or '←') instead of '<-'?\n"
     "    Correct: ans(x) :- R(x)"),
    (re.compile(r"=>"),
     "Rules are written head first, with ':-' (or '←'), not "
     "'body => head'.\n"
     "    Correct: ans(x) :- R(x)"),
    (re.compile(r"\bselect\b", re.IGNORECASE),
     "NeuroLang uses Datalog syntax, not SQL. Use 'ans(x) :- predicate(x)'\n"
     "    instead of 'SELECT x FROM predicate'."),
    (re.compile(r"\bwhere\b", re.IGNORECASE),
     "In Datalog, conditions go in the rule body separated by commas, not\n"
     "    with WHERE. Example: ans(x) :- R(x), (x > 0)"),
    (re.compile(r"\bnot\b", re.IGNORECASE),
     "Use '~' for negation in Datalog, not 'not'.\n"
     "    Correct: ans(x) :- R(x), ~S(x)"),
]

_COMMON_SQUALL_MISTAKES = [
    (re.compile(r"\bselect\b", re.IGNORECASE),
     "Use 'obtain' instead of 'select' in SQUALL.\n"
     "    Correct: obtain every X that ..."),
    (re.compile(r"\bget\b", re.IGNORECASE),
     "Use 'obtain' instead of 'get' in SQUALL.\n"
     "    Correct: obtain every X that ..."),
    (re.compile(r"\bfind\b", re.IGNORECASE),
     "Use 'obtain' instead of 'find' in SQUALL.\n"
     "    Correct: obtain every X that ..."),
]

_HINT_TABLES = {
    "Datalog": _COMMON_DATALOG_MISTAKES,
    "SQUALL": _COMMON_SQUALL_MISTAKES,
}


def _suggest_fixes(text: str, table: list) -> list:
    """Return the hints of *table* whose pattern matches *text*."""
    return [msg for pattern, msg in table if pattern.search(text)]


def _suggest_datalog_fixes(text: str) -> list:
    """Return a list of suggestion strings for a Datalog query."""
    return _suggest_fixes(text, _COMMON_DATALOG_MISTAKES)


def _suggest_squall_fixes(text: str) -> list:
    """Return a list of suggestion strings for a SQUALL query."""
    return _suggest_fixes(text, _COMMON_SQUALL_MISTAKES)


def _error_line_text(source_lines: list, line) -> str:
    """Return the source text of 1-based *line*, or '' if unavailable."""
    if line is None or not 1 <= line <= len(source_lines):
        return ""
    return source_lines[line - 1]


def _suggestions_block(source_lines: list, line, language: str) -> list:
    """Return the 'Suggestions:' lines for the hints matching *line*."""
    table = _HINT_TABLES.get(language, _COMMON_DATALOG_MISTAKES)
    suggestions = _suggest_fixes(_error_line_text(source_lines, line), table)
    if not suggestions:
        return []
    return [_c("yellow", "  Suggestions:")] + [f"    {s}" for s in suggestions]


# ── Error formatters ───────────────────────────────────────────────────────


_MAX_EXPECTED = 8


def _compiled_grammar(language: str):
    """Return the compiled Lark grammar of the *language* parser."""
    if language == "SQUALL":
        from .datalog.squall_syntax_lark import compiled_grammar
    else:
        from .datalog.standard_syntax import compiled_grammar
    return compiled_grammar()


def _describe_terminal(grammar, name: str) -> str:
    """Return the literal text of terminal *name*, or *name* itself."""
    try:
        pattern = grammar.get_terminal(name).pattern
    except KeyError:
        return name
    return repr(pattern.value) if pattern.type == "str" else name


def _format_expected(names, language: str) -> str:
    """Render a capped, de-duplicated list of expected terminals."""
    grammar = _compiled_grammar(language)
    shown = sorted({_describe_terminal(grammar, n) for n in names})
    if len(shown) > _MAX_EXPECTED:
        shown = shown[:_MAX_EXPECTED] + ["…"]
    return ", ".join(_c("cyan", e) for e in shown)


def _end_position(source_lines: list) -> Tuple[Optional[int], Optional[int]]:
    """Return the (line, column) just after the last non-blank character."""
    for lineno in range(len(source_lines), 0, -1):
        text = source_lines[lineno - 1].rstrip()
        if text:
            return lineno, len(text) + 1
    return None, None


def _lark_error_details(exc: UnexpectedInput) -> tuple:
    """Return (kind, found, expected terminal names) for a Lark error."""
    if isinstance(exc, UnexpectedToken) and exc.token.type != "$END":
        return "unexpected token", repr(str(exc.token)), exc.expected
    if isinstance(exc, UnexpectedCharacters):
        return "unexpected character", repr(exc.char), exc.allowed
    return "unexpected end of input", None, getattr(exc, "expected", None)


def _format_lark_error(
    exc: UnexpectedInput,
    source_lines: list,
    *,
    language: str = "Datalog",
) -> str:
    """Format a Lark parse error with context, expectations and hints."""
    kind, found, expected = _lark_error_details(exc)
    line, column = exc.line, exc.column
    if found is None or line is None or line < 1:
        line, column = _end_position(source_lines)

    parts = [_c("bold", f"{language} parse error") + f" — {kind}"]
    if line is not None:
        ctx = _format_source_window(source_lines, line, column or 1)
        if ctx:
            parts.append(ctx)
    if found is not None:
        parts.append(
            f"  {_c('red', 'Found:')} {_c('bold', found)}"
            f" at line {line}, column {column}"
        )
    if expected:
        parts.append(
            f"  {_c('yellow', 'Expected:')} one of "
            + _format_expected(expected, language)
        )
    parts += _suggestions_block(source_lines, line, language)
    return "\n".join(parts)


# ── Fuzzy-name matching for undefined symbols ──────────────────────────────


def _suggest_similar_names(bad_name: str, known_names) -> list:
    """Return up to three *known_names* close to *bad_name*."""
    by_lower = {}
    for name in known_names:
        by_lower.setdefault(name.lower(), name)
    matches = difflib.get_close_matches(bad_name.lower(), by_lower, n=3)
    return [by_lower[m] for m in matches]


_SYMBOL_NAME = re.compile(r"Symbol\s+(?:not found\s+)?(\S+)")


def _format_symbol_not_found(
    exc: SymbolNotFoundError,
    source_lines: list,
    known_symbols=(),
) -> str:
    """Format a SymbolNotFoundError with similar-name suggestions."""
    msg = str(exc)
    parts = [_c("bold", "Undefined predicate") + " — symbol not found"]

    line = getattr(exc, "line", None)
    column = getattr(exc, "column", None)
    if line is not None:
        ctx = _format_source_window(source_lines, line, column or 1)
        if ctx:
            parts.append(ctx)
        parts.append(
            f"  {_c('red', 'Location:')} line {line}"
            + (f", column {column}" if column else "")
        )

    parts.append(f"  {_c('red', 'Error:')} {msg}")

    m = _SYMBOL_NAME.search(msg)
    if m:
        pred_name = m.group(1)
        suggestions = _suggest_similar_names(pred_name, known_symbols)
        if suggestions:
            parts.append(
                f"  {_c('yellow', 'Did you mean:')} "
                + ", ".join(_c("bold", s) for s in suggestions)
            )
        else:
            parts.append(
                f"  {_c('yellow', 'Tip:')} Predicate '{pred_name}' has not been "
                f"defined. Use ``nl.add_tuple_set(...)`` to register it, or "
                f"check the spelling."
            )

    parts += _suggestions_block(source_lines, line, "Datalog")
    return "\n".join(parts)


def _format_wrong_arguments(
    exc: WrongArgumentsInPredicateError,
    source_lines: list,
) -> str:
    """Format a WrongArgumentsInPredicateError with context."""
    msg = str(exc)
    parts = [_c("bold", "Arity mismatch") + " — wrong number of arguments"]

    line = getattr(exc, "line", None)
    column = getattr(exc, "column", None)
    if line is not None:
        ctx = _format_source_window(source_lines, line, column or 1)
        if ctx:
            parts.append(ctx)
        parts.append(
            f"  {_c('red', 'Location:')} line {line}"
            + (f", column {column}" if column else "")
        )

    parts.append(f"  {_c('red', 'Error:')} {msg}")
    parts.append(
        f"  {_c('yellow', 'Tip:')} Every predicate call must provide exactly "
        f"the same number of arguments as its definition. "
        f"Check the predicate's definition and count the arguments."
    )

    return "\n".join(parts)


def _format_squall_semantic_error(
    exc: SquallSemanticError,
    source_lines: list,
) -> str:
    """Format a SquallSemanticError with source context."""
    parts = [_c("bold", "SQUALL semantic error")]

    # Use source_lines if provided, otherwise fall back to exc.source_line
    ctx_source = source_lines if source_lines else (
        [exc.source_line] if exc.source_line else []
    )

    if ctx_source and exc.column is not None:
        ctx = _format_source_window(
            ctx_source, exc.line or 1, exc.column, context_lines=1
        )
        if ctx:
            parts.append(ctx)

    if exc.line is not None:
        parts.append(
            f"  {_c('red', 'Location:')} line {exc.line}"
            + (f", column {exc.column}" if exc.column else "")
        )

    parts.append(f"  {_c('red', 'Error:')} {exc.message}")
    return "\n".join(parts)


def _format_generic_error(
    exc: Exception,
    source_lines: list,
) -> str:
    """Format a generic NeuroLangException with context."""
    msg = str(exc)
    parts = [_c("bold", "Error") + f" — {type(exc).__name__}"]

    # Try to extract line info from the exception
    line = getattr(exc, "line", None)
    column = getattr(exc, "column", None)
    if line is not None:
        ctx = _format_source_window(source_lines, line, column or 1)
        if ctx:
            parts.append(ctx)

    parts.append(f"  {_c('red', 'Message:')} {msg}")

    return "\n".join(parts)


# ── Main formatting entry point ────────────────────────────────────────────


def format_error(
    exc: Exception,
    source_text: str = "",
    source_lines: Optional[list] = None,
    *,
    language: str = "Datalog",
    known_symbols=(),
) -> str:
    """
    Format any NeuroLang exception into a readable error message.

    Parameters
    ----------
    exc : Exception
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

    Returns
    -------
    str
        A formatted, human-readable error message.
    """
    if source_lines is None:
        source_lines = source_text.splitlines() if source_text else []

    if isinstance(exc, SquallSemanticError):
        return _format_squall_semantic_error(exc, source_lines)

    if isinstance(exc, SymbolNotFoundError):
        return _format_symbol_not_found(exc, source_lines, known_symbols)

    if isinstance(exc, WrongArgumentsInPredicateError):
        return _format_wrong_arguments(exc, source_lines)

    if isinstance(exc, UnexpectedInput):
        return _format_lark_error(exc, source_lines, language=language)

    if isinstance(exc, NeuroLangException):
        # Parser wrappers: report the original Lark error.
        cause = getattr(exc, "__cause__", None)
        if isinstance(cause, UnexpectedInput):
            return _format_lark_error(cause, source_lines, language=language)
        return _format_generic_error(exc, source_lines)

    # Fallback for non-NeuroLang exceptions
    return f"{_c('bold', 'Error:')} {type(exc).__name__}: {exc}"


# ── CLI integration helper ────────────────────────────────────────────────


def print_formatted_error(
    exc: Exception,
    source_text: str = "",
    source_lines: Optional[list] = None,
    *,
    language: str = "Datalog",
    known_symbols=(),
) -> None:
    """
    Print a formatted error message to stderr.

    Parameters
    ----------
    exc : Exception
        The exception to format and print.
    source_text : str
        The original query/program text.
    source_lines : list, optional
        Pre-split source lines.
    language : str
        "Datalog" or "SQUALL", see :func:`format_error`.
    known_symbols : iterable of str
        Symbol names defined in the engine, see :func:`format_error`.
    """
    msg = format_error(
        exc, source_text, source_lines,
        language=language, known_symbols=known_symbols,
    )
    print(msg, file=sys.stderr, flush=True)
