"""
Error formatting and reporting for NeuroLang frontend.

Provides rich, readable error messages for Datalog and SQUALL parse/semantic
errors, with source context, location pointers, and actionable suggestions.
All modifications are confined to the frontend module.
"""

import re
import sys
import textwrap
from typing import Optional, Tuple

from lark.exceptions import (
    LarkError,
    UnexpectedCharacters,
    UnexpectedToken,
    VisitError,
)

from ..exceptions import (
    NeuroLangException,
    ParserError,
    SquallSemanticError,
    SymbolNotFoundError,
    UnexpectedCharactersError,
    UnexpectedTokenError,
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
    (re.compile(r"ans\s*\(.*\)\s*:-\s*$", re.IGNORECASE),
     "The rule body is empty after ':-'. Add at least one predicate, e.g.\n"
     "    ans(x) :- predicate(x)"),
    (re.compile(r"ans\s*\(.*\)\s*:-", re.IGNORECASE),
     None),  # valid start, no suggestion needed
    (re.compile(r"\.\s*$"),
     "Datalog rules use ':-' (not '.') as the implication operator.\n"
     "    Correct: ans(x) :- R(x)\n"
     "    Not:     ans(x) :- R(x)."),
    (re.compile(r"\bans\b", re.IGNORECASE),
     None),  # ans is valid
    (re.compile(r"<-[^>]"),
     "Did you mean ':-' (colon-dash) instead of '<-'?\n"
     "    Correct: ans(x) :- R(x)"),
    (re.compile(r"=>"),
     "Did you mean ':-' (colon-dash) instead of '=>'?\n"
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
    (re.compile(r"\btrue\b", re.IGNORECASE),
     None),
    (re.compile(r"\bfalse\b", re.IGNORECASE),
     None),
]

_COMMON_SQUALL_MISTAKES = [
    (re.compile(r"\bobtain\b", re.IGNORECASE),
     None),  # valid keyword
    (re.compile(r"\bdefine\b", re.IGNORECASE),
     None),  # valid keyword
    (re.compile(r"\bfor\s+every\b", re.IGNORECASE),
     None),  # valid
    (re.compile(r"\bfor\s+each\b", re.IGNORECASE),
     "Use 'for every' instead of 'for each' in SQUALL.\n"
     "    Correct: define as R for every X ?x where ..."),
    (re.compile(r"\bselect\b", re.IGNORECASE),
     "Use 'obtain' instead of 'select' in SQUALL.\n"
     "    Correct: obtain every X that ..."),
    (re.compile(r"\bget\b", re.IGNORECASE),
     "Use 'obtain' instead of 'get' in SQUALL.\n"
     "    Correct: obtain every X that ..."),
    (re.compile(r"\bfind\b", re.IGNORECASE),
     "Use 'obtain' instead of 'find' in SQUALL.\n"
     "    Correct: obtain every X that ..."),
    (re.compile(r"\bwhere\b", re.IGNORECASE),
     None),  # valid in SQUALL
    (re.compile(r"\bthat\b", re.IGNORECASE),
     None),  # valid
    (re.compile(r"\bwhich\b", re.IGNORECASE),
     "Use 'that' instead of 'which' in relative clauses.\n"
     "    Correct: every study that reports ..."),
]


def _suggest_datalog_fixes(text: str) -> list:
    """Return a list of suggestion strings for a Datalog query."""
    suggestions = []
    for pattern, msg in _COMMON_DATALOG_MISTAKES:
        if pattern.search(text):
            if msg:
                suggestions.append(msg)
    return suggestions


def _suggest_squall_fixes(text: str) -> list:
    """Return a list of suggestion strings for a SQUALL query."""
    suggestions = []
    for pattern, msg in _COMMON_SQUALL_MISTAKES:
        if pattern.search(text):
            if msg:
                suggestions.append(msg)
    return suggestions


# ── Error formatters ───────────────────────────────────────────────────────


def _format_lark_unexpected_token(
    exc: UnexpectedToken,
    source_lines: list,
    source_text: str,
    *,
    language: str = "Datalog",
) -> str:
    """Format a Lark UnexpectedToken error with context and suggestions."""
    line = getattr(exc, "line", None)
    column = getattr(exc, "column", None)
    token = getattr(exc, "token", None)
    expected = getattr(exc, "expected", None)
    token_value = str(token) if token is not None else "?"
    token_type = token.type if token is not None else "?"

    parts = [_c("bold", f"{language} parse error") + " — unexpected token"]

    # Source context
    if line is not None:
        ctx = _format_source_window(source_lines, line, column or 1)
        if ctx:
            parts.append(ctx)

    # What was found
    if token_value:
        parts.append(
            f"  {_c('red', 'Found:')} {_c('bold', repr(str(token_value)))}"
            f" at line {line}, column {column}"
        )

    # What was expected
    if expected:
        expected_list = sorted(expected, key=lambda x: (len(str(x)), str(x)))
        # Limit to most relevant expectations
        if len(expected_list) > 8:
            expected_list = expected_list[:8] + ["..."]
        parts.append(
            f"  {_c('yellow', 'Expected:')} one of "
            + ", ".join(_c("cyan", repr(e)) for e in expected_list)
        )

    # Suggestions
    suggestions = _suggest_datalog_fixes(source_text)
    if suggestions:
        parts.append(_c("yellow", "  Suggestions:"))
        for s in suggestions:
            parts.append(f"    {s}")

    return "\n".join(parts)


def _format_lark_unexpected_characters(
    exc: UnexpectedCharacters,
    source_lines: list,
    source_text: str,
    *,
    language: str = "Datalog",
) -> str:
    """Format a Lark UnexpectedCharacters error with context."""
    line = getattr(exc, "line", None)
    column = getattr(exc, "column", None)
    char = getattr(exc, "char", None)
    allowed = getattr(exc, "allowed", None)

    parts = [_c("bold", f"{language} parse error") + " — unexpected character"]

    if line is not None:
        ctx = _format_source_window(source_lines, line, column or 1)
        if ctx:
            parts.append(ctx)

    if char:
        parts.append(
            f"  {_c('red', 'Found:')} unexpected character "
            f"{_c('bold', repr(char))} at line {line}, column {column}"
        )

    if allowed:
        parts.append(
            f"  {_c('yellow', 'Allowed:')} "
            + ", ".join(_c("cyan", repr(a)) for a in allowed)
        )

    suggestions = _suggest_datalog_fixes(source_text)
    if suggestions:
        parts.append(_c("yellow", "  Suggestions:"))
        for s in suggestions:
            parts.append(f"    {s}")

    return "\n".join(parts)


# ── Fuzzy-name matching for undefined symbols ──────────────────────────────


def _edit_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(
                curr[j] + 1,        # insert
                prev[j + 1] + 1,    # delete
                prev[j] + cost,     # substitute
            ))
        prev = curr
    return prev[-1]


def _suggest_similar_names(bad_name: str, known_names: set, max_suggestions: int = 3) -> list:
    """Find known names within edit-distance threshold of *bad_name*.

    The threshold scales with name length: 2 for short names (≤5 chars),
    3 for medium (≤10 chars), 4 for longer names.
    """
    if not bad_name:
        return []
    max_distance = 2 if len(bad_name) <= 5 else (3 if len(bad_name) <= 10 else 4)
    scored = sorted(
        ((n, _edit_distance(bad_name.lower(), n.lower())) for n in known_names),
        key=lambda x: x[1],
    )
    return [n for n, d in scored if d <= max_distance][:max_suggestions]


_COMMON_DATALOG_PREDICATES = {
    "ans",
    "true",
    "false",
    # NeuroSynth engine predicates (common ones)
    "term_in_study_tfidf", "term_in_study", "study_in_peak",
    "peak_reported", "region_reported", "study",
    "term", "peak", "region", "study_in_study",
    "term_in_peak_tfidf", "term_in_peak",
    "study_in_peak_tfidf",
    # SQUALL internal predicates
    "Activation", "Voxel", "Study", "Term", "Image", "Peak",
    "Template", "Mask", "Region", "Label", "TissueClass",
    "PeakReported", "TermInStudyTFIDF",
}


def _format_symbol_not_found(
    exc: SymbolNotFoundError,
    source_text: str,
    source_lines: list,
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

    # Extract the predicate name from the error message
    pred_name = None
    for pattern in [
        r"Symbol not found\s+(.+)",
        r"Symbol\s+(.+?)\s+not found",
        r"Symbol\s+(.+?)\s+not",
        r"Predicate\s+(.+?)\s+not found",
        r"'([^']+)'",
    ]:
        m = re.search(pattern, msg, re.IGNORECASE)
        if m:
            pred_name = m.group(1)
            break

    parts.append(f"  {_c('red', 'Error:')} {msg}")

    if pred_name:
        suggestions = _suggest_similar_names(
            pred_name, _COMMON_DATALOG_PREDICATES
        )
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

    # Also run general Datalog suggestions
    datalog_suggestions = _suggest_datalog_fixes(source_text)
    if datalog_suggestions:
        parts.append(_c("yellow", "  Suggestions:"))
        for s in datalog_suggestions:
            parts.append(f"    {s}")

    return "\n".join(parts)


def _format_wrong_arguments(
    exc: WrongArgumentsInPredicateError,
    source_text: str,
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

    suggestions = _suggest_squall_fixes(
        exc.source_line or (ctx_source[0] if ctx_source else "")
    )
    if suggestions:
        parts.append(_c("yellow", "  Suggestions:"))
        for s in suggestions:
            parts.append(f"    {s}")

    return "\n".join(parts)


def _format_generic_error(
    exc: Exception,
    source_text: str,
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


# ── Language detection ──────────────────────────────────────────────────────


_SQUALL_KEYWORDS = {"obtain", "define", "for every", "for each", "that", "which"}


def _looks_like_squall(text: str) -> bool:
    """Heuristic: does *text* look like SQUALL rather than Datalog?"""
    if not text:
        return False
    lower = text.lower()
    # SQUALL queries start with 'obtain', 'define', or contain 'for every'
    for kw in _SQUALL_KEYWORDS:
        if kw in lower:
            return True
    return False


# ── Main formatting entry point ────────────────────────────────────────────


def format_error(
    exc: Exception,
    source_text: str = "",
    source_lines: Optional[list] = None,
    *,
    language: str = "Datalog",
) -> str:
    """Format any NeuroLang exception into a readable error message.

    Parameters
    ----------
    exc : Exception
        The exception to format.
    source_text : str
        The original query/program text (used for suggestions).
    source_lines : list, optional
        Pre-split source lines. If not provided, derived from source_text.
    language : str
        "Datalog" or "SQUALL" — used in error headers and suggestions.

    Returns
    -------
    str
        A formatted, human-readable error message.
    """
    if source_lines is None:
        source_lines = source_text.splitlines() if source_text else []

    # Auto-detect language from source text if not explicitly set
    if language == "Datalog" and _looks_like_squall(source_text):
        language = "SQUALL"

    if isinstance(exc, SquallSemanticError):
        return _format_squall_semantic_error(exc, source_lines)

    if isinstance(exc, SymbolNotFoundError):
        return _format_symbol_not_found(exc, source_text, source_lines)

    if isinstance(exc, WrongArgumentsInPredicateError):
        return _format_wrong_arguments(exc, source_text, source_lines)

    if isinstance(exc, UnexpectedToken):
        return _format_lark_unexpected_token(
            exc, source_lines, source_text, language=language
        )

    if isinstance(exc, UnexpectedCharacters):
        return _format_lark_unexpected_characters(
            exc, source_lines, source_text, language=language
        )

    if isinstance(exc, UnexpectedTokenError):
        # Our wrapper — try to get Lark original
        cause = getattr(exc, "__cause__", None)
        if isinstance(cause, (UnexpectedToken, UnexpectedCharacters)):
            return format_error(cause, source_text, source_lines, language=language)
        return _format_generic_error(exc, source_text, source_lines)

    if isinstance(exc, UnexpectedCharactersError):
        cause = getattr(exc, "__cause__", None)
        if isinstance(cause, (UnexpectedToken, UnexpectedCharacters)):
            return format_error(cause, source_text, source_lines, language=language)
        return _format_generic_error(exc, source_text, source_lines)

    if isinstance(exc, ParserError):
        return _format_generic_error(exc, source_text, source_lines)

    if isinstance(exc, NeuroLangException):
        return _format_generic_error(exc, source_text, source_lines)

    # Fallback for non-NeuroLang exceptions
    return f"{_c('bold', 'Error:')} {exc}"


# ── CLI integration helper ────────────────────────────────────────────────


def print_formatted_error(
    exc: Exception,
    source_text: str = "",
    source_lines: Optional[list] = None,
) -> None:
    """Print a formatted error message to stderr.

    Parameters
    ----------
    exc : Exception
        The exception to format and print.
    source_text : str
        The original query/program text.
    source_lines : list, optional
        Pre-split source lines.
    """
    msg = format_error(exc, source_text, source_lines)
    print(msg, file=sys.stderr, flush=True)


def format_lark_parse_error(
    exc: LarkError,
    source_text: str,
) -> str:
    """Format a Lark parse error (from either Datalog or SQUALL parser).

    Handles VisitError by unwrapping the inner cause.
    """
    if isinstance(exc, VisitError):
        cause = exc.__cause__ if exc.__cause__ is not None else exc.orig_exc
        if isinstance(cause, SquallSemanticError):
            return format_error(cause, source_text)
        return format_error(cause, source_text)

    return format_error(exc, source_text)
