"""Utility to enrich NeuroLang exceptions with source context information.

Can be used by error handlers (e.g. in frontend query execution paths,
REST APIs, or CLI tools) to attach query text, engine type, and source
location to exceptions *before* they reach the user.
"""

from typing import Optional, cast

from ..exceptions import NeuroLangException


def enrich_exception(
    exc: Exception,
    query_text: Optional[str] = None,
    engine_type: Optional[str] = None,
    line: Optional[int] = None,
    column: Optional[int] = None,
    source_line: Optional[str] = None,
) -> Exception:
    """Attach source context and query info to an exception, then return it.

    The returned exception is the same object (so ``raise`` preserves the
    original traceback).  If *exc* is a `NeuroLangException` it is mutated
    in place; otherwise a new wrapper may be returned.

    Parameters
    ----------
    exc : Exception
        The exception to enrich.
    query_text : str, optional
        The full query text that caused the error.
    engine_type : str, optional
        A human-readable engine identifier (e.g. ``"datalog"``,
        ``"squall"``).
    line : int, optional
        1-based source line number.
    column : int, optional
        1-based source column number.
    source_line : str, optional
        The relevant source line text.

    Returns
    -------
    Exception
        The enriched exception (same object), suitable for re-raising.

    Examples
    --------
    >>> from neurolang.exceptions import SymbolNotFoundError
    >>> exc = SymbolNotFoundError("Unknown symbol 'foo'")
    >>> enriched = enrich_exception(
    ...     exc, query_text="ans(x) :- foo(x)",
    ...     engine_type="datalog", line=1)
    >>> enriched.line
    1
    >>> isinstance(enriched, SymbolNotFoundError)
    True
    """
    if isinstance(exc, NeuroLangException):
        # Only set attributes that are not already set on the instance.
        # Subclasses such as SquallSemanticError and ParserError already
        # set line/column in their own __init__.
        if line is not None and getattr(exc, "line", None) is None:
            exc.line = line
        if column is not None and getattr(exc, "column", None) is None:
            exc.column = column
        if source_line is not None and getattr(exc, "source_line", None) is None:
            exc.source_line = source_line

    # Attach context as generic attributes for universal access.
    if query_text is not None:
        exc.query_text = query_text  # type: ignore[attr-defined]
    if engine_type is not None:
        exc.engine_type = engine_type  # type: ignore[attr-defined]

    return exc


def format_user_error(exc: Exception) -> str:
    """Format an exception into a user-facing error message string.

    Uses `error_summary()` if available, otherwise falls back to
    ``str(exc)``.

    Parameters
    ----------
    exc : Exception
        The exception to format.

    Returns
    -------
    str
        A human-readable error message.
    """
    if isinstance(exc, NeuroLangException):
        summary = exc.error_summary()
        parts = [summary["short_message"]]
        detail = summary["detail"]
        if detail and detail != parts[0]:
            parts.append(f"\n{detail}")
        suggestion = summary["suggestion"]
        if suggestion:
            parts.append(f"\nSuggestion: {suggestion}")
        return "".join(parts)
    return str(exc)
