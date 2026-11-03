"""
Translation of Lark parse errors into NeuroLang exceptions.

Shared by the Datalog and SQUALL parsers so that both raise the same
exception types, with 1-based line and column numbers.
"""

from contextlib import contextmanager

from lark.exceptions import LarkError, UnexpectedCharacters, UnexpectedToken

from nl_expr.exceptions import (
    NeuroLangException,
    UnexpectedCharactersError,
    UnexpectedTokenError,
)


@contextmanager
def lark_errors_as_neurolang(message_prefix=""):
    """
    Re-raise Lark errors raised inside the block as NeuroLang exceptions.

    ``UnexpectedToken`` and ``UnexpectedCharacters`` become
    :class:`UnexpectedTokenError` and :class:`UnexpectedCharactersError`;
    any other ``LarkError`` becomes a :class:`NeuroLangException` whose
    message is *message_prefix* followed by the Lark message. The original
    Lark error is kept as ``__cause__``.
    """
    try:
        yield
    except UnexpectedToken as e:
        raise UnexpectedTokenError(
            str(e), line=e.line, column=e.column
        ) from e
    except UnexpectedCharacters as e:
        raise UnexpectedCharactersError(
            str(e), line=e.line, column=e.column
        ) from e
    except LarkError as e:
        raise NeuroLangException(f"{message_prefix}{e}") from e
