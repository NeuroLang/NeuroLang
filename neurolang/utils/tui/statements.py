"""
Splitting of the REPL input into statements.

A statement is a dot-command, which takes one line, or a Datalog or
SQUALL program, which may span several lines: it ends on the first line
where it is complete (see :func:`is_complete`) or at an empty line.
The same rules decide whether Enter runs the input at the prompt and
how a batch read from a pipe is split.
"""

from typing import Callable, Iterable, Iterator

DATALOG_CONTINUATIONS = (",", ":-", "←", "&", "|", "∧", "∨")
QUOTES = "'\""
OPENING = "([{"
CLOSING = ")]}"


def bracket_depth(text: str) -> int:
    """Return the number of brackets of *text* left open, outside quotes."""
    depth = 0
    quote = None
    for char in text:
        if quote:
            quote = None if char == quote else quote
        elif char in QUOTES:
            quote = char
        elif char in OPENING:
            depth += 1
        elif char in CLOSING:
            depth -= 1
    return depth


def is_complete(text: str, squall: bool) -> bool:
    """
    Return whether *text* is a complete statement.

    Dot-commands are always complete. A program is incomplete while
    brackets are open; a SQUALL program is complete when it ends with a
    period and a Datalog one unless it ends with ``,``, ``:-`` or a
    logical connective.
    """
    stripped = text.strip()
    if not stripped or stripped.startswith("."):
        return True
    if bracket_depth(stripped) > 0:
        return False
    if squall:
        return stripped.endswith(".")
    return not stripped.endswith(DATALOG_CONTINUATIONS)


def split_statements(
    lines: Iterable[str], squall: Callable[[], bool]
) -> Iterator[str]:
    """
    Yield the statements of *lines*.

    *squall* tells whether the input is SQUALL; it is called for each
    line so that a ``.mode`` command run by the consumer of a statement
    applies to the following ones.
    """
    buffer = []
    for line in lines:
        line = line.rstrip("\r\n")
        if not line.strip() or line.lstrip().startswith("."):
            if buffer:
                yield "\n".join(buffer)
                buffer = []
            if line.strip():
                yield line.strip()
            continue
        buffer.append(line)
        if is_complete("\n".join(buffer), squall()):
            yield "\n".join(buffer)
            buffer = []
    if buffer:
        yield "\n".join(buffer)


def split_at_commands(text: str) -> list:
    """
    Return the dot-commands of *text*, a line each, and the programs
    between them, in order.

    A program keeps its lines, blank ones included: it is one submission.
    """
    parts, program = [], []
    for line in text.splitlines():
        if line.lstrip().startswith("."):
            if "\n".join(program).strip():
                parts.append("\n".join(program).strip())
            program = []
            parts.append(line.strip())
        else:
            program.append(line)
    if "\n".join(program).strip():
        parts.append("\n".join(program).strip())
    return parts
