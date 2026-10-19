"""
Splitting of the REPL input into statements.

A statement is a dot-command, which takes one line, or a Datalog or
SQUALL program, which may span several lines: it ends on the first line
where it is complete (see :func:`is_complete`) or at an empty line. A
``%`` comment line between statements is no statement.
The same rules decide whether Enter runs the input at the prompt and
how a batch read from a pipe is split.
"""

from typing import Callable, Iterable, Iterator

from neurolang.frontend.datalog.standard_syntax import strip_comment

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
    period and a Datalog one, its ``%`` comments left out, unless it
    ends with ``,``, ``:-`` or a logical connective.
    """
    stripped = text.strip()
    if not stripped or stripped.startswith("."):
        return True
    if not squall:
        stripped = "\n".join(
            strip_comment(line) for line in stripped.splitlines()
        ).strip()
    if bracket_depth(stripped) > 0:
        return False
    if squall:
        return stripped.endswith(".")
    return not stripped.endswith(DATALOG_CONTINUATIONS)


def is_comment(text: str) -> bool:
    """Return whether *text* is only ``%`` comment lines."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(lines) and all(line.startswith("%") for line in lines)


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
            yield from _ended(buffer, line.strip())
            buffer = []
        elif buffer or not is_comment(line):
            buffer.append(line)
            if is_complete("\n".join(buffer), squall()):
                yield "\n".join(buffer)
                buffer = []
    yield from _ended(buffer, "")


def _ended(buffer: list, command: str) -> Iterator[str]:
    """Yield the statement of the lines of *buffer*, if any, then the
    dot-command *command*, if any."""
    if buffer:
        yield "\n".join(buffer)
    if command:
        yield command


def split_at_commands(text: str) -> list:
    """
    Return the dot-commands of *text*, a line each, and the programs
    between them, in order.

    A program keeps its lines, blank ones included: it is one submission.
    One of ``%`` comment lines only is left out.
    """
    parts, program = [], []
    for line in text.splitlines():
        if line.lstrip().startswith("."):
            parts += _program_part(program)
            program = []
            parts.append(line.strip())
        else:
            program.append(line)
    return parts + _program_part(program)


def _program_part(lines: list) -> list:
    """Return the program of *lines* in a list, or [] if it is empty or
    only comments."""
    program = "\n".join(lines).strip()
    return [program] if program and not is_comment(program) else []
