"""
Syntax highlighting of Datalog and SQUALL queries.

:func:`tokens` splits a line of a query into fragments of the kinds
``keyword``, ``operator``, ``predicate``, ``string`` and ``number`` (or
``""`` for the rest). In Datalog, the grammar's keywords (``PROB``,
``AGGREGATE``…), its operators (``:-``, ``~``, ``//``, comparisons…) and
the predicates applied to arguments are highlighted; in SQUALL, the
keywords of its grammar. Strings and numbers are highlighted in both.

The REPL styles the fragments as it edits a line, and :func:`highlight`
colours a query with ANSI codes for a terminal.
"""

import functools
import os
import re

_TOKEN = re.compile(
    r"""(?P<string>"[^"]*"?|'[^']*'?)"""
    r"|(?P<number>\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"|(?P<word>[A-Za-z_][A-Za-z0-9_]*)(?P<call>\s*\()?"
    r"|(?P<operator>:~:|:-|::|:=|-:|//|==|<=|>=|!=|\*\*|[^\w\s])"
)

# The keywords of the Datalog grammar.
DATALOG_KEYWORDS = frozenset({
    "AGGREGATE", "EXISTS", "MARG", "PROB", "SUCC", "ans", "exists",
    "False", "True", "lambda", "st",
})

# The operators of the Datalog grammar that are highlighted.
DATALOG_OPERATORS = frozenset({
    ":-", "\N{LEFTWARDS ARROW}", "-:", "\N{RIGHTWARDS ARROW}", "~",
    "\N{NOT SIGN}", "//", "|", "::", ":~:", "^", ":=", "==", "!=", "<",
    "<=", ">", ">=", "=", "&", "\N{LOGICAL AND}", "\N{THERE EXISTS}",
})

# The ANSI colours of the kinds of fragments, readable on light and dark
# terminals, as the REPL shows them.
ANSI = {
    "keyword": "1;35",
    "operator": "1;34",
    "predicate": "34",
    "string": "32",
    "number": "36",
}


@functools.lru_cache(maxsize=None)
def _squall_words() -> frozenset:
    """Return the keywords of the SQUALL grammar, as a set."""
    from neurolang.utils.squall_completion import squall_keywords

    return frozenset(squall_keywords())


def _word_kind(word: str, call: bool, squall: bool) -> str:
    """Return the kind of *word*, followed by ``(`` if *call*."""
    if squall:
        return "keyword" if word in _squall_words() else ""
    if word in DATALOG_KEYWORDS:
        return "keyword"
    return "predicate" if call else ""


def _fragments(match, squall: bool):
    """Yield the ``(kind, text)`` fragments of a token *match*."""
    kind = match.lastgroup
    if kind == "call":
        word = match.group("word")
        yield _word_kind(word, True, squall), word
        yield "", match.group("call")
    elif kind == "word":
        yield _word_kind(match.group(0), False, squall), match.group(0)
    elif kind == "operator":
        operator = match.group(0)
        highlighted = not squall and operator in DATALOG_OPERATORS
        yield ("operator" if highlighted else ""), operator
    else:
        yield kind, match.group(0)


def tokens(line: str, squall: bool = False):
    """Yield the ``(kind, text)`` fragments of a *line* of a query."""
    end = 0
    for match in _TOKEN.finditer(line):
        if match.start() > end:
            yield "", line[end:match.start()]
        yield from _fragments(match, squall)
        end = match.end()
    if end < len(line):
        yield "", line[end:]


def ansi(text: str, language: str = "datalog") -> str:
    """Return the query *text* in *language* with ANSI colours."""
    squall = language == "squall"
    lines = []
    for line in text.split("\n"):
        lines.append("".join(
            f"\033[{ANSI[kind]}m{fragment}\033[0m" if kind else fragment
            for kind, fragment in tokens(line, squall)
        ))
    return "\n".join(lines)


def use_color(stream) -> bool:
    """Whether to colour what is written on *stream*: a terminal."""
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def highlight(text: str, language: str, stream) -> str:
    """Return the query *text* coloured if *stream* is a terminal."""
    return ansi(text, language) if use_color(stream) else text
