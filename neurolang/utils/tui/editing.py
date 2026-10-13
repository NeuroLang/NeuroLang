"""
Line editing of the interactive REPL.

It provides the completion of predicates, their columns and the
grammar (:class:`~.completion.NeuroLangReplCompleter`), with a keyword
fallback for SQUALL, whose Earley parser cannot complete interactively,
syntax highlighting (:class:`NeuroLangLexer`), the bottom toolbar,
suggestions from the history and the vi and emacs editing
modes (``.mode vi|emacs``).
"""

import re
from html import escape
from typing import Callable, Optional

from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.document import Document
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.styles import Style

from neurolang.frontend.datalog.standard_syntax import COMPILED_GRAMMAR
from neurolang.utils.cli import _declared_columns
from neurolang.utils.interactive_parsing import LarkCompleter

from .commands import MODES
from .completion import (
    KEYWORDS as DATALOG_KEYWORDS,
    NeuroLangReplCompleter,
    predicate_catalog,
)

SQUALL_KEYWORDS = list(dict.fromkeys([
    "define", "as", "obtain", "every", "a", "an", "the",
    "in", "where", "that", "with", "without", "if", "then",
    "and", "or", "not", "for", "of", "from", "to", "by",
    "at", "on", "is", "are", "has", "have", "there",
    "probability", "conditioned", "inferred", "choose",
    "peaks", "voxels", "studies", "terms", "experiments",
    "reported", "which", "atlas",
    "all", "each", "some", "no", "any",
    "most", "few", "both", "neither",
]))


_TOKEN = re.compile(
    r"""(?P<string>"[^"]*"?|'[^']*'?)"""
    r"|(?P<number>\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"|(?P<word>[A-Za-z_][A-Za-z0-9_]*)(?P<call>\s*\()?"
    r"|(?P<operator>:~:|:-|::|:=|-:|//|==|<=|>=|!=|\*\*|[^\w\s])"
)

# The operators of the Datalog grammar that are highlighted.
DATALOG_OPERATORS = frozenset({
    ":-", "\N{LEFTWARDS ARROW}", "-:", "\N{RIGHTWARDS ARROW}", "~",
    "\N{NOT SIGN}", "//", "|", "::", ":~:", "^", ":=", "==", "!=", "<",
    "<=", ">", ">=", "=", "&", "\N{LOGICAL AND}", "\N{THERE EXISTS}",
})


class NeuroLangLexer(Lexer):
    """
    Highlight a line of Datalog or SQUALL.

    *is_squall* tells whether the input is SQUALL. In Datalog, the
    grammar's keywords (``PROB``, ``EXISTS``…), its operators (``:-``,
    ``~``, ``//``, comparisons…) and the predicates applied to arguments
    are highlighted; in SQUALL, the :data:`SQUALL_KEYWORDS`. Strings and
    numbers are highlighted in both. The fragments are styled with the
    classes ``keyword``, ``operator``, ``predicate``, ``string`` and
    ``number`` of :data:`STYLE`.
    """

    def __init__(self, is_squall: Optional[Callable[[], bool]] = None):
        """Set up the lexer of the language *is_squall* tells."""
        self._is_squall = is_squall or (lambda: False)

    def lex_document(self, document: Document):
        """Return the function giving the styled fragments of a line."""
        squall = self._is_squall()

        def get_line(lineno: int):
            if lineno >= len(document.lines):
                return []
            return list(self.tokens(document.lines[lineno], squall))

        return get_line

    def tokens(self, line: str, squall: Optional[bool] = None):
        """Yield the ``(style, text)`` fragments of *line*."""
        if squall is None:
            squall = self._is_squall()
        end = 0
        for match in _TOKEN.finditer(line):
            if match.start() > end:
                yield "", line[end:match.start()]
            yield from _styled(match, squall)
            end = match.end()
        if end < len(line):
            yield "", line[end:]


def _word_style(word: str, call: bool, squall: bool) -> str:
    """Return the style of *word*, followed by ``(`` if *call*."""
    if squall:
        return "class:keyword" if word in _SQUALL_WORDS else ""
    if word in DATALOG_KEYWORDS:
        return "class:keyword"
    return "class:predicate" if call else ""


def _styled(match, squall: bool):
    """Yield the styled fragments of a token *match* of :data:`_TOKEN`."""
    kind = match.lastgroup
    if kind == "call":
        word = match.group("word")
        yield _word_style(word, True, squall), word
        yield "", match.group("call")
    elif kind == "word":
        yield _word_style(match.group(0), False, squall), match.group(0)
    elif kind == "operator":
        operator = match.group(0)
        highlighted = not squall and operator in DATALOG_OPERATORS
        yield ("class:operator" if highlighted else ""), operator
    else:
        yield f"class:{kind}", match.group(0)


_SQUALL_WORDS = frozenset(SQUALL_KEYWORDS)

# Readable on light and dark terminals: ANSI colours, no hex.
STYLE = Style.from_dict({
    "keyword": "bold ansimagenta",
    "operator": "bold ansiblue",
    "predicate": "ansiblue",
    "string": "ansigreen",
    "number": "ansicyan",
})


def make_completer(app) -> NeuroLangReplCompleter:
    """
    Return the completer of the REPL *app*.

    It reads the predicates from the engine at each completion, naming
    their columns with those the engine declares and those of the
    relations loaded with ``.load``.
    """
    try:
        lark = LarkCompleter(COMPILED_GRAMMAR)
    except Exception:
        lark = None
    declared = _declared_columns(app.settings.engine_name)
    return NeuroLangReplCompleter(
        lark_completer=lark,
        predicates=lambda: predicate_catalog(
            app._nl, declared, app.loaded_relations
        ),
        squall_keywords=SQUALL_KEYWORDS,
        is_squall=lambda: app.settings.squall,
    )


def toolbar(app) -> HTML:
    """Return the bottom toolbar: engine, mode, sizes and editing mode."""
    mode = "SQ" if app.settings.squall else "DL"
    rows = sum(len(block.frame) for block in app.results)
    result = f"rows:{rows}" if app.results else "—"
    editing = "VI" if app.settings.vi_mode else "EM"
    return HTML(
        f"<b>nl({escape(app.settings.engine_name)}:{mode})</b>  "
        f"{result}  [{editing}]"
    )


def session_options(app) -> dict:
    """Return the prompt session options for line editing in *app*."""
    app.completer = make_completer(app)
    return dict(
        completer=app.completer,
        complete_while_typing=True,
        lexer=NeuroLangLexer(lambda: app.settings.squall),
        auto_suggest=AutoSuggestFromHistory(),
        bottom_toolbar=lambda: toolbar(app),
        style=STYLE,
        vi_mode=app.settings.vi_mode,
        enable_open_in_editor=True,
    )


def _editing_mode(vi: bool):
    """Return the ``.mode`` setter of the vi or emacs editing mode."""
    def set_mode(app) -> str:
        app.settings.vi_mode = vi
        if app.session is not None:
            app.session.editing_mode = (
                EditingMode.VI if vi else EditingMode.EMACS
            )
        return "vi" if vi else "emacs"
    return set_mode


MODES["vi"] = _editing_mode(True)
MODES["emacs"] = _editing_mode(False)
