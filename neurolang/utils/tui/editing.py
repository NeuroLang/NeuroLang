"""
Line editing of the interactive REPL.

It provides the completion of predicates, their columns and the
Datalog and SQUALL grammars (:class:`~.completion.NeuroLangReplCompleter`),
syntax highlighting (:class:`NeuroLangLexer`), the bottom toolbar,
suggestions from the history and the vi and emacs editing modes
(``.mode vi|emacs``).
"""

import functools
import re
from typing import Callable, Optional

from prompt_toolkit.application.current import get_app
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.document import Document
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.styles import Style

from neurolang.frontend.datalog.standard_syntax import COMPILED_GRAMMAR
from neurolang.utils import engine_registry
from neurolang.utils.cli import _declared_columns
from neurolang.utils.interactive_parsing import LarkCompleter

from .catalog import predicate_catalog
from .commands import MODES, sort_spec_text
from .completion import KEYWORDS as DATALOG_KEYWORDS, NeuroLangReplCompleter
from .squall_completion import SquallGrammarOptions, squall_keywords
from .value_completion import ColumnValues



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
    are highlighted; in SQUALL, the keywords of its grammar
    (:func:`~.squall_completion.squall_keywords`), which its completion
    offers. Strings and numbers are highlighted in both. The fragments
    are styled with the classes ``keyword``, ``operator``, ``predicate``,
    ``string`` and ``number`` of :data:`STYLE`.
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
        return "class:keyword" if word in _squall_words() else ""
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


@functools.lru_cache(maxsize=None)
def _squall_words() -> frozenset:
    """Return the keywords of the SQUALL grammar, as a set."""
    return frozenset(squall_keywords())


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
    signatures, hidden = _engine_sorts(app.settings.engine_name)
    return NeuroLangReplCompleter(
        lark_completer=lark,
        predicates=lambda: predicate_catalog(
            app._nl, declared, app.loaded_relations, signatures
        ),
        squall_keywords=squall_keywords(),
        is_squall=lambda: app.settings.squall,
        squall_grammar=SquallGrammarOptions(),
        hidden=hidden,
        column_values=ColumnValues(app._nl),
    )


def _engine_sorts(engine_name: str):
    """
    Return the arguments of the engine's predicates and the hidden ones.

    The result is ``(signatures, hidden)`` (see
    :func:`~neurolang.utils.engine_registry.get_signatures`), empty for
    no engine.
    """
    if not engine_name:
        return {}, frozenset()
    return (
        engine_registry.get_signatures(engine_name),
        engine_registry.get_hidden_predicates(engine_name),
    )


TOOLBAR_HINTS = (".help", "Tab completes", "Ctrl+D quits")


def toolbar_fields(settings) -> list:
    """
    Return the settings shown in the toolbar.

    They are the engine, the language, the page size, the sort if set,
    the output format if not ``table`` and the editing mode.
    """
    fields = [
        settings.engine_name or "NeuroLang",
        "SQUALL" if settings.squall else "Datalog",
        f"page {settings.page_size}",
    ]
    if settings.sort_by:
        fields.append(f"sort {sort_spec_text(settings.sort_by, ' ')}")
    if settings.output_format != "table":
        fields.append(settings.output_format)
    fields.append("vi" if settings.vi_mode else "emacs")
    return fields


def toolbar_text(settings, width: int = 80) -> str:
    """
    Return the toolbar of the session *settings* fitting in *width*.

    The :data:`TOOLBAR_HINTS` are dropped from the last one until it
    fits, then the text is cut with an ellipsis.
    """
    fields = toolbar_fields(settings)
    hints = list(TOOLBAR_HINTS)
    text = " · ".join(fields + hints)
    while hints and len(text) > width - 1:
        hints.pop()
        text = " · ".join(fields + hints)
    if len(text) > width - 1:
        text = text[:max(width - 2, 0)] + "…"
    return " " + text


def _terminal_width() -> int:
    """Return the width of the running prompt's terminal, or 80."""
    try:
        return get_app().output.get_size().columns
    except Exception:
        return 80


def toolbar(app) -> str:
    """Return the bottom toolbar of the REPL *app*: its settings and keys."""
    return toolbar_text(app.settings, _terminal_width())


def session_options(app) -> dict:
    """Return the prompt session options for line editing in *app*."""
    app.completer = make_completer(app)
    return dict(
        completer=app.completer,
        complete_while_typing=True,
        # The SQUALL grammar can take tens of milliseconds to complete a
        # long sentence: complete without blocking the typing.
        complete_in_thread=True,
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
