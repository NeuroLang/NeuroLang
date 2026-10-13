"""
Line editing of the interactive REPL.

It provides the completion of predicates, their columns and the
grammar (:class:`~.completion.NeuroLangReplCompleter`), with a keyword
fallback for SQUALL, whose Earley parser cannot complete interactively,
syntax highlighting (:class:`NeuroLangLexer`), the bottom toolbar,
suggestions from the history and the vi and emacs editing
modes (``.mode vi|emacs``).
"""

from html import escape

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
from .completion import NeuroLangReplCompleter, predicate_catalog

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


OPERATOR_CHARS = frozenset(":-+~¬(),.=!|&")


class NeuroLangLexer(Lexer):
    """Highlight the keywords, operators and capitalised words of a line."""

    KEYWORDS = frozenset({
        "ans", "define", "as", "obtain", "every", "a", "an", "the",
        "where", "that", "with", "without", "if", "then",
        "and", "or", "not", "for", "of", "from", "to", "by",
        "exists", "EXISTS", "MARG", "PROB", "SUCC",
        "lambda", "True", "False",
    })

    OPERATORS = frozenset({":-", ":-~", "::", ":=", "->", "||"})

    def lex_document(self, document: Document):
        """Return the function giving the styled fragments of a line."""
        def get_line(lineno: int):
            if lineno >= len(document.lines):
                return []
            return list(self.tokens(document.lines[lineno]))

        return get_line

    def tokens(self, line: str):
        """Yield the ``(style, text)`` fragments of *line*."""
        i = 0
        while i < len(line):
            if line[i].isalnum() or line[i] == "_":
                j = _run_end(line, i, lambda c: c.isalnum() or c == "_")
                yield self._word_style(line[i:j]), line[i:j]
            elif line[i] in OPERATOR_CHARS:
                j = _run_end(line, i, OPERATOR_CHARS.__contains__)
                token = line[i:j]
                yield ("#888888" if token in self.OPERATORS else ""), token
            else:
                j = i + 1
                yield "", line[i]
            i = j

    def _word_style(self, word: str) -> str:
        """Return the style of *word*: keyword, capitalised or plain."""
        if word in self.KEYWORDS:
            return "bold #ff8800"
        if word[0].isupper():
            return "#44ff44"
        return ""


def _run_end(line: str, start: int, accept) -> int:
    """Return the end of the run of characters of *line* from *start*."""
    end = start
    while end < len(line) and accept(line[end]):
        end += 1
    return end


STYLE = Style([("keyword", "bold #ff8800")])


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
        lexer=NeuroLangLexer(),
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
