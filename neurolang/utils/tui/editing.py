"""
Line editing of the interactive REPL.

It provides grammar-aware completion (:class:`NeuroLangReplCompleter`),
with a keyword fallback for SQUALL, whose Earley parser cannot complete
interactively, syntax highlighting (:class:`NeuroLangLexer`), the bottom
toolbar, suggestions from the history and the vi and emacs editing
modes (``.mode vi|emacs``).
"""

from html import escape
from typing import Callable, Optional

from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.styles import Style

from neurolang.datalog import WrappedRelationalAlgebraSet
from neurolang.frontend.datalog.standard_syntax import COMPILED_GRAMMAR
from neurolang.utils import engine_registry
from neurolang.utils.interactive_parsing import LarkCompleter

from .commands import COMMANDS, MODES

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


def _word_prefix(text: str) -> str:
    """Return the word being typed at the end of *text*."""
    return text[text.rfind(" ") + 1:]


class NeuroLangReplCompleter(Completer):
    """
    Complete dot-commands, the Datalog grammar and engine predicates.

    Datalog input is completed with the LALR grammar's *lark_completer*;
    SQUALL input, or any input without *lark_completer*, with the
    *squall_keywords*. *is_squall* tells whether the input is SQUALL.
    """

    def __init__(
        self,
        lark_completer: Optional[LarkCompleter] = None,
        engine_predicates: Optional[set] = None,
        squall_keywords: Optional[list] = None,
        is_squall: Optional[Callable[[], bool]] = None,
    ):
        """Set up the completion sources."""
        self._lark = lark_completer
        self.predicates = engine_predicates or set()
        self._squall_keywords = squall_keywords or []
        self._is_squall = is_squall or (lambda: False)

    def get_completions(self, document, complete_event):
        """Yield the completions of the text before the cursor."""
        text = document.text_before_cursor
        if text.startswith("."):
            yield from self._command_completions(text)
            return
        use_grammar = self._lark is not None and not self._is_squall()
        if use_grammar:
            yield from self._grammar_completions(text)
        prefix = _word_prefix(text)
        if prefix[:1].isalpha() or prefix[:1] == "_":
            yield from self._completions(
                sorted(self.predicates), prefix, "predicate"
            )
        if not use_grammar:
            yield from self._completions(
                self._squall_keywords, prefix, "keyword"
            )

    @staticmethod
    def _command_completions(text: str):
        """Yield the dot-commands starting with *text*."""
        for name in COMMANDS:
            if name.startswith(text):
                yield Completion(name, start_position=-len(text))

    @staticmethod
    def _completions(words, prefix: str, meta: str):
        """Yield the *words* starting with *prefix*, described by *meta*."""
        for word in words:
            if word.startswith(prefix):
                yield Completion(
                    word, start_position=-len(prefix), display_meta=meta
                )

    def _grammar_completions(self, text: str):
        """Yield the tokens the Datalog grammar accepts after *text*."""
        try:
            result = self._lark.complete(text)
        except Exception:
            return
        seen = set()
        for category, container in result.token_options.items():
            for value in container.get("values", set()):
                if not value or value in seen:
                    continue
                seen.add(value)
                yield Completion(
                    value,
                    start_position=-len(result.prefix),
                    display=value.strip("<>"),
                    display_meta=category,
                )


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


def engine_predicates(nl, engine_name: str) -> set:
    """Return the names of the engine's sets and declared predicates."""
    names = {
        getattr(symbol, "name", str(symbol))
        for symbol, value in nl.program_ir.symbol_table.items()
        if isinstance(
            getattr(value, "value", value), WrappedRelationalAlgebraSet
        )
    }
    if engine_name:
        names.update(engine_registry.get_predicates(engine_name))
    return names


def make_completer(app) -> NeuroLangReplCompleter:
    """Return the completer of the REPL *app*."""
    try:
        lark = LarkCompleter(COMPILED_GRAMMAR)
    except Exception:
        lark = None
    return NeuroLangReplCompleter(
        lark_completer=lark,
        engine_predicates=engine_predicates(
            app._nl, app.settings.engine_name
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
        f"preds:{len(app.completer.predicates)}  {result}  [{editing}]"
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
