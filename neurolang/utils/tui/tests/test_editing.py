"""Tests for the line editing of the interactive REPL."""

import io
import threading

import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

from prompt_toolkit.document import Document  # noqa: E402
from prompt_toolkit.enums import EditingMode  # noqa: E402
from rich.console import Console  # noqa: E402

from neurolang.frontend import NeurolangPDL  # noqa: E402
from neurolang.frontend.datalog.standard_syntax import (  # noqa: E402
    COMPILED_GRAMMAR,
)
from neurolang.utils.interactive_parsing import LarkCompleter  # noqa: E402
from neurolang.utils.tui import InteractiveTuiApp, TuiSettings  # noqa: E402
from neurolang.utils.tui.commands import COMMANDS  # noqa: E402
from neurolang.utils.tui.editing import (  # noqa: E402
    SQUALL_KEYWORDS,
    NeuroLangLexer,
    NeuroLangReplCompleter,
    toolbar,
)


def _texts(completer, text):
    return [c.text for c in completer.get_completions(Document(text), None)]


@pytest.fixture
def completer():
    return NeuroLangReplCompleter(
        lark_completer=LarkCompleter(COMPILED_GRAMMAR)
    )


class TestNeuroLangReplCompleter:
    def test_dot_command_completions(self, completer):
        assert ".help" in _texts(completer, ".h")

    def test_dot_command_partial_engines(self, completer):
        assert _texts(completer, ".eng") == [".engines"]

    def test_dot_command_empty_dot(self, completer):
        assert set(_texts(completer, ".")) == set(COMMANDS)

    def test_session_commands_use_dot_prefix(self, completer):
        assert _texts(completer, ".re") == [".rewritten"]
        assert _texts(completer, ".da") == [".datalog"]
        assert _texts(completer, "/") == []

    def test_non_dot_does_not_crash(self, completer):
        assert isinstance(_texts(completer, "ans(x) :- term"), list)

    def test_grammar_completion(self, completer):
        assert ":-" in _texts(completer, "ans(x) ")

    def test_squall_keywords(self):
        completer = NeuroLangReplCompleter(squall_keywords=SQUALL_KEYWORDS)
        assert "obtain" in _texts(completer, "ob")

    def test_squall_mode_uses_keywords(self):
        squall = {"on": False}
        completer = NeuroLangReplCompleter(
            lark_completer=LarkCompleter(COMPILED_GRAMMAR),
            squall_keywords=SQUALL_KEYWORDS,
            is_squall=lambda: squall["on"],
        )
        assert "obtain" not in _texts(completer, "ob")
        squall["on"] = True
        assert "obtain" in _texts(completer, "ob")

    def test_engine_predicates(self):
        completer = NeuroLangReplCompleter(
            engine_predicates={"term", "study", "peak"}
        )
        texts = _texts(completer, "st")
        assert "study" in texts
        assert "term" not in texts

    def test_predicates_skipped_for_non_identifiers(self):
        completer = NeuroLangReplCompleter(engine_predicates={"term"})
        assert "term" not in _texts(completer, "12")


def test_lexer_styles():
    tokens = list(NeuroLangLexer().tokens("ans(X) :- P(x)"))
    assert ("bold #ff8800", "ans") in tokens
    assert ("#44ff44", "X") in tokens
    assert ("#888888", ":-") in tokens
    assert "".join(text for _, text in tokens) == "ans(X) :- P(x)"


def test_lexer_lines():
    get_line = NeuroLangLexer().lex_document(Document("a\nobtain b"))
    assert get_line(1)[0] == ("bold #ff8800", "obtain")
    assert get_line(5) == []


@pytest.fixture
def app():
    nl = NeurolangPDL()
    nl.add_tuple_set([(1,), (2,)], name="P")
    console = Console(file=io.StringIO(), width=100)
    return InteractiveTuiApp(
        nl, TuiSettings(engine_name="neurosynth"), console=console,
        err_console=console,
    )


def test_session_completes_engine_predicates(app):
    session = app._make_session()
    assert session.completer is app.completer
    assert {"P", "peak_reported"} <= app.completer.predicates


def test_toolbar(app):
    app._make_session()
    assert "nl(neurosynth:DL)" in toolbar(app).value
    assert "—" in toolbar(app).value
    app._execute_and_display("ans(x) :- P(x)")
    app._handle_dot_command(".mode squall")
    text = toolbar(app).value
    assert "nl(neurosynth:SQ)" in text
    assert "rows:2" in text
    assert "[EM]" in text


def test_vi_and_emacs_modes(app):
    app.session = app._make_session()
    app._handle_dot_command(".mode vi")
    assert app.session.editing_mode == EditingMode.VI
    assert "[VI]" in toolbar(app).value
    app._handle_dot_command(".mode emacs")
    assert app.session.editing_mode == EditingMode.EMACS
    app._handle_dot_command(".mode nano")
    assert "Use one of: datalog, squall, vi, emacs" in (
        app.console.file.getvalue()
    )


def test_banner_mentions_tab(app):
    app._print_banner()
    assert "Tab completes" in app.console.file.getvalue()


def test_tab_completes_at_the_prompt(app):
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    with create_pipe_input() as pipe:
        # Completion is asynchronous: type the keys as a person would.
        pipe.send_text(".hel")
        timers = [
            threading.Timer(0.5, pipe.send_text, ["\t"]),
            threading.Timer(1.0, pipe.send_text, ["\r\x04"]),
        ]
        for timer in timers:
            timer.start()
        app.run(app._make_session(input=pipe, output=DummyOutput()))
        for timer in timers:
            timer.join()
    assert "Dot commands:" in app.console.file.getvalue()
