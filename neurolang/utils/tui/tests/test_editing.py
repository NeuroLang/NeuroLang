"""Tests for the line editing of the interactive REPL."""

import io
import threading
import time

import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

from prompt_toolkit.completion import CompleteEvent  # noqa: E402
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
from neurolang.utils.tui.completion import (  # noqa: E402
    PredicateInfo,
    argument_context,
)
from neurolang.utils.tui.editing import (  # noqa: E402
    NeuroLangLexer,
    STYLE,
    NeuroLangReplCompleter,
    toolbar,
    toolbar_text,
)
from neurolang.utils.tui.squall_completion import (  # noqa: E402
    SquallGrammarOptions,
    current_sentence,
    squall_keywords,
)


def _texts(completer, text):
    return [c.text for c in completer.get_completions(Document(text), None)]


def _catalog(*names):
    infos = {name: PredicateInfo(name, "stored") for name in names}
    return lambda: infos


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
        completer = NeuroLangReplCompleter(squall_keywords=squall_keywords())
        assert "obtain" in _texts(completer, "ob")

    def test_squall_mode_uses_keywords(self):
        squall = {"on": False}
        completer = NeuroLangReplCompleter(
            lark_completer=LarkCompleter(COMPILED_GRAMMAR),
            squall_keywords=squall_keywords(),
            is_squall=lambda: squall["on"],
        )
        assert "obtain" not in _texts(completer, "ob")
        squall["on"] = True
        assert "obtain" in _texts(completer, "ob")

    def test_engine_predicates(self):
        completer = NeuroLangReplCompleter(
            predicates=_catalog("term", "study", "peak")
        )
        texts = _texts(completer, "st")
        assert "study" in texts
        assert "term" not in texts

    def test_predicates_skipped_for_non_identifiers(self):
        completer = NeuroLangReplCompleter(predicates=_catalog("term"))
        assert "term" not in _texts(completer, "12")


def _styles(line, squall=False):
    return dict(
        (text, style)
        for style, text in NeuroLangLexer(lambda: squall).tokens(line)
    )


def test_lexer_datalog_styles():
    line = 'ans(X, PROB) :- the(a), ~every(a), p(a, "s t", 0.5) // q(X)'
    tokens = list(NeuroLangLexer().tokens(line))
    assert "".join(text for _, text in tokens) == line
    styles = _styles(line)
    assert styles["ans"] == styles["PROB"] == "class:keyword"
    assert styles[":-"] == styles["~"] == styles["//"] == "class:operator"
    assert styles["the"] == styles["every"] == "class:predicate"
    assert styles["a"] == ""
    assert styles['"s t"'] == "class:string"
    assert styles["0.5"] == "class:number"


def test_lexer_no_capitalised_word_style():
    styles = _styles("ans(X) :- P(X, Y), Y > 2")
    assert styles["X"] == styles["Y"] == ""
    assert styles[">"] == "class:operator"


def test_lexer_squall_keywords():
    line = 'define as Active every study that the a "x" mentions.'
    styles = _styles(line, squall=True)
    for word in ("define", "as", "every", "that", "the", "a"):
        assert styles[word] == "class:keyword", word
    assert styles["Active"] == styles["study"] == ""
    assert styles['"x"'] == "class:string"
    assert all(word in squall_keywords() for word in ("every", "the", "a"))


def test_lexer_follows_mode():
    squall = {"on": False}
    lexer = NeuroLangLexer(lambda: squall["on"])
    assert lexer.lex_document(Document("the(a)"))(0)[0] == (
        "class:predicate", "the"
    )
    squall["on"] = True
    get_line = lexer.lex_document(Document("x\nobtain b"))
    assert get_line(1)[0] == ("class:keyword", "obtain")
    assert get_line(5) == []


def test_style_uses_ansi_colours():
    rules = dict(STYLE.style_rules)
    assert {"keyword", "operator", "predicate", "string", "number"} <= set(
        rules
    )
    assert not any("#" in value for value in rules.values())


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
    assert session.style is STYLE
    app.settings.squall = True
    assert session.lexer.lex_document(Document("the"))(0) == [
        ("class:keyword", "the")
    ]
    app.settings.squall = False
    assert "P" in _texts(app.completer, "ans(x) :- ")


def test_toolbar(app):
    app._make_session()
    assert toolbar(app) == (
        " neurosynth · Datalog · page 20 · emacs · .help · "
        "Tab completes · Ctrl+D quits"
    )
    app._handle_dot_command(".mode squall")
    app._handle_dot_command(".sort x:desc")
    app._handle_dot_command(".format csv")
    app._handle_dot_command(".head 5")
    assert toolbar_text(app.settings, 200) == (
        " neurosynth · SQUALL · page 5 · sort x:desc · csv · emacs · "
        ".help · Tab completes · Ctrl+D quits"
    )


def test_toolbar_fits_the_terminal(app):
    app.settings.sort_by = [("study_id", True), ("term", False)]
    text = toolbar_text(app.settings, 80)
    assert len(text) <= 80
    assert text.endswith("emacs · .help")
    narrow = toolbar_text(app.settings, 30)
    assert len(narrow) == 30 and narrow.endswith("…")
    assert "neurosynth · Datalog" in narrow


def test_vi_and_emacs_modes(app):
    app.session = app._make_session()
    app._handle_dot_command(".mode vi")
    assert app.session.editing_mode == EditingMode.VI
    assert " vi · " in toolbar(app)
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


@pytest.fixture
def neurosynth_app():
    """Return a REPL on a small engine named like neurosynth's."""
    nl = NeurolangPDL()
    nl.add_tuple_set([(1, 2, 3, "s1")], name="peak_reported")
    nl.add_tuple_set([("s1", "pain", 0.5)], name="term_in_study_tfidf")
    nl.execute_datalog_program(
        "mentions(s, t) :- term_in_study_tfidf(s, t, v)"
    )
    console = Console(file=io.StringIO(), width=100)
    app = InteractiveTuiApp(
        nl, TuiSettings(engine_name="neurosynth"), console=console,
        err_console=console,
    )
    app._make_session()
    return app


def _completions(app, text, requested=True):
    event = CompleteEvent(
        text_inserted=not requested, completion_requested=requested
    )
    return list(app.completer.get_completions(Document(text), event))


def _words(app, text):
    return [c.text for c in _completions(app, text)]


def _metas(app, text):
    return {c.text: c.display_meta_text for c in _completions(app, text)}


class TestDynamicCompletion:
    def test_predicate_ranked_before_grammar(self, neurosynth_app):
        texts = _words(neurosynth_app, "ans(x) :- pe")
        assert texts[0] == "peak_reported"
        assert "(" not in texts

    def test_no_placeholders_or_punctuation(self, neurosynth_app):
        for text in ("ans(x) :- ", "ans(x) :- peak_reported(", "ans(x) "):
            for completion in _completions(neurosynth_app, text):
                word = completion.text
                assert not word.startswith("<"), (text, word)
                assert word not in {"(", ")", "@", "...", "-", ",", "lambda"}

    def test_grammar_operator_after_head(self, neurosynth_app):
        assert _words(neurosynth_app, "ans(x) ") == [
            ":-"
        ]

    def test_keywords_after_predicates(self, neurosynth_app):
        texts = _words(neurosynth_app, "ans(x) :- ")
        assert texts.index("peak_reported") < texts.index("EXISTS")
        assert texts.index("PROB") < texts.index("~")

    def test_engine_idb_offered(self, neurosynth_app):
        metas = _metas(neurosynth_app, "ans(x) :- me")
        assert metas["mentions"] == "mentions(s, t) · derived"

    def test_session_rule_offered_once_defined(self, neurosynth_app):
        assert "myp" not in _metas(neurosynth_app, "ans(x) :- my")
        neurosynth_app._execute_and_display(
            "myp(x) :- peak_reported(x, y, z, s)"
        )
        metas = _metas(neurosynth_app, "ans(x) :- my")
        assert metas["myp"] == "myp(x) · session"

    def test_loaded_relation_offered(self, neurosynth_app, tmp_path):
        path = tmp_path / "regions.csv"
        path.write_text("region,label\n1,amygdala\n")
        neurosynth_app._handle_dot_command(f".load {path}")
        metas = _metas(neurosynth_app, "ans(x) :- reg")
        assert metas["regions"] == "regions(region, label) · loaded"
        assert _metas(neurosynth_app, "ans(x) :- regions(1, ")["label"] == (
            "column 2 of regions"
        )

    def test_declared_columns_and_meta(self, neurosynth_app):
        metas = _metas(neurosynth_app, "ans(x) :- pe")
        assert metas["peak_reported"] == (
            "peak_reported(i, j, k, study_id) · stored"
        )
        assert _metas(neurosynth_app, "ans(x) :- te")[
            "term_in_study_tfidf"
        ] == "term_in_study_tfidf(study_id, term, tfidf) · stored"

    def test_unknown_columns_shown_as_arity(self, neurosynth_app):
        neurosynth_app._nl.add_tuple_set([(1, 2)], name="pairs")
        assert _metas(neurosynth_app, "ans(x) :- pa")["pairs"] == (
            "pairs/2 · stored"
        )

    def test_yaml_predicates_absent_from_engine_not_offered(
        self, neurosynth_app
    ):
        texts = _words(neurosynth_app, "ans(x) :- ")
        assert "voxel" not in texts and "selected_study" not in texts

    def test_column_names_inside_arguments(self, neurosynth_app):
        completions = _completions(
            neurosynth_app, "ans(x) :- peak_reported(x, "
        )
        assert completions[0].text == "j"
        assert completions[0].display_meta_text == (
            "column 2 of peak_reported"
        )
        texts = [c.text for c in completions]
        assert {"x", "i", "k", "study_id"} <= set(texts)
        assert "peak_reported" not in texts

    def test_idb_parameters_as_columns(self, neurosynth_app):
        texts = _words(neurosynth_app, "q(y) :- mentions(")
        assert texts[:2] == ["s", "y"]

    def test_aggregations_in_rule_head(self, neurosynth_app):
        metas = _metas(neurosynth_app, "q(x, co")
        assert metas["count"] == "aggregation"
        assert "count" not in _metas(neurosynth_app, "q(x) :- p(x, co")

    def test_functions_ranked_after_relations(self, neurosynth_app):
        texts = _words(neurosynth_app, "ans(x) :- me")
        assert texts == ["mentions", "mean"]

    def test_probabilistic_choice(self, neurosynth_app):
        neurosynth_app._nl.add_uniform_probabilistic_choice_over_set(
            [("s1",), ("s2",)], name="selected_study"
        )
        assert _metas(neurosynth_app, "ans(x) :- sel")["selected_study"] == (
            "selected_study(study_id) · probabilistic"
        )

    def test_no_popup_while_typing_without_a_word(self, neurosynth_app):
        assert _completions(neurosynth_app, "ans(x) :- ", False) == []
        assert _completions(neurosynth_app, "ans(x) :- pe", False)

    def test_invalid_prefix_still_offers_predicates(self, neurosynth_app):
        assert "peak_reported" in _words(neurosynth_app, "ans(x) :- ) pe")

    def test_squall_offers_predicates_and_keywords(self, neurosynth_app):
        neurosynth_app.settings.squall = True
        metas = _metas(neurosynth_app, "obtain every pe")
        assert list(metas)[0] == "peak_reported"
        assert metas["peak_reported"].endswith("· stored")

    def test_completion_is_fast(self, neurosynth_app):
        # On the full neurosynth engine a completion takes about 1 ms.
        start = time.perf_counter()
        for _ in range(20):
            _completions(neurosynth_app, "ans(x) :- peak_reported(x, y")
            _completions(neurosynth_app, "ans(x) :- pe")
        assert (time.perf_counter() - start) / 40 < 0.05


def test_predicate_info_signature():
    assert PredicateInfo("f", "function").signature() == "f"
    assert PredicateInfo("p", "stored", None, 2).signature() == "p/2"
    assert PredicateInfo("p", "stored", ("a",), 1).meta() == "p(a) · stored"


def test_argument_context():
    assert argument_context("ans(x) :- p(a, (b), ") == ("p", 2)
    assert argument_context("ans(x) :- p(a)") is None
    assert argument_context("ans(x) :- (x ") is None


def test_help_lists_key_bindings(app):
    app._handle_dot_command(".help")
    out = app.console.file.getvalue()
    for key in (
        "Tab", "Alt+Enter", "Ctrl+X Ctrl+E", "$EDITOR", "Ctrl+C", "Ctrl+D",
        "Up/Down", "Ctrl+R",
    ):
        assert key in out, key


def test_ctrl_x_ctrl_e_opens_the_editor(app):
    from prompt_toolkit.keys import Keys

    session = app._make_session()
    assert session.enable_open_in_editor
    bindings = session.app.key_bindings.get_bindings_for_keys(
        (Keys.ControlX, Keys.ControlE)
    )
    assert any(
        b.handler.__name__ == "edit_and_execute" for b in bindings
    )


@pytest.fixture
def squall_app(neurosynth_app):
    neurosynth_app.settings.squall = True
    return neurosynth_app


class TestSquallGrammarCompletion:
    def test_sentence_start(self, squall_app):
        words = _words(squall_app, "")
        assert {"obtain", "define", "every", "the"} <= set(words)
        assert "is" not in words

    def test_after_obtain(self, squall_app):
        words = _words(squall_app, "obtain ")
        assert words[0] == "mentions"
        assert {"every", "the", "a"} <= set(words)
        assert _words(squall_app, "obtain ev") == ["every"]

    def test_after_every_predicates_first(self, squall_app):
        words = _words(squall_app, "obtain every ")
        assert words.index("peak_reported") < words.index("top")
        assert "every" not in words

    def test_verbs_after_that(self, squall_app):
        words = _words(squall_app, "obtain every study that ")
        assert {"mentions", "has", "is", "does"} <= set(words)
        assert "." not in words

    def test_end_of_sentence(self, squall_app):
        completions = _completions(squall_app, "obtain every study that is ")
        assert "." not in [c.text for c in completions]
        metas = _metas(squall_app, "obtain every study that mentions ")
        assert metas["."] == "end of the sentence"

    def test_next_sentence(self, squall_app):
        assert "obtain" in _words(squall_app, "obtain every study. ob")

    def test_no_punctuation(self, squall_app):
        for text in ("", "obtain ", "obtain every study that mentions "):
            for word in _words(squall_app, text):
                assert word == "." or word[0].isalpha(), (text, word)

    def test_invalid_sentence_falls_back(self, squall_app):
        # "count" is reserved: after "the" a noun is expected.
        words = _words(squall_app, "obtain the count of every pe")
        assert words[0] == "peak_reported"
        assert _words(squall_app, "obtain the count of ev") == ["every"]
        assert _words(squall_app, "obtain the count of every ") == []

    def test_options_are_cached(self):
        options = SquallGrammarOptions()
        first = options.options("obtain every ")
        assert options.options("obtain every ") is first
        assert options.options("obtain every x. obtain every ") == first

    def test_latency(self, squall_app):
        clause = "that mentions a term and "
        sentence = "obtain every study that reports a peak and " + clause * 4
        assert 140 < len(sentence) <= 200
        start = time.perf_counter()
        words = _words(squall_app, sentence)
        _words(squall_app, "obtain every study that ")
        assert (time.perf_counter() - start) / 2 < 0.2
        assert "mentions" in words and "has" in words

    def test_long_sentence_falls_back(self, squall_app):
        sentence = "obtain every study that " + "mentions a term and " * 10
        assert len(sentence) > 200
        start = time.perf_counter()
        assert _words(squall_app, sentence) == []
        assert _words(squall_app, sentence + "ha") == ["had", "has", "have"]
        assert time.perf_counter() - start < 0.05


def test_current_sentence():
    assert current_sentence("obtain a. obtain every ") == "obtain every "
    assert current_sentence("obtain a.") == ""
    assert current_sentence("obtain x greater than 0.5 ") == (
        "obtain x greater than 0.5 "
    )


def test_squall_keywords_from_grammar():
    words = squall_keywords()
    assert {"obtain", "every", "all", "an", "count", "probably"} <= set(words)
    assert not {"(", "D", "_", "--", "n't"} & set(words)
