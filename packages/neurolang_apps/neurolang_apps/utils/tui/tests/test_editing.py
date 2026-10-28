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
from neurolang_apps.utils.tui import (  # noqa: E402
    InteractiveTuiApp,
    TuiSettings,
)
from neurolang_apps.utils.tui.commands import COMMANDS  # noqa: E402
from neurolang_apps.utils.predicate_catalog import PredicateInfo  # noqa: E402
from neurolang_apps.utils.tui.completion import argument_context  # noqa: E402
from neurolang_apps.utils.tui.editing import (  # noqa: E402
    NeuroLangLexer,
    STYLE,
    NeuroLangReplCompleter,
    toolbar,
    toolbar_text,
)
from neurolang_apps.utils.squall_completion import (  # noqa: E402
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
        assert _texts(completer, ".rew") == [".rewritten"]
        assert _texts(completer, ".da") == [".datalog"]
        assert _texts(completer, "/") == []

    def test_non_dot_does_not_crash(self, completer):
        assert isinstance(_texts(completer, "ans(x) :- term"), list)

    def test_grammar_completion(self, completer):
        assert ":-" in _texts(completer, "ans(x) ")

    def test_unlexable_character_does_not_crash(self, completer):
        # The lexer rejects "$"; the grammar then offers nothing.
        assert _texts(completer, "a(x) :- b(x)\nans(x) $ ") == []

    def test_comment_lines_are_skipped(self, completer):
        text = "% the studies\na(x) :- b(x) % b is stored\nans(x) "
        assert ":-" in _texts(completer, text)

    def test_nothing_offered_inside_a_comment(self, completer):
        assert _texts(completer, "ans(x) :- b(x) % b is ") == []

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
        session = app._make_session(input=pipe, output=DummyOutput())

        def wait_for(text):
            deadline = time.monotonic() + 30
            while (
                session.default_buffer.text != text and
                time.monotonic() < deadline
            ):
                time.sleep(0.01)

        def type_keys():
            # Completion is asynchronous: wait for each key to take effect
            # before typing the next one, however slow the machine.
            pipe.send_text(".hel")
            wait_for(".hel")
            pipe.send_text("\t")
            wait_for(".help")
            pipe.send_text("\r\x04")

        typist = threading.Thread(target=type_keys)
        typist.start()
        app.run(session)
        typist.join()
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



def test_conditional_predicate_offered_with_its_own_arity(neurosynth_app):
    """
    A conditional is offered as itself, not as one of its helpers.

    ``marg`` compiles to ``marg`` and the helpers ``marg^cond_num`` and
    ``marg^cond_den``, also named ``marg`` but of other arities.
    """
    app = neurosynth_app
    nl = app._nl
    nl.add_tuple_set([("pain",)], name="term")
    nl.add_uniform_probabilistic_choice_over_set(
        [("s1",)], name="selected_study"
    )
    app._handle_dot_command(".mode squall")
    app._execute_and_display(
        "define as marg with inferred probability every Term where a "
        "Selected_study mentions the Term conditioned to the "
        "Selected_study that mentions 'pain'."
    )
    app._handle_dot_command(".mode datalog")
    assert _metas(app, "ans(t) :- mar")["marg"].endswith("/2 · session")

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
        assert metas["mentions"] == (
            "mentions(study(study_id), term) · derived"
        )

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
            "peak_reported(voxel(i, j, k), study(study_id)) · stored"
        )
        assert _metas(neurosynth_app, "ans(x) :- te")[
            "term_in_study_tfidf"
        ] == "term_in_study_tfidf(study(study_id), term, tfidf) · stored"

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
        neurosynth_app._execute_and_display("myp(a, b) :- mentions(a, b)")
        texts = _words(neurosynth_app, "q(y) :- myp(")
        assert texts[:2] == ["a", "y"]

    def test_declared_columns_of_engine_rules(self, neurosynth_app):
        texts = _words(neurosynth_app, "q(y) :- mentions(")
        assert texts[:2] == ["study_id", "y"]

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
        metas = _metas(neurosynth_app, "obtain every Te")
        assert list(metas)[0] == "Term_in_study_tfidf"
        assert metas["Term_in_study_tfidf"] == "study, term, tfidf · stored"
        assert _words(neurosynth_app, "obtain every te") == [
            "term_in_study_tfidf"
        ]

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
        assert "mentions" not in words
        assert {"every", "the", "a"} <= set(words)
        assert _words(squall_app, "obtain ev") == ["every"]

    def test_after_every_predicates_first(self, squall_app):
        words = _words(squall_app, "obtain every ")
        assert words.index("Term_in_study_tfidf") < words.index("top")
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
        words = _words(squall_app, "obtain the count of every te")
        assert words[0] == "term_in_study_tfidf"
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
        # build the grammar first: this measures each completion, not the
        # one-time construction of the parser (about 4 times longer)
        _words(squall_app, "obtain every ")
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


@pytest.fixture
def sorts_app():
    """Return a SQUALL REPL on the vocabulary of the neurosynth engine."""
    nl = NeurolangPDL()
    nl.add_tuple_set([(10,), (11,)], name="study")
    nl.add_tuple_set([("pain",)], name="term")
    nl.add_tuple_set([(1, 2, 3)], name="voxel")
    nl.add_tuple_set([(1, 2, 3, 10)], name="peak_reported")
    nl.add_tuple_set([(10, "pain", 0.5)], name="term_in_study_tfidf")
    nl.add_uniform_probabilistic_choice_over_set(
        [(10,), (11,)], name="selected_study"
    )
    nl.execute_datalog_program(
        "mentions(s, t) :- term_in_study_tfidf(s, t, v)\n"
        "reports(s, i, j, k) :- peak_reported(i, j, k, s)"
    )
    console = Console(file=io.StringIO(), width=100)
    app = InteractiveTuiApp(
        nl, TuiSettings(engine_name="neurosynth", squall=True),
        console=console, err_console=console,
    )
    app._make_session()
    return app


class TestSquallSorts:
    def test_nouns_are_capitalised_with_their_entities(self, sorts_app):
        metas = _metas(sorts_app, "obtain every ")
        assert metas["Voxel"] == "voxel(i, j, k) · in 3D · stored"
        assert metas["Study"] == "study · stored"
        assert "cos" not in metas and "voxel" not in metas
        assert _words(sorts_app, "obtain every Stu") == ["Study"]

    def test_verbs_by_the_sort_of_their_subject(self, sorts_app):
        metas = _metas(sorts_app, "obtain every Study that ")
        verbs = [word for word, meta in metas.items() if "→" in meta]
        assert verbs == ["mentions", "reports"]
        assert metas["reports"] == "study → voxel(i, j, k) · derived"
        assert "is" in metas

    def test_no_verb_from_another_sort(self, sorts_app):
        words = _words(sorts_app, "obtain every Term that ")
        assert "mentions" not in words and "reports" not in words

    def test_object_clause_on_the_object_sort(self, sorts_app):
        assert "a Study mentions" in _words(sorts_app, "obtain every Term that ")
        metas = _metas(sorts_app, "obtain every Voxel in 3D that ")
        assert metas["a Study reports"] == (
            "object clause · study → voxel(i, j, k) · derived"
        )

    def test_verb_of_an_object_clause(self, sorts_app):
        words = _words(sorts_app, "obtain every Voxel that a Study ")
        assert "reports" in words and "mentions" not in words

    def test_inverse_verbs_are_not_offered(self, sorts_app):
        for text in ("obtain every Term that ", "obtain every Voxel that "):
            assert not any(
                word.startswith("~") for word in _words(sorts_app, text)
            )

    def test_in_3d_after_a_multi_column_noun(self, sorts_app):
        words = _words(sorts_app, "obtain every Voxel ")
        assert words[:2] == ["in 3D", "(?i; ?j; ?k)"]
        assert _words(sorts_app, "obtain every Voxel i")[0] == "in 3D"
        assert "in 1D" not in _words(sorts_app, "obtain every Study ")

    def test_with_probability_after_an_inferred_probability(self, sorts_app):
        sorts_app._execute_and_display(
            "define as Active_voxel with inferred probability "
            "every Voxel in 3D that a Selected_study reports."
        )
        words = _words(sorts_app, "obtain every Active_voxel ")
        assert words[:2] == ["in 3D", "in 3D with Probability"]
        assert "with Probability" not in _words(
            sorts_app, "obtain every Selected_study "
        )

    def test_hidden_predicates_not_offered_in_squall(self, sorts_app):
        assert "Peak_reported" not in _words(sorts_app, "obtain every ")
        assert "peak_reported" not in _words(sorts_app, "obtain every pe")
        sorts_app.settings.squall = False
        assert "peak_reported" in _words(sorts_app, "ans(x) :- pe")

    def test_entity_columns_completed_together_in_datalog(self, sorts_app):
        sorts_app.settings.squall = False
        completions = _completions(sorts_app, "ans(s) :- reports(s, ")
        assert completions[0].text == "i, j, k"
        assert completions[0].display_meta_text == "voxel(i, j, k) of reports"
        assert completions[1].text == "i"
        first = _words(sorts_app, "ans(s) :- peak_reported(")
        assert first[:2] == ["i, j, k", "i"]
        assert "i, j, k" not in _words(sorts_app, "ans(s) :- reports(")
        assert _metas(sorts_app, "ans(s) :- rep")["reports"] == (
            "reports(study(study_id), voxel(i, j, k)) · derived"
        )

    def test_suggested_sentences_run(self, sorts_app):
        for sentence in (
            "obtain every Voxel in 3D that a Study reports.",
            "obtain every Term that a Study mentions.",
            "obtain every Study that reports a Voxel in 3D.",
            "obtain every Voxel (?i; ?j; ?k) that a Study reports.",
        ):
            sorts_app._execute_and_display(sentence)
            assert not sorts_app.failed, sentence
            assert len(sorts_app.results[-1].frame) == 1, sentence


@pytest.fixture
def values_app(sorts_app):
    """Return the SQUALL REPL of :func:`sorts_app` with several terms."""
    sorts_app._nl.add_tuple_set(
        [("pain",), ("parietal",), ("fear",), ("working memory",),
         ("alzheimer's",)],
        name="term",
    )
    return sorts_app


class TestQuotedValues:
    def test_terms_after_mentions(self, values_app):
        metas = _metas(values_app, "obtain every Study that mentions '")
        assert list(metas) == [
            "fear'", "pain'", "parietal'", "working memory'"
        ]
        assert metas["pain'"] == "term"

    def test_terms_filtered_by_their_prefix(self, values_app):
        assert _words(values_app, "obtain every Study that mentions 'pa") == [
            "pain'", "parietal'"
        ]
        assert _words(
            values_app, "obtain every Study that mentions 'working m"
        ) == ["working memory'"]

    def test_double_quotes(self, values_app):
        words = _words(values_app, 'obtain every Study that mentions "al')
        assert words == ["alzheimer's\""]

    def test_the_completion_replaces_the_quoted_text(self, values_app):
        (completion,) = _completions(
            values_app, "obtain every Study that mentions 'working m"
        )
        assert completion.start_position == -len("working m")

    def test_while_typing_after_the_quote_alone(self, values_app):
        text = "obtain every Study that mentions '"
        assert _words(values_app, text) and not [
            c.text for c in _completions(values_app, text, requested=False)
        ]
        assert _completions(values_app, text + "p", requested=False)

    def test_values_in_datalog(self, values_app):
        values_app.settings.squall = False
        assert _words(values_app, "ans(s) :- mentions(s, 'pa") == [
            "pain'", "parietal'"
        ]

    def test_no_values_for_an_unknown_verb(self, values_app):
        assert _words(values_app, "obtain every Study that plays '") == []

    def test_values_are_read_once(self, values_app, monkeypatch):
        from neurolang_apps.utils.tui import value_completion

        calls = []
        read = value_completion.ColumnValues._read

        def counting(self, name, position):
            calls.append(name)
            return read(self, name, position)

        monkeypatch.setattr(value_completion.ColumnValues, "_read", counting)
        for prefix in ("", "p", "pa"):
            _words(values_app, "obtain every Study that mentions '" + prefix)
        assert calls == ["term"]


class TestTupleVariables:
    def test_the_next_columns_of_the_noun(self, sorts_app):
        metas = _metas(sorts_app, "obtain every Voxel (?i; ")
        assert list(metas) == ["?j", "?j; ?k)"]
        assert metas["?j"] == "column 2 of voxel"

    def test_filtered_by_the_variable_typed(self, sorts_app):
        assert _words(sorts_app, "obtain every Voxel (?i; ?j; ?") == [
            "?k", "?k)"
        ]

    def test_from_the_first_column(self, sorts_app):
        assert _words(sorts_app, "obtain every Voxel (") == [
            "?i", "?i; ?j; ?k)"
        ]

    def test_nothing_after_the_last_column(self, sorts_app):
        assert _words(sorts_app, "obtain every Voxel (?i; ?j; ?k; ") == []


def test_arity_error_notes_the_entities(sorts_app):
    """Check a wrong arity on a voxel predicate tells its columns."""
    sorts_app.settings.squall = False
    sorts_app._execute_and_display("ans(s) :- reports(s, i)")
    out = sorts_app.console.file.getvalue()
    assert (
        "note: reports relates study(study_id) to voxel(i, j, k): "
        "write reports(study_id, i, j, k)"
    ) in out


def test_squall_error_notes_the_entities(sorts_app, monkeypatch):
    """Check an error on a SQUALL noun of several columns tells how."""
    def failing(program):
        raise ValueError("Relations must have the same arity")

    monkeypatch.setattr(sorts_app, "_execute", failing)
    sorts_app._execute_and_display("obtain every Voxel that a Study reports.")
    out = sorts_app.console.file.getvalue()
    assert '"every Voxel in 3D" or "every Voxel (?i; ?j; ?k)"' in out
    assert '"a Study reports a Voxel in 3D"' in out


def test_one_declaration_of_the_sorts(monkeypatch, tmp_path):
    """Check completion, .predicates and the primer read the same sorts."""
    from neurolang_apps.utils import engine_registry

    path = tmp_path / "engines.yaml"
    path.write_text(
        "engines:\n  grid:\n"
        "    sorts:\n"
        "      voxel: {columns: [x, y, z], type: int, description: a cell}\n"
        "      study: {columns: [study_id], type: int}\n"
        "    predicates:\n"
        "      voxel: {arguments: [voxel]}\n"
        "      reports: {arguments: [study, voxel]}\n"
    )
    monkeypatch.setattr(engine_registry, "_ENGINES_YAML", path)
    nl = NeurolangPDL()
    nl.add_tuple_set([(1, 2, 3)], name="voxel")
    nl.add_tuple_set([(10, 1, 2, 3)], name="reports")
    console = Console(file=io.StringIO(), width=200)
    app = InteractiveTuiApp(
        nl, TuiSettings(engine_name="grid", squall=True), console=console,
        err_console=console,
    )
    app._make_session()
    assert _metas(app, "obtain every ")["Voxel"].startswith(
        "voxel(x, y, z) · in 3D"
    )
    assert _words(app, "obtain every Voxel (?x; ") == ["?y", "?y; ?z)"]
    app._handle_dot_command(".predicates")
    app._handle_dot_command(".help squall")
    out = console.file.getvalue()
    assert "reports(study(study_id), voxel(x, y, z))" in out
    assert "voxel(x, y, z)         in 3D · a cell" in out
