"""Tests of the notes on constants matching no value of their sort."""

import pytest

from neurolang_apps.utils.tui.constant_notes import constant_notes


@pytest.fixture
def terms():
    """Return an engine with the neurosynth ``mentions`` of three terms."""
    from neurolang.frontend import NeurolangPDL

    nl = NeurolangPDL()
    nl.add_tuple_set([(10,), (11,)], name="study")
    nl.add_tuple_set(
        [(10, "pain", 0.5), (11, "fear", 0.4), (10, "memory", 0.5)],
        name="term_in_study_tfidf",
    )
    nl.execute_datalog_program(
        "mentions(s, t) :- term_in_study_tfidf(s, t, v), (v > 0.03)"
    )
    return nl


@pytest.mark.parametrize("program, note", [
    (
        "ans(s) :- mentions(s, 'Pain')",
        "'Pain' is not a term of the engine: mentions(...) matches no row "
        "with it; did you mean 'pain'?",
    ),
    (
        "ans(s) :- mentions(s, 'pian')",
        "'pian' is not a term of the engine: mentions(...) matches no row "
        "with it; did you mean 'pain'?",
    ),
    (
        "ans(s) :- mentions(s, 'fear conditioning')",
        "'fear conditioning' is not a term of the engine: mentions(...) "
        "matches no row with it.",
    ),
    (
        "ans(s) :- mentions(s, pain)",
        "pain in mentions(...) is a variable, matching every term: write "
        "'pain', quoted, for the term.",
    ),
    (
        "near(s, t) :- mentions(s, t)\nans(s) :- near(s, 'pian')",
        "'pian' is not a term of the engine: near(...) matches no row "
        "with it; did you mean 'pain'?",
    ),
])
def test_constants_matching_nothing_are_noted(terms, program, note):
    assert constant_notes(terms, "neurosynth", program, False) == [note]


@pytest.mark.parametrize("program", [
    "ans(s) :- mentions(s, 'pain')",
    "ans(s) :- mentions(s, t)",
    "ans(s, pain) :- mentions(s, pain)",
    "ans(s) :- mentions(s, 'Pain'",
])
def test_known_constants_and_variables_are_not_noted(terms, program):
    assert constant_notes(terms, "neurosynth", program, False) == []


def test_without_declared_sorts_nothing_is_noted(terms):
    assert constant_notes(terms, "", "ans(s) :- mentions(s, 'x')", False) \
        == []


def test_squall_constants_are_noted(terms):
    assert constant_notes(
        terms, "neurosynth", "obtain every Study that mentions 'Pain'.", True
    ) == [
        "'Pain' is not a term of the engine: mentions(...) matches no row "
        "with it; did you mean 'pain'?"
    ]


def test_the_repl_prints_the_notes(terms):
    import io

    from rich.console import Console

    from neurolang_apps.utils.tui.app import InteractiveTuiApp, TuiSettings

    console = Console(file=io.StringIO(), width=200)
    app = InteractiveTuiApp(
        terms, TuiSettings(engine_name="neurosynth"), console=console,
        err_console=console,
    )
    app._execute_and_display("ans(s) :- mentions(s, pain)")
    assert (
        "note: pain in mentions(...) is a variable, matching every term"
        in console.file.getvalue()
    )


def test_the_assistant_tools_observe_the_notes(terms):
    from neurolang_apps.utils.llm.agent import Toolbox
    from neurolang_apps.utils.llm.context import EngineContext
    from neurolang_apps.utils.llm.validation import check_query, run_preview

    note = "Note: 'Pain' is not a term of the engine"
    query = "ans(s) :- mentions(s, 'Pain')"
    check = check_query(terms, query, "datalog", "neurosynth")
    assert check.ok and check.notes.startswith(note)
    assert not check_query(terms, query, "datalog").notes
    preview = run_preview(terms, query, "datalog", engine_name="neurosynth")
    assert preview.text.startswith(note) and preview.text.endswith("0 rows")

    toolbox = Toolbox(EngineContext(terms, "neurosynth"), "datalog")
    for tool in ("check_query", "run_query", "final_answer"):
        observation, _ = getattr(toolbox, f"tool_{tool}")(query)
        assert note in observation
