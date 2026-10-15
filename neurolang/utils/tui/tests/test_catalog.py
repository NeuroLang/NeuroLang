"""Tests for the catalog of the predicates of an engine."""

import io

import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

from rich.console import Console  # noqa: E402

from neurolang.frontend import NeurolangPDL  # noqa: E402
from neurolang.utils import engine_registry  # noqa: E402
from neurolang.utils.tui import InteractiveTuiApp, TuiSettings  # noqa: E402
from neurolang.utils.tui.catalog import (  # noqa: E402
    PredicateInfo,
    predicate_catalog,
)


@pytest.fixture
def engine():
    """
    Return an engine with a predicate of each kind.

    The engine has the relation ``peak_reported``, the rule ``named``,
    the probabilistic choice ``selected_study`` and the function
    ``inc``.
    """
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [(1, 2, 3, "s1"), (4, 5, 6, "s2")], name="peak_reported"
    )
    nl.add_tuple_set([("alice",), ("bob",)], name="person")
    nl.add_uniform_probabilistic_choice_over_set(
        [("s1",), ("s2",)], name="selected_study"
    )
    nl.execute_datalog_program("named(x) :- person(x)")

    @nl.add_symbol
    def inc(x: int) -> int:
        return x + 1

    return nl


@pytest.fixture
def session(engine):
    """Return the *engine* where a session scope defines ``active``."""
    engine.program_ir.push_scope()
    engine.execute_datalog_program("active(who) :- person(who)")
    return engine


def test_catalog_kinds_and_columns(session):
    catalog = predicate_catalog(
        session,
        columns={
            "peak_reported": ["i", "j", "k", "study_id"],
            "selected_study": ["study_id"],
        },
    )
    assert catalog["peak_reported"] == PredicateInfo(
        "peak_reported", "stored", ("i", "j", "k", "study_id"), 4
    )
    assert catalog["person"] == PredicateInfo("person", "stored", None, 1)
    assert catalog["named"] == PredicateInfo("named", "derived", ("x",), 1)
    assert catalog["active"] == PredicateInfo(
        "active", "session", ("who",), 1
    )
    assert catalog["selected_study"] == PredicateInfo(
        "selected_study", "probabilistic", ("study_id",), 1, True
    )
    assert catalog["inc"] == PredicateInfo("inc", "function", None, 1)


def test_catalog_lists_builtin_euclidean_function(engine):
    catalog = predicate_catalog(engine)
    assert catalog["euclidean"] == PredicateInfo(
        "euclidean", "function", None, 6
    )


def test_catalog_loaded_relation(session):
    session.add_tuple_set([("a", 1)], name="scores")
    catalog = predicate_catalog(session, loaded={"scores": ["Name", "Score"]})
    assert catalog["scores"] == PredicateInfo(
        "scores", "loaded", ("Name", "Score"), 2
    )


def test_catalog_columns_must_fit(session):
    catalog = predicate_catalog(session, columns={"person": ["a", "b"]})
    assert catalog["person"].columns is None


def test_catalog_leaves_out_generated_symbols(session):
    session.execute_squall_program(
        "define as marg with inferred probability every Person "
        "conditioned to the Person that is named."
    )
    catalog = predicate_catalog(session)
    assert catalog["marg"] == PredicateInfo("marg", "session", None, 2)
    assert not [name for name in catalog if name.startswith(("_", "fresh"))]
    assert "PROB" not in catalog


def test_squall_rule_has_no_generated_column_names(session):
    session.execute_squall_program(
        "define as Fan every person that is named."
    )
    fan = predicate_catalog(session)["fan"]
    assert fan.columns is None
    assert fan.arity == 1
    assert fan.signature() == "fan/1"


@pytest.mark.parametrize(
    "info, signature",
    [
        (PredicateInfo("r", "stored", ("a", "b"), 2), "r(a, b)"),
        (PredicateInfo("r", "stored", None, 2), "r/2"),
        (PredicateInfo("f", "function"), "f"),
    ],
)
def test_signature(info, signature):
    assert info.signature() == signature


def _app(engine, console):
    """Return a REPL on the neurosynth-like *engine* printing to *console*."""
    return InteractiveTuiApp(
        engine, TuiSettings(engine_name="neurosynth"), console=console,
        err_console=console,
    )


def test_predicates_lists_kinds_and_columns(engine):
    console = Console(file=io.StringIO(), width=200)
    app = _app(engine, console)
    app._execute_and_display("active(who) :- person(who)")
    console.file.truncate(0)
    console.file.seek(0)
    app._handle_dot_command(".predicates")
    lines = [
        " ".join(line.split())
        for line in console.file.getvalue().splitlines()
    ]
    functions = [line for line in lines if " function " in line]
    assert "inc function inc/1" in functions
    assert [line for line in lines if line not in functions] == [
        "Predicate Kind Columns",
        "active session active(who)",
        "peak_reported stored "
        "peak_reported(voxel(i, j, k), study(study_id))",
        "person stored person/1",
        "named derived named(x)",
        "selected_study probabilistic selected_study(study_id)",
    ]
    assert lines.index("inc function inc/1") > lines.index(
        "selected_study probabilistic selected_study(study_id)"
    )


def test_predicates_is_a_table_on_a_terminal(engine):
    console = Console(file=io.StringIO(), width=200, force_terminal=True)
    _app(engine, console)._handle_dot_command(".predicates")
    out = console.file.getvalue()
    assert "╭" in out and "Predicate" in out
    assert "peak_reported(voxel(i, j, k), study(study_id))" in out


def test_predicates_shows_the_entities_of_a_rule(engine):
    console = Console(file=io.StringIO(), width=200)
    app = _app(engine, console)
    app._execute_and_display(
        "reports(s, i, j, k) :- peak_reported(i, j, k, s)"
    )
    console.file.truncate(0)
    console.file.seek(0)
    app._handle_dot_command(".predicates")
    lines = [
        " ".join(line.split())
        for line in console.file.getvalue().splitlines()
    ]
    assert "reports session reports(study(study_id), voxel(i, j, k))" in lines


def test_catalog_arguments_from_signatures(session):
    signatures = engine_registry.get_signatures("neurosynth")
    catalog = predicate_catalog(session, signatures=signatures)
    peaks = catalog["peak_reported"]
    assert [argument.sort for argument in peaks.arguments] == [
        "voxel", "study"
    ]
    assert peaks.signature() == (
        "peak_reported(voxel(i, j, k), study(study_id))"
    )
    assert peaks.meta() == f"{peaks.signature()} · stored"
    assert catalog["person"].arguments is None
