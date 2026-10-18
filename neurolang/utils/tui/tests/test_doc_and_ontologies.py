"""Tests for ``.doc`` and for the predicates of the ontologies."""

import io

import pandas as pd
import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

from prompt_toolkit.completion import CompleteEvent  # noqa: E402
from prompt_toolkit.document import Document  # noqa: E402
from rich.console import Console  # noqa: E402

from neurolang.frontend import NeurolangPDL  # noqa: E402
from neurolang.utils.tui import InteractiveTuiApp, TuiSettings  # noqa: E402
from neurolang.utils.tui.catalog import listed, predicate_catalog  # noqa: E402

GRAPH = """
@prefix ex: <http://example.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

ex:Animal a rdfs:Class .
ex:Dog a rdfs:Class ; rdfs:subClassOf ex:Animal .
ex:rex a ex:Dog ; ex:name "Rex" .
"""


@pytest.fixture
def engine(tmp_path):
    """A neurosynth-like engine that also loaded a small ontology."""
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [(1, 2, 3, 10), (4, 5, 6, 11)], name="peak_reported"
    )
    nl.add_tuple_set([(10,), (11,)], name="study")
    nl.add_uniform_probabilistic_choice_over_set(
        pd.DataFrame({"study_id": [10, 11]}), name="selected_study"
    )
    path = tmp_path / "vocab.ttl"
    path.write_text(GRAPH)
    nl.load_ontology(str(path), individuals=True, names="prefixed")
    return nl


@pytest.fixture
def app(engine):
    console = Console(file=io.StringIO(), width=120)
    return InteractiveTuiApp(
        engine, TuiSettings(engine_name="neurosynth"), console=console,
        err_console=console,
    )


def _output(app):
    return app.console.file.getvalue()


def _run(app, line):
    app.console.file.truncate(0)
    app.console.file.seek(0)
    app._handle_dot_command(line)
    return _output(app)


def test_the_frontend_remembers_the_predicates_of_the_ontologies(engine):
    registry = engine.ontology_symbols
    assert registry["ex:name"][0].endswith("vocab.ttl")
    assert "ex:Dog" in registry and "rdfs:subClassOf" in registry
    assert "peak_reported" not in registry


def test_the_catalog_gives_them_the_ontology_kind(engine):
    catalog = predicate_catalog(engine)
    assert catalog["ex:name"].kind == "ontology"
    assert catalog["peak_reported"].kind == "stored"
    assert "ex:name" not in {info.name for info in listed(catalog)}
    assert "ex:name" in {
        info.name for info in listed(catalog, ontologies=True)
    }


def test_predicates_leave_out_the_ontologies_by_default(app):
    output = _run(app, ".predicates")
    assert "peak_reported" in output and "study" in output
    assert "ex:name" not in output and "rdfs:subClassOf" not in output
    assert "ontology predicates not shown" in output
    assert ".predicates ontologies" in output


def test_predicates_list_the_ontologies_on_request(app):
    only = _run(app, ".predicates ontologies")
    assert "ex:name" in only and "rdfs:subClassOf" in only
    assert "peak_reported" not in only and "not shown" not in only
    everything = _run(app, ".predicates all")
    assert "ex:name" in everything and "peak_reported" in everything
    assert "not shown" not in everything


def test_predicates_reject_an_unknown_argument(app):
    assert "Usage: .predicates [all|ontologies]" in _run(
        app, ".predicates some"
    )
    assert app.failed


def test_doc_describes_a_predicate_of_the_engine(app):
    output = _run(app, ".doc peak_reported")
    assert "peak_reported(voxel(i, j, k), study(study_id)) · stored" in output
    assert "Reported activation peaks in voxel coordinates" in output
    assert "1. voxel(i, j, k): a voxel of the MNI template grid" in output
    assert "2. study(study_id)" in output
    assert "2 facts" in output


def test_doc_shows_the_rules_of_a_session_predicate(app):
    app._execute_and_display("near(x) :- study(x)")
    output = _run(app, ".doc near")
    assert "near(x) · session" in output
    assert "1 rule" in output and "near(x) :- study(x)" in output
    assert "(no documentation: .doc near TEXT writes one)" in output


def test_doc_describes_a_predicate_of_an_ontology(app):
    output = _run(app, ".doc ex:name")
    assert "· ontology" in output
    assert "from the ontology vocab.ttl" in output
    assert "1 fact" in output


def test_doc_writes_a_note_that_shows_first_and_clear_forgets_it(app):
    assert "documented peak_reported" in _run(
        app, ".doc peak_reported where the studies report a peak"
    )
    output = _run(app, ".doc peak_reported")
    lines = output.splitlines()
    assert lines[1] == "where the studies report a peak"
    assert "Engine: Reported activation peaks" in output
    _run(app, ".clear")
    assert "Engine:" not in _run(app, ".doc peak_reported")
    assert "where the studies report" not in _output(app)


def test_doc_suggests_the_predicate_it_does_not_find(app):
    output = _run(app, ".doc peak_reportd")
    assert (
        "Unknown predicate 'peak_reportd'. Did you mean peak_reported?"
        in output
    )
    assert app.failed
    assert "Usage: .doc PREDICATE [TEXT]" in _run(app, ".doc")


def _completions(app, text):
    if app.completer is None:
        app._make_session()
    return [
        c.text for c in app.completer.get_completions(
            Document(text), CompleteEvent(completion_requested=True)
        )
    ]


def test_doc_completes_the_names_of_the_predicates(app):
    assert "peak_reported" in _completions(app, ".doc pe")
    assert "study" not in _completions(app, ".doc pe")
    assert ".doc" in _completions(app, ".do")


def test_completion_offers_the_ontologies_after_two_characters(app):
    assert not [n for n in _completions(app, "r") if n.startswith("rdfs:")]
    assert "rdfs:subClassOf" in _completions(app, "rd")
    assert "ex:name" in _completions(app, ".doc ex")


def test_doc_names_the_kind_of_a_probabilistic_predicate_once(app):
    output = _run(app, ".doc selected_study")
    assert output.splitlines()[0].count("probabilistic") == 1
    assert "2 facts" in output
