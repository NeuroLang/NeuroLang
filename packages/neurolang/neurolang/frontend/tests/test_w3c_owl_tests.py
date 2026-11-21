"""The W3C OWL test cases in the fragment NeuroLang reads: see
``w3c_owl_tests/README.rst``."""

import os
from pathlib import Path

import pytest
import rdflib
from rdflib import OWL, RDF, RDFS, BNode, Literal, URIRef

from nl_core.datalog.ontologies_parser import OntologyParser
from ...frontend import NeurolangPDL

DATA = Path(__file__).parent / "w3c_owl_tests"
BASE = "http://www.w3.org/2002/03owlt/"
OTEST = rdflib.Namespace("http://www.w3.org/2002/03owlt/testOntology#")
RTEST = rdflib.Namespace(
    "http://www.w3.org/2000/10/rdf-tests/rdfcore/testSchema#"
)
ENTAILMENT = ("PositiveEntailmentTest", "NegativeEntailmentTest")
READ_PREDICATES = {
    RDF.type, RDFS.subClassOf, OWL.onProperty, OWL.someValuesFrom,
    OWL.allValuesFrom, OWL.hasValue, OWL.intersectionOf, RDF.first,
    RDF.rest, RDFS.label, RDFS.comment, RDFS.subPropertyOf, RDFS.domain,
    RDFS.range, OWL.inverseOf, OWL.equivalentProperty,
}
READ_TYPES = {
    OWL.Class, OWL.Restriction, OWL.ObjectProperty, OWL.DatatypeProperty,
    OWL.Thing, OWL.Ontology, OWL.SymmetricProperty, OWL.TransitiveProperty,
}
PROPERTY_AXIOMS = (
    RDFS.subPropertyOf, RDFS.domain, RDFS.range, OWL.inverseOf,
    OWL.equivalentProperty,
)


def _vocabulary(term):
    return isinstance(term, URIRef) and str(term).startswith(
        (str(RDF), str(RDFS), str(OWL))
    )


def _parse(suite, iri):
    graph = rdflib.Graph()
    graph.parse(_path(suite, iri), format="xml", publicID=iri)
    return graph


def _path(suite, iri):
    return str(DATA / suite / (iri[len(BASE):] + ".rdf"))


def read_test(suite, manifest):
    """The type, premises and conclusions of the test of ``manifest``."""
    name = os.path.relpath(manifest, DATA / suite)[:-len(".rdf")]
    graph = rdflib.Graph()
    graph.parse(manifest, format="xml", publicID=BASE + name)
    test, kind = next(
        (test, kind) for kind in ENTAILMENT
        for test in graph.subjects(RDF.type, OTEST[kind])
    )
    return dict(
        id=f"{suite}/{name}", suite=suite, kind=kind,
        premises=[str(d) for d in graph.objects(test, RTEST.premiseDocument)],
        conclusions=[
            str(d) for d in graph.objects(test, RTEST.conclusionDocument)
        ],
    )


def classify(graph):
    """Why the premises ``graph`` are out of the fragment, or None."""
    name = graph.namespace_manager.qname
    for s, p, o in graph:
        if _vocabulary(p) and p not in READ_PREDICATES:
            return f"premises use {name(p)}"
        if p == RDF.type and _vocabulary(o) and o not in READ_TYPES:
            return f"premises use {name(o)}"
        if p in PROPERTY_AXIOMS and isinstance(o, BNode):
            return f"{name(p)} with a class expression"
        if p == RDF.type and isinstance(o, BNode):
            return "an individual is typed by a class expression"
        if p == OWL.intersectionOf and (None, RDFS.subClassOf, s) not in graph:
            return "intersectionOf other than as a superclass"
        if p == RDFS.subClassOf and isinstance(s, BNode):
            return "a class expression as a subclass"
    return None


def conclusion_atoms(graph):
    """The conclusion ``graph`` as atoms ``(predicate IRI, arguments)``,
    a blank node being a variable; None if it is not a set of facts."""
    variables = {}

    def term(t):
        if isinstance(t, BNode):
            return variables.setdefault(t, f"v{len(variables)}")
        if isinstance(t, Literal) and not isinstance(t.toPython(), str):
            return repr(t.toPython())
        return f"'{t}'"

    atoms = []
    for s, p, o in graph:
        if p == RDF.type and o in (OWL.Thing, OWL.Ontology):
            continue
        if (_vocabulary(p) and p != RDF.type) or (
            p == RDF.type and (_vocabulary(o) or isinstance(o, BNode))
        ):
            return None
        if p == RDF.type:
            atoms.append((o, (term(s),)))
        else:
            atoms.append((p, (term(s), term(o))))
    return atoms or None


TESTS = sorted(
    (
        read_test(suite, manifest)
        for suite in ("approved", "proposed")
        for manifest in (DATA / suite).glob("*/Manifest*.rdf")
    ),
    key=lambda test: test["id"],
)
SKIPPED = [
    line.split("\t")
    for line in (DATA / "skipped.tsv").read_text().splitlines()
    if not line.startswith("#")
]


@pytest.mark.parametrize("test", TESTS, ids=[t["id"] for t in TESTS])
def test_w3c_owl_entailment(test):
    """The conclusion is entailed exactly when the test says so."""
    for premises in test["premises"]:
        assert classify(_parse(test["suite"], premises)) is None
    paths = [_path(test["suite"], p) for p in test["premises"]]
    nl = NeurolangPDL()
    nl.load_ontology(paths, format="xml", individuals=True, names="prefixed")
    names = OntologyParser(paths, load_format="xml", names="prefixed")
    entailed = True
    for conclusion in test["conclusions"]:
        atoms = conclusion_atoms(_parse(test["suite"], conclusion))
        assert atoms is not None
        known = nl.ontology_predicates
        body = [
            f"`{names._parse_name(p)}`({', '.join(args)})"
            for p, args in atoms
        ]
        # A predicate the premises never name has no values.
        entailed &= all(
            names._parse_name(p) in known for p, _ in atoms
        ) and bool(nl.execute_datalog_program(
            f"ans() :- {', '.join(body)}"
        ))
    assert entailed == (test["kind"] == "PositiveEntailmentTest")


@pytest.mark.parametrize(
    "test, kind, reason", SKIPPED, ids=[s[0] for s in SKIPPED]
)
def test_w3c_owl_out_of_the_fragment(test, kind, reason):
    pytest.skip(f"{kind}: {reason}")
