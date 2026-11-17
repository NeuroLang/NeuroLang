"""Ontologies from a SPARQL endpoint (sparql_endpoint), with the endpoint
answered here: the tests reach no address."""
import email.message
import io
import re
import textwrap
from pathlib import Path
from urllib.parse import parse_qs

import pytest
import yaml

from neurolang.frontend import NeurolangPDL
from neurolang_apps.utils import engine_registry, sparql_endpoint
from neurolang_apps.utils.sparql_endpoint import (
    DEFAULTS, DEFAULTS_FILE, EndpointError, construct_query, fetch_triples,
)

ENDPOINT = "https://example.org/sparql"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
EX = "http://example.org/"
TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
CLASS = "http://www.w3.org/2002/07/owl#Class"
# What classes_under gives: the classes, typed, and their superclasses.
TRIPLES = "".join(
    f"<{EX}{c}> <{TYPE}> <{CLASS}> .\n"
    for c in ("Employee", "Manager", "Intern")
).encode() + (
    f"<{EX}Manager> <{RDFS}subClassOf> <{EX}Employee> .\n"
    f"<{EX}Intern> <{RDFS}subClassOf> <{EX}Employee> .\n"
).encode()


class Reply(io.BytesIO):
    def __init__(self, data, kind):
        super().__init__(data)
        self.headers = email.message.Message()
        self.headers["Content-Type"] = kind


@pytest.fixture
def endpoint(monkeypatch):
    """The endpoint: it answers each request with ``answer["data"]``, of
    the type ``answer["kind"]``, and keeps the requests."""
    answer = {"data": TRIPLES, "kind": "application/n-triples",
              "requests": []}

    def urlopen(request, timeout):
        answer["requests"].append((request, timeout))
        return Reply(answer["data"], answer["kind"])

    monkeypatch.setattr(sparql_endpoint, "urlopen", urlopen)
    return answer


def _query(request):
    return parse_qs(request.data.decode())["query"][0]


@pytest.mark.parametrize("entry, expected", [
    ({"query": "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o } LIMIT 5"},
     "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o } LIMIT 5"),
    ({"graph": f"{EX}g"}, f"GRAPH <{EX}g>"),
    ({"classes_under": f"{EX}Employee"}, f"rdfs:subClassOf* <{EX}Employee>"),
])
def test_the_query_of_an_entry(entry, expected):
    assert expected in construct_query({"endpoint": ENDPOINT, **entry})


@pytest.mark.parametrize("entry", [{}, {"graph": "g", "query": "q"}])
def test_an_endpoint_is_never_read_whole(entry):
    with pytest.raises(EndpointError, match="never read whole"):
        construct_query({"endpoint": ENDPOINT, **entry})


def test_the_triples_are_fetched_once(endpoint, tmp_path):
    entry = {"endpoint": ENDPOINT, "graph": f"{EX}g"}
    path = fetch_triples(entry, tmp_path)
    assert path.suffix == ".nt" and path.read_bytes() == TRIPLES
    (request, timeout), = endpoint["requests"]
    assert request.full_url == ENDPOINT and request.get_method() == "POST"
    assert f"GRAPH <{EX}g>" in _query(request)
    assert "application/n-triples" in request.get_header("Accept")
    assert timeout == DEFAULTS["timeout"]
    assert fetch_triples(entry, tmp_path) == path  # from the cache
    assert len(endpoint["requests"]) == 1
    fetch_triples({**entry, "graph": f"{EX}h"}, tmp_path)  # another query
    assert len(endpoint["requests"]) == 2


def test_a_turtle_reply_is_counted_by_rdflib(endpoint, tmp_path):
    endpoint["data"] = (
        f"@prefix ex: <{EX}> . @prefix rdfs: <{RDFS}> .\n"
        "ex:A rdfs:subClassOf ex:B . ex:C rdfs:subClassOf ex:B .\n"
    ).encode()
    endpoint["kind"] = "text/turtle"
    entry = {"endpoint": ENDPOINT, "graph": "g", "max_triples": 1}
    with pytest.raises(EndpointError, match="gives 2 triples"):
        fetch_triples(entry, tmp_path)
    assert fetch_triples({**entry, "max_triples": 2}, tmp_path).suffix == (
        ".ttl"
    )


@pytest.mark.parametrize("limits, kind, message", [
    ({"max_triples": 1}, "application/n-triples",
     "over the 1 of max_triples"),
    ({"max_megabytes": 1e-5}, "application/n-triples", "max_megabytes"),
    ({}, "text/html", "answered text/html, not RDF triples"),
])
def test_a_reply_is_refused(endpoint, tmp_path, limits, kind, message):
    endpoint["kind"] = kind
    entry = {"endpoint": ENDPOINT, "graph": "g", **limits}
    with pytest.raises(EndpointError, match=re.escape(message)):
        fetch_triples(entry, tmp_path)
    assert list(tmp_path.iterdir()) == []  # nothing kept


def test_an_engine_loads_the_triples_of_an_endpoint(endpoint, tmp_path):
    nl = NeurolangPDL()
    nl.add_tuple_set([(f"{EX}m1",)], name="Manager")
    entry = {"endpoint": ENDPOINT, "classes_under": f"{EX}Employee"}
    engine_registry.load_ontology_entry(nl, entry, tmp_path, "my_engine")
    assert set(nl.execute_datalog_program("ans(x) :- Employee(x)")) == {
        (f"{EX}m1",)
    }
    assert list((tmp_path / "my_engine" / "sparql").iterdir())
    assert engine_registry.ontology_sources([entry]) == [
        f"{ENDPOINT} (classes_under: {EX}Employee)"
    ]


def test_the_docs_quote_the_defaults():
    docs = Path(__file__).parents[5] / "doc" / "engine_configuration.rst"
    if not docs.exists():  # not a checkout
        return
    block = docs.read_text().split("\n   max_triples:", 1)[1]
    block = "   max_triples:" + block.split("\n\n", 1)[0]
    quoted = yaml.safe_load(textwrap.dedent(block))
    assert quoted == DEFAULTS == yaml.safe_load(DEFAULTS_FILE.read_text())
