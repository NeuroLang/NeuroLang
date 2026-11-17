"""
Ontologies from a SPARQL endpoint: the triples a ``CONSTRUCT`` query of the
endpoint gives, kept in a file loaded as the files of an engine's
``ontologies`` are (:func:`~neurolang_apps.utils.engine_registry.
load_ontology_entry`).

An entry with ``endpoint`` (its address) says which triples with one of:

* ``query``: a ``CONSTRUCT`` query;
* ``graph``: the IRI of a named graph, all of its triples;
* ``classes_under``: the IRI of a class, it and its subclasses, at any
  depth, with their ``rdfs:subClassOf`` and labels.

An endpoint is never read whole. A reply of more than ``max_triples``
triples, or ``max_megabytes`` megabytes, is refused, and the endpoint has
``timeout`` seconds to answer: the defaults are those of
``sparql_endpoint.yaml``, which an entry may change. A reply is kept in
the cache directory, named by the endpoint and its query, and read from
there afterwards; deleting it fetches it again.
"""
import hashlib
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import yaml

DEFAULTS_FILE = Path(__file__).with_name("sparql_endpoint.yaml")
DEFAULTS = yaml.safe_load(DEFAULTS_FILE.read_text(encoding="utf-8"))

#: The keys of an entry saying which triples of the endpoint.
SELECTORS = ("query", "graph", "classes_under")

# The formats asked for, N-Triples first (its triples are its lines), and
# the extension of the file of each, from which rdflib guesses it.
ACCEPT = (
    "application/n-triples, text/turtle;q=0.9, application/rdf+xml;q=0.8"
)
EXTENSIONS = {
    "application/n-triples": ".nt",
    "text/plain": ".nt",
    "text/turtle": ".ttl",
    "application/x-turtle": ".ttl",
    "application/rdf+xml": ".rdf",
}

CLASSES_UNDER = """\
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
CONSTRUCT {{
  ?c a owl:Class ; rdfs:subClassOf ?d ; rdfs:label ?label .
}} WHERE {{
  ?c rdfs:subClassOf* <{iri}> .
  OPTIONAL {{ ?c rdfs:subClassOf ?d . FILTER(isIRI(?d)) }}
  OPTIONAL {{ ?c rdfs:label ?label }}
}}"""


class EndpointError(ValueError):
    """The triples of an endpoint could not be read."""


def construct_query(entry: Dict[str, Any]) -> str:
    """Return the ``CONSTRUCT`` query of the entry *entry*."""
    given = [key for key in SELECTORS if entry.get(key)]
    if len(given) != 1:
        raise EndpointError(
            f"the ontology of the endpoint {entry.get('endpoint')} needs one "
            f"of {', '.join(SELECTORS)}, not {', '.join(given) or 'none'}: "
            "an endpoint is never read whole"
        )
    if given[0] == "query":
        return entry["query"]
    if given[0] == "graph":
        return (
            "CONSTRUCT { ?s ?p ?o } WHERE { GRAPH <%s> { ?s ?p ?o } }"
            % entry["graph"]
        )
    return CLASSES_UNDER.format(iri=entry["classes_under"])


def describe(entry: Dict[str, Any]) -> str:
    """Return how the ontology of the entry *entry* is named, to the
    user: its endpoint and what it reads of it."""
    key = next((key for key in SELECTORS if entry.get(key)), None)
    return f"{entry['endpoint']} ({key}: {entry[key]})" if key else \
        entry["endpoint"]


def fetch_triples(entry: Dict[str, Any], cache_dir: Path) -> Path:
    """
    Return the file of the triples of the entry *entry*, an endpoint and
    which of its triples, fetched into *cache_dir* unless it holds them.
    """
    query = construct_query(entry)
    key = hashlib.sha256(
        f"{entry['endpoint']}\n{query}".encode("utf-8")
    ).hexdigest()[:20]
    cache_dir = Path(cache_dir)
    for cached in sorted(cache_dir.glob(f"{key}.*")):
        return cached
    limits = {name: entry.get(name, value) for name, value in DEFAULTS.items()}
    request = Request(
        entry["endpoint"],
        data=urlencode({"query": query}).encode("utf-8"),
        headers={
            "Accept": ACCEPT,
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "NeuroLang",
        },
    )
    most_bytes = int(float(limits["max_megabytes"]) * (1 << 20))
    with urlopen(request, timeout=float(limits["timeout"])) as reply:
        kind = reply.headers.get_content_type()
        data = reply.read(most_bytes + 1)
    if len(data) > most_bytes:
        raise EndpointError(
            f"the reply of {describe(entry)} is over "
            f"{limits['max_megabytes']} MB (max_megabytes)"
        )
    extension = EXTENSIONS.get(kind)
    if extension is None:
        raise EndpointError(
            f"{entry['endpoint']} answered {kind}, not RDF triples"
        )
    triples = _count_triples(data, extension)
    if triples > int(limits["max_triples"]):
        raise EndpointError(
            f"{describe(entry)} gives {triples} triples, over the "
            f"{limits['max_triples']} of max_triples: ask for fewer, or "
            "raise max_triples"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}{extension}"
    path.write_bytes(data)
    return path


def _count_triples(data: bytes, extension: str) -> int:
    """The number of triples of the reply *data* of the format of the
    files of *extension*: the lines of N-Triples, else those rdflib
    reads."""
    if extension == ".nt":
        return sum(
            1 for line in data.splitlines()
            if line.strip() and not line.lstrip().startswith(b"#")
        )
    import rdflib

    graph = rdflib.Graph()
    graph.parse(data=data, format="turtle" if extension == ".ttl" else "xml")
    return len(graph)
