W3C OWL test cases
==================

These are test cases of the W3C *OWL Web Ontology Language Test Cases*
(https://www.w3.org/TR/owl-test/), from the archives
https://www.w3.org/2002/03owlt/approved.zip (``approved/``) and
https://www.w3.org/2002/03owlt/proposed.zip (``proposed/``), downloaded on
2026-11-14 and copied without modification.  They are distributed under the
W3C Software Notice and License, reproduced in ``NOTICE``; each file keeps its
copyright notice.  The OWL 2 test repository that the OWL 2 conformance
document points to (owl.semanticweb.org) no longer resolves.

``test_w3c_owl_tests.py`` runs them.  Of the 84 tests of the two archives
(entailment, non-entailment, consistency, inconsistency, imports and
not-OWL-feature tests), the ones kept here are the entailment and
non-entailment tests whose premises use only what NeuroLang reads
(``rdfs:subClassOf``; ``owl:someValuesFrom``, ``owl:allValuesFrom`` and
``owl:hasValue`` restrictions; ``owl:intersectionOf`` as a superclass;
``rdfs:subPropertyOf``, ``rdfs:domain``, ``rdfs:range``, ``owl:inverseOf``,
``owl:equivalentProperty``, ``owl:SymmetricProperty`` and
``owl:TransitiveProperty``, on named classes and properties;
``rdf:type`` of a named class; labels and comments) and whose conclusion is
a set of facts on individuals, read as a boolean query whose blank nodes
are existential variables.  A positive entailment test passes if the query
is true (its atoms are certain answers), a negative one if it is false.

``skipped.tsv`` lists every other test of the two archives with the reason it
is left out: its type, or the first construct that the test module's
``classify`` (premises) or ``conclusion_atoms`` (conclusion) rejects, applied
to the full archives.  In short: NeuroLang does not
check consistency; it does not read ``owl:imports``, cardinalities,
functional and inverse functional properties, ``owl:samePropertyAs``,
equality of individuals or classes, ``unionOf``,
``complementOf``, ``oneOf`` or ``disjointWith``; and a conclusion that is an
axiom on classes or properties is not a query on individuals.
