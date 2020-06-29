import io

import pytest
from rdflib import RDF

from ...exceptions import NeuroLangNotImplementedError
from ...expression_walker import ExpressionBasicEvaluator
from ...expressions import Symbol
from ...logic import Union
from ..chase import Chase
from ..constraints_representation import DatalogConstraintsProgram
from ..expressions import Implication
from ..ontologies_parser import OntologyParser
from ..ontologies_rewriter import OntologyRewriter


class Datalog(DatalogConstraintsProgram, ExpressionBasicEvaluator):
    pass


def test_all_values_from():
    """
    Test case acquired from:
    http://owl.semanticweb.org/page/TestCase:WebOnt-allValuesFrom-001.html
    Since the original test is a test of entailment, a query is derived that
    simulates the information implied by the Conclusion ontology
    """

    premise_ontology = """
    <rdf:RDF
        xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
        xmlns:owl="http://www.w3.org/2002/07/owl#"
        xmlns:first="http://www.w3.org/2002/03owlt/allValuesFrom/premises001#"
        xml:base="http://www.w3.org/2002/03owlt/allValuesFrom/premises001" >
        <owl:Ontology/>
        <owl:Class rdf:ID="r">
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="#p"/>
                <owl:allValuesFrom rdf:resource="#c"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        </owl:Class>
        <owl:ObjectProperty rdf:ID="p"/>
        <owl:Class rdf:ID="c"/>
        <first:r rdf:ID="i">
        <rdf:type rdf:resource="http://www.w3.org/2002/07/owl#Thing"/>
        <first:p>
            <owl:Thing rdf:ID="o" />
        </first:p>
        </first:r>
    </rdf:RDF>
    """

    conclusion_ontology = """
    <rdf:RDF
    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:first="http://www.w3.org/2002/03owlt/allValuesFrom/premises001#"
    xmlns:owl="http://www.w3.org/2002/07/owl#"
    xml:base="http://www.w3.org/2002/03owlt/allValuesFrom/conclusions001" >
        <first:c rdf:about="premises001#o">
            <rdf:type rdf:resource="http://www.w3.org/2002/07/owl#Thing"/>
        </first:c>
        <owl:Class rdf:about="premises001#c"/>
    </rdf:RDF>  
    """

    rdf_type = Symbol(str(RDF.type))
    answer = Symbol("answer")
    x = Symbol("x")
    y = Symbol("y")
    test_base_q = Union((Implication(answer(x, y), rdf_type(x, y)),))

    onto = OntologyParser(io.StringIO(premise_ontology))
    predicate_tuples, union_of_constraints = onto.parse_ontology()

    triples = predicate_tuples[onto.get_triples_symbol()]
    pointers = predicate_tuples[onto.get_pointers_symbol()]

    dl = Datalog()
    dl.add_extensional_predicate_from_tuples(
        onto.get_triples_symbol(), triples
    )
    dl.add_extensional_predicate_from_tuples(
        onto.get_pointers_symbol(), pointers
    )

    orw = OntologyRewriter(test_base_q, union_of_constraints)
    rewrite = orw.Xrewrite()

    uc = ()
    for imp in rewrite:
        uc += (imp[0],)
    uc = Union(uc)

    dl.walk(uc)
    dc = Chase(dl)
    solution_instance = dc.build_chase_solution()

    resp = list(solution_instance["answer"].value.unwrapped_iter())

    assert (
        "http://www.w3.org/2002/03owlt/allValuesFrom/premises001#o",
        "http://www.w3.org/2002/03owlt/allValuesFrom/premises001#c",
    ) in resp


def test_some_values_from():
    premise_ontology = """
    <rdf:RDF 
        xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
        xmlns:owl="http://www.w3.org/2002/07/owl#"
        xmlns:first="http://www.w3.org/2002/03owlt/someValuesFrom/premises001#"
        xml:base="http://www.w3.org/2002/03owlt/someValuesFrom/premises001" >
        <owl:Ontology/>
        <owl:Class rdf:ID="r">
            <rdfs:subClassOf>
                <owl:Restriction>
                    <owl:onProperty rdf:resource="#p2"/>
                    <owl:someValuesFrom rdf:resource="#c"/>
                </owl:Restriction>
            </rdfs:subClassOf>
        </owl:Class>
        <owl:ObjectProperty rdf:ID="p"/>
        <owl:Class rdf:ID="c"/>
        <first:r rdf:ID="i">
            <rdf:type rdf:resource="http://www.w3.org/2002/07/owl#Thing"/>
            <first:p>
                <owl:Thing rdf:ID="o" />
            </first:p>
        </first:r>
        </rdf:RDF>
    """

    p2 = Symbol("http://www.w3.org/2002/03owlt/someValuesFrom/premises001#p2")
    answer = Symbol("answer")
    x = Symbol("x")
    y = Symbol("y")
    test_base_q = Union((Implication(answer(x, y), p2(x, y)),))

    onto = OntologyParser(io.StringIO(premise_ontology))
    predicate_tuples, union_of_constraints = onto.parse_ontology()

    triples = predicate_tuples[onto.get_triples_symbol()]
    pointers = predicate_tuples[onto.get_pointers_symbol()]

    dl = Datalog()
    dl.add_extensional_predicate_from_tuples(
        onto.get_triples_symbol(), triples
    )
    dl.add_extensional_predicate_from_tuples(
        onto.get_pointers_symbol(), pointers
    )

    orw = OntologyRewriter(test_base_q, union_of_constraints)
    rewrite = orw.Xrewrite()

    uc = ()
    for imp in rewrite:
        uc += (imp[0],)
    uc = Union(uc)

    dl.walk(uc)
    dc = Chase(dl)
    solution_instance = dc.build_chase_solution()

    resp = list(solution_instance["answer"].value.unwrapped_iter())

    assert (
        "http://www.w3.org/2002/03owlt/someValuesFrom/premises001#i",
        "http://www.w3.org/2002/03owlt/someValuesFrom/premises001#c",
    ) in resp


def test_has_value():
    test_case = """
    <rdf:RDF
        xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
        xmlns:owl="http://www.w3.org/2002/07/owl#"
        xmlns:first="http://www.w3.org/2002/03owlt/hasValue/premises001#"
        xml:base="http://www.w3.org/2002/03owlt/hasValue/premises001" >
        <owl:Ontology/>
        <owl:Class rdf:ID="r">
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="#p2"/>
                <owl:hasValue rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</owl:hasValue>
            </owl:Restriction>
        </rdfs:subClassOf>
        </owl:Class>
        <owl:ObjectProperty rdf:ID="p"/>
        <owl:ObjectProperty rdf:ID="p2"/>
        <owl:Class rdf:ID="c"/>
        <first:r rdf:ID="i">
            <rdf:type rdf:resource="http://www.w3.org/2002/07/owl#Thing"/>
            <first:p>
                <owl:Thing rdf:ID="o" />
            </first:p>
        </first:r>
    </rdf:RDF>
    """

    answer = Symbol("answer")
    x = Symbol("x")
    y = Symbol("y")
    p2 = Symbol("http://www.w3.org/2002/03owlt/hasValue/premises001#p2")
    test_base_q = Union((Implication(answer(x, y), p2(x, y)),))

    onto = OntologyParser(io.StringIO(test_case))
    predicate_tuples, union_of_constraints = onto.parse_ontology()

    triples = predicate_tuples[onto.get_triples_symbol()]
    pointers = predicate_tuples[onto.get_pointers_symbol()]

    dl = Datalog()
    dl.add_extensional_predicate_from_tuples(
        onto.get_triples_symbol(), triples
    )
    dl.add_extensional_predicate_from_tuples(
        onto.get_pointers_symbol(), pointers
    )

    orw = OntologyRewriter(test_base_q, union_of_constraints)
    rewrite = orw.Xrewrite()

    uc = ()
    for imp in rewrite:
        uc += (imp[0],)
    uc = Union(uc)

    dl.walk(uc)
    dc = Chase(dl)
    solution_instance = dc.build_chase_solution()

    resp = list(solution_instance["answer"].value.unwrapped_iter())

    assert (
        "http://www.w3.org/2002/03owlt/hasValue/premises001#i",
        "true",
    ) in resp


@pytest.mark.skip()
def test_min_cardinality():
    """
    Test case based on:
    http://owl.semanticweb.org/page/TestCase:WebOnt-cardinality-001.html

    Since the original test is a test of entailment between cardinality
    and minCardinality and maxCardinality, a query is derived that
    simulates the information implied.
    """

    test_case = """
    <rdf:RDF
        xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
        xmlns:owl="http://www.w3.org/2002/07/owl#"
        xml:base="http://www.w3.org/2002/03owlt/cardinality/conclusions001" >
        <owl:Class rdf:about="premises001#c">
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="premises001#p"/>
                <owl:minCardinality
                    rdf:datatype="http://www.w3.org/2001/XMLSchema#nonNegativeInteger">
                    2
                </owl:minCardinality>
            </owl:Restriction>
        </rdfs:subClassOf>
        </owl:Class>
        <owl:ObjectProperty rdf:about="premises001#p"/>
    </rdf:RDF>
    """

    answer = Symbol("answer")
    x = Symbol("x")
    y = Symbol("y")
    p2 = Symbol("http://www.w3.org/2002/03owlt/hasValue/premises001#p")
    test_base_q = Union((Implication(answer(x, y), p2(x, y)),))

    onto = OntologyParser(io.StringIO(test_case))
    predicate_tuples, union_of_constraints = onto.parse_ontology()

    triples = predicate_tuples[onto.get_triples_symbol()]
    pointers = predicate_tuples[onto.get_pointers_symbol()]

    dl = Datalog()
    dl.add_extensional_predicate_from_tuples(
        onto.get_triples_symbol(), triples
    )
    dl.add_extensional_predicate_from_tuples(
        onto.get_pointers_symbol(), pointers
    )

    orw = OntologyRewriter(test_base_q, union_of_constraints)
    rewrite = orw.Xrewrite()

    uc = ()
    for imp in rewrite:
        uc += (imp[0],)
    uc = Union(uc)

    dl.walk(uc)
    dc = Chase(dl)
    solution_instance = dc.build_chase_solution()

    resp = list(solution_instance["answer"].value.unwrapped_iter())


@pytest.mark.skip()
def test_max_cardinality():
    """
    Test case based on:
    http://owl.semanticweb.org/page/TestCase:WebOnt-cardinality-001.html

    Since the original test is a test of entailment between cardinality
    and minCardinality and maxCardinality, a query is derived that
    simulates the information implied.
    """

    test_case = """
    <rdf:RDF
        xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
        xmlns:owl="http://www.w3.org/2002/07/owl#"
        xml:base="http://www.w3.org/2002/03owlt/cardinality/conclusions001" >
        <owl:Class rdf:about="premises001#c">
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="premises001#p"/>
                <owl:maxCardinality 
                    rdf:datatype="http://www.w3.org/2001/XMLSchema#nonNegativeInteger">
                    2
                </owl:maxCardinality>
            </owl:Restriction>
        </rdfs:subClassOf>
        </owl:Class>
        <owl:ObjectProperty rdf:about="premises001#p"/>
    </rdf:RDF>
    """

    answer = Symbol("answer")
    x = Symbol("x")
    y = Symbol("y")
    p2 = Symbol("http://www.w3.org/2002/03owlt/hasValue/premises001#p")
    test_base_q = Union((Implication(answer(x, y), p2(x, y)),))

    onto = OntologyParser(io.StringIO(test_case))
    predicate_tuples, union_of_constraints = onto.parse_ontology()

    triples = predicate_tuples[onto.get_triples_symbol()]
    pointers = predicate_tuples[onto.get_pointers_symbol()]

    dl = Datalog()
    dl.add_extensional_predicate_from_tuples(
        onto.get_triples_symbol(), triples
    )
    dl.add_extensional_predicate_from_tuples(
        onto.get_pointers_symbol(), pointers
    )

    orw = OntologyRewriter(test_base_q, union_of_constraints)
    rewrite = orw.Xrewrite()

    uc = ()
    for imp in rewrite:
        uc += (imp[0],)
    uc = Union(uc)

    dl.walk(uc)
    dc = Chase(dl)
    solution_instance = dc.build_chase_solution()

    resp = list(solution_instance["answer"].value.unwrapped_iter())
