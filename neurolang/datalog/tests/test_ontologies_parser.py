import io

import pytest

from ...exceptions import NeuroLangNotImplementedError
from ...expression_walker import ExpressionBasicEvaluator
from ..constraints_representation import DatalogConstraintsProgram
from ..ontologies_parser import OntologiesParser


class Datalog(DatalogConstraintsProgram, ExpressionBasicEvaluator):
    pass


def test_all_values_from():

    test_case = '''
    <rdf:RDF
    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
    xmlns:owl="http://www.w3.org/2002/07/owl#"
    xmlns:first="http://www.w3.org/2002/03owlt/allValuesFrom/premises001#"
    xml:base="http://www.w3.org/2002/03owlt/allValuesFrom/premises001" >
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
    '''

    expected = '''
    first:r rdf:type owl:Class .
    _:a rdf:type owl:Restriction .
    _:a owl:onProperty first:p .
    _:a owl:allValuesFrom first:c .
    first:r rdfs:subClassOf _:a .
    first:p rdf:type owl:ObjectProperty .
    first:c rdf:type owl:Class .
    first:i rdf:type first:r .
    first:i rdf:type owl:Thing .
    first:o rdf:type owl:Thing .
    first:i first:p first:o .
    '''

    dl = Datalog()
    onto = OntologiesParser(io.StringIO(test_case))
    dl = onto.parse_ontology(dl)

    for restriction in dl.get_constraints().expressions:
        print(restriction)


def test_has_value():

    test_case = '''
    <rdf:RDF
    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
    xmlns:owl="http://www.w3.org/2002/07/owl#"
    xml:base="http://www.w3.org/2002/03owlt/equivalentProperty/premises004" >
        <owl:ObjectProperty rdf:ID="p">
            <rdfs:domain rdf:resource="#d"/>
        </owl:ObjectProperty>
        <owl:ObjectProperty rdf:ID="q">
            <rdfs:domain rdf:resource="#d"/>
        </owl:ObjectProperty>
        <owl:FunctionalProperty rdf:about="#q"/>
        <owl:FunctionalProperty rdf:about="#p"/>
        <owl:Thing rdf:ID="v"/>
        <owl:Class rdf:ID="d">
            <owl:equivalentClass>
                    <owl:Restriction>
                        <owl:onProperty rdf:resource="#p"/>
                        <owl:hasValue rdf:resource="#v"/>
                    </owl:Restriction>
            </owl:equivalentClass>
            <owl:equivalentClass>
                    <owl:Restriction>
                        <owl:onProperty rdf:resource="#q"/>
                        <owl:hasValue rdf:resource="#v"/>
                    </owl:Restriction>
            </owl:equivalentClass>
        </owl:Class>
    </rdf:RDF>
    '''

    expected = '''
    first:p rdf:type owl:ObjectProperty .
    first:p rdfs:domain first:d .
    first:q rdf:type owl:ObjectProperty .
    first:q rdfs:domain first:d .
    first:q rdf:type owl:FunctionalProperty .
    first:p rdf:type owl:FunctionalProperty .
    first:v rdf:type owl:Thing .
    first:d rdf:type owl:Class .
    _:a rdf:type owl:Restriction .
    _:a owl:onProperty first:p .
    _:a owl:hasValue first:v .
    first:d owl:equivalentClass _:a .
    _:c rdf:type owl:Restriction .
    _:c owl:onProperty first:q .
    _:c owl:hasValue first:v .
    first:d owl:equivalentClass _:c .
    '''

    dl = Datalog()
    onto = OntologiesParser(io.StringIO(test_case))
    dl = onto.parse_ontology(dl)


def test_not_implemented():

    test_case = '''
    <rdf:RDF
    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
    xmlns:owl="http://www.w3.org/2002/07/owl#"
    xmlns:first="http://www.w3.org/2002/03owlt/someValuesFrom/premises001#"
    xml:base="http://www.w3.org/2002/03owlt/someValuesFrom/premises001" >
        <owl:Class rdf:ID="r">
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="#p"/>
                <owl:someValuesFrom rdf:resource="#c"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        </owl:Class>
        <owl:ObjectProperty rdf:ID="p"/>
        <owl:Class rdf:ID="c"/>
        <first:r rdf:ID="i"/>
    </rdf:RDF>
    '''

    dl = Datalog()
    onto = OntologiesParser(io.StringIO(test_case))

    with pytest.raises(NeuroLangNotImplementedError):
        dl = onto.parse_ontology(dl)
