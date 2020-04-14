import pytest

from ..constraints_representation import DatalogConstraintsProgram
from ..ontologies_parser import OntologiesParser
from ...expression_walker import ExpressionBasicEvaluator
import io


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

    io.StringIO(test_case)
    pass


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

    pass


def test_min_cardinality():

    test_case = '''
    <rdf:RDF
        xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xmlns:owl="http://www.w3.org/2002/07/owl#"
        xmlns:first="http://www.w3.org/2002/03owlt/disjointWith/inconsistent010#"
        xml:base="http://www.w3.org/2002/03owlt/disjointWith/inconsistent010" >

    <owl:Restriction rdf:nodeID="n">
        <owl:onProperty>
            <owl:ObjectProperty rdf:about="#p" />
        </owl:onProperty>
        <owl:minCardinality rdf:datatype=
    "http://www.w3.org/2001/XMLSchema#int"
        >1</owl:minCardinality>
        <owl:disjointWith rdf:nodeID="n"/>
    </owl:Restriction>
    <owl:Thing>
        <first:p>
            <owl:Thing/>
        </first:p>
    </owl:Thing>

    </rdf:RDF>
    '''

    pass
