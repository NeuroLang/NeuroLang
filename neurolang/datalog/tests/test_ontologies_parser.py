import io

import pytest

from ... import expression_walker as ew
from ...exceptions import NeuroLangNotImplementedError
from ...expression_walker import ExpressionBasicEvaluator
from ...expressions import Constant, ExpressionBlock, Symbol
from ..aggregation import DatalogWithAggregationMixin
from ..chase import Chase
from ..constraints_representation import DatalogConstraintsProgram
from ..expressions import Implication, TranslateToLogic
from ..ontologies_parser import OntologyParser
from ..ontologies_rewriter import OntologyRewriter


class Datalog(DatalogConstraintsProgram, ExpressionBasicEvaluator):
    pass


class DatalogTranslator(
    TranslateToLogic, ew.IdentityWalker, DatalogWithAggregationMixin
):
    pass


Eb_ = ExpressionBlock


def test_all_values_from():

    test_case = """
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
        <owl:Class rdf:ID="c"/>
        <owl:ObjectProperty rdf:ID="p"/>
    </rdf:RDF>
    """

    test_base = """
    <rdf:RDF
    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
    xmlns:owl="http://www.w3.org/2002/07/owl#"
    xmlns:first="http://www.w3.org/2002/03owlt/allValuesFrom/premises001#"
    xml:base="http://www.w3.org/2002/03owlt/allValuesFrom/premises001" >
    <owl:Class rdf:ID="c"/>
    <owl:ObjectProperty rdf:ID="p"/>
        <first:r rdf:ID="i">
            <rdf:type rdf:resource="http://www.w3.org/2002/07/owl#Thing"/>
            <first:p>
                <owl:Thing rdf:ID="o" />
            </first:p>
        </first:r>
    </rdf:RDF>
    """

    expected = """
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

    # x = Symbol("x")
    # y = Symbol("y")
    # owl_class = Symbol("owl_Class")
    # elem1 = Constant("elem1")

    owl_class(elem1)

    dl = Datalog()
    onto = OntologyParser(io.StringIO(test_case))
    dl = onto.parse_ontology(dl)
    sigmaB = dl.get_constraints()

    # q = I_(p(b), hasCollaborator(a, db, b))
    # qB = EB_((q,))

    dt = DatalogTranslator()
    qB = dt.walk(qB)

    orw = OntologyRewriter(qB, sigmaB)
    rewrite = orw.Xrewrite()


def test_has_value():

    test_case = """
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
    """

    expected = """
    <rdf:RDF
    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
    xmlns:owl="http://www.w3.org/2002/07/owl#"
    xml:base="http://www.w3.org/2002/03owlt/equivalentProperty/conclusions004" >
        <owl:ObjectProperty rdf:about="premises004#p">
            <owl:equivalentProperty>
                <owl:ObjectProperty rdf:about="premises004#q"/>
            </owl:equivalentProperty>
        </owl:ObjectProperty>
    </rdf:RDF>
    """

    dl = Datalog()
    onto = OntologyParser(io.StringIO(test_case))
    dl = onto.parse_ontology(dl)


def test_not_implemented():

    test_case = """
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
    """

    dl = Datalog()
    onto = OntologyParser(io.StringIO(test_case))

    with pytest.raises(NeuroLangNotImplementedError):
        dl = onto.parse_ontology(dl)
