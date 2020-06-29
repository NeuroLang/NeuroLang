import io

from ...datalog.basic_representation import DatalogProgram
from ...datalog.constraints_representation import DatalogConstraintsProgram
from ...datalog.expressions import Implication
from ...datalog.ontologies_parser import OntologyParser
from ...expression_walker import ExpressionBasicEvaluator
from ...expressions import Symbol
from ...logic import Conjunction, Union
from ...region_solver import RegionSolver
from ..cplogic.program import CPLogicMixin
from ..expression_processing import separate_deterministic_probabilistic_code


class DeterministicSolver(
    RegionSolver, DatalogProgram, ExpressionBasicEvaluator
):
    pass


class ProbabilisticSolver(
    RegionSolver, CPLogicMixin, DatalogProgram, ExpressionBasicEvaluator
):
    pass


class DeterministicOntologySolver(
    RegionSolver, DatalogConstraintsProgram, ExpressionBasicEvaluator
):
    pass


class ProbabilisticOntologySolver(
    RegionSolver,
    CPLogicMixin,
    DatalogConstraintsProgram,
    ExpressionBasicEvaluator,
):
    pass


def test_deterministic_code():
    det_solver = DeterministicSolver()
    pass


def test_probabilistic_code():
    prob_solver = ProbabilisticSolver()
    pass


def test_deterministic_constraints_code():
    det_onto_solver = DeterministicOntologySolver()

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
    p2 = Symbol("http://www.w3.org/2002/03owlt/hasValue/premises001#p")
    test_base_q = Union((Implication(answer(x, y), p2(x, y)),))

    onto = OntologyParser(io.StringIO(test_case))
    predicate_tuples, union_of_constraints = onto.parse_ontology()
    det_onto_solver.walk(union_of_constraints)
    triples = predicate_tuples[onto.get_triples_symbol()]
    pointers = predicate_tuples[onto.get_pointers_symbol()]

    det_onto_solver.add_extensional_predicate_from_tuples(
        onto.get_triples_symbol(), triples
    )
    det_onto_solver.add_extensional_predicate_from_tuples(
        onto.get_pointers_symbol(), pointers
    )

    det_onto_solver.walk(test_base_q)
    det, prob = separate_deterministic_probabilistic_code(det_onto_solver)

    assert len(prob.formulas) == 0
    assert len(det.formulas) == 1


def test_probabilistic_constraints_code():
    prob_onto_solver = ProbabilisticOntologySolver()
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

    answer1 = Symbol("answer1")
    answer2 = Symbol("answer2")
    x = Symbol("x")
    y = Symbol("y")
    prob = Symbol("prob")
    p2 = Symbol("http://www.w3.org/2002/03owlt/hasValue/premises001#p")
    test_base_q1 = Union((Implication(answer1(x, y), p2(x, y)),))
    test_base_q2 = Union(
        (Implication(answer2(x), Conjunction((answer1(x, y), prob(x)))),)
    )

    onto = OntologyParser(io.StringIO(test_case))
    predicate_tuples, union_of_constraints = onto.parse_ontology()
    prob_onto_solver.walk(union_of_constraints)
    triples = predicate_tuples[onto.get_triples_symbol()]
    pointers = predicate_tuples[onto.get_pointers_symbol()]

    prob_onto_solver.add_extensional_predicate_from_tuples(
        onto.get_triples_symbol(), triples
    )
    prob_onto_solver.add_extensional_predicate_from_tuples(
        onto.get_pointers_symbol(), pointers
    )

    iterable = [
        "http://www.w3.org/2002/03owlt/hasValue/premises001#i",
        "http://www.w3.org/2002/03owlt/hasValue/premises001#b",
    ]
    probability = 1 / len(iterable)
    iterable = [(probability,) + tuple(tupl) for tupl in iterable]
    prob_onto_solver.add_probabilistic_choice_from_tuples(prob, iterable)
    prob_onto_solver.walk(test_base_q1)
    prob_onto_solver.walk(test_base_q2)
    det, prob = separate_deterministic_probabilistic_code(prob_onto_solver)

    a = 1
    assert len(prob.formulas) == 1
    assert len(det.formulas) == 1
