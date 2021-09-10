import io
import operator

from ...datalog.basic_representation import DatalogProgram
from ...datalog.constraints_representation import DatalogConstraintsProgram
from ...datalog.expressions import Implication
from ...datalog.ontologies_parser import OntologyParser
from ...expression_walker import ExpressionBasicEvaluator
from ...expressions import Constant, Symbol
from ...logic import Conjunction, Union
from ..cplogic.program import CPLogicMixin
from ..expression_processing import separate_deterministic_probabilistic_code


class ProbabilisticSolver(
    CPLogicMixin, DatalogProgram, ExpressionBasicEvaluator
):
    pass


class ProbabilisticOntologySolver(
    CPLogicMixin, DatalogConstraintsProgram, ExpressionBasicEvaluator
):
    pass


def test_builtin():
    answer1 = Symbol("answer1")
    equals = Constant(operator.eq)
    y = Symbol("y")
    x = Symbol("x")
    sym1 = Symbol("sym1")
    sym2 = Symbol("sym2")

    prob_solver = ProbabilisticSolver()
    s1 = [(1,), (2,), (3,), (4,), (5,)]
    s2 = [(1,), (2,), (3,)]
    prob_solver.add_extensional_predicate_from_tuples(sym1, s1)
    prob_solver.add_extensional_predicate_from_tuples(sym2, s2)

    test_q1 = Union(
        (
            Implication(
                answer1(y), Conjunction((sym1(y), sym2(x), equals(y, x)))
            ),
        )
    )
    prob_solver.walk(test_q1)

    det, prob = separate_deterministic_probabilistic_code(prob_solver)

    assert len(det.formulas) == 1
    assert len(prob.formulas) == 0


def test_probabilistic_code():
    prob_solver = ProbabilisticSolver()
    answer1 = Symbol("answer1")
    answer2 = Symbol("answer2")
    x = Symbol("x")
    y = Symbol("y")
    prob1 = Symbol("prob1")
    prob2 = Symbol("prob2")
    prob3 = Symbol("prob3")

    d1 = [(1,), (2,), (3,), (4,), (5,)]
    d2 = [(2, "a"), (3, "b"), (4, "d"), (5, "c"), (7, "z")]
    d3 = [("a",), ("b",), ("c",)]

    p_d1 = 1 / len(d1)
    p_d2 = 1 / len(d2)
    p_d3 = 1 / len(d3)
    iterable_d1 = [(p_d1,) + tupl for tupl in d1]
    iterable_d2 = [(p_d2,) + tupl for tupl in d2]
    iterable_d3 = [(p_d3,) + tupl for tupl in d3]
    prob_solver.add_probabilistic_choice_from_tuples(prob1, iterable_d1)
    prob_solver.add_probabilistic_choice_from_tuples(prob2, iterable_d2)
    prob_solver.add_probabilistic_choice_from_tuples(prob3, iterable_d3)

    test_q1 = Union(
        (Implication(answer1(y), Conjunction((prob2(x, y), prob1(x)))),)
    )
    test_q2 = Union(
        (Implication(answer2(x), Conjunction((answer1(x), prob3(x)))),)
    )

    prob_solver.walk(test_q1)
    prob_solver.walk(test_q2)
    det, prob = separate_deterministic_probabilistic_code(prob_solver)

    assert len(prob.formulas) == 2
    assert len(det.formulas) == 0


def test_probabilistic_ontology_code():
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

    assert len(prob.formulas) == 1
    assert len(det.formulas) == 1
