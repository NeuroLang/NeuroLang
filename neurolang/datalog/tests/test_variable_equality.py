import operator

from ...datalog.basic_representation import DatalogProgram
from ...expression_walker import ExpressionWalker
from ...expressions import Constant, Symbol
from ...logic import Conjunction, Implication, Union
from ..expression_processing import (
    PropagatedEqualityRemover,
    VariableEqualityPropagator,
)

P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
x = Symbol("x")
y = Symbol("y")
z = Symbol("z")
a = Constant("a")
b = Constant("b")
c = Constant("c")

EQ = Constant(operator.eq)


class DatalogWithVariableEqualityPropagation(
    VariableEqualityPropagator,
    PropagatedEqualityRemover,
    DatalogProgram,
    ExpressionWalker,
):
    pass


def test_propagation_to_one_conjunct():
    rule = Implication(R(x), Conjunction((P(x, y), EQ(y, a))))
    program = DatalogWithVariableEqualityPropagation()
    program.walk(rule)
    assert R in program.intensional_database()
    expected = Union((Implication(R(x), Conjunction((P(x, a),))),))
    assert expected == program.intensional_database()[R]


def test_propagation_to_two_conjuncts():
    rule = Implication(R(x, y), Conjunction((P(x, y), EQ(y, a), Q(y, y))))
    program = DatalogWithVariableEqualityPropagation()
    program.walk(rule)
    assert R in program.intensional_database()
    expected = Union((Implication(R(x, a), Conjunction((P(x, a), Q(a, a)))),))
    assert expected == program.intensional_database()[R]


def test_single_equality_antecedent():
    rule = Implication(R(x), Conjunction((EQ(x, a))))
    program = DatalogWithVariableEqualityPropagation()
    program.walk(rule)
    assert R in program.intensional_database()
    expected = Union((Implication(R(x, a), Conjunction((P(x, a), Q(a, a)))),))
    assert expected == program.intensional_database()[R]
