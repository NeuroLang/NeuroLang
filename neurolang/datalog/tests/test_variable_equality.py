import operator

from ...datalog.basic_representation import DatalogProgram
from ...expression_walker import ExpressionWalker
from ...expressions import Constant, Symbol
from ...logic import Conjunction, Implication, Union
from ..expression_processing import (
    EliminateTrivialTrueCases,
    VariableEqualityUnifier,
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
    VariableEqualityUnifier,
    EliminateTrivialTrueCases,
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
    rule = Implication(R(x), Conjunction((EQ(x, a),)))
    program = DatalogWithVariableEqualityPropagation()
    program.walk(rule)
    assert R not in program.intensional_database()
    assert R in program.extensional_database()


def test_between_vars_equality_propagation():
    rule = Implication(R(x, y), Conjunction((P(x, y), EQ(y, x), Q(y, y))))
    program = DatalogWithVariableEqualityPropagation()
    program.walk(rule)
    assert R in program.intensional_database()
    expecteds = [
        Union((Implication(R(x, x), Conjunction((P(x, x), Q(x, x)))),)),
        Union((Implication(R(y, y), Conjunction((P(y, y), Q(y, y)))),)),
    ]
    result = program.intensional_database()[R]
    assert any(hash(expected) == hash(result) for expected in expecteds)


def test_multiple_between_vars_equalities():
    rule = Implication(
        R(x, y, z), Conjunction((P(z, x), EQ(y, z), EQ(y, x), Q(y, y)))
    )
    program = DatalogWithVariableEqualityPropagation()
    program.walk(rule)
    assert R in program.intensional_database()
    expecteds = [
        Union((Implication(R(v, v, v), Conjunction((P(v, v), Q(v, v)))),))
        for v in (x, y, z)
    ]
    result = program.intensional_database()[R]
    assert any(hash(expected) == hash(result) for expected in expecteds)


def test_mix_between_var_eqs_var_to_const_eq():
    rule = Implication(
        R(x, y, z),
        Conjunction((P(z, x), EQ(y, z), EQ(y, a), Q(y, y), EQ(b, x))),
    )
    program = DatalogWithVariableEqualityPropagation()
    program.walk(rule)
    assert R in program.intensional_database()
    expected = Union(
        (
            Implication(
                R(b, a, a),
                Conjunction((P(a, b), Q(a, a))),
            ),
        )
    )
    result = program.intensional_database()[R]
    assert hash(result) == hash(expected)
