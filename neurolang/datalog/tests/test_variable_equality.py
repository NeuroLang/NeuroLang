import operator

from ...datalog.basic_representation import DatalogProgram
from ...expression_walker import ExpressionWalker
from ...expressions import Constant, Symbol
from ...logic import Conjunction, Implication, Union
from ..expression_processing import (
    CollapseConjunctiveAntecedents,
    RemoveDuplicatedAntecedentPredicates,
    TranslateToLogic,
    UnifyVariableEqualitiesMixin,
)

P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
Z = Symbol("Z")
x = Symbol("x")
y = Symbol("y")
z = Symbol("z")
a = Constant("a")
b = Constant("b")
c = Constant("c")

EQ = Constant(operator.eq)


class DatalogWithVariableEqualityPropagation(
    TranslateToLogic,
    CollapseConjunctiveAntecedents,
    UnifyVariableEqualitiesMixin,
    RemoveDuplicatedAntecedentPredicates,
    DatalogProgram,
    ExpressionWalker,
):
    pass


def test_propagation_to_one_conjunct():
    rule = Implication(R(x), Conjunction((P(x, y), EQ(y, a))))
    program = DatalogWithVariableEqualityPropagation()
    program.walk(rule)
    assert R in program.intensional_database()
    expected = Union((Implication(R(x), P(x, a)),))
    result = program.intensional_database()[R]
    assert result == expected


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


def test_collapsable_conjunction():
    rule = Implication(
        R(x, y),
        Conjunction(
            (
                Conjunction((P(z), Q(y, x), EQ(y, z))),
                Conjunction((Q(z, y), P(z), P(y), EQ(z, a))),
                Z(b),
                Z(z),
            )
        ),
    )
    program = DatalogWithVariableEqualityPropagation()
    program.walk(rule)
    expected = Union(
        (
            Implication(
                R(x, a),
                Conjunction((P(a), Q(a, x), Q(a, a), Z(b), Z(a))),
            ),
        )
    )
    result = program.intensional_database()[R]
    assert isinstance(result, Union)
    assert len(result.formulas) == 1
    assert isinstance(result.formulas[0], Implication)
    assert result.formulas[0].consequent == expected.formulas[0].consequent
    assert set(result.formulas[0].antecedent.formulas) == set(
        expected.formulas[0].antecedent.formulas
    )
