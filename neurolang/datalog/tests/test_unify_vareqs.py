import operator
import typing

import pytest

from ...datalog.aggregation import (
    DatalogWithAggregationMixin,
    TranslateToLogicWithAggregation,
)
from ...datalog.basic_representation import DatalogProgram
from ...exceptions import ForbiddenExpressionError
from ...expression_walker import ExpressionWalker
from ...expressions import Constant, Symbol
from ...logic import Conjunction, Implication, Union
from ...type_system import Unknown
from ...utils.testing.logic import logic_exp_commutative_equal
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


class DatalogWithVariableEqualityUnification(
    TranslateToLogic,
    CollapseConjunctiveAntecedents,
    UnifyVariableEqualitiesMixin,
    RemoveDuplicatedAntecedentPredicates,
    DatalogProgram,
    ExpressionWalker,
):
    pass


def test_one_conjunct():
    rule = Implication(R(x), Conjunction((P(x, y), EQ(y, a))))
    program = DatalogWithVariableEqualityUnification()
    program.walk(rule)
    assert R in program.intensional_database()
    expected = Union((Implication(R(x), P(x, a)),))
    result = program.intensional_database()[R]
    assert result == expected


def test_two_conjuncts():
    rule = Implication(R(x, y), Conjunction((P(x, y), EQ(y, a), Q(y, y))))
    program = DatalogWithVariableEqualityUnification()
    program.walk(rule)
    expected = Union((Implication(R(x, a), Conjunction((P(x, a), Q(a, a)))),))
    assert R in program.intensional_database()
    result = program.intensional_database()[R]
    assert logic_exp_commutative_equal(result, expected)


def test_single_equality_antecedent():
    rule = Implication(R(x), Conjunction((EQ(x, a),)))
    program = DatalogWithVariableEqualityUnification()
    program.walk(rule)
    assert R not in program.intensional_database()
    assert R in program.extensional_database()


def test_between_vars_equalities():
    rule = Implication(R(x, y), Conjunction((P(x, y), EQ(y, x), Q(y, y))))
    program = DatalogWithVariableEqualityUnification()
    program.walk(rule)
    assert R in program.intensional_database()
    expecteds = [
        Union((Implication(R(x, x), Conjunction((P(x, x), Q(x, x)))),)),
        Union((Implication(R(y, y), Conjunction((P(y, y), Q(y, y)))),)),
    ]
    result = program.intensional_database()[R]
    assert any(
        logic_exp_commutative_equal(result, expected) for expected in expecteds
    )


def test_multiple_between_vars_equalities():
    rule = Implication(
        R(x, y, z), Conjunction((P(z, x), EQ(y, z), EQ(y, x), Q(y, y)))
    )
    program = DatalogWithVariableEqualityUnification()
    program.walk(rule)
    expecteds = [
        Union((Implication(R(v, v, v), Conjunction((P(v, v), Q(v, v)))),))
        for v in (x, y, z)
    ]
    assert R in program.intensional_database()
    result = program.intensional_database()[R]
    assert any(
        logic_exp_commutative_equal(result, expected) for expected in expecteds
    )


def test_mix_between_var_eqs_var_to_const_eq():
    rule = Implication(
        R(x, y, z),
        Conjunction((P(z, x), EQ(y, z), EQ(y, a), Q(y, y), EQ(b, x))),
    )
    program = DatalogWithVariableEqualityUnification()
    program.walk(rule)
    expected = Union(
        (
            Implication(
                R(b, a, a),
                Conjunction((P(a, b), Q(a, a))),
            ),
        )
    )
    assert R in program.intensional_database()
    result = program.intensional_database()[R]
    assert logic_exp_commutative_equal(result, expected)


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
    program = DatalogWithVariableEqualityUnification()
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
    assert logic_exp_commutative_equal(result, expected)


def test_extra_var_eq_const_eq():
    rule = Implication(
        Z(x, y),
        Conjunction((Q(x, y), EQ(z, y), EQ(y, a))),
    )
    program = DatalogWithVariableEqualityUnification()
    program.walk(rule)
    expected = Union((Implication(Z(x, a), Q(x, a)),))
    result = program.intensional_database()[Z]
    assert logic_exp_commutative_equal(result, expected)
    rule = Implication(
        Z(x, y, z),
        Conjunction((Q(x, y), EQ(z, y), EQ(y, a))),
    )
    program = DatalogWithVariableEqualityUnification()
    program.walk(rule)
    expected = Union((Implication(Z(x, a, a), Q(x, a)),))
    result = program.intensional_database()[Z]
    assert logic_exp_commutative_equal(result, expected)


class DatalogWithVariableEqualityUnificationAndAggregation(
    TranslateToLogicWithAggregation,
    UnifyVariableEqualitiesMixin,
    RemoveDuplicatedAntecedentPredicates,
    DatalogWithAggregationMixin,
    DatalogProgram,
    ExpressionWalker,
):
    def function_sum(
        self,
        x: typing.AbstractSet,
        y: typing.AbstractSet,
    ) -> Unknown:
        return sum(v + w for v, w in zip(x, y))


def test_aggregation():
    rule = Implication(
        P(x, Constant(sum)(y, z)),
        Conjunction(
            (
                R(x, y, z),
                EQ(z, a),
            )
        ),
    )
    program = DatalogWithVariableEqualityUnificationAndAggregation()
    with pytest.raises(ForbiddenExpressionError):
        program.walk(rule)
