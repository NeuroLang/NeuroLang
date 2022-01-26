import operator

import pytest

from ...datalog.expression_processing import UnifyVariableEqualitiesMixin
from ...expression_walker import (
    ExpressionWalker,
    ResolveSymbolMixin,
    TypedSymbolTableEvaluatorMixin
)
from ...expressions import Constant, FunctionApplication, Symbol
from ...logic import Conjunction, Disjunction, Implication, Negation
from .. import ranking
from ..cplogic.program import CPLogicProgram
from ..probabilistic_ra_utils import (
    ProbabilisticFactSet,
    generate_probabilistic_symbol_table_for_query
)
from ..transforms import convert_rule_to_ucq

EQ = Constant(operator.eq)

P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
w = Symbol("w")
x = Symbol("x")
y = Symbol("y")
z = Symbol("z")
ans = Symbol("ans")
a = Constant("a")
b = Constant("b")
d = Constant("d")


class CPLogicProgramWithVarEqUnification(
    UnifyVariableEqualitiesMixin,
    CPLogicProgram,
):
    pass


class Ranking(
    ranking.Rank,
    ResolveSymbolMixin,
    ExpressionWalker
):
    def __init__(self, symbol_table):
        self.symbol_table = symbol_table


def test_ranked():
    w = Symbol('w')
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    P = Symbol('P')
    Q = Symbol('Q')
    R = Symbol('R')
    S = Symbol('S')

    query = Implication(P(x), Conjunction((R(x, y), S(y, z))))
    query = convert_rule_to_ucq(query)

    assert ranking.is_ranked(query)

    query = Implication(
        P(x),
        Conjunction((R(x, y), S(y, z), R(x, z, y)))
    )
    query = convert_rule_to_ucq(query)
    assert not ranking.is_ranked(query)

    query = Implication(
        P(x),
        Disjunction((Conjunction((R(x, y), S(y, z))), R(x, z, y)))
    )
    query = convert_rule_to_ucq(query)
    assert not ranking.is_ranked(query)


def test_ranking_self_join():
    query = Implication(ans(w, x, y, z), Conjunction((P(w, x), P(y, z))))
    cpl = CPLogicProgramWithVarEqUnification()
    cpl.add_probabilistic_facts_from_tuples(
        P, [(0.2, "a", "b"), (1.0, "a", "c"), (0.7, "b", "b")]
    )
    symbol_table = generate_probabilistic_symbol_table_for_query(cpl, query)
    ranking = Ranking(symbol_table)
    res = ranking.walk(query.antecedent)

    assert res
