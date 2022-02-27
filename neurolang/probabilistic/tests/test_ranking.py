import operator
from re import I
from typing import AbstractSet

from pytest import fixture

from ...datalog.expression_processing import extract_logic_atoms
from ...datalog.translate_to_named_ra import TranslateToNamedRA
from ...datalog.wrapped_collections import WrappedRelationalAlgebraSet
from ...expressions import Constant, Symbol
from ...logic import Conjunction, Negation, Disjunction
from ...relational_algebra import Projection, Selection, int2columnint_constant
from ...relational_algebra.optimisers import (
    PushUnnamedSelectionsUp,
    RelationalAlgebraOptimiser
)
from ...relational_algebra.relational_algebra import RelationalAlgebraSolver
from ..ranking import verify_that_the_query_is_ranked, partially_rank_query

RAO = RelationalAlgebraOptimiser()


EQ = Constant(operator.eq)
NE = Constant(operator.ne)

P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
x = Symbol("x")
y = Symbol("y")
z = Symbol("z")
ans = Symbol("ans")
a = Constant("a")
b = Constant("b")
d = Constant("d")


def test_query_ranked():
    query = Conjunction(
        (Q(x, y), Q(x, z), Q(y, z), P(x, y), Negation(Q(y, z)))
    )
    assert verify_that_the_query_is_ranked(query)

    query = Disjunction((Q(x, y), Conjunction((Q(x, z), P(z, y)))))

    assert verify_that_the_query_is_ranked(query)


def test_query_not_ranked():
    query = Conjunction(
        (Q(x, y), Q(x, z), Q(z, x), P(x, y), Negation(Q(y, z)))
    )
    assert not verify_that_the_query_is_ranked(query)

    query = Conjunction(
        (Q(x, y), Q(x, z), P(z, x), P(x, y), Negation(Q(y, z)))
    )
    assert not verify_that_the_query_is_ranked(query)

    query = Disjunction((Q(x, y), Conjunction((Q(x, z), P(y, y)))))

    assert not verify_that_the_query_is_ranked(query)

    query = Conjunction((Q(x, z), P(y, x), EQ(x, y)))

    assert not verify_that_the_query_is_ranked(query)


@fixture
def R1():
    return Constant[AbstractSet](
        WrappedRelationalAlgebraSet([(chr(ord('a') + i),) for i in range(10)])
    )


@fixture
def R2():
    return Constant[AbstractSet](
        WrappedRelationalAlgebraSet([
            (i, chr(ord('a') + i * 2), i * 2)
            for i in range(10)
        ])
    )


@fixture
def R3():
    return Constant[AbstractSet](
        WrappedRelationalAlgebraSet([
            (i,)
            for i in range(10)
        ])
    )


@fixture
def R4():
    return Constant[AbstractSet](
        WrappedRelationalAlgebraSet([
            (
                chr(ord('a') + i * 2),
                chr(ord('a') + i)
            )
            for i in range(10)
        ])
    )


def test_ranking_repeated(R2, R4):
    query = Conjunction((R(x, y, z), Q(y, y)))
    ra_query = TranslateToNamedRA().walk(query)

    symbol_table = dict({Q: R4, R: R2})
    ras = RelationalAlgebraSolver(symbol_table)
    ra_sol = ras.walk(ra_query)

    res = partially_rank_query(query, symbol_table)

    assert verify_that_the_query_is_ranked(res)

    ras = RelationalAlgebraSolver(symbol_table)
    ra_query = TranslateToNamedRA().walk(res)
    ranked_sol = ras.walk(ra_query)

    assert ra_sol == ranked_sol

    query = Conjunction((R(x, y, x), Q(z, y)))
    ra_query = TranslateToNamedRA().walk(query)

    symbol_table = dict({Q: R4, R: R2})
    ras = RelationalAlgebraSolver(symbol_table)
    ra_sol = ras.walk(ra_query)

    res = partially_rank_query(query, symbol_table)

    assert verify_that_the_query_is_ranked(res)

    ras = RelationalAlgebraSolver(symbol_table)
    ra_query = TranslateToNamedRA().walk(res)
    ranked_sol = ras.walk(ra_query)

    assert ra_sol == ranked_sol


def test_cant_rank_repeated():
    query = Conjunction((Q(x, z), Q(y, y)))
    symbol_table = dict({Q: R4, R: R2})

    res = partially_rank_query(query, symbol_table)

    assert not verify_that_the_query_is_ranked(res)
