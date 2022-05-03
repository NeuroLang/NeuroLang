import operator
from typing import AbstractSet

from pytest import fixture

from ...datalog.translate_to_named_ra import TranslateToNamedRA
from ...datalog.wrapped_collections import WrappedRelationalAlgebraSet
from ...expressions import Constant, Symbol
from ...logic import Conjunction, Disjunction, ExistentialPredicate, Negation
from ...logic.expression_processing import extract_logic_free_variables
from ...relational_algebra.relational_algebra import RelationalAlgebraSolver
from ..ranking import partially_rank_query, verify_that_the_query_is_ranked

EQ = Constant(operator.eq)

P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
x = Symbol("x")
y = Symbol("y")
z = Symbol("z")


def add_all_existentials(query):
    for v in extract_logic_free_variables(query):
        query = ExistentialPredicate(v, query)
    return query


def test_query_ranked():
    query = add_all_existentials(Conjunction(
        (Q(x, y), Q(x, z), Q(y, z), P(x, y), Negation(Q(y, z)))
    ))
    assert verify_that_the_query_is_ranked(query)

    query = add_all_existentials(
        Disjunction((Q(x, y), Conjunction((Q(x, z), P(z, y)))))
    )

    assert verify_that_the_query_is_ranked(query)

    query = add_all_existentials(
        Disjunction((Q(x, y), Conjunction((Q(x, z), P(z, x)))))
    )

    assert verify_that_the_query_is_ranked(query)


def test_query_not_ranked():
    query = Conjunction(
        (Q(x, y), Q(x, z), Q(z, x), P(x, y), Negation(Q(y, x)))
    )
    assert not verify_that_the_query_is_ranked(query)

    query = Conjunction(
        (Q(x, y), Q(x, z), P(z, x), P(x, z), Negation(Q(y, z)))
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
