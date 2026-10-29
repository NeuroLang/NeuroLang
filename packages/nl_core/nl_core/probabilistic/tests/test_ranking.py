import operator
from typing import AbstractSet, Tuple

import pytest
from pytest import fixture

from ...datalog.translate_to_named_ra import TranslateToNamedRA
from ...datalog.wrapped_collections import WrappedRelationalAlgebraSet
from nl_expr.expressions import Constant, Symbol
from nl_expr.exceptions import NotRankedException
from nl_expr.logic import (
    Conjunction,
    Disjunction,
    ExistentialPredicate,
    Negation,
)
from nl_expr.logic.expression_processing import extract_logic_atoms
from ...relational_algebra import int2columnint_constant
from ...relational_algebra.relational_algebra import RelationalAlgebraSolver
from ..probabilistic_ra_utils import ProbabilisticChoiceSet
from ..transforms import add_existentials_except
from ..ranking import (
    partially_rank_query,
    rank_query,
    verify_that_the_query_is_ranked
)

EQ = Constant(operator.eq)
NE = Constant(operator.ne)

P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
x = Symbol("x")
y = Symbol("y")
z = Symbol("z")


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
    assert not verify_that_the_query_is_ranked(_boolean(query))

    query = Conjunction(
        (Q(x, y), Q(x, z), P(z, x), P(x, y), Negation(Q(y, z)))
    )
    assert not verify_that_the_query_is_ranked(_boolean(query))

    query = Disjunction((Q(x, y), Conjunction((Q(x, z), P(y, y)))))

    assert not verify_that_the_query_is_ranked(_boolean(query))

    query = Conjunction((Q(x, z), P(y, x), EQ(x, y)))

    assert not verify_that_the_query_is_ranked(_boolean(query))


def _boolean(query):
    return add_existentials_except(query, set())


def test_answer_variables_need_no_order():
    """The answer variables, the free ones, are constants: only the
    bound variables are ordered."""
    query = Conjunction((Q(x, y), Q(y, x)))
    assert verify_that_the_query_is_ranked(query)
    assert verify_that_the_query_is_ranked(
        ExistentialPredicate(y, Conjunction((Q(x, y), Q(y, x), P(y))))
    )
    assert not verify_that_the_query_is_ranked(_boolean(query))


def test_rank_query_answer_variable_at_a_ranked_position():
    query = ExistentialPredicate(y, Conjunction((Q(x, y), Q(y, y))))
    symbol_table = {Q: _pairs((1, 2), (2, 1), (1, 1))}
    with pytest.raises(NotRankedException, match="answer variable"):
        rank_query(query, symbol_table)


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

    assert not verify_that_the_query_is_ranked(_boolean(res))


def test_query_with_comparison_ranked():
    query = Conjunction((Q(x, y), P(z, y), NE(z, x)))
    assert verify_that_the_query_is_ranked(query)


def _pairs(*tuples):
    types = tuple(type(value) for value in tuples[0])
    return Constant[AbstractSet[Tuple[types]]](
        WrappedRelationalAlgebraSet(tuples), verify_type=False
    )


def test_rank_query_symmetric_pair():
    query = ExistentialPredicate(x, ExistentialPredicate(
        y, Conjunction((Q(x, y), Q(y, x)))
    ))
    symbol_table = {Q: _pairs((1, 2), (2, 1), (1, 1))}
    assert not verify_that_the_query_is_ranked(query)
    res = rank_query(query, symbol_table)
    assert verify_that_the_query_is_ranked(res)
    assert Q not in {atom.functor for atom in extract_logic_atoms(res)}


@pytest.mark.parametrize("tuples", (
    ((frozenset({1}), frozenset({1, 2})), (frozenset({1, 2}), frozenset())),
    ((1, "a"), (2, "b")),
), ids=("sets", "integers and strings"))
def test_rank_query_needs_a_total_order(tuples):
    """Ranking compares the values of the columns of the ranked
    variables: their type must have a total order and an equality, the
    same for all of them."""
    query = ExistentialPredicate(x, ExistentialPredicate(
        y, Conjunction((Q(x, y), Q(y, x)))
    ))
    with pytest.raises(NotRankedException):
        rank_query(query, {Q: _pairs(*tuples)})


def test_rank_query_with_a_choice_not_to_rank():
    """Only the relations with variables to rank are split: a
    probabilistic choice elsewhere in the query does not prevent it."""
    query = ExistentialPredicate(x, ExistentialPredicate(
        y, ExistentialPredicate(z, Conjunction((Q(x, y), Q(y, x), P(z))))
    ))
    symbol_table = {
        Q: _pairs((1, 2), (2, 1), (1, 1)),
        P: ProbabilisticChoiceSet(R, int2columnint_constant(0)),
    }
    res = rank_query(query, symbol_table)
    assert verify_that_the_query_is_ranked(res)
    assert P in {atom.functor for atom in extract_logic_atoms(res)}
