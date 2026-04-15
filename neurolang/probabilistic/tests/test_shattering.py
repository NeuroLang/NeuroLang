import operator
from typing import AbstractSet

from pytest import fixture

from ...datalog.expression_processing import extract_logic_atoms
from ...datalog.translate_to_named_ra import TranslateToNamedRA
from ...datalog.wrapped_collections import WrappedRelationalAlgebraSet
from ...expressions import Constant, Symbol
from ...logic import Conjunction, Negation
from ...relational_algebra import Projection, Selection, int2columnint_constant
from ...relational_algebra.optimisers import (
    PushUnnamedSelectionsUp,
    RelationalAlgebraOptimiser
)
from ...relational_algebra.relational_algebra import RelationalAlgebraSolver
from ..shattering import sets_per_symbol, shatter_constants

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


def test_symbols_per_set_unitary():
    query = Conjunction((Q(x), Q(a), Q(b), P(x), Negation(Q(a))))

    res = sets_per_symbol(query)

    c0 = int2columnint_constant(0)
    expected = {
        Q: set([
            Projection(Selection(Q, EQ(c0, a)), (c0,)),
            Projection(Selection(Q, EQ(c0, b)), (c0,)),
            Projection(
                Selection(Q, Conjunction((NE(c0, a), NE(c0, b)))),
                (int2columnint_constant(0),)
            )
        ])
    }

    assert res == expected


def test_symbols_per_set_unitary_ne():
    query = Conjunction((Q(x), Q(a), Q(b), P(x), Negation(Q(a)), NE(d, x)))

    res = sets_per_symbol(query)

    c0 = int2columnint_constant(0)
    expected = {
        Q: set([
            Projection(Selection(Q, EQ(c0, a)), (c0,)),
            Projection(Selection(Q, EQ(c0, b)), (c0,)),
            Projection(Selection(Q, EQ(c0, d)), (c0,)),
            Projection(
                Selection(Q, Conjunction((NE(c0, a), NE(c0, b), NE(c0, d)))),
                (c0,)
            ),
        ]),
        P: set([
            Projection(Selection(P, EQ(c0, d)), (c0,)),
            Projection(
                Selection(P, NE(c0, d)),
                (c0,)
            )

        ])
    }

    assert res == expected


def test_symbols_per_set_binary():
    query = Conjunction((Q(x, b), Q(a, x), Q(b, y), NE(y, b)))

    res = sets_per_symbol(query)

    c0 = int2columnint_constant(0)
    c1 = int2columnint_constant(1)
    expected = {
        Q: set([
            Projection(
                Selection(Q, Conjunction((EQ(c0, a), EQ(c1, b)))), (c0, c1,)
            ),
            Projection(
                Selection(Q, Conjunction((EQ(c0, b), EQ(c1, b)))), (c0, c1,)
            ),
            Projection(
                Selection(Q, Conjunction((EQ(c0, a), NE(c1, b)))), (c0, c1,)
            ),
            Projection(
                Selection(Q, Conjunction((EQ(c0, b), NE(c1, b)))), (c0, c1,)
            ),
            Projection(
                Selection(Q, Conjunction((NE(c0, a), NE(c0, b), EQ(c1, b)))),
                (c0, c1)
            ),
            Projection(
                Selection(Q, Conjunction((NE(c0, a), NE(c0, b), NE(c1, b)))),
                (c0, c1)
            )
        ])
    }

    assert res == expected


@fixture
def R1():
    return Constant[AbstractSet](
        WrappedRelationalAlgebraSet([(chr(ord('a') + i),) for i in range(10)])
    )


@fixture
def R2():
    return Constant[AbstractSet](
        WrappedRelationalAlgebraSet([
            (chr(ord('a') + i * 2), i)
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


class RelationalAlgebraPushInSolver(
    PushUnnamedSelectionsUp, RelationalAlgebraSolver
):
    pass


def test_shatter_unitary(R1, R2, R3):
    query = Conjunction((Q(x), Q(a), Q(b), R(y), Negation(P(b, y))))

    ra_query = TranslateToNamedRA().walk(query)

    symbol_table = dict({Q: R1, P: R2, R: R3})
    ras = RelationalAlgebraSolver(symbol_table)
    ra_sol = ras.walk(ra_query)

    res = shatter_constants(query, symbol_table, True)

    ras = RelationalAlgebraPushInSolver(symbol_table)
    ra_query = TranslateToNamedRA().walk(res)
    shattered_sol = ras.walk(ra_query)

    assert all(
        all(isinstance(arg, Symbol) for arg in atom.args)
        for atom in extract_logic_atoms(res)
    )
    assert ra_sol == shattered_sol


def test_shatter_binary_1(R4):
    query = Conjunction((Q(x, a), Q(y, a), Q(x, y)))
    ra_query = TranslateToNamedRA().walk(query)

    symbol_table = dict({Q: R4})
    ras = RelationalAlgebraSolver(symbol_table)
    ra_sol = ras.walk(ra_query)

    res = shatter_constants(query, symbol_table, True)

    ras = RelationalAlgebraPushInSolver(symbol_table)
    ra_query = TranslateToNamedRA().walk(res)
    shattered_sol = ras.walk(ra_query)

    assert all(
        all(isinstance(arg, Symbol) for arg in atom.args)
        for atom in extract_logic_atoms(res)
    )
    assert ra_sol == shattered_sol


def test_shatter_binary_2(R1, R4):
    query = Conjunction((Q(x, b), Q(a, x), Q(x, y), R(z)))
    ra_query = TranslateToNamedRA().walk(query)

    symbol_table = dict({Q: R4, R: R1})
    ras = RelationalAlgebraSolver(symbol_table)
    ra_sol = ras.walk(ra_query)

    res = shatter_constants(query, symbol_table, True)

    ras = RelationalAlgebraPushInSolver(symbol_table)
    ra_query = TranslateToNamedRA().walk(res)
    shattered_sol = ras.walk(ra_query)

    assert all(
        all(isinstance(arg, Symbol) for arg in atom.args)
        for atom in extract_logic_atoms(res)
    )
    assert ra_sol == shattered_sol


def test_shatter_binary_3(R4):
    query = Conjunction((Q(x, b), Q(y, y)))
    ra_query = TranslateToNamedRA().walk(query)

    symbol_table = dict({Q: R4})
    ras = RelationalAlgebraSolver(symbol_table)
    ra_sol = ras.walk(ra_query)

    res = shatter_constants(query, symbol_table, True)

    ras = RelationalAlgebraPushInSolver(symbol_table)
    ra_query = TranslateToNamedRA().walk(res)
    shattered_sol = ras.walk(ra_query)

    assert all(
        all(isinstance(arg, Symbol) for arg in atom.args)
        for atom in extract_logic_atoms(res)
    )
    assert ra_sol == shattered_sol


def test_shatter_binary_inequality(R1, R4):
    query = Conjunction((Q(x, a), Q(a, x), Q(y, y), NE(x, b), R(z)))
    ra_query = TranslateToNamedRA().walk(query)

    symbol_table = dict({Q: R4, R: R1})
    ras = RelationalAlgebraSolver(symbol_table)
    ra_sol = ras.walk(ra_query)

    res = shatter_constants(query, symbol_table, True)

    ras = RelationalAlgebraPushInSolver(symbol_table)
    ra_query = TranslateToNamedRA().walk(res)
    shattered_sol = ras.walk(ra_query)

    assert all(
        all(isinstance(arg, Symbol) for arg in atom.args)
        for atom in extract_logic_atoms(res)
    )
    assert ra_sol == shattered_sol
