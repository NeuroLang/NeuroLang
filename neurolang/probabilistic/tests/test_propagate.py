import typing

import pytest

from ...datalog.basic_representation import WrappedRelationalAlgebraSet
from ...exceptions import ForbiddenDisjunctionError
from ...expressions import Constant, FunctionApplication, Symbol
from ...logic import Implication, Union
from ...relational_algebra import (
    Difference,
    NameColumns,
    NamedRelationalAlgebraFrozenSet,
    NaturalJoin,
    Projection,
    RelationalAlgebraSolver,
    int2columnint_constant,
    str2columnstr_constant
)
from ..cplogic.program import CPLogicProgram
from ..expressions import ProbabilisticPredicate
from ..propagate import EQ, PushUp, PushUpWalker, remove_push_up_from_top
from ..query_resolution import QueryBasedProbFactToDetRule

P = Symbol("P")
Q = Symbol("Q")
x = Symbol("x")
y = Symbol("y")
p = Symbol("p")


@pytest.fixture
def R1():
    return WrappedRelationalAlgebraSet([(i, i * 2) for i in range(10)])


@pytest.fixture
def R2():
    return WrappedRelationalAlgebraSet([(i * 2, i * 3) for i in range(10)])


def test_unnamed_to_named(R1):
    r1 = Constant[typing.AbstractSet](R1)
    query = NameColumns(
        Projection(
            PushUp(
                r1,
                EQ(int2columnint_constant(1), str2columnstr_constant('b'))
            ),
            (int2columnint_constant(0),)
        ),
        (str2columnstr_constant('a'),)
    )
    query = PushUpWalker().walk(query)
    query = remove_push_up_from_top(query)
    solution = RelationalAlgebraSolver({}).walk(query)

    assert solution.value == NamedRelationalAlgebraFrozenSet(('a', 'b'), R1)

    query = NameColumns(
        Projection(
            PushUp(
                r1,
                EQ(int2columnint_constant(0), str2columnstr_constant('a'))
            ),
            (int2columnint_constant(1),)
        ),
        (str2columnstr_constant('b'),)
    )
    query = PushUpWalker().walk(query)
    query = remove_push_up_from_top(query)
    solution = RelationalAlgebraSolver({}).walk(query)

    assert solution.value == NamedRelationalAlgebraFrozenSet(('a', 'b'), R1)


def test_natural_join(R1, R2):
    r1 = Constant[typing.AbstractSet](R1)
    r2 = Constant[typing.AbstractSet](R2)
    r1_ = NameColumns(
        Projection(
            PushUp(
                r1,
                EQ(int2columnint_constant(1), str2columnstr_constant('b'))
            ),
            (int2columnint_constant(0),)
        ),
        (str2columnstr_constant('a'),)
    )
    r2_ = NameColumns(
        Projection(
            PushUp(
                r2,
                EQ(int2columnint_constant(1), str2columnstr_constant('c'))
            ),
            (int2columnint_constant(0),)
        ),
        (str2columnstr_constant('a'),)
    )
    query = NaturalJoin(r1_, r2_)
    query = PushUpWalker().walk(query)
    query = remove_push_up_from_top(query)
    solution = RelationalAlgebraSolver({}).walk(query)

    assert solution.value == (
        NamedRelationalAlgebraFrozenSet(('a', 'b'), R1).naturaljoin(
            NamedRelationalAlgebraFrozenSet(('a', 'c'), R2)
        )
    )


def test_difference(R1, R2):
    r1 = Constant[typing.AbstractSet](R1)
    r2 = Constant[typing.AbstractSet](R2)
    r1_ = NameColumns(
        Projection(
            PushUp(
                r1,
                EQ(int2columnint_constant(1), str2columnstr_constant('b'))
            ),
            (int2columnint_constant(0),)
        ),
        (str2columnstr_constant('a'),)
    )
    r2_ = NameColumns(
        Projection(
            PushUp(
                r2,
                EQ(int2columnint_constant(1), str2columnstr_constant('b'))
            ),
            (int2columnint_constant(0),)
        ),
        (str2columnstr_constant('a'),)
    )
    query = Difference(r1_, r2_)
    query = PushUpWalker().walk(query)
    query = remove_push_up_from_top(query)
    solution = RelationalAlgebraSolver({}).walk(query)

    assert solution.value == (
        NamedRelationalAlgebraFrozenSet(('a', 'b'), R1) -
        NamedRelationalAlgebraFrozenSet(('a', 'b'), R2)
    )
