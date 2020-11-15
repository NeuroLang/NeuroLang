import operator

import pytest

from ...exceptions import UnsupportedProgramError
from ...expression_walker import ExpressionWalker
from ...expressions import Constant, Symbol
from ...logic import (
    FALSE,
    Conjunction,
    Disjunction,
    ExistentialPredicate,
    Implication,
    Union,
)
from ..basic_representation import DatalogProgram
from ..expression_processing import extract_logic_predicates, flatten_query

EQ = Constant(operator.eq)

P = Symbol("P")
R = Symbol("R")
Z = Symbol("Z")
Q = Symbol("Q")
x = Symbol("x")
y = Symbol("y")
z = Symbol("z")

a = Constant("a")
b = Constant("b")


class TestDatalogProgram(DatalogProgram, ExpressionWalker):
    pass


def test_flatten_extensional_predicate_query():
    result = flatten_query(P(x), TestDatalogProgram())
    assert result == Conjunction((P(x),))


def test_flatten_with_renaming():
    code = Union(
        (
            Implication(Z(x), P(x)),
            Implication(R(y), P(y)),
            Implication(Q(y), Conjunction((Z(y), R(x)))),
        )
    )
    program = TestDatalogProgram()
    program.walk(code)
    result = flatten_query(Q(x), program)
    assert len(result.formulas) == 2
    assert P(x) in result.formulas
    assert any(
        pred.functor == P and pred.args[0] != x for pred in result.formulas
    )


def test_flatten_query_multiple_preds():
    query = Conjunction((P(x), Z(y)))
    result = flatten_query(query, TestDatalogProgram())
    assert len(result.formulas) == 2
    assert P(x) in result.formulas
    assert Z(y) in result.formulas


def test_existential():
    code = Union(
        (
            Implication(P(x), Conjunction((Z(x, y), Z(y, x)))),
            Implication(Q(x), Conjunction((P(x), P(y), R(z)))),
        )
    )
    program = TestDatalogProgram()
    program.walk(code)
    query = Q(z)
    result = flatten_query(query, program)
    assert any(
        formula.functor == R and formula.args[0] != z
        for formula in result.formulas
    )
    assert any(
        formula.functor == Z
        and formula.args[0] == z
        and "fresh" in formula.args[1].name
        for formula in result.formulas
    )
    match = next(
        formula
        for formula in result.formulas
        if formula.functor == Z
        and "fresh" in formula.args[0].name
        and "fresh" in formula.args[1].name
    )
    expected = match.functor(*reversed(match.args))
    assert expected in result.formulas


def test_flatten_with_top_level_disjunction():
    code = Union(
        (
            Implication(R(x), Z(x)),
            Implication(R(y), P(y)),
            Implication(Q(x), R(x)),
        )
    )
    program = TestDatalogProgram()
    program.walk(code)
    result = flatten_query(Q(x), program)
    assert isinstance(result, Disjunction)
    assert len(result.formulas) == 2
    assert P(x) in result.formulas
    assert Z(x) in result.formulas


def test_flatten_with_2nd_level_disjunction():
    code = Union(
        (
            Implication(R(x), Z(x)),
            Implication(R(y), P(y)),
            Implication(Q(x, z), Conjunction((R(x), Z(z)))),
        )
    )
    program = TestDatalogProgram()
    program.walk(code)
    result = flatten_query(Q(x, z), program)
    assert isinstance(result, Conjunction)
    assert len(result.formulas) == 2
    assert (
        isinstance(result.formulas[0], Disjunction)
        and set((P(x), Z(x))) == set(result.formulas[0].formulas)
    ) or (
        isinstance(result.formulas[1], Disjunction)
        and set((P(x), Z(x))) == set(result.formulas[1].formulas)
    )
    assert Z(z) == result.formulas[0] or Z(z) == result.formulas[1]


def test_flatten_recursive():
    code = Union(
        (
            Implication(R(x, y), Conjunction((Z(y), Z(x)))),
            Implication(R(x, y), Conjunction((Z(x), R(x, y)))),
        )
    )
    program = TestDatalogProgram()
    program.walk(code)
    with pytest.raises(UnsupportedProgramError):
        flatten_query(R(x, y), program)


def test_incorrect_program():
    code = Union(
        (
            Implication(
                R(x), ExistentialPredicate(y, Conjunction((Z(y), Z(x))))
            ),
        )
    )
    program = TestDatalogProgram()
    program.walk(code)
    assert flatten_query(R(x, y), program) == FALSE


def test_flatten_with_variable_equality():
    rule = Implication(R(x, y), Conjunction((Z(x), P(x, y), EQ(x, a))))
    program = TestDatalogProgram()
    program.walk(rule)
    flat = flatten_query(R(x, y), program)
    assert set(flat.formulas) == set(extract_logic_predicates(rule.antecedent))


def test_flatten_repeated_variable_in_rule():
    """
    Given a program with the rule `R(x, x) :- Q(x, x, z)` and a (single
    predicate) conjunctive query `R(x, y)`, we expect the resulting flattened
    query to be either `Q(y, y, _freshvar_), x = y` or `Q(x, x, _freshvar_), y
    = x`.

    """
    rule = Implication(R(x, x), Q(x, x, z))
    program = TestDatalogProgram()
    program.walk(rule)
    flat = flatten_query(R(x, y), program)
    assert len(flat.formulas) == 2
    assert (
        Constant(operator.eq)(x, y) in flat.formulas
        and any(
            (formula.functor == Q and formula.args[:2] == (y, y))
            for formula in flat.formulas
        )
    ) or (
        Constant(operator.eq)(y, x) in flat.formulas
        and any(
            (formula.functor == Q and formula.args[:2] == (x, x))
            for formula in flat.formulas
        )
    )


def test_flatten_not_unifiable_becomes_false():
    code = Union(
        (
            Implication(R(Constant(2), Constant(3)), Conjunction((Q(y, x),))),
            Implication(Z(x), Conjunction((R(x, x),))),
        )
    )
    program = TestDatalogProgram()
    program.walk(code)
    flat = flatten_query(Z(z), program)
    assert flat == FALSE


def test_flatten_query_double_vareq():
    code = Implication(P(x), Conjunction((R(x),)))
    program = TestDatalogProgram()
    program.walk(code)
    query = Conjunction((P(a), P(b)))
    flat = flatten_query(query, program)
    assert isinstance(flat, Conjunction)
    assert set(flat.formulas) == {R(a), R(b)}
