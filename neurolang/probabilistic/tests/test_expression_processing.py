import pytest

from ...exceptions import UnexpectedExpressionError
from ...expressions import Constant, Symbol
from ...logic import Union
from ..expression_processing import (
    add_to_union,
    get_probchoice_variable_equalities,
    group_preds_by_functor,
)

P = Symbol("P")
Q = Symbol("Q")
x = Symbol("x")
y = Symbol("y")
z = Symbol("z")
a = Constant("a")
b = Constant("b")


def test_concatenate_to_union():
    union1 = Union((P(x),))
    union2 = Union((Q(z), Q(y), Q(x)))
    union3 = Union((P(z), P(y)))
    assert P(y) in add_to_union(union1, [P(y)]).formulas
    for expression in union2.formulas:
        new_union = add_to_union(union1, union2)
        assert expression in new_union.formulas
        new_union = add_to_union(union1, union2)
        assert expression in new_union.formulas
    for expression in union1.formulas + union2.formulas + union3.formulas:
        new_union = add_to_union(add_to_union(union1, union2), union3)
        assert expression in new_union.formulas


def test_concatenate_to_union_not_expression():
    union = Union((P(x),))
    with pytest.raises(UnexpectedExpressionError):
        add_to_union(union, ["this_is_not_an_expression"])
    with pytest.raises(UnexpectedExpressionError):
        add_to_union(union, None)


def test_get_probchoice_variable_equalities():
    pchoice_pred_symbs = {P, Q}
    predicates = {P(x, y), Q(y)}
    assert (
        get_probchoice_variable_equalities(predicates, pchoice_pred_symbs)
        == set()
    )
    predicates = {P(x, y, z), P(y, y, z), Q(z)}
    equalities = get_probchoice_variable_equalities(
        predicates, pchoice_pred_symbs
    )
    assert equalities == {(x, y)}
    assert get_probchoice_variable_equalities(set(), set()) == set()
    predicates = {P(x, y, z), P(x, z, y), Q(z)}
    equalities = get_probchoice_variable_equalities(
        predicates, pchoice_pred_symbs
    )
    assert equalities == {(y, z)}
    predicates = {P(x), P(y)}
    equalities = get_probchoice_variable_equalities(
        predicates, pchoice_pred_symbs
    )
    assert equalities == {(x, y)}
    predicates = {P(x)}
    equalities = get_probchoice_variable_equalities(
        predicates, pchoice_pred_symbs
    )
    assert equalities == set()
    predicates = {P(x), P(y), P(z)}
    equalities = get_probchoice_variable_equalities(
        predicates, pchoice_pred_symbs
    )
    assert equalities in [
        {(x, y), (x, z)},
        {(x, y), (y, z)},
        {(x, z), (y, z)},
    ]


def test_group_preds_by_pred_symb():
    predicates = [P(x, y), Q(x)]
    grouped = group_preds_by_functor(predicates, filter_set={Q})
    assert grouped == {Q: {Q(x)}}
    predicates = [P(x, y), Q(x)]
    grouped = group_preds_by_functor(predicates, filter_set=set())
    assert grouped == dict()
