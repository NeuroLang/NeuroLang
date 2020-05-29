import pytest

from ...exceptions import UnexpectedExpressionError
from ...expressions import Symbol
from ...logic import Union
from ..expression_processing import add_to_union

P = Symbol("P")
Q = Symbol("Q")
x = Symbol("x")
y = Symbol("y")
z = Symbol("z")


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
