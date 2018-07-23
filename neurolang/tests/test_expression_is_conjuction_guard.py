import operator

from neurolang.expressions import (
    Symbol, Constant, FunctionApplication, ToBeInferred
)
from neurolang.solver import expression_is_conjunction_guard


def test_bool_constant_is_conjunction():
    expression = Constant[bool](True)
    assert expression_is_conjunction_guard(expression)


def test_nonbool_constant_is_not_a_conjunction():
    expression = Constant[int](3)
    assert not expression_is_conjunction_guard(expression)


def test_disjunction_is_not_a_conjunction():
    expression = FunctionApplication[bool](
        Constant(operator.or_), (Constant[bool](True), Constant[bool](False))
    )
    assert not expression_is_conjunction_guard(expression)


def test_simple_conjunction_is_conjunction():
    expression = FunctionApplication[bool](
        Constant(operator.and_), (Constant[bool](True), Constant[bool](False))
    )
    assert expression_is_conjunction_guard(expression)


def test_nested_conjuctions_is_conjuction():
    expression_a = FunctionApplication[bool](
        Constant(operator.and_), (Constant[bool](True), Constant[bool](False))
    )
    expression_b = FunctionApplication[bool](
        Constant(operator.and_), (Constant[bool](False), Constant[bool](False))
    )
    final_expression = FunctionApplication[bool](
        Constant(operator.and_), (expression_a, expression_b)
    )
    assert expression_is_conjunction_guard(expression_a)
    assert expression_is_conjunction_guard(expression_b)
    assert expression_is_conjunction_guard(final_expression)


def test_non_constant_arg_is_not_a_conjunction():
    non_constant_arg = FunctionApplication[bool](
        Symbol('inferior_of'), (Constant[int](1), Constant[int](2))
    )
    expression = FunctionApplication[bool](
        Constant(operator.and_), (Constant[bool](True), non_constant_arg)
    )
    assert not expression_is_conjunction_guard(expression)
