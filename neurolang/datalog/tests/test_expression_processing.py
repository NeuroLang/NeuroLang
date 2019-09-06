from ...expressions import (Constant, ExistentialPredicate, ExpressionBlock,
                            Symbol)
from .. import Fact, Implication
from ..expression_processing import (
    extract_datalog_free_variables, is_conjunctive_expression,
    is_conjunctive_expression_with_nested_predicates)

S_ = Symbol
C_ = Constant
Imp_ = Implication
B_ = ExpressionBlock
EP_ = ExistentialPredicate
T_ = Fact


def test_conjunctive_expression():
    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    x = S_('x')
    y = S_('y')

    assert is_conjunctive_expression(
        Q()
    )

    assert is_conjunctive_expression(
        Q(x)
    )

    assert is_conjunctive_expression(
        Q(x) & R(y, C_(1))
    )

    assert not is_conjunctive_expression(
        R(x) | R(y)
    )

    assert not is_conjunctive_expression(
        R(x) & R(y) | R(x)
    )

    assert not is_conjunctive_expression(
        ~R(x)
    )

    assert not is_conjunctive_expression(
        R(Q(x))
    )


def test_conjunctive_expression_with_nested():
    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    x = S_('x')
    y = S_('y')

    assert is_conjunctive_expression_with_nested_predicates(
        Q()
    )

    assert is_conjunctive_expression_with_nested_predicates(
        Q(x)
    )

    assert is_conjunctive_expression_with_nested_predicates(
        Q(x) & R(y, C_(1))
    )

    assert is_conjunctive_expression_with_nested_predicates(
        Q(x) & R(Q(y), C_(1))
    )

    assert not is_conjunctive_expression_with_nested_predicates(
        R(x) | R(y)
    )

    assert not is_conjunctive_expression_with_nested_predicates(
        R(x) & R(y) | R(x)
    )

    assert not is_conjunctive_expression_with_nested_predicates(
        ~R(x)
    )


def test_extract_free_variables():
    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    x = S_('x')
    y = S_('y')

    emptyset = set()
    assert extract_datalog_free_variables(Q()) == emptyset
    assert extract_datalog_free_variables(x) == {x}
    assert extract_datalog_free_variables(Q(x, y)) == {x, y}
    assert extract_datalog_free_variables(Q(x, C_(1))) == {x}
    assert extract_datalog_free_variables(Q(x) & R(y)) == {x, y}
    assert extract_datalog_free_variables(EP_(x, Q(x, y))) == {y}
    assert extract_datalog_free_variables(Imp_(R(x), Q(x, y))) == {y}
    assert extract_datalog_free_variables(Imp_(R(x), Q(y) & Q(x))) == {y}
    assert extract_datalog_free_variables(Q(R(y))) == {y}
    assert extract_datalog_free_variables(Q(x) | R(y)) == {x, y}
    assert extract_datalog_free_variables(~(R(y))) == {y}
