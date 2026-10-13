import typing

import pytest

from ...datalog.wrapped_collections import (
    WrappedNamedRelationalAlgebraFrozenSetMixin,
    WrappedRelationalAlgebraSetMixin,
)
from ...exceptions import UnexpectedExpressionError
from ...expressions import Constant, Symbol
from ...logic import Union
from ...utils.relational_algebra_set import pandas as pandas_ras
from ..exceptions import DistributionDoesNotSumToOneError
from ..expression_processing import (
    add_to_union,
    check_probabilistic_choice_set_probabilities_sum_to_one,
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


def _ras_backends():
    backends = [pandas_ras]
    try:
        from ...utils.relational_algebra_set import polars as polars_ras
        backends.append(polars_ras)
    except ImportError:  # pragma: no cover
        pass
    return backends


def _wrapped_set_classes(backend):
    class WrappedSet(
        WrappedRelationalAlgebraSetMixin, backend.RelationalAlgebraSet
    ):
        pass

    class WrappedNamedSet(
        WrappedNamedRelationalAlgebraFrozenSetMixin,
        backend.NamedRelationalAlgebraFrozenSet,
    ):
        pass

    return WrappedSet, WrappedNamedSet


def _choice_constant(backend, tuples, named=False):
    wrapped_set, wrapped_named_set = _wrapped_set_classes(backend)
    if named:
        ras = wrapped_named_set(columns=("p", "x", "y"), iterable=tuples)
    else:
        ras = wrapped_set(tuples)
    return Constant[typing.AbstractSet](
        ras, auto_infer_type=False, verify_type=False
    )


@pytest.mark.parametrize("backend", _ras_backends())
@pytest.mark.parametrize("named", [False, True])
def test_probabilistic_choice_sums_to_one(backend, named):
    tuples = [(0.1 * (i % 4 + 1), "a", str(i)) for i in range(4)]
    tuples = tuples + [(0.1 * (i % 4 + 1), "a", str(i)) for i in range(4)]
    ra_set = _choice_constant(backend, tuples, named)
    check_probabilistic_choice_set_probabilities_sum_to_one(ra_set)


@pytest.mark.parametrize("backend", _ras_backends())
@pytest.mark.parametrize("named", [False, True])
def test_probabilistic_choice_does_not_sum_to_one(backend, named):
    tuples = [(0.5, "a", "a"), (0.2, "a", "b"), (0.1, "b", "b")]
    ra_set = _choice_constant(backend, tuples, named)
    with pytest.raises(DistributionDoesNotSumToOneError, match=r"sum to 1\. Got 0\.(8|79)"):
        check_probabilistic_choice_set_probabilities_sum_to_one(ra_set)


@pytest.mark.parametrize("backend", _ras_backends())
def test_probabilistic_choice_empty_does_not_sum_to_one(backend):
    ra_set = _choice_constant(backend, [])
    with pytest.raises(DistributionDoesNotSumToOneError, match="Got 0"):
        check_probabilistic_choice_set_probabilities_sum_to_one(ra_set)

