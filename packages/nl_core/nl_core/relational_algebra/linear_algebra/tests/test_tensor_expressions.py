"""Tensor expressions: the rewrites to normal form are equivalences, and
the scipy backend computes sums of products as relational algebra does."""
import numpy as np
import pytest

from ....utils.relational_algebra_set import RelationalAlgebraFrozenSet
from ..expressions import (
    Contraction,
    Diagonal,
    Factor,
    WeightedFactor,
    RenameAxes,
    SliceAxis,
    TensorProduct,
    TensorRatio,
)
from ..rewrites import NotApplicable, normal_form
from ..scipy_backend import evaluate


def _relation(rows):
    return RelationalAlgebraFrozenSet(rows)


def _tensor(frame, axes, value="p"):
    """A frame of the entries of a tensor, as a dict of axes to value."""
    return {
        tuple(row[a] for a in axes): row[value]
        for _, row in frame.iterrows() if row[value] != 0
    }


@pytest.fixture
def relations():
    rng = np.random.default_rng(0)
    act = _relation({(v, s) for v in range(4) for s in range(5)
                     if rng.random() < .5})
    weights = rng.random(5)
    sel = _relation([(float(w), s) for s, w in enumerate(weights)])
    return act, sel


def _brute(act, sel, keep):
    """sum over the rows of act join sel of the weight of s, by kept axes."""
    weight = {s: w for w, s in sel}
    sums = {}
    for v, s in act:
        key = tuple({"v": v, "s": s}[a] for a in keep)
        sums[key] = sums.get(key, 0) + weight[s]
    return sums


def _solve(expression, axes):
    normal = normal_form(expression)
    return _tensor(evaluate(normal, "p"), axes)


def test_contraction_is_a_sum_of_products(relations):
    act, sel = relations
    plan = Contraction(TensorProduct((
        Factor(act, ("v", "s"), (0, 1), (), ()),
        WeightedFactor(sel, ("s",), (1,), 0, (), ()),
    )), ("v",))
    assert _solve(plan, ("v",)) == pytest.approx(_brute(act, sel, ("v",)))


def test_renames_and_slices_are_pushed_to_factors(relations):
    act, sel = relations
    inner = Contraction(TensorProduct((
        Factor(act, (0, 1), (0, 1), (), ()),
        WeightedFactor(sel, ("s",), (1,), 0, (), ()),
    )), (0,))
    plan = SliceAxis(RenameAxes(
        RenameAxes(
            Contraction(TensorProduct((
                RenameAxes(
                    Factor(act, (0, 1), (0, 1), (), ()),
                    ((0, "v"), (1, "s")),
                ),
                WeightedFactor(sel, ("s",), (1,), 0, (), ()),
            )), ("v",)),
            (("v", "w"),),
        ), ()), "w", 2)
    normal = normal_form(plan)
    assert not any(
        isinstance(e, (RenameAxes, SliceAxis))
        for e in _walk(normal)
    )
    expected = {k: p for k, p in _brute(act, sel, ("v",)).items() if k == (2,)}
    assert _tensor(evaluate(normal, "p"), ("w",)) == pytest.approx(expected)
    assert inner.axes == (0,)


def test_diagonal_on_a_factor():
    pairs = _relation({(a, b) for a in range(3) for b in range(3)})
    plan = Diagonal(Factor(pairs, ("a", "b"), (0, 1), (), ()), "a", "b")
    assert _solve(plan, ("a", "b")) == {(a, a): 1.0 for a in range(3)}


def test_diagonal_across_terms_is_not_applicable():
    one = _relation({(0,), (1,)})
    plan = Diagonal(TensorProduct((
        Factor(one, ("a",), (0,), (), ()), Factor(one, ("b",), (0,), (), ()),
    )), "a", "b")
    with pytest.raises(NotApplicable):
        normal_form(plan)


def test_ratio_divides_at_the_entries_of_both(relations):
    act, sel = relations
    numerator = Contraction(TensorProduct((
        Factor(act, ("v", "s"), (0, 1), (), ()),
        WeightedFactor(sel, ("s",), (1,), 0, (), ()),
    )), ("v",))
    denominator = Contraction(WeightedFactor(sel, ("s",), (1,), 0, (), ()), ())
    total = sum(w for w, _ in sel)
    expected = {k: p / total for k, p in _brute(act, sel, ("v",)).items()}
    assert _solve(TensorRatio(numerator, denominator), ("v",)) == (
        pytest.approx(expected)
    )


def _walk(expression):
    yield expression
    for child in getattr(expression, "terms", ()) or ():
        yield from _walk(child)
    for name in ("term", "numerator", "denominator"):
        if hasattr(expression, name):
            yield from _walk(getattr(expression, name))
