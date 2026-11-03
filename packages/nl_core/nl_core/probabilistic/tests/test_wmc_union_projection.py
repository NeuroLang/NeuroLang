"""Tests of the union of the rules of a query in weighted model counting."""

import pytest

from nl_expr.expressions import Symbol
from nl_expr.logic import Conjunction, Disjunction, ExistentialPredicate
from ...relational_algebra import str2columnstr_constant

pytest.importorskip("pysdd")

from ..weighted_model_counting import _project_disjuncts  # noqa: E402


def test_project_disjuncts_keeps_the_columns_of_the_union():
    """Each disjunct is quantified on the variables it alone has, those
    of the query and of the other conjuncts kept."""
    M, D = Symbol("M"), Symbol("D")
    t, s, s2, u = (Symbol(n) for n in ("t", "s", "s2", "u"))
    union = Disjunction((
        Conjunction((M(s, t),)), Conjunction((M(s, u), M(s2, t))),
    ))
    projected = _project_disjuncts(
        Conjunction((union, D(u))), (str2columnstr_constant("t"),)
    )
    first, second = projected.formulas[0].formulas
    assert first == ExistentialPredicate(s, Conjunction((M(s, t),)))
    quantified = set()
    while isinstance(second, ExistentialPredicate):
        quantified.add(second.head)
        second = second.body
    assert quantified == {s, s2}
    assert projected.formulas[1] == D(u)
    alone = _project_disjuncts(union, (str2columnstr_constant("t"),))
    assert isinstance(alone, Disjunction)
