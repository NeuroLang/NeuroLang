from typing import AbstractSet

from ...expressions import Constant, Symbol
from ...logic import Conjunction
from ...relational_algebra import ColumnInt
from ...utils.relational_algebra_set import NamedRelationalAlgebraFrozenSet
from .. import containment, dalvi_suciu_lift
from ..probabilistic_ra_utils import DeterministicFactSet, ProbabilisticFactSet

R = Symbol("R")
S = Symbol("S")
x = Symbol("x")
y = Symbol("y")


def symbol_table_with(wrapper):
    """Map ``R`` to a fresh table wrapped as ``wrapper``."""
    fresh = Symbol.fresh()
    if wrapper is ProbabilisticFactSet:
        table = wrapper(fresh, Constant[ColumnInt](ColumnInt(0)))
    else:
        table = wrapper(fresh)
    return {
        R: table,
        fresh: Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(
                iterable=[(1, 2), (3, 4)], columns=("a", "b")
            )
        ),
    }


def test_containment_cache_hit():
    q1 = Conjunction((R(x, y), S(y)))
    q2 = Conjunction((R(x, y), S(y)))
    assert containment.is_contained(q1, q2)
    assert containment.is_contained(q1, q2)
    info = containment.is_contained.cache_info()
    assert (info.hits, info.misses) == (1, 1)
    containment.clear_cache()
    assert containment.is_contained.cache_info().currsize == 0


def test_dalvi_suciu_lift_cache_shared_across_fresh_tables():
    query = Conjunction((R(x, y),))
    plan = dalvi_suciu_lift.dalvi_suciu_lift(
        query, symbol_table_with(ProbabilisticFactSet)
    )
    plan_fresh_table = dalvi_suciu_lift.dalvi_suciu_lift(
        query, symbol_table_with(ProbabilisticFactSet)
    )
    assert plan_fresh_table is plan
    assert len(dalvi_suciu_lift._dalvi_suciu_lift_cache) == 1
    dalvi_suciu_lift.clear_cache()
    assert len(dalvi_suciu_lift._dalvi_suciu_lift_cache) == 0


def test_dalvi_suciu_lift_cache_depends_on_relation_kind():
    query = Conjunction((R(x, y), R(y, x)))
    plans = set()
    for wrapper in (ProbabilisticFactSet, DeterministicFactSet):
        dalvi_suciu_lift.clear_cache()
        expected = dalvi_suciu_lift.dalvi_suciu_lift(
            query, symbol_table_with(wrapper)
        )
        for other in (ProbabilisticFactSet, DeterministicFactSet):
            dalvi_suciu_lift.dalvi_suciu_lift(
                query, symbol_table_with(other)
            )
        assert dalvi_suciu_lift.dalvi_suciu_lift(
            query, symbol_table_with(wrapper)
        ) == expected
        plans.add(expected)
    assert len(plans) == 2
