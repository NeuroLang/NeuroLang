from typing import AbstractSet
from ...expressions import Symbol
from ...logic import Conjunction
from .. import containment
from .. import dalvi_suciu_lift
from ..probabilistic_ra_utils import ProbabilisticFactSet
from ...expressions import Constant
from ...utils.relational_algebra_set import NamedRelationalAlgebraFrozenSet


def test_containment_cache_is_populated():
    """Repeated identical containment checks should be cached."""
    R = Symbol("R")
    S = Symbol("S")
    x = Symbol("x")
    y = Symbol("y")

    q1 = Conjunction((R(x, y), S(y)))
    q2 = Conjunction((R(x, y), S(y)))

    containment.clear_cache()
    assert containment.is_contained.cache_info().hits == 0
    assert containment.is_contained.cache_info().misses == 0

    # first call populates cache
    containment.is_contained(q1, q2)
    assert containment.is_contained.cache_info().misses == 1
    assert containment.is_contained.cache_info().hits == 0

    # second identical call should hit
    containment.is_contained(q1, q2)
    assert containment.is_contained.cache_info().hits == 1


def test_dalvi_suciu_lift_cache_is_populated():
    """Repeated identical lifted-plan calls should be cached."""
    R = Symbol("R")
    x = Symbol("x")
    y = Symbol("y")
    query = Conjunction((R(x, y),))

    ra_set = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            iterable=[(1, 2), (3, 4)],
            columns=("a", "b"),
        )
    )
    fresh_R = Symbol("R_data")
    symbol_table = {
        R: ProbabilisticFactSet(fresh_R, Constant(0)),
        fresh_R: ra_set,
    }

    dalvi_suciu_lift.clear_cache()
    assert len(dalvi_suciu_lift._dalvi_suciu_lift_cache) == 0

    plan1 = dalvi_suciu_lift.dalvi_suciu_lift(query, symbol_table)
    assert len(dalvi_suciu_lift._dalvi_suciu_lift_cache) == 1

    plan2 = dalvi_suciu_lift.dalvi_suciu_lift(query, symbol_table)
    assert len(dalvi_suciu_lift._dalvi_suciu_lift_cache) == 1
    # cached result should be the same object
    assert plan1 is plan2


def test_containment_cache_clear():
    """Cache clearing should reset the internal state."""
    R = Symbol("R")
    x = Symbol("x")
    q = Conjunction((R(x),))

    containment.is_contained(q, q)
    assert containment.is_contained.cache_info().misses >= 1

    containment.clear_cache()
    assert containment.is_contained.cache_info().misses == 0
    assert containment.is_contained.cache_info().hits == 0
