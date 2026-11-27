"""
Joins of hundreds of operands: the plans of a query of many conjuncts are
not trees as deep as the conjuncts are many, which the walkers that
optimise and solve them recurse on, one level per join.
"""
from typing import AbstractSet

from nl_expr.expressions import Constant, Symbol
from nl_expr.logic import Conjunction
from ...relational_algebra import NaturalJoin
from ...relational_algebra.optimisers import JOIN_CHUNK, join_in_chunks
from ...utils.relational_algebra_set import NamedRelationalAlgebraFrozenSet
from .. import dalvi_suciu_lift
from ..probabilistic_ra_utils import ProbabilisticFactSet
from ..query_resolution import RAQueryOptimiser


def _join_depth(expression):
    depth, to_visit = 0, [(expression, 1)]
    while to_visit:
        node, level = to_visit.pop()
        depth = max(depth, level)
        if isinstance(node, NaturalJoin):
            to_visit.append((node.relation_left, level + 1))
            to_visit.append((node.relation_right, level + 1))
    return depth


def test_plan_of_independent_atoms_is_a_shallow_tree():
    """A conjunction of atoms of different relations on one variable
    is the independent join of the plans of its components."""
    x = Symbol("x")
    relations = [Symbol(f"R{i}") for i in range(60)]
    symbol_table = {
        relation: ProbabilisticFactSet(Symbol.fresh(), "p")
        for relation in relations
    }
    plan = dalvi_suciu_lift.dalvi_suciu_lift(
        Conjunction(tuple(relation(x) for relation in relations)),
        symbol_table
    )

    assert dalvi_suciu_lift.is_pure_lifted_plan(plan)
    assert _join_depth(plan) < 2 * JOIN_CHUNK


def test_plan_optimiser_walks_a_join_of_many_operands():
    """The optimiser of the plans walked a chain of 250 joins or more
    with a RecursionError."""
    operands = [
        Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(("x", f"y{i}"), [(0, i)])
        )
        for i in range(300)
    ]
    join = join_in_chunks(operands)

    assert RAQueryOptimiser().walk(join) == join
