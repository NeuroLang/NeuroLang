"""
Tests for the rewriting of PushSumsThroughJoins: the projections of a
provenance plan pushed through its ordered joins. The probabilities it
gives are checked in neurolang/frontend/tests/test_push_sums_through_joins.py.
"""

from typing import AbstractSet

from nl_expr.expressions import Constant
from ...relational_algebra import (
    NamedRelationalAlgebraFrozenSet,
    NaturalJoin,
    Projection,
    str2columnstr_constant,
)
from ...relational_algebra_provenance import NaturalJoinInverse
from ..query_resolution import PushSumsThroughJoins


def _relation(*columns):
    """A named relation of the *columns*, one row of zeros."""
    return Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(columns, [(0,) * len(columns)])
    )


def _project(relation, *columns):
    return Projection(
        relation, tuple(str2columnstr_constant(c) for c in columns)
    )


def test_a_projection_is_pushed_through_ordered_joins():
    """π[t]((voxel(v) ⋈ activated(v, s)) ⋈ terms(s, t)): v is summed out
    before the join with terms, s once joined with it."""
    voxel, activated, terms = (
        _relation("v"), _relation("v", "s"), _relation("s", "t")
    )
    query = _project(NaturalJoin(NaturalJoin(voxel, activated), terms), "t")
    pushed = PushSumsThroughJoins().walk(query)
    assert pushed == _project(
        NaturalJoin(
            _project(NaturalJoin(voxel, activated), "s"), terms
        ),
        "t",
    )


def test_a_projection_keeping_every_column_is_left_alone():
    left, right = _relation("v", "s"), _relation("s", "t")
    query = _project(NaturalJoin(left, right), "v", "s", "t")
    assert PushSumsThroughJoins().walk(query) is query


def test_a_division_is_not_a_product():
    """A sum does not distribute over the division of NaturalJoinInverse:
    its projection stays above it."""
    left, right = _relation("v", "s"), _relation("s")
    query = _project(NaturalJoinInverse(left, right), "s")
    assert PushSumsThroughJoins().walk(query) is query
