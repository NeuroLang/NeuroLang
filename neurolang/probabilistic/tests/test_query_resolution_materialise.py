from typing import AbstractSet

from ...expressions import Constant
from ...relational_algebra import (
    Materialise,
    Projection,
    str2columnstr_constant
)
from ...utils.relational_algebra_set import NamedRelationalAlgebraFrozenSet
from ..query_resolution import RAQueryOptimiser


def test_query_optimiser_does_not_rewrite_relations_to_materialise():
    """A relation to materialise is computed as it is: the optimiser of
    the plans does not walk into it, its expression having been
    optimised when it was built.
    """
    relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(("x",), [(1,), (2,)])
    )
    trivial_projection = Projection(
        relation, (str2columnstr_constant("x"),)
    )
    materialise = Materialise(trivial_projection)

    assert RAQueryOptimiser().walk(trivial_projection) == relation
    assert RAQueryOptimiser().walk(materialise) == materialise
