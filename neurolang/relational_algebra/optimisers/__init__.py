"""
Static optimisers of relational algebra expressions.

The optimisers are pattern walkers rewriting relational algebra
expressions into equivalent, cheaper ones. They are grouped by the
operation they act upon:

- :mod:`.selections`: rewriting and pushing selections;
- :mod:`.projections`: eliminating, simplifying and pushing projections;
- :mod:`.renames`: simplifying and pushing column renamings;
- :mod:`.join_ordering`: flattening and reordering joins and products;
- :mod:`.columns`: column helpers shared by the optimisers.

:class:`RelationalAlgebraOptimiser` composes them. Every optimiser, and
the helpers used outside of this package, are importable from here.
"""

from ... import expression_walker as ew
from .columns import _ra_columns
from .join_ordering import (
    DegenerateNaturalJoinToProduct,
    FlattenAndReorderJoins,
    GreedyJoinOrdering,
    ProductSimplification,
    _estimated_size,
    _sort_join_operands
)
from .projections import (
    EliminateTrivialProjections,
    PushProjectionsDown,
    PushProjectionThroughProduct,
    SimplifyExtendedProjectionsWithConstants
)
from .renames import RenameOptimizations
from .selections import (
    AND,
    EQ,
    PushInSelections,
    PushUnnamedSelectionsUp,
    RewriteSelections
)

__all__ = [
    "AND",
    "EQ",
    "DegenerateNaturalJoinToProduct",
    "EliminateTrivialProjections",
    "FlattenAndReorderJoins",
    "GreedyJoinOrdering",
    "ProductSimplification",
    "PushInSelections",
    "PushProjectionThroughProduct",
    "PushProjectionsDown",
    "PushUnnamedSelectionsUp",
    "RelationalAlgebraOptimiser",
    "RenameOptimizations",
    "RewriteSelections",
    "SimplifyExtendedProjectionsWithConstants",
    "_estimated_size",
    "_ra_columns",
    "_sort_join_operands",
]


class RelationalAlgebraOptimiser(
    RewriteSelections,
    ProductSimplification,
    EliminateTrivialProjections,
    PushInSelections,
    RenameOptimizations,
    PushUnnamedSelectionsUp,
    PushProjectionsDown,
    FlattenAndReorderJoins,
    ew.ExpressionWalker,
):
    """
    Mixing that optimises through relational algebra expressions by
    rewriting.
    equi-selection/product compositions into equijoins.

    Pipeline order within each convergence pass:
      1. Existing optimisers (selection pushdown, projection elimination, …)
      2. PushProjectionsDown            — push projections through
         NaryNaturalJoin operands to reduce join sizes
      3. FlattenAndReorderJoins        — flatten join tree into
         NaryNaturalJoin, greedily reorder to minimise cross products
    """

    pass
