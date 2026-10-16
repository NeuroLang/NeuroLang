"""
Probabilistic semirings for provenance computation.

The abstract :class:`~neurolang.provenance_semiring.Semiring` and the
default :class:`~neurolang.provenance_semiring.ProbabilitySemiring` live in
the core module :mod:`neurolang.provenance_semiring`; they are re-exported
here together with the max-product (Viterbi) semiring.
"""

from ..expressions import Constant, FunctionApplication
from ..provenance_semiring import ProbabilitySemiring, Semiring
from ..relational_algebra import (
    FunctionApplicationListMember,
    GroupByAggregation
)


class MaxProductSemiring(Semiring):

    """
    Max-product (Viterbi) semiring: (max, ×) over [0, 1].

    Every ⊕, including independent existential and universal projections,
    keeps the most likely derivation by taking the maximum.
    """

    def add(self, a, b):
        """Return ``max(a, b)``."""
        return max(a, b)

    def mul(self, a, b):
        """Return ``a * b``."""
        return a * b

    def zero(self):
        """Return 0."""
        return 0.0

    def one(self):
        """Return 1."""
        return 1.0

    def agg_add(self):
        """Return ``max``."""
        return max

    def add_expression(self, operands):
        """Return the expression taking the maximum of ``operands``."""
        if len(operands) == 1:
            return operands[0]
        return FunctionApplication(Constant(max), tuple(operands))

    def monus_expression(self, left, right):
        """Return the expression ``left * (1 - right)``."""
        return ProbabilitySemiring().monus_expression(left, right)

    def independent_projection(
        self, relation, provenance_column, columns, attributes
    ):
        """
        Keep the maximum provenance per group.

        Parameters are those of :meth:`Semiring.independent_projection`.
        """
        return GroupByAggregation(
            relation,
            attributes,
            (
                FunctionApplicationListMember(
                    self.agg_add_expression((provenance_column,)),
                    provenance_column,
                ),
            ),
        )

    def independent_projection_universal(
        self, relation, provenance_column, columns, attributes
    ):
        """
        Keep the maximum provenance per group.

        Parameters are those of :meth:`Semiring.independent_projection`.
        """
        return self.independent_projection(
            relation, provenance_column, columns, attributes
        )


__all__ = [
    "Semiring",
    "ProbabilitySemiring",
    "MaxProductSemiring",
]
