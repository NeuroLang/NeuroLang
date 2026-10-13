"""
Semirings driving the provenance relational algebra solvers.

A :class:`Semiring` gathers every operation that the provenance solvers
of :mod:`neurolang.relational_algebra_provenance` apply to provenance
columns. Solvers delegate to the semiring held in their ``semiring``
attribute and never inspect its concrete class, so new semirings can be
added (e.g. in :mod:`neurolang.probabilistic.semiring`) without changing
the solvers.
"""

import math
import operator
from abc import ABC, abstractmethod

from .expressions import Constant, FunctionApplication
from .relational_algebra import (
    ExtendedProjection,
    FunctionApplicationListMember,
    GroupByAggregation,
    Selection,
    Union
)

MUL = Constant(operator.mul)
SUB = Constant(operator.sub)
LT = Constant(operator.lt)
EQ = Constant(operator.eq)
EXP = Constant(math.exp)
LOG = Constant(math.log)
ONE = Constant[float](1.)
LARGE_NEG = Constant[float](-1e20)


class Semiring(ABC):

    """
    Abstract base for (T, ⊕, ⊗, 0, 1) semirings used in provenance.

    Besides the value-level operations, a semiring builds the relational
    algebra expressions that the provenance solvers use to combine
    provenance columns: ⊗ of two columns (joins), n-ary ⊕ of columns
    (weighted joins), ⊕-aggregation (projections), the independent
    existential and universal projections, and the monus (difference).
    """

    def __init__(self, name=None):
        """
        Create the semiring.

        Parameters
        ----------
        name : str, optional
            Name of the semiring, defaults to the class name.

        """
        self._name = name or self.__class__.__name__

    @property
    def name(self):
        """Return the name of the semiring."""
        return self._name

    @abstractmethod
    def add(self, a, b):
        """Semiring addition (⊕) of two values."""

    @abstractmethod
    def mul(self, a, b):
        """Semiring multiplication (⊗) of two values."""

    @abstractmethod
    def zero(self):
        """Additive identity element."""

    @abstractmethod
    def one(self):
        """Multiplicative identity element."""

    @abstractmethod
    def agg_add(self):
        """Return the ⊕-aggregation callable (e.g. ``sum`` or ``max``)."""

    def mul_expression(self, left, right):
        """Return the expression computing ``left`` ⊗ ``right``."""
        return left * right

    @abstractmethod
    def add_expression(self, operands):
        """Return the expression computing the ⊕ of ``operands``."""

    def agg_add_expression(self, args, **kwargs):
        """Return the ⊕-aggregation applied to ``args``."""
        return FunctionApplication(Constant(self.agg_add()), args, **kwargs)

    @abstractmethod
    def monus_expression(self, left, right):
        """Return the expression computing ``left`` ⊖ ``right``."""

    @abstractmethod
    def independent_projection(
        self, relation, provenance_column, columns, attributes
    ):
        """
        Project ``relation`` on ``attributes`` assuming independent tuples.

        Parameters
        ----------
        relation : RelationalAlgebraOperation
            relation holding the provenance column and ``columns``.
        provenance_column : Constant[ColumnStr]
            provenance column of ``relation``, kept in the result.
        columns : iterable of Constant[ColumnStr]
            non-provenance columns of ``relation``.
        attributes : tuple of Constant[ColumnStr]
            columns to group by.

        Returns
        -------
        RelationalAlgebraOperation
            relation with ``attributes`` and the existentially combined
            provenance column.

        """

    @abstractmethod
    def independent_projection_universal(
        self, relation, provenance_column, columns, attributes
    ):
        """
        Universally project ``relation`` assuming independent tuples.

        Parameters are those of :meth:`independent_projection`.
        """

    def __repr__(self):
        """Return the representation of the semiring."""
        return f"<Semiring: {self._name}>"


class ProbabilitySemiring(Semiring):

    """
    Probability semiring: (+, ×) over [0, 1].

    This is the default semiring of the provenance solvers. Independent
    existential projections compute 1 - ∏(1 - p) and universal ones ∏ p,
    both in log space.
    """

    def add(self, a, b):
        """Return ``a + b``."""
        return a + b

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
        """Return ``sum``."""
        return sum

    def add_expression(self, operands):
        """Return the expression summing ``operands``."""
        return sum(operands[1:], operands[0])

    def monus_expression(self, left, right):
        """Return the expression ``left * (1 - right)``."""
        return MUL(left, SUB(Constant(1), right))

    def independent_projection(
        self, relation, provenance_column, columns, attributes
    ):
        """
        Compute 1 - ∏(1 - p) per group, in log space.

        Parameters are those of :meth:`Semiring.independent_projection`.
        """
        prov_col = provenance_column
        proj_list = [
            FunctionApplicationListMember(col, col) for col in columns
        ]
        proj_list_log = proj_list + [
            FunctionApplicationListMember(LOG(ONE - prov_col), prov_col)
        ]
        proj_list_0 = proj_list + [
            FunctionApplicationListMember(LARGE_NEG, prov_col)
        ]
        relation_log = ExtendedProjection(
            Selection(relation, FunctionApplication(LT, (prov_col, ONE))),
            proj_list_log
        )
        relation_zero = ExtendedProjection(
            Selection(relation, FunctionApplication(EQ, (prov_col, ONE))),
            proj_list_0
        )
        result = self._log_sum_by_group(
            Union(relation_log, relation_zero), prov_col, attributes
        )
        return ExtendedProjection(
            result,
            [FunctionApplicationListMember(col, col) for col in result.groupby]
            + [FunctionApplicationListMember(ONE - EXP(prov_col), prov_col)]
        )

    def independent_projection_universal(
        self, relation, provenance_column, columns, attributes
    ):
        """
        Compute ∏ p per group, in log space.

        Parameters are those of :meth:`Semiring.independent_projection`.
        """
        prov_col = provenance_column
        proj_list = [
            FunctionApplicationListMember(col, col) for col in columns
        ]
        proj_list.append(
            FunctionApplicationListMember(LOG(prov_col), prov_col)
        )
        result = self._log_sum_by_group(
            ExtendedProjection(relation, proj_list), prov_col, attributes
        )
        return ExtendedProjection(
            result,
            [FunctionApplicationListMember(col, col) for col in result.groupby]
            + [FunctionApplicationListMember(EXP(prov_col), prov_col)]
        )

    @staticmethod
    def _log_sum_by_group(relation, prov_col, attributes):
        return GroupByAggregation(
            relation,
            groupby=attributes,
            aggregate_functions=(
                FunctionApplicationListMember(
                    FunctionApplication(Constant(sum), (prov_col,)),
                    prov_col,
                ),
            ),
        )


__all__ = [
    "Semiring",
    "ProbabilitySemiring",
]
