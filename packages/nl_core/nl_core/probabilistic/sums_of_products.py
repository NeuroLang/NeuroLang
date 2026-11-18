"""
Provenance plans of sums of products, computed as tensor contractions.

The plan of a query of choices, probabilistic facts and deterministic
relations, its sums pushed through its joins
(:class:`~.query_resolution.PushSumsThroughJoins`), gives each tuple of
the result the sum, over the tuples of the join of the relations it
merges, of the product of their probabilities (1 for a deterministic
tuple). :class:`ContractionPlan` translates it into the tensor expressions of
:mod:`~..relational_algebra.linear_algebra.expressions`;
:class:`SolveAsContractions`, a step of the provenance query solver,
normalises them (:mod:`~..relational_algebra.linear_algebra.rewrites`)
and computes them with a backend
(:mod:`~..relational_algebra.linear_algebra.scipy_backend`). Plans of
other operations are left to relational algebra.
"""
from typing import AbstractSet

from nl_expr.config import config
from nl_expr.expression_pattern_matching import NeuroLangPatternMatchingNoMatch
from nl_expr.expression_walker import PatternWalker, add_match
from nl_expr.expressions import Constant, FunctionApplication, Symbol

from ..relational_algebra import (
    ColumnInt,
    ColumnStr,
    NamedRelationalAlgebraFrozenSet,
    NameColumns,
    NaturalJoin,
    Product as ProductOperation,
    Projection,
    RenameColumn,
    RenameColumns,
    Selection,
    eq_,
    str2columnstr_constant,
)
from ..relational_algebra.relational_algebra import Column
from ..relational_algebra.linear_algebra.expressions import (
    Contraction,
    Diagonal,
    Factor,
    RenameAxes,
    SliceAxis,
    TensorProduct,
    TensorRatio,
)
from ..relational_algebra.linear_algebra.rewrites import (
    NotApplicable,
    normal_form,
)
from ..relational_algebra.linear_algebra.scipy_backend import evaluate
from ..relational_algebra_provenance import (
    NaturalJoinInverse,
    ProvenanceAlgebraSet,
)
from .probabilistic_ra_utils import (
    DeterministicFactSet,
    ProbabilisticChoiceSet,
    ProbabilisticFactSet,
)


def _is_value(constant):
    return not isinstance(constant.value, Column)


def _unnamed(term):
    """Whether the axes of *term* are the numbers of its columns."""
    return set(term.axes) == set(range(len(term.axes)))


class ContractionPlan(PatternWalker):
    """
    Write a provenance plan in tensor expressions: a deterministic
    relation is a factor of 1s, a set of probabilistic facts or choices
    a factor of their probabilities, a natural join a product, a
    projection a contraction keeping its columns, a selection of a
    column equal to a constant a slice, two columns equal a diagonal,
    a renaming of columns a renaming of axes and the division of a
    :class:`NaturalJoinInverse` a ratio. Any other operation is no match
    (:class:`~nl_expr.expression_pattern_matching.NeuroLangPatternMatchingNoMatch`).
    """

    def __init__(self, symbol_table):
        self.symbol_table = symbol_table

    def _relation(self, symbol):
        value = self.symbol_table.get(symbol)
        if not isinstance(value, Constant[AbstractSet]):
            raise NotApplicable(f"{symbol} is not a relation")
        return value.value

    def _unnamed(self, expression):
        """Return the factor of the unnamed *expression*."""
        factor = self.walk(expression)
        if not isinstance(factor, Factor) or not _unnamed(factor):
            raise NotApplicable(f"{expression} is not a relation's columns")
        return factor

    @add_match(Symbol)
    def symbol(self, symbol):
        if symbol not in self.symbol_table:
            raise NotApplicable(f"{symbol} is not in the symbol table")
        return self.walk(self.symbol_table[symbol])

    @add_match(DeterministicFactSet(Symbol))
    def deterministic(self, fact_set):
        relation = self._relation(fact_set.relation)
        columns = tuple(range(relation.arity))
        return Factor(relation, columns, columns)

    @add_match(ProbabilisticFactSet(Symbol, Constant[ColumnInt]))
    def probabilistic(self, fact_set):
        relation = self._relation(fact_set.relation)
        value = fact_set.probability_column.value
        positions = tuple(c for c in range(relation.arity) if c != value)
        return Factor(relation, tuple(range(len(positions))), positions, value)

    @add_match(ProbabilisticChoiceSet(Symbol, Constant[ColumnInt]))
    def choice(self, choice_set):
        return self.probabilistic(choice_set)

    @add_match(
        ProvenanceAlgebraSet(Constant[AbstractSet], Constant[ColumnStr])
    )
    def provenance_set(self, provenance_set):
        relation = provenance_set.relation.value
        columns = tuple(relation.columns)
        value = columns.index(provenance_set.provenance_column.value)
        positions = tuple(c for c in range(len(columns)) if c != value)
        return Factor(
            relation, tuple(columns[c] for c in positions), positions, value
        )

    @add_match(Selection(..., FunctionApplication(
        eq_, (Constant[ColumnInt], Constant[ColumnInt])
    )))
    def diagonal(self, selection):
        first, second = (c.value for c in selection.formula.args)
        return Diagonal(self._unnamed(selection.relation), first, second)

    @add_match(Selection(..., FunctionApplication(
        eq_, (Constant[ColumnStr], Constant[ColumnStr])
    )))
    def diagonal_axes(self, selection):
        first, second = (c.value for c in selection.formula.args)
        return Diagonal(self.walk(selection.relation), first, second)

    @add_match(
        Selection(..., FunctionApplication(eq_, (Constant[ColumnInt], ...))),
        lambda selection: _is_value(selection.formula.args[1]),
    )
    def slice_column(self, selection):
        column, constant = selection.formula.args
        return SliceAxis(
            self._unnamed(selection.relation), column.value, constant.value
        )

    @add_match(
        Selection(..., FunctionApplication(eq_, (Constant[ColumnStr], ...))),
        lambda selection: _is_value(selection.formula.args[1]),
    )
    def slice_axis(self, selection):
        axis, constant = selection.formula.args
        return SliceAxis(
            self.walk(selection.relation), axis.value, constant.value
        )

    @add_match(NameColumns)
    def name_columns(self, name_columns):
        term = self.walk(name_columns.relation)
        names = [c.value for c in name_columns.column_names]
        if not _unnamed(term) or len(term.axes) != len(names):
            raise NotApplicable(f"{name_columns} names no columns")
        return RenameAxes(term, tuple(enumerate(names)))

    @add_match(RenameColumn)
    def rename_column(self, rename_column):
        return self._rename(
            rename_column.relation,
            ((rename_column.src, rename_column.dst),),
        )

    @add_match(RenameColumns)
    def rename_columns(self, rename_columns):
        return self._rename(rename_columns.relation, rename_columns.renames)

    def _rename(self, relation, renames):
        term = self.walk(relation)
        renames = {src.value: dst.value for src, dst in renames}
        axes = [renames.get(a, a) for a in term.axes]
        if len(set(axes)) < len(axes):
            raise NotApplicable("a renaming merging columns")
        return RenameAxes(term, tuple(renames.items()))

    @add_match(
        NaturalJoin, lambda join: type(join) is NaturalJoin
    )
    def natural_join(self, join):
        return self._product((join.relation_left, join.relation_right))

    @add_match(ProductOperation)
    def product(self, product):
        return self._product(product.relations)

    def _product(self, relations):
        terms = []
        for relation in relations:
            term = self.walk(relation)
            terms += term.terms if isinstance(term, TensorProduct) else [term]
        if any(isinstance(a, int) for t in terms for a in t.axes):
            raise NotApplicable("a join of unnamed columns")
        return TensorProduct(tuple(terms))

    @add_match(Projection)
    def project(self, projection):
        term = self.walk(projection.relation)
        keep = tuple(a.value for a in projection.attributes)
        if len(set(keep)) < len(keep) or not set(keep) <= set(term.axes):
            raise NotApplicable(f"{projection} keeps no columns of its own")
        if term.axes and _unnamed(term):  # columns numbered as kept
            renames = {a: i for i, a in enumerate(keep)}
            renames.update(
                (a, Symbol.fresh().name) for a in term.axes if a not in keep
            )
            term = RenameAxes(term, tuple(renames.items()))
            keep = tuple(range(len(keep)))
        return Contraction(term, keep)

    @add_match(NaturalJoinInverse)
    def natural_join_inverse(self, join):
        return TensorRatio(
            self.walk(join.relation_left), self.walk(join.relation_right)
        )


class SolveAsContractions:
    """
    Step of the provenance query solver computing a plan of sums of
    products (:class:`ContractionPlan`) as tensor contractions: the plan
    becomes the provenance set of its result, whose probabilities are
    float64 sums of products of those of the relations, its rows
    those of the result but, possibly, some of probability 0. Other
    plans, and plans of a relation alone, are left as they are.

    It is configured in the section ``PROBABILISTIC_SOLVER`` of the
    configuration: ``contractions`` turns it on, ``dense_cells`` is the
    number of cells of the tensors of a product under which it is
    computed with dense arrays.
    """

    def __init__(self, symbol_table):
        self.symbol_table = symbol_table

    def walk(self, expression):
        try:
            plan = normal_form(
                ContractionPlan(self.symbol_table).walk(expression)
            )
            if isinstance(plan, Factor) or any(
                not isinstance(a, str) for a in plan.axes
            ):
                return expression
            column = Symbol.fresh().name
            frame = evaluate(plan, column, config.getint(
                "PROBABILISTIC_SOLVER", "dense_cells"
            ))
        except (NotApplicable, NeuroLangPatternMatchingNoMatch):
            return expression
        return ProvenanceAlgebraSet(
            Constant[AbstractSet](
                NamedRelationalAlgebraFrozenSet(tuple(frame.columns), frame)
            ),
            str2columnstr_constant(column),
        )
