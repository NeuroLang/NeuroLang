"""
Rewrites of tensor expressions (:mod:`.expressions`) into the normal form
a backend computes: summed axes named apart, renamings, slices and
diagonals on factors only, and products flat. Every rewrite is an
equivalence: the tensor is unchanged. :func:`normal_form` applies them.
"""
import itertools

from nl_expr.expression_walker import ExpressionWalker, add_match

from .expressions import (
    Contraction,
    Diagonal,
    Factor,
    RenameAxes,
    SliceAxis,
    TensorProduct,
    TensorRatio,
)


class NotApplicable(Exception):
    """The expression has no normal form a backend computes."""


class SummedAxis(str):
    """The name of an axis a contraction sums, unique to it."""


_summed_names = itertools.count()


class RenameSummedAxesApart(ExpressionWalker):
    """
    Give the axes each contraction sums names of their own, as bound
    variables are renamed apart: two contractions summing an axis of the
    same name (the numerator and the denominator of a ratio) then read
    different axes. A contraction's summed axes are bound by it, so the
    tensor is unchanged; inner contractions are renamed first, and a
    renaming of a contraction does not reach the axes it sums.
    """

    @add_match(
        Contraction,
        lambda c: any(
            not isinstance(a, SummedAxis)
            for a in set(c.term.axes) - set(c.keep)
        ),
    )
    def rename_apart(self, contraction):
        renames = tuple(
            (a, SummedAxis(f"{a}\N{MIDDLE DOT}{next(_summed_names)}"))
            for a in contraction.term.axes
            if a not in contraction.keep and not isinstance(a, SummedAxis)
        )
        return type(contraction)(
            RenameAxes(contraction.term, renames), contraction.keep
        )


class PushAxisOperationsToFactors(ExpressionWalker):
    """
    Move :class:`RenameAxes`, :class:`SliceAxis` and :class:`Diagonal`
    down to the factors, and flatten products of products.

    - a renaming of a factor renames its axes; of a contraction, the
      axes it keeps (the summed ones are its own); of a product or a
      ratio, each term;
    - a slice of a factor slices the column of the axis (a factor
      without the axis is unchanged); of a contraction summing the axis,
      nothing; else each term reading the axis;
    - a diagonal of a factor ties the columns of the two axes; of a
      product, the one term reading both, if there is one.

    A diagonal across terms is not a tensor operation of this form:
    :class:`NotApplicable`.
    """

    @add_match(RenameAxes(Factor, ...))
    def rename_factor(self, expression):
        renames = dict(expression.renames)
        factor = expression.term
        return factor.replace(axes=tuple(renames.get(a, a) for a in factor.axes))

    @add_match(RenameAxes(Contraction, ...))
    def rename_contraction(self, expression):
        contraction = expression.term
        renames = dict(expression.renames)
        if any(
            new in contraction.term.axes and new not in contraction.keep
            for old, new in renames.items() if old in contraction.keep
        ):  # a summed axis named as a kept one: rename it first
            raise NotApplicable("a renaming onto a summed axis")
        inner = tuple((a, n) for a, n in renames.items() if a in contraction.keep)
        return self.walk(type(contraction)(
            RenameAxes(contraction.term, inner),
            tuple(renames.get(a, a) for a in contraction.keep),
        ))

    @add_match(RenameAxes(TensorProduct, ...))
    def rename_product(self, expression):
        return self.walk(TensorProduct(tuple(
            RenameAxes(t, expression.renames) for t in expression.term.terms
        )))

    @add_match(RenameAxes(TensorRatio, ...))
    def rename_ratio(self, expression):
        ratio = expression.term
        return self.walk(TensorRatio(
            RenameAxes(ratio.numerator, expression.renames),
            RenameAxes(ratio.denominator, expression.renames),
        ))

    @add_match(SliceAxis(Factor, ..., ...))
    def slice_factor(self, expression):
        factor = expression.term
        if expression.axis not in factor.axes:
            return factor
        position = factor.positions[factor.axes.index(expression.axis)]
        return factor.replace(
            slices=factor.slices + ((position, expression.constant),)
        )

    @add_match(SliceAxis(Contraction, ..., ...))
    def slice_contraction(self, expression):
        contraction = expression.term
        if expression.axis not in contraction.keep:
            return contraction
        return self.walk(type(contraction)(
            SliceAxis(contraction.term, expression.axis, expression.constant),
            contraction.keep,
        ))

    @add_match(SliceAxis(TensorProduct, ..., ...))
    def slice_product(self, expression):
        return self.walk(TensorProduct(tuple(
            SliceAxis(t, expression.axis, expression.constant)
            for t in expression.term.terms
        )))

    @add_match(SliceAxis(TensorRatio, ..., ...))
    def slice_ratio(self, expression):
        ratio = expression.term
        return self.walk(TensorRatio(
            SliceAxis(ratio.numerator, expression.axis, expression.constant),
            SliceAxis(ratio.denominator, expression.axis, expression.constant),
        ))

    @add_match(Diagonal(Factor, ..., ...))
    def diagonal_factor(self, expression):
        factor = expression.term
        first, second = (
            factor.positions[factor.axes.index(a)]
            for a in (expression.first, expression.second)
        )
        return factor.replace(diagonals=factor.diagonals + ((first, second),))

    @add_match(Diagonal(TensorProduct, ..., ...))
    def diagonal_product(self, expression):
        pair = {expression.first, expression.second}
        terms = expression.term.terms
        holding = [i for i, t in enumerate(terms) if pair <= set(t.axes)]
        if not holding:
            raise NotApplicable("a diagonal across the terms of a product")
        i = holding[0]
        return self.walk(TensorProduct(
            terms[:i]
            + (Diagonal(terms[i], expression.first, expression.second),)
            + terms[i + 1:]
        ))

    @add_match(
        TensorProduct,
        lambda product: any(isinstance(t, TensorProduct) for t in product.terms),
    )
    def flatten_product(self, product):
        return self.walk(TensorProduct(tuple(
            s for t in product.terms
            for s in (t.terms if isinstance(t, TensorProduct) else (t,))
        )))


def normal_form(expression):
    """Return *expression* in the normal form a backend computes."""
    return PushAxisOperationsToFactors().walk(
        RenameSummedAxesApart().walk(expression)
    )
