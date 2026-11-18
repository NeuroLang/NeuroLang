"""
Tensor expressions: sums of products of relations, as NeuroLang
expressions.

A relation whose tuples carry a number (a probability, or 1) is a tensor
with an axis per column: its entry at a tuple is the tuple's number, 0
off the relation. Over the semiring ``(+, ×)`` of the reals a natural
join multiplies the numbers of the tuples it pairs and a projection sums
those of the tuples it merges, so a plan of joins and projections is a
contraction of the tensors of its relations.

The expressions of this module write such a plan:

- :class:`Factor`, the tensor of a relation;
- :class:`TensorProduct`, the product of tensors over the union of their
  axes (a natural join);
- :class:`Contraction`, the sum of a tensor over the axes it does not
  keep (a projection), and :class:`NoisyOrContraction`, their noisy-or
  ``1 - prod(1 - p)`` (an independent projection);
- :class:`TensorRatio`, a tensor divided by another over the axes of the
  latter (a natural join dividing the numbers);
- :class:`RenameAxes`, :class:`SliceAxis` and :class:`Diagonal`, a tensor
  with axes renamed, restricted to a constant on an axis, or to its
  entries with two axes equal.

They are produced by a translator from another language (the provenance
plans of the probabilistic solver), normalised by the walkers of
:mod:`.rewrites`, which leave the last three on factors only, and
computed by a backend (:mod:`.scipy_backend`). An axis name is one
variable.
"""
from typing import Any, Optional, Tuple

from nl_expr.expressions import Definition


class TensorExpression(Definition):
    """A tensor; ``axes`` are the names of its axes, in order."""

    @property
    def axes(self) -> Tuple:
        raise NotImplementedError()


class Factor(TensorExpression):
    """
    The tensor of *relation*, a relation set: an axis per column of
    *positions*, named by *axes*, its entry at a tuple the number in the
    column *value*, or 1 without one.

    Its tuples are those of the relation equal to the constant of each
    ``(position, constant)`` of *slices*, and with equal values in the
    two columns of each ``(position, position)`` of *diagonals*.
    """

    def __init__(
        self, relation: Any, axes: Tuple, positions: Tuple[int, ...],
        value: Optional[int] = None, slices: Tuple = (),
        diagonals: Tuple = (),
    ):
        self.relation = relation
        self.factor_axes = tuple(axes)
        self.positions = tuple(positions)
        self.value = value
        self.slices = tuple(slices)
        self.diagonals = tuple(diagonals)

    @property
    def axes(self):
        return self.factor_axes

    def replace(self, **changes):
        """Return this factor with the fields of *changes* replaced."""
        fields = dict(
            relation=self.relation, axes=self.factor_axes,
            positions=self.positions, value=self.value,
            slices=self.slices, diagonals=self.diagonals,
        )
        fields.update(changes)
        return Factor(**fields)

    def __repr__(self):
        value = "1" if self.value is None else f"#{self.value}"
        cuts = "".join(
            f"[{p}={c!r}]" for p, c in self.slices
        ) + "".join(f"[{p}=={q}]" for p, q in self.diagonals)
        return f"T<{value}>{cuts}{self.factor_axes}"


class TensorProduct(TensorExpression):
    """The product of the tensors of *terms*, over the union of their
    axes."""

    def __init__(self, terms: Tuple):
        self.terms = tuple(terms)

    @property
    def axes(self):
        return tuple(dict.fromkeys(a for t in self.terms for a in t.axes))

    def __repr__(self):
        return "(" + " \N{MULTIPLICATION SIGN} ".join(map(repr, self.terms)) + ")"


class Contraction(TensorExpression):
    """The sum of the tensor *term* over its axes not in *keep*, the
    axes of the result, in that order."""

    def __init__(self, term: TensorExpression, keep: Tuple):
        self.term = term
        self.keep = tuple(keep)

    @property
    def axes(self):
        return self.keep

    def __repr__(self):
        return f"\N{N-ARY SUMMATION}_{self.keep}{self.term!r}"


class NoisyOrContraction(Contraction):
    """
    The noisy-or of the tensor *term* over its axes not in *keep*: at
    each kept entry, ``1 - prod(1 - p)`` of the entries *p* it merges,
    the probability that one of independent events holds (an independent
    projection of a lifted plan). Entries absent from *term* are events
    of probability 0.
    """

    def __repr__(self):
        return f"\N{N-ARY COPRODUCT}_{self.keep}{self.term!r}"


class TensorRatio(TensorExpression):
    """The tensor *numerator* divided by *denominator*, whose axes are
    axes of the numerator, at the entries of both."""

    def __init__(self, numerator: TensorExpression,
                 denominator: TensorExpression):
        self.numerator = numerator
        self.denominator = denominator

    @property
    def axes(self):
        return self.numerator.axes

    def __repr__(self):
        return f"({self.numerator!r} / {self.denominator!r})"


class RenameAxes(TensorExpression):
    """The tensor *term* with its axes renamed by *renames*, pairs
    ``(axis, new name)``."""

    def __init__(self, term: TensorExpression, renames: Tuple):
        self.term = term
        self.renames = tuple(renames)

    @property
    def axes(self):
        renames = dict(self.renames)
        return tuple(renames.get(a, a) for a in self.term.axes)

    def __repr__(self):
        return f"\N{GREEK SMALL LETTER RHO}_{self.renames}{self.term!r}"


class SliceAxis(TensorExpression):
    """The entries of *term* whose axis *axis* is *constant*; the axis
    stays."""

    def __init__(self, term: TensorExpression, axis: Any, constant: Any):
        self.term = term
        self.axis = axis
        self.constant = constant

    @property
    def axes(self):
        return self.term.axes

    def __repr__(self):
        return f"\N{GREEK SMALL LETTER SIGMA}_{self.axis}={self.constant!r}{self.term!r}"


class Diagonal(TensorExpression):
    """The entries of *term* whose axes *first* and *second* are equal;
    both axes stay."""

    def __init__(self, term: TensorExpression, first: Any, second: Any):
        self.term = term
        self.first = first
        self.second = second

    @property
    def axes(self):
        return self.term.axes

    def __repr__(self):
        return f"\N{GREEK SMALL LETTER SIGMA}_{self.first}={self.second}{self.term!r}"
