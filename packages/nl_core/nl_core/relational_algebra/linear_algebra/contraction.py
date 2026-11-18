"""
Sums of products of relations as contractions of sparse tensors.

A relation whose tuples carry a number, a probability or 1, is a tensor
with an axis per column: its entry at a tuple is the tuple's number, and
0 off the relation. A natural join multiplies the numbers of the tuples
it pairs, a projection sums those of the tuples it merges: over the
semiring ``(+, ×)`` of the reals, a plan of joins and projections is a
contraction of the tensors of its relations, the axes it projects away
summed, the others kept.

The plan is written in the expressions of this module:

- :class:`Factor`, a relation, its tuples sliced by constants, with the
  numbers of a column of it, or 1;
- :class:`Product`, the product of tensors over the union of their axes
  (a natural join);
- :class:`Contraction`, the sum of a tensor over the axes it does not
  keep (a projection);
- :class:`Ratio`, a tensor divided by another over the axes of the
  latter (a natural join dividing the numbers).

:func:`evaluate` computes one. An axis name is one variable: the
entities of the axes that one engine's view groups (a voxel ``(i, j,
k)``) are numbered by the view's :class:`~.encoding.EntityIndex`, and
the values of each other axis by an index of its own. The terms of a
product are multiplied pairwise, in their order, the axes no later term
reads summed out as soon as possible: each pair as a sparse matrix
product (SpGEMM) of the two tensors laid out as matrices of their kept
axes by their shared ones, as a scaling when the axes of one are kept
ones of the other, or as a dense ``numpy.einsum`` when both and the
result are small. A relation held by an engine's
:class:`~.encoding.SparseView` (same columns) is read as its matrix.
Numbers are float64; entries of 0 may be left out of the result.
"""
import math
import string
from typing import Any, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from .encoding import EntityIndex, views_of_columns


class NotApplicable(Exception):
    """The expression is not a sum of products :func:`evaluate`
    computes."""


class Factor(NamedTuple):
    """
    The tensor of *relation*: an axis per column of *positions*, named
    by *axes*, its entry at a tuple the number in the column *value*, or
    1 without one.

    Its tuples are those of the relation equal to the constant of each
    ``(position, constant)`` of *slices*, and with equal values in the
    two columns of each ``(position, position)`` of *diagonals*.
    """

    relation: Any
    axes: Tuple
    positions: Tuple[int, ...]
    value: Optional[int] = None
    slices: Tuple = ()
    diagonals: Tuple = ()


class Product(NamedTuple):
    """The product of the tensors of *terms*, over the union of their
    axes."""

    terms: Tuple

    @property
    def axes(self):
        return tuple(dict.fromkeys(a for t in self.terms for a in t.axes))


class Contraction(NamedTuple):
    """The sum of the tensor *term* over its axes not in *keep*, the
    axes of the result, in that order."""

    term: Any
    keep: Tuple

    @property
    def axes(self):
        return self.keep


class Ratio(NamedTuple):
    """The tensor *numerator* divided by *denominator*, of axes the
    numerator has, at the entries of both."""

    numerator: Any
    denominator: Any

    @property
    def axes(self):
        return self.numerator.axes


def _children(expression):
    if isinstance(expression, Contraction):
        return (expression.term,)
    if isinstance(expression, Product):
        return expression.terms
    if isinstance(expression, Ratio):
        return (expression.numerator, expression.denominator)
    return ()


def _rebuild(expression, children):
    if isinstance(expression, Contraction):
        return expression._replace(term=children[0])
    if isinstance(expression, Product):
        return Product(tuple(children))
    return Ratio(*children)


def rename(expression, renames):
    """Return *expression* with its axes renamed by *renames*, a mapping
    of the axes to rename to their new names (axes it sums are not)."""
    if isinstance(expression, Factor):
        return expression._replace(
            axes=tuple(renames.get(a, a) for a in expression.axes)
        )
    if isinstance(expression, Contraction):
        inner = {a: n for a, n in renames.items() if a in expression.keep}
        return Contraction(
            rename(expression.term, inner),
            tuple(renames.get(a, a) for a in expression.keep),
        )
    return _rebuild(expression, [
        rename(child, renames) for child in _children(expression)
    ])


def select(expression, axis, constant):
    """Return *expression* restricted to its entries whose *axis* is
    *constant*: each factor of the axis sliced."""
    if isinstance(expression, Factor):
        if axis not in expression.axes:
            return expression
        position = expression.positions[expression.axes.index(axis)]
        return expression._replace(
            slices=expression.slices + ((position, constant),)
        )
    if isinstance(expression, Contraction) and axis not in expression.keep:
        return expression
    return _rebuild(expression, [
        select(child, axis, constant) for child in _children(expression)
    ])


def _factors(expression):
    if isinstance(expression, Factor):
        return [expression]
    return [f for c in _children(expression) for f in _factors(c)]


def _contractions(expression):
    found = [expression] if isinstance(expression, Contraction) else []
    return found + [
        c for child in _children(expression) for c in _contractions(child)
    ]


class _Tensor:
    """
    A sparse tensor over *groups* of axes, each numbered by an index:
    the numbers of the entities of each group at each entry (*codes*)
    and the entries (*values*), or, of two groups, the CSR *matrix* of
    the first by the second.
    """

    def __init__(self, groups, codes=None, values=None, matrix=None):
        self.groups = tuple(groups)
        self.matrix = matrix
        self._codes, self._values = codes, values

    def _entries(self):
        if self._codes is None:
            entries = self.matrix.tocoo()
            self._codes = [
                entries.row.astype(np.int64), entries.col.astype(np.int64)
            ]
            self._values = entries.data
        return self._codes, self._values

    @property
    def codes(self):
        return self._entries()[0]

    @property
    def values(self):
        return self._entries()[1]


class _Evaluation:
    """The evaluation of an expression, given the groups of its axes,
    the index numbering the entities of each, the rows of its factors
    and the views read as matrices (by the factor's id)."""

    def __init__(self, groups, indices, frames, views, dense_cells):
        self.group_of = {a: g for g in groups for a in g}
        self.indices = indices
        self.frames = frames
        self.views = views
        self.dense_cells = dense_cells

    def size(self, group):
        return len(self.indices[group])

    def cells(self, groups):
        return math.prod(self.size(g) for g in groups)

    def groups(self, axes):
        return tuple(dict.fromkeys(self.group_of[a] for a in axes))

    def keys(self, tensor, groups):
        """Return the key of each entry of *tensor* over *groups*, mixed
        radix of their numbers."""
        if self.cells(groups) >= 2 ** 62:
            raise NotApplicable("too many combinations of entities")
        keys = np.zeros(len(tensor.values), dtype=np.int64)
        for group in groups:
            keys *= self.size(group)
            keys += tensor.codes[tensor.groups.index(group)]
        return keys

    def unkey(self, keys, groups):
        """Return the numbers of the entities of *groups* of *keys*."""
        codes = []
        for group in reversed(groups):
            keys, code = np.divmod(keys, self.size(group))
            codes.append(code)
        return codes[::-1]

    def leaf(self, factor):
        found = self.views.get(id(factor))
        if found is not None:
            view, (rows, columns) = found
            return _Tensor(
                (self.group_of[rows[0]], self.group_of[columns[0]]),
                matrix=view.weighted(),
            )
        frame = self.frames[id(factor)]
        groups = self.groups(factor.axes)
        present = np.ones(len(frame), dtype=bool)
        codes = []
        for group in groups:
            code = self.indices[group].encode(_columns(frame, factor, group))
            present &= code >= 0
            codes.append(code)
        values = (
            np.ones(len(frame)) if factor.value is None
            else frame.iloc[:, factor.value].to_numpy(np.float64)
        )
        return _Tensor(
            groups, [c[present].astype(np.int64) for c in codes],
            values[present],
        )

    def sum_out(self, tensor, keep):
        """Return the sum of *tensor* over its groups not in *keep*."""
        keep = tuple(g for g in tensor.groups if g in keep)
        if keep == tensor.groups:
            return tensor
        matrix = tensor.matrix
        if matrix is not None and not keep:
            return _Tensor((), [], np.array([matrix.sum()])[:matrix.nnz])
        if matrix is not None:
            axis = 1 - tensor.groups.index(keep[0])
            sums = np.asarray(matrix.sum(axis=axis)).ravel()
            stored = (
                np.diff(matrix.indptr) if axis == 1
                else np.bincount(matrix.indices, minlength=matrix.shape[1])
            )
            codes = np.flatnonzero(stored)
            return _Tensor(keep, [codes], sums[codes])
        if not keep:
            values = tensor.values
            return _Tensor((), [], np.array([values.sum()])[:len(values)])
        keys, inverse = np.unique(
            self.keys(tensor, keep), return_inverse=True
        )
        return _Tensor(
            keep, self.unkey(keys, keep),
            np.bincount(inverse, weights=tensor.values),
        )

    def scale(self, tensor, by, divide=False):
        """Return *tensor* times *by*, or divided by it, at the entries
        of both: the groups of *by* are groups of *tensor*."""
        with np.errstate(divide="ignore"):
            factors = 1 / by.values if divide else by.values
        if not by.groups:
            factor = factors[0] if len(factors) else 0.
            if tensor.matrix is not None:
                matrix = tensor.matrix * factor
                matrix.eliminate_zeros()
                return _Tensor(tensor.groups, matrix=matrix)
            values = tensor.values * factor
            present = values != 0
            return _Tensor(
                tensor.groups, [c[present] for c in tensor.codes],
                values[present],
            )
        size = self.cells(by.groups)
        if len(by.groups) == 1:
            by_keys = by.codes[0]
        else:
            keys, by_keys, size = _compact(
                self.keys(tensor, by.groups), self.keys(by, by.groups),
                size=size,
            )
        vector = np.zeros(size)
        vector[by_keys] = factors
        matrix = tensor.matrix
        if len(by.groups) == 1 and matrix is not None:
            matrix = matrix.copy()
            matrix.data *= vector[
                matrix.indices if tensor.groups.index(by.groups[0])
                else np.repeat(
                    np.arange(matrix.shape[0]), np.diff(matrix.indptr)
                )
            ]
            matrix.eliminate_zeros()
            return _Tensor(tensor.groups, matrix=matrix)
        if len(by.groups) == 1:
            keys = tensor.codes[tensor.groups.index(by.groups[0])]
        values = tensor.values * vector[keys]
        present = values != 0
        return _Tensor(
            tensor.groups, [c[present] for c in tensor.codes],
            values[present],
        )

    def matrix(self, tensor, rows, columns):
        """Return the matrix of *tensor*, of the group *rows* by the
        group *columns*, if it is held as one."""
        if tensor.matrix is None or not len(rows) == len(columns) == 1:
            return None
        if (rows[0], columns[0]) == tensor.groups:
            return tensor.matrix
        if (columns[0], rows[0]) == tensor.groups:
            return tensor.matrix.T
        return None

    def contract(self, left, right, keep):
        """Return the product of *left* and *right* summed over their
        groups not in *keep*."""
        left = self.sum_out(
            left, [g for g in left.groups if g in keep or g in right.groups]
        )
        right = self.sum_out(
            right, [g for g in right.groups if g in keep or g in left.groups]
        )
        for big, small in ((left, right), (right, left)):
            if set(small.groups) <= set(big.groups) & set(keep):
                return self.scale(big, small)
        shared = [g for g in left.groups if g in right.groups]
        batch = [g for g in shared if g in keep]
        rows = batch + [g for g in left.groups if g not in shared]
        inner = batch + [g for g in shared if g not in keep]
        columns = [g for g in right.groups if g not in shared]
        if max(
            self.cells(left.groups), self.cells(right.groups),
            self.cells(rows + columns),
        ) <= self.dense_cells:
            return self.dense(left, right, rows + columns)
        left_matrix = self.matrix(left, rows, inner)
        right_matrix = self.matrix(right, inner, columns)
        inner_size = self.cells(inner)
        if len(inner) > 1:  # neither is held as a matrix
            inner_left, inner_right, inner_size = _compact(
                self.keys(left, inner), self.keys(right, inner),
                size=inner_size,
            )
        row_unique = column_unique = None
        if left_matrix is None:
            row_keys, row_unique, row_size = _compact_one(
                self.keys(left, rows), self.cells(rows)
            )
            if len(inner) <= 1:
                inner_left = self.keys(left, inner)
            left_matrix = sparse.csr_array(
                (left.values, (row_keys, inner_left)),
                shape=(row_size, inner_size),
            )
        if right_matrix is None:
            column_keys, column_unique, column_size = _compact_one(
                self.keys(right, columns), self.cells(columns)
            )
            if len(inner) <= 1:
                inner_right = self.keys(right, inner)
            right_matrix = sparse.csr_array(
                (right.values, (inner_right, column_keys)),
                shape=(inner_size, column_size),
            )
        product = sparse.coo_array(left_matrix @ right_matrix)
        row_keys = product.row.astype(np.int64)
        column_keys = product.col.astype(np.int64)
        if row_unique is not None:
            row_keys = row_unique[row_keys]
        if column_unique is not None:
            column_keys = column_unique[column_keys]
        return _Tensor(
            rows + columns,
            self.unkey(row_keys, rows) + self.unkey(column_keys, columns),
            product.data,
        )

    def dense(self, left, right, out):
        """Return the product of *left* and *right*, of groups *out*, by
        ``numpy.einsum`` over dense arrays."""
        letters = dict(zip(
            dict.fromkeys(left.groups + right.groups), string.ascii_letters
        ))
        arrays = []
        for tensor in (left, right):
            array = np.zeros([self.size(g) for g in tensor.groups])
            array[tuple(tensor.codes)] = tensor.values
            arrays.append(array)
        result = np.einsum(
            ",".join(
                "".join(letters[g] for g in t.groups) for t in (left, right)
            ) + "->" + "".join(letters[g] for g in out),
            *arrays,
        )
        if not out:
            result = result.reshape(1)
            return _Tensor((), [], result[result != 0])
        codes = list(np.nonzero(result))
        return _Tensor(out, codes, result[tuple(codes)])

    def evaluate(self, expression):
        if isinstance(expression, Factor):
            return self.leaf(expression)
        if isinstance(expression, Ratio):
            return self.scale(
                self.evaluate(expression.numerator),
                self.evaluate(expression.denominator), divide=True,
            )
        keep = self.groups(expression.axes)
        terms = expression.terms if isinstance(expression, Product) else (
            expression.term.terms if isinstance(expression.term, Product)
            else (expression.term,)
        )
        tensors = [self.evaluate(t) for t in terms]
        result = tensors.pop(0)
        while tensors:
            later = {g for t in tensors[1:] for g in t.groups}
            result = self.contract(
                result, tensors.pop(0), set(keep) | later
            )
        return self.sum_out(result, keep)

    def frame(self, tensor, axes, value):
        """Return the entries of *tensor*, its axes in the order of
        *axes*, and its numbers as the column *value*."""
        parts = [
            self.indices[g].decode(codes, g)
            for g, codes in zip(tensor.groups, tensor.codes)
        ]
        parts.append(pd.DataFrame({value: tensor.values}))
        return pd.concat(parts, axis=1)[list(axes) + [value]]

def _compact(*keys, size):
    """Return *keys* numbered in the order of their distinct values when
    there are more possible keys than keys, and the number of keys."""
    if size <= 4 * sum(len(k) for k in keys) + 1024:
        return keys + (size,)
    unique = np.unique(np.concatenate(keys))
    return tuple(np.searchsorted(unique, k) for k in keys) + (len(unique),)


def _compact_one(keys, size):
    """Return *keys* compacted as :func:`_compact` does, the key of each
    number (or None if unchanged), and the number of keys."""
    if size <= 4 * len(keys) + 1024:
        return keys, None, size
    unique, keys = np.unique(keys, return_inverse=True)
    return keys, unique, len(unique)


def _rows(factor):
    """Return the rows of the relation of *factor*, sliced."""
    frame = factor.relation.as_pandas_dataframe()
    if frame is None:
        frame = pd.DataFrame(columns=range(len(factor.positions)))
    for position, constant in factor.slices:
        frame = frame[frame.iloc[:, position] == constant]
    for first, second in factor.diagonals:
        frame = frame[frame.iloc[:, first] == frame.iloc[:, second]]
    return frame


def _columns(frame, factor, group):
    return frame.iloc[:, [
        factor.positions[factor.axes.index(axis)] for axis in group
    ]]


def _view_of(factor):
    """Return the view an engine built of the relation of *factor*, and
    the axes of its rows and of its columns, if there is one: a view
    without values, of a relation not sliced, every column an axis."""
    if factor.slices or factor.diagonals or factor.value is not None:
        return None
    frame = getattr(factor.relation, "_container", None)
    if not isinstance(frame, pd.DataFrame) or len(frame.columns) != len(
        factor.positions
    ):
        return None
    axis_of = {
        frame.columns[p]: a for p, a in zip(factor.positions, factor.axes)
    }
    for view, column_of in views_of_columns(frame):
        if view.value is None:
            return view, (
                tuple(axis_of[column_of[c]] for c in view.rows),
                tuple(axis_of[column_of[c]] for c in view.columns),
            )
    return None


def _plan(expression):
    """
    Return the groups of axes of *expression*, the index numbering the
    entities of each, the rows of the factors not read as a view and the
    view of each other (by the factor's id).

    Each view groups the axes of its rows and those of its columns, if
    every factor and contraction has all the axes of a group or none,
    and a group is numbered by one index. The other axes are each a
    group, numbered by an index of the values of the smallest factor of
    it: the natural join keeps no other.
    """
    factors = _factors(expression)
    for contraction in _contractions(expression):
        inside = {id(f) for f in _factors(contraction.term)}
        summed = set(contraction.term.axes) - set(contraction.keep)
        if any(
            set(f.axes) & summed for f in factors if id(f) not in inside
        ):
            raise NotApplicable(f"the axes {summed} sum a part only")
    for ratio in _ratios(expression):
        if not set(ratio.denominator.axes) <= set(ratio.numerator.axes):
            raise NotApplicable("a denominator of axes not divided")
    axis_sets = [set(f.axes) for f in factors] + [
        set(c.keep) for c in _contractions(expression)
    ]
    group_of, indices, views = {}, {}, {}
    for factor in factors:
        found = _view_of(factor)
        if found is None:
            continue
        view, (rows, columns) = found
        sides = ((rows, view.row_index), (columns, view.column_index))
        if all(
            all(set(g) <= s or not set(g) & s for s in axis_sets)
            and all(group_of.get(a, g) == g for a in g)
            and indices.get(g, index) is index
            for g, index in sides
        ):
            for g, index in sides:
                group_of.update(dict.fromkeys(g, g))
                indices[g] = index
            views[id(factor)] = found
    viewed = set(indices)
    frames, kinds = {}, {}
    for factor in factors:
        if id(factor) in views:
            continue
        frames[id(factor)] = frame = _rows(factor)
        for axis, position in zip(factor.axes, factor.positions):
            kinds.setdefault(axis, set()).add(frame.dtypes.iloc[position].kind)
            group = group_of.setdefault(axis, (axis,))
            columns = _columns(frame, factor, group)
            if group not in viewed and (
                group not in indices or len(columns) < len(indices[group])
            ):
                indices[group] = columns
    if any(len(k) > 1 for k in kinds.values()):
        raise NotApplicable("an axis of values of different kinds")
    indices = {
        g: i if g in viewed else EntityIndex([i]) for g, i in indices.items()
    }
    return set(group_of.values()), indices, frames, views


def _ratios(expression):
    found = [expression] if isinstance(expression, Ratio) else []
    return found + [
        r for child in _children(expression) for r in _ratios(child)
    ]


def evaluate(expression, value, dense_cells=0):
    """
    Return the entries of the tensor of *expression*.

    Parameters
    ----------
    expression : Factor, Product, Contraction or Ratio
        The tensor; an axis is one variable, each name is summed by at
        most one contraction, over the factors it holds.
    value : str
        Name of the column of the entries.
    dense_cells : int
        Products whose operands and result have at most as many cells
        are computed over dense arrays.

    Returns
    -------
    pandas.DataFrame
        A row per entry, of a column per axis of *expression*, in its
        order, of the values of the relations, and the entry, float64, in
        the column *value*. Entries of 0 may be left out.

    Raises
    ------
    NotApplicable
        If the axes of *expression* are not variables (a name summed in
        a part of it and read in another, an axis of values of different
        kinds in two factors) or are too many to be keyed; nothing is
        computed then.
    """
    evaluation = _Evaluation(*_plan(expression), dense_cells)
    tensor = evaluation.evaluate(expression)
    return evaluation.frame(tensor, expression.axes, value)
