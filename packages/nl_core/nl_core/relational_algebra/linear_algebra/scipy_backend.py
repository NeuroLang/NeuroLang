"""
A backend computing tensor expressions (:mod:`.expressions`) in normal
form (:mod:`.rewrites`) with scipy's sparse matrices and numpy.

Axes are grouped and numbered: the axes an engine's view groups (a voxel
``(i, j, k)``) are numbered by the view's :class:`~.encoding.EntityIndex`,
each other axis by an index of its values. The terms of a product are
multiplied pairwise, in their order, the axes no later term reads summed
out as soon as possible: each pair as a sparse matrix product (SpGEMM) of
the two tensors laid out as matrices of their kept axes by their shared
ones, as a scaling when the axes of one are kept ones of the other, or as
a dense ``numpy.einsum`` when both and the result are small. A relation
held by an engine's :class:`~.encoding.SparseView` (same columns) is read
as its matrix. Numbers are float64; entries of 0 may be left out of the
result.
"""
import math
import string
from typing import Any, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from nl_expr.config import config
from nl_expr.expression_walker import PatternWalker, add_match

from .encoding import EntityIndex, views_of_columns
from .expressions import (
    Contraction,
    Factor,
    NoisyOrContraction,
    TensorExpression,
    TensorProduct,
    TensorRatio,
    WeightedFactor,
)
from .rewrites import NotApplicable


def _children(expression):
    if isinstance(expression, Contraction):
        return (expression.term,)
    if isinstance(expression, TensorProduct):
        return expression.terms
    if isinstance(expression, TensorRatio):
        return (expression.numerator, expression.denominator)
    if isinstance(expression, Factor):
        return ()
    raise NotApplicable(f"{expression} is not in normal form")


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

    def __init__(
        self, groups, codes=None, values=None, matrix=None, by_columns=None
    ):
        self.groups = tuple(groups)
        self.matrix = matrix
        # the matrix (an engine's view, as it is) by its column entities,
        # a ColumnIndex; a tensor computed from it has none
        self.by_columns = by_columns
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
                by_columns=getattr(view, "by_columns", None),
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
            frame.iloc[:, factor.value].to_numpy(np.float64)
            if isinstance(factor, WeightedFactor) else np.ones(len(frame))
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
            # A product by the diagonal of the factors present: by rows,
            # it reads the rows of those alone (the voxels of an image
            # out of the 346 M entries of activated, where copying the
            # matrix and scaling each entry took 25 s); by columns, every
            # entry once, never copied. The diagonal's indices have the
            # matrix's type, which the product would convert otherwise.
            present = np.flatnonzero(vector).astype(matrix.indices.dtype)
            diagonal = sparse.csr_array(
                (vector[present], (present, present)), shape=(size, size)
            )
            matrix = (
                matrix @ diagonal if tensor.groups.index(by.groups[0])
                else diagonal @ matrix
            )
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

    def matrix_vector(self, held, vector, keep):
        """Return the product of the matrix *held* and the tensor *vector*
        of one of its two groups, summed over that group, as a sparse
        matrix-vector product with a dense vector: the matrix (an
        engine's view) is read as it is, never copied; None if they are
        not such a pair."""
        if (
            held.matrix is None or len(vector.groups) != 1
            or vector.groups[0] not in held.groups
            or vector.groups[0] in keep
        ):
            return None
        group = vector.groups[0]
        other = held.groups[1 - held.groups.index(group)]
        if other not in keep:
            return None
        codes, values = vector.codes[0], vector.values
        rows = held.matrix
        if held.groups[0] == group and rows.format == "csr" and _few(
            np.diff(rows.indptr)[codes].sum(), rows.nnz
        ):
            # a few rows (the voxels of an image) read alone, not the
            # whole view by its transpose: 10 voxels of activated, 1.7 s
            # warm and 55 s cold for its 1 billion entries
            out = values @ rows[codes]
        elif held.groups[1] == group and held.by_columns is not None:
            order = np.argsort(codes)
            if _selective(held.by_columns, codes[order]):
                out = values[order] @ held.by_columns.matrix(codes[order])
            else:
                out = rows @ self._dense(group, codes, values)
        else:
            dense = self._dense(group, codes, values)
            out = (rows if held.groups[1] == group else rows.T) @ dense
        codes = np.flatnonzero(out)
        return _Tensor((other,), [codes], out[codes])

    def _dense(self, group, codes, values):
        """The dense vector of the entities of *group*, *values* at
        *codes*."""
        dense = np.zeros(self.size(group))
        dense[codes] = values
        return dense

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
        for held, vector in ((left, right), (right, left)):
            product = self.matrix_vector(held, vector, keep)
            if product is not None:
                return product
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
        held = (left_matrix is not None, right_matrix is not None)
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
            left_matrix = _csr(
                left.values, row_keys, inner_left, (row_size, inner_size)
            )
        if right_matrix is None:
            column_keys, column_unique, column_size = _compact_one(
                self.keys(right, columns), self.cells(columns)
            )
            if len(inner) <= 1:
                inner_right = self.keys(right, inner)
            right_matrix = _csr(
                right.values, inner_right, column_keys,
                (inner_size, column_size),
            )
        # a view (the right operand, as products are taken) by an operand
        # that selects few of its column entities (contracted): both
        # restricted to them, the view read through its index
        if held == (False, True) and right.by_columns is not None and (
            inner + columns == list(right.groups[::-1])
        ):
            selected = np.unique(left_matrix.indices)
            if _selective(right.by_columns, selected):
                left_matrix = left_matrix[:, selected]
                right_matrix = right.by_columns.matrix(selected)
                held = (False, False)
        product = sparse.coo_array(_product(left_matrix, right_matrix, held))
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

    def frame(self, tensor, axes, value):
        """Return the entries of *tensor*, its axes in the order of
        *axes*, and its numbers as the column *value*."""
        parts = [
            self.indices[g].decode(codes, g)
            for g, codes in zip(tensor.groups, tensor.codes)
        ]
        parts.append(pd.DataFrame({value: tensor.values}))
        return pd.concat(parts, axis=1)[list(axes) + [value]]

class ScipyTensorSolver(PatternWalker):
    """Compute a tensor expression in normal form as a :class:`_Tensor`,
    over the groups, indices, rows and views of *evaluation*."""

    def __init__(self, evaluation):
        self.evaluation = evaluation

    @add_match(Factor)
    def factor(self, factor):
        return self.evaluation.leaf(factor)

    @add_match(TensorRatio)
    def ratio(self, ratio):
        return self.evaluation.scale(
            self.walk(ratio.numerator), self.walk(ratio.denominator),
            divide=True,
        )

    @add_match(TensorProduct)
    def product(self, product):
        return self._products(product.terms, product.axes)

    @add_match(NoisyOrContraction)
    def noisy_or(self, contraction):
        """``1 - exp(sum log(1 - p))``: a probability of 1 gives a log of
        ``-inf`` and a noisy-or of 1; absent entries, of probability 0,
        add nothing."""
        term = contraction.term
        terms = term.terms if isinstance(term, TensorProduct) else (term,)
        evaluation = self.evaluation
        tensor = self._products(terms, term.axes)
        with np.errstate(divide="ignore"):
            logs = np.log1p(-np.minimum(tensor.values, 1.))
        sums = evaluation.sum_out(
            _Tensor(tensor.groups, tensor.codes, logs),
            evaluation.groups(contraction.keep),
        )
        values = -np.expm1(sums.values)
        present = values != 0
        return _Tensor(
            sums.groups, [c[present] for c in sums.codes], values[present]
        )

    @add_match(Contraction)
    def contraction(self, contraction):
        term = contraction.term
        terms = term.terms if isinstance(term, TensorProduct) else (term,)
        return self._products(terms, contraction.keep)

    def _products(self, terms, axes):
        evaluation = self.evaluation
        keep = evaluation.groups(axes)
        # The smallest tensors first, the matrices of views last: the small
        # ones combine into a vector before the large one is read, which a
        # product with a vector then reads as it is.
        tensors = sorted(
            (self.walk(t) for t in terms),
            key=lambda t: (
                (True, t.matrix.nnz) if t.matrix is not None
                else (False, len(t.values))
            ),
        )
        result = tensors.pop(0)
        while tensors:
            later = {g for t in tensors[1:] for g in t.groups}
            result = evaluation.contract(
                result, tensors.pop(0), set(keep) | later
            )
        return evaluation.sum_out(result, keep)

    @add_match(TensorExpression)
    def other(self, expression):
        raise NotApplicable(f"{expression} is not in normal form")


def _product(left, right, held):
    """
    Return the product of the sparse matrices *left* and *right*, of
    which those *held* (a pair of booleans) are an engine's views, read
    as they are.

    A view by an operand of a few rows is the product of the view by
    that operand as a dense matrix (contractions multiply the product
    of the terms so far by the next term: a view is the right operand): one pass over the view, in
    its own layout. A sparse product would cost the whole view all the
    same, in the layout it is not held in (a selection of 2 895 studies
    by the 1 billion entries of neurostore's voxel by study view: 2.8 s
    rather than 12 s). Otherwise the operand of fewer entries takes the
    layout of the other (a view transposed is a CSC matrix).
    """
    if held == (False, True) and left.shape[0] * max(
        left.shape[1], right.shape[1]
    ) <= config.getint("PROBABILISTIC_SOLVER", "few_cells"):
        return (right.T @ left.T.toarray()).T
    small, big = (left, right) if left.nnz <= right.nnz else (right, left)
    small = small.asformat(big.format)
    return small @ big if left.nnz <= right.nnz else big @ small


def _few(entries, nnz):
    """Whether *entries* of a view of *nnz* are few enough to be read alone
    (``[PROBABILISTIC_SOLVER] index_share`` of them at most)."""
    return entries <= config.getfloat(
        "PROBABILISTIC_SOLVER", "index_share"
    ) * nnz


def _selective(index, columns):
    """Whether the entries of the column entities *columns* of a view are
    few enough to be read through its *index*
    (:class:`~.encoding.ColumnIndex`)."""
    return _few(index.entries(columns), index.nnz)


def _csr(values, rows, columns, shape):
    """
    Return the CSR matrix of *shape* of the *values* at (*rows*,
    *columns*), with indices of 32 bits when they fit, as those of the
    views: scipy keeps the type of the numbers it is given (64 bits for
    keys), and a product of matrices of indices of two types converts
    both to 64 bits (the 1 billion indices of neurostore's voxel by study
    view: 12 s rather than 5 s).
    """
    if max(*shape, len(values)) <= np.iinfo(np.int32).max:
        rows, columns = rows.astype(np.int32), columns.astype(np.int32)
    return sparse.csr_array((values, (rows, columns)), shape=shape)


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
    if factor.slices or factor.diagonals or isinstance(
        factor, WeightedFactor
    ):
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

    Relational algebra is left the factors of missing values in an axis
    (which it joins with each other), of a tuple of two probabilities
    (two facts, or two outcomes of a choice) and of values without an
    order: :class:`NotApplicable`.
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
        axes = frame.iloc[:, list(factor.positions)]
        if axes.isna().to_numpy().any():  # joined with each other
            raise NotApplicable("missing values in a column of axes")
        try:
            duplicated = isinstance(factor, WeightedFactor) and (
                axes.duplicated().any()
            )
        except TypeError as error:  # values without a hash
            raise NotApplicable(str(error)) from error
        if duplicated:
            raise NotApplicable("a tuple of two probabilities")
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
    try:
        indices = {
            g: i if g in viewed else EntityIndex([i])
            for g, i in indices.items()
        }
    except (TypeError, ValueError) as error:  # values without an order,
        raise NotApplicable(str(error)) from error  # or too many
    return set(group_of.values()), indices, frames, views


def _ratios(expression):
    found = [expression] if isinstance(expression, TensorRatio) else []
    return found + [
        r for child in _children(expression) for r in _ratios(child)
    ]


def evaluate(expression, value, dense_cells=0):
    """
    Return the entries of the tensor of *expression*.

    Parameters
    ----------
    expression : TensorExpression
        The tensor, in the normal form of :mod:`.rewrites`; an axis is one
        variable, each name is summed by at most one contraction, over
        the factors it holds.
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
    tensor = ScipyTensorSolver(evaluation).walk(expression)
    return evaluation.frame(tensor, expression.axes, value)
