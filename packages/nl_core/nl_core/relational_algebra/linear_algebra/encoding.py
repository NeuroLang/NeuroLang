"""
Relations encoded for linear algebra.

An entity of a sort spans one column or several: a voxel ``(i, j, k)`` is
one entity, a study another. :class:`EntityIndex` numbers the entities of
a sort ``0 … n - 1`` and :class:`SparseView` holds a relation between a
row entity and a column entity, with the value of a third column or
without, as a ``scipy.sparse`` CSR matrix over their numbers. The view is
the relation itself, under set semantics: it spans every column of the
relation, and its stored entries are the relation's tuples.

Which columns are an entity, and of which sort, is not known here: it is
given by the caller (an engine's declaration of its sorts).
"""
import sys
import weakref
from collections import Counter
from typing import Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from nl_expr.expressions import Constant, Symbol
from scipy import sparse

from ...datalog.expressions import Conjunction, Implication, Union


def _integers(values):
    """Return *values* as a numpy array if they are integers without
    missing values, else None."""
    values = pd.Series(values, copy=False)
    if pd.api.types.is_integer_dtype(values.dtype) and not values.hasnans:
        return values.to_numpy()
    return None


def _offsets(values, low):
    """Return *values* minus *low*, none below it, in 64 bits of their
    signedness."""
    return np.subtract(
        values, low, dtype=np.uint64 if values.dtype.kind == "u" else np.int64
    )


def _positions(values, dictionary):
    """
    Return the position of each of *values* in *dictionary*, -1 if absent.

    *dictionary* is a sorted ``pandas.Index`` of distinct values. Integers
    spanning no more values than there are *values* are looked up in a
    table of that span; others are hashed once (``pandas.factorize``,
    which does not go through Python objects for Arrow strings) and only
    their distinct values looked up in the *dictionary*. Missing values
    are absent.
    """
    integers = _integers(values)
    if (
        integers is not None and len(dictionary)
        and pd.api.types.is_integer_dtype(dictionary.dtype)
    ):
        low, high = int(dictionary[0]), int(dictionary[-1])
        limits = np.iinfo(integers.dtype)
        if high - low < len(integers) and limits.min <= low <= high <= (
            limits.max
        ):  # offsets from low, in the dtype of each, of no more than span
            lookup = np.full(high - low + 1, -1, dtype=np.int64)
            lookup[_offsets(dictionary.to_numpy(), low)] = np.arange(
                len(dictionary)
            )
            positions = np.full(len(integers), -1, dtype=np.int64)
            inside = (integers >= low) & (integers <= high)
            positions[inside] = lookup[_offsets(integers[inside], low)]
            return positions
    codes, uniques = pd.factorize(values)
    lookup = np.append(dictionary.get_indexer(uniques), -1)
    return lookup[codes]


def _dictionary(series_list):
    """
    Return the distinct values of the series of *series_list*, sorted, as
    a ``pandas.Index``, and the position in it of each value of each
    series (int32), -1 for missing values.

    Integers spanning no more values than there are rows are marked in a
    table of that span, then looked up in it; other values are hashed
    once (``pandas.factorize``) and only their distinct values sorted and
    looked up.
    """
    integers = [_integers(series) for series in series_list]
    rows = sum(len(series) for series in series_list)
    if rows and all(values is not None for values in integers):
        low = min(int(values.min()) for values in integers if len(values))
        high = max(int(values.max()) for values in integers if len(values))
        dtype = np.result_type(*(values.dtype for values in integers))
        if dtype.kind == "f":  # signed and unsigned 64-bit integers
            dtype = np.dtype(
                np.int64 if high <= np.iinfo(np.int64).max
                else np.uint64 if low >= 0 else object
            )
        integers = [values.astype(dtype, copy=False) for values in integers]
        if high - low < rows:  # offsets from low, of no more than rows
            present = np.zeros(high - low + 1, dtype=bool)
            for values in integers:
                present[_offsets(values, low)] = True
            lookup = (np.cumsum(present) - 1).astype(np.int32)
            wide = np.uint64 if dtype.kind == "u" else np.int64
            dictionary = (np.flatnonzero(present).astype(wide) + low).astype(
                dtype
            )
            return pd.Index(dictionary, dtype=dtype), [
                lookup[_offsets(values, low)] for values in integers
            ]
        series_list = [pd.Series(values, copy=False) for values in integers]
    factorized = [pd.factorize(series) for series in series_list]
    uniques = [pd.Series(values) for _, values in factorized]
    dictionary = pd.Index(
        pd.concat(uniques, ignore_index=True).unique() if uniques else []
    ).sort_values()
    if len(dictionary) >= 2 ** 31:
        raise ValueError("a column holds fewer than 2**31 distinct values")
    return dictionary, [
        np.append(dictionary.get_indexer(values), -1).astype(np.int32)[codes]
        for codes, values in factorized
    ]


def _stored(dictionary: pd.Index) -> np.ndarray:
    """Return the values of *dictionary* as an array a file holds."""
    if dictionary.dtype.kind in "iufb":
        return dictionary.to_numpy()
    values = dictionary.to_numpy(dtype=object)
    if not all(isinstance(value, str) for value in values):
        raise TypeError("a dictionary of values neither numbers nor texts")
    return values.astype(str) if len(values) else np.array([], dtype="U1")


def _loaded(values: np.ndarray) -> pd.Index:
    """Return the dictionary of the array *values* (:func:`_stored`)."""
    if values.dtype.kind == "U":
        return pd.Index(values.astype(object), dtype="str")
    return pd.Index(values, copy=False)


class EntityIndex:
    """
    The entities of a sort, numbered ``0 … n - 1`` in the order of their
    values (lexicographic over their columns), and back.

    An entity spans the same number of columns, its *width*, in every
    relation; its values are those of the given frames. Each column of
    the entity has a dictionary, its distinct values sorted; an entity is
    the tuple of the positions of its values in them, a single integer
    key in mixed radix.

    Parameters
    ----------
    frames : sequence of pandas.DataFrame
        Tables of the entity's columns, in order, e.g. the voxel columns
        ``(i, j, k)`` of each relation holding voxels.
    """

    def __init__(self, frames: Sequence[pd.DataFrame]):
        self._fit(frames)

    @classmethod
    def numbered(cls, frames: Sequence[pd.DataFrame]):
        """
        Return the index of the entities of *frames* and the numbers of
        the entities of the rows of each, as :meth:`encode` gives them,
        the frames being read once.
        """
        index = cls.__new__(cls)
        return index, index._fit(frames)

    def _fit(self, frames):
        """Index the entities of *frames*; return the numbers of the
        entities of their rows."""
        frames = [pd.DataFrame(frame) for frame in frames]
        if not frames:
            raise ValueError("an entity index needs a table of its columns")
        self.width = frames[0].shape[1]
        if any(frame.shape[1] != self.width for frame in frames):
            raise ValueError(
                "the tables of an entity index have different widths"
            )
        self._dictionaries, positions = zip(*(
            _dictionary([frame.iloc[:, column] for frame in frames])
            for column in range(self.width)
        ))
        sizes = [len(dictionary) for dictionary in self._dictionaries]
        if np.prod(sizes, dtype=float) >= 2 ** 62:
            raise ValueError(
                "the entity has too many combinations of values to be keyed"
            )
        self._strides = np.cumprod([1] + sizes[:0:-1])[::-1]
        if self.width == 1:
            self._entities = pd.Index(np.arange(sizes[0]))
            return list(positions[0])
        keys = []
        for rows in zip(*positions):  # in mixed radix, in place
            key = rows[0].astype(np.int64)
            absent = rows[0] < 0
            for row, size in zip(rows[1:], sizes[1:]):
                key *= size
                key += row
                absent |= row < 0
            if absent.any():
                key[absent] = -1
            keys.append(key)
        del positions, rows
        present = [None if (key >= 0).all() else key >= 0 for key in keys]
        self._entities, found = _dictionary([
            pd.Series(key if mask is None else key[mask], copy=False)
            for key, mask in zip(keys, present)
        ])
        if len(self._entities) >= 2 ** 31:
            raise ValueError("an entity index holds fewer than 2**31 entities")
        numbers = []
        for mask, number in zip(present, found):
            if mask is not None:
                number, scattered = np.full(len(mask), -1, np.int32), number
                number[mask] = scattered
            numbers.append(number)
        return numbers

    def state(self) -> dict:
        """
        Return the arrays the index is made of, by name, for
        :meth:`from_state`: numbers, and texts as fixed-width unicode,
        arrays a file can hold and give back memory-mapped.

        Raises
        ------
        TypeError
            If a column holds values neither numbers nor texts.
        """
        arrays = {
            "width": np.asarray(self.width),
            "strides": np.asarray(self._strides, dtype=np.int64),
            "entities": self._entities.to_numpy(),
        }
        for column, dictionary in enumerate(self._dictionaries):
            arrays[f"dictionary{column}"] = _stored(dictionary)
        return arrays

    @classmethod
    def from_state(cls, arrays: Mapping[str, np.ndarray]) -> "EntityIndex":
        """Return the index whose :meth:`state` is *arrays*."""
        index = cls.__new__(cls)
        index.width = int(arrays["width"])
        index._strides = np.asarray(arrays["strides"])
        index._entities = pd.Index(arrays["entities"], copy=False)
        index._dictionaries = tuple(
            _loaded(arrays[f"dictionary{column}"])
            for column in range(index.width)
        )
        return index

    def __len__(self):
        return len(self._entities)

    def __repr__(self):
        return f"EntityIndex({len(self)} entities of width {self.width})"

    def _keys(self, frame):
        """Return the key of each row of *frame*, -1 if a value is absent."""
        keys = np.zeros(len(frame), dtype=np.int64)
        absent = np.zeros(len(frame), dtype=bool)
        for column, (dictionary, stride) in enumerate(
            zip(self._dictionaries, self._strides)
        ):
            positions = _positions(frame.iloc[:, column], dictionary)
            absent |= positions < 0
            keys += positions * stride
        keys[absent] = -1
        return keys

    def _numbers(self, keys):
        """Return the numbers of the entities of *keys*, -1 if absent."""
        if self.width > 1:
            keys = _positions(keys, self._entities)
        return keys.astype(np.int32)

    def encode(self, frame: pd.DataFrame) -> np.ndarray:
        """
        Return the number of the entity of each row of *frame*.

        Parameters
        ----------
        frame : pandas.DataFrame
            The entity's columns, in order.

        Returns
        -------
        numpy.ndarray of int32
            The entity numbers, -1 for the rows whose entity is not in the
            index.
        """
        frame = pd.DataFrame(frame)
        if frame.shape[1] != self.width:
            raise ValueError(
                f"an entity spans {self.width} columns, not {frame.shape[1]}"
            )
        return self._numbers(self._keys(frame))

    def decode(self, codes, columns: Optional[Sequence] = None):
        """
        Return the entities numbered *codes*, as a table of their columns.

        Parameters
        ----------
        codes : array of int
            Entity numbers, each in ``0 … len(self) - 1``.
        columns : sequence, optional
            Names of the columns, by default ``0 … width - 1``.

        Returns
        -------
        pandas.DataFrame
        """
        keys = self._entities.to_numpy()[np.asarray(codes, dtype=np.intp)]
        if columns is None:
            columns = range(self.width)
        return pd.DataFrame({
            name: dictionary.take(keys // stride % len(dictionary))
            for name, dictionary, stride in zip(
                columns, self._dictionaries, self._strides
            )
        })


class SparseView:
    """
    A relation as a sparse matrix whose rows are numbered by one entity
    and its columns by another: entry ``(r, c)`` is stored when the
    relation holds a tuple of the entities ``r`` and ``c``, and its value
    is the tuple's *value* column, or ``True`` when there is none.

    The view spans every column of the relation, which is taken as a set
    (its duplicate tuples are dropped first), so its stored entries are
    the relation's tuples; with a *value*, the value must be a function
    of the row and column entities. Explicit zeros are kept.

    The matrix is a ``scipy.sparse.csr_array`` with int32 indices; its
    transpose ``matrix.T`` is the CSC matrix of the transposed relation,
    without a copy. Its entries are booleans when there is no value: to
    multiply views (sums of products, as ``R @ R.T``), take
    :meth:`weighted`.

    Parameters
    ----------
    relation : RelationalAlgebraFrozenSet
        The relation.
    rows, columns : sequence
        Columns of the relation spanned by the row entity and by the
        column entity.
    row_index, column_index : EntityIndex
        The numbering of each entity; every entity of the relation must
        be in it.
    value : optional
        The column of the relation holding the values of the entries.
    codes : pair of arrays, optional
        The numbers of the row and of the column entities of the rows of
        the relation, as the indices :meth:`~EntityIndex.encode` them, if
        they are already known.
    """

    def __init__(
        self, relation, rows: Sequence, columns: Sequence,
        row_index: EntityIndex, column_index: EntityIndex, value=None,
        codes=None,
    ):
        self.rows, self.columns = tuple(rows), tuple(columns)
        self.value = value
        self.row_index, self.column_index = row_index, column_index
        spanned = self.rows + self.columns + (
            () if value is None else (value,)
        )
        frame = relation.as_pandas_dataframe()
        if frame is None:  # an empty relation, of no columns yet
            frame = pd.DataFrame(columns=list(spanned))
        if Counter(spanned) != Counter(frame.columns):
            raise ValueError(
                f"a view spans each column of its relation once: "
                f"{list(spanned)} for the columns {list(frame.columns)}"
            )
        if codes is None:
            codes = (
                row_index.encode(frame[list(self.rows)]),
                column_index.encode(frame[list(self.columns)]),
            )
        row_codes, column_codes = codes
        if (row_codes < 0).any() or (column_codes < 0).any():
            raise ValueError("an entity of the relation is not in its index")
        data = (
            np.ones(len(frame), dtype=bool) if value is None
            else frame[value].to_numpy()
        )
        self.matrix = sparse.coo_array(
            (data, (row_codes, column_codes)),
            shape=(len(row_index), len(column_index)),
        ).tocsr()
        if self.matrix.nnz != len(frame):
            raise ValueError(
                f"the column {value!r} is not a function of the row and "
                "column entities"
            )

    def state(self) -> dict:
        """
        Return the arrays of the view's matrix and of its float64
        :meth:`weighted` matrix, by name, for :meth:`from_state`.
        """
        return {
            "indptr": self.matrix.indptr, "indices": self.matrix.indices,
            "data": self.matrix.data, "weighted": self.weighted().data,
        }

    @classmethod
    def from_state(
        cls, arrays: Mapping[str, np.ndarray], rows: Sequence,
        columns: Sequence, row_index: EntityIndex,
        column_index: EntityIndex, value=None,
    ) -> "SparseView":
        """
        Return the view whose :meth:`state` is *arrays*, of the columns
        *rows* and *columns* (and *value*) of its relation, numbered by
        the indices: its matrices are those of the arrays, not copies
        (when the arrays are mapped from a file, their pages are the
        file's, shared by the processes reading it).
        """
        view = cls.__new__(cls)
        view.rows, view.columns = tuple(rows), tuple(columns)
        view.value = value
        view.row_index, view.column_index = row_index, column_index
        shape = (len(row_index), len(column_index))

        def matrix(data):
            return sparse.csr_array(
                (data, arrays["indices"], arrays["indptr"]), shape=shape,
                copy=False,
            )

        view.matrix = matrix(arrays["data"])
        view._weighted = {np.dtype(np.float64): matrix(arrays["weighted"])}
        return view

    def weighted(self, dtype=np.float64):
        """
        Return the matrix with its entries as numbers of *dtype*: 1 for
        each tuple of a view without a value, its value otherwise.

        The matrix shares the indices of the view; only its entries are
        new (8 bytes a tuple in float64). Products of views are to be
        taken over it: the boolean ``matrix`` multiplies as a logical
        or, and ``R @ R.T`` would not count the studies two regions
        share.

        The matrix of each *dtype* is made once and kept with the view:
        one made before the REPL server forks its sessions
        (:func:`build_sparse_views` makes that of float64) is shared by
        them, rather than made again by each query of each session.
        """
        dtype = np.dtype(dtype)
        kept = self.__dict__.setdefault("_weighted", {})
        if dtype not in kept:
            data = self.matrix.data
            data = (
                np.ones(len(data), dtype=dtype) if self.value is None
                else data.astype(dtype, copy=False)
            )
            kept[dtype] = sparse.csr_array(
                (data, self.matrix.indices, self.matrix.indptr),
                shape=self.matrix.shape, copy=False,
            )
        return kept[dtype]

    def __repr__(self):
        shape = "x".join(map(str, self.matrix.shape))
        return f"SparseView({shape}, {self.matrix.nnz} tuples)"

    def to_frame(self) -> pd.DataFrame:
        """Return the tuples of the view, as a table of the relation's
        columns (rows, then columns, then value)."""
        entries = self.matrix.tocoo()
        parts = [
            self.row_index.decode(entries.row, self.rows),
            self.column_index.decode(entries.col, self.columns),
        ]
        if self.value is not None:
            parts.append(pd.DataFrame({self.value: entries.data}))
        return pd.concat(parts, axis=1)


class ColumnIndex:
    """
    The entries of a :class:`SparseView` by its column entities: for each,
    the numbers of the row entities it is stored with, and their values
    for a view with a value (the CSR structure of the view's transpose).
    The entries of a few columns are read alone (a few studies of the
    voxel by study view of ``activated``), rather than the whole view.

    The entries of a view without a value are 1, as in its
    :meth:`~SparseView.weighted` matrix, and are not stored.

    Parameters
    ----------
    indptr, indices : numpy.ndarray
        The entries of column ``c`` are the row numbers
        ``indices[indptr[c]:indptr[c + 1]]``.
    rows : int
        The number of row entities.
    values : numpy.ndarray, optional
        The values of the entries, as ``indices``.
    """

    def __init__(self, indptr, indices, rows: int, values=None):
        self.indptr, self.indices = indptr, indices
        self.rows, self.values = rows, values

    @classmethod
    def of(cls, view: "SparseView") -> "ColumnIndex":
        """Return the index of *view* by its column entities."""
        matrix = (view.matrix if view.value is None else view.weighted())
        by_columns = matrix.T.tocsr()
        return cls(
            by_columns.indptr, by_columns.indices, matrix.shape[0],
            None if view.value is None else by_columns.data,
        )

    def state(self) -> dict:
        """Return the arrays of the index, by name, for :meth:`from_state`."""
        arrays = {
            "indptr": self.indptr, "indices": self.indices,
            "rows": np.array([self.rows]),
        }
        if self.values is not None:
            arrays["values"] = self.values
        return arrays

    @classmethod
    def from_state(cls, arrays: Mapping[str, np.ndarray]) -> "ColumnIndex":
        """Return the index whose :meth:`state` is *arrays* (not copies)."""
        return cls(
            arrays["indptr"], arrays["indices"], int(arrays["rows"][0]),
            arrays.get("values"),
        )

    @property
    def nnz(self) -> int:
        return int(self.indptr[-1])

    def entries(self, columns: np.ndarray) -> int:
        """Return the number of entries of the column entities *columns*."""
        return int((self.indptr[columns + 1] - self.indptr[columns]).sum())

    def matrix(self, columns: np.ndarray):
        """Return the CSR matrix of the column entities *columns* (sorted,
        distinct numbers) by the row entities, of float64 entries: their
        entries alone are read, a slice each."""
        starts, ends = self.indptr[columns], self.indptr[columns + 1]
        indptr = np.zeros(len(columns) + 1, dtype=np.int64)
        np.cumsum(ends - starts, out=indptr[1:])
        indices = np.concatenate(
            [self.indices[a:b] for a, b in zip(starts, ends)]
            or [np.zeros(0, dtype=self.indices.dtype)]
        )
        values = np.ones(len(indices)) if self.values is None else (
            np.concatenate([self.values[a:b] for a, b in zip(starts, ends)])
            .astype(np.float64, copy=False)
        )
        if indptr[-1] <= np.iinfo(np.int32).max:
            indptr = indptr.astype(np.int32)
        return sparse.csr_array(
            (values, indices, indptr), shape=(len(columns), self.rows)
        )


# The views built on the frame holding the rows of a relation, by the
# frame's id, and the frame, by the identities of its columns (see
# :func:`views_of_columns`), for as long as the frame lives.
_VIEWS = {}
_FRAMES = {}


def _column_identity(column: pd.Series):
    """
    Return what identifies the values of *column* for as long as the
    arrays holding them live: the addresses, sizes and types of its
    buffers; None if they are not known.
    """
    array = column.array
    arrow = getattr(array, "_pa_array", None)
    if arrow is not None:
        return (str(arrow.type), len(arrow)) + tuple(
            (buffer.address, buffer.size)
            for chunk in arrow.chunks for buffer in chunk.buffers()
            if buffer is not None
        )
    values = array.to_numpy() if isinstance(
        array, pd.arrays.NumpyExtensionArray
    ) else None
    if values is None:
        return None
    return (
        values.dtype.str, values.shape, values.strides,
        values.__array_interface__["data"][0],
    )


def _frame_identity(frame: pd.DataFrame):
    """Return the identities of the columns of *frame*, as a set, or None
    if one is not known."""
    identities = [_column_identity(frame[c]) for c in frame.columns]
    if not identities or None in identities:
        return None
    return frozenset(identities)


def _forget(frame_id, identity):
    _VIEWS.pop(frame_id, None)
    _FRAMES.pop(identity, None)


def views_of_columns(frame: pd.DataFrame):
    """
    Return the views built on a frame of the same columns as *frame*, the
    same arrays, as a frame computed by a rule renaming a stored relation
    has them, with the column of *frame* each of the view's is.

    Parameters
    ----------
    frame : pandas.DataFrame
        The rows of a relation.

    Returns
    -------
    list of (SparseView, dict)
        The views, each with the column of *frame* of each column of
        the frame it was built on.
    """
    identity = _frame_identity(frame)
    reference = None if identity is None else _FRAMES.get(identity)
    other = None if reference is None else reference()
    if other is None or len(other) != len(frame) or (
        len(set(identity)) != len(frame.columns)
    ):
        return []
    column_of = {_column_identity(frame[c]): c for c in frame.columns}
    columns = {c: column_of[_column_identity(other[c])] for c in other.columns}
    return [
        (view, columns) for view in _VIEWS.get(id(other), {}).values()
        if view.matrix.nnz == len(other)
    ]


def sparse_view(
    relation, rows: Sequence, columns: Sequence,
    row_index: EntityIndex, column_index: EntityIndex, value=None,
    codes=None,
) -> SparseView:
    """
    Return the :class:`SparseView` of *relation*, built once.

    The view is kept for as long as the frame holding the relation's rows
    lives (with the pandas backend, the relation and the sets made from it
    without changing its rows share it); it is built again if that frame
    was changed in place (``RelationalAlgebraSet.discard``), which is
    seen by its number of rows. The arguments are those of
    :class:`SparseView`.
    """
    return _registered(
        relation,
        (tuple(rows), tuple(columns), value, row_index, column_index),
        lambda: SparseView(
            relation, rows, columns, row_index, column_index, value, codes
        ),
    )


def _registered(relation, key, make) -> SparseView:
    """Return the view *key* of the frame of *relation*, or *make()*,
    kept with the frame (:func:`sparse_view`)."""
    frame = relation.as_pandas_dataframe()
    if frame is None:
        return make()
    views = _VIEWS.get(id(frame))
    if views is None:
        views = _VIEWS[id(frame)] = {}
        identity = _frame_identity(frame)
        if identity is not None:
            _FRAMES[identity] = weakref.ref(frame)
        weakref.finalize(frame, _forget, id(frame), identity)
    view = views.get(key)
    if view is None or view.matrix.nnz != len(frame):
        view = views[key] = make()
    return view


def stored_relation(symbol_table, symbol: Symbol):
    """
    Return the stored relation of the predicate *symbol*, and the column
    of it each argument of the predicate is.

    It is the predicate's own set, or that of the stored predicate the
    only rule of the predicate renames, as the rule
    ``activated(i, j, k, s) :- fresh(i, j, k, s)`` the ``EUCLIDEAN`` sugar
    leaves, whose variables are distinct and the same in its head and in
    its body, in any order.

    Raises
    ------
    ValueError
        If the predicate is neither.
    """
    value = symbol_table.get(symbol)
    if isinstance(value, Constant):
        return value.value, tuple(value.value.columns)
    rules = value.formulas if isinstance(value, Union) else ()
    if len(rules) == 1 and isinstance(rules[0], Implication):
        head, body = rules[0].consequent, rules[0].antecedent
        if isinstance(body, Conjunction) and len(body.formulas) == 1:
            body = body.formulas[0]
        stored = symbol_table.get(getattr(body, "functor", None))
        if (
            isinstance(stored, Constant)
            and len(set(head.args)) == len(head.args) == len(body.args)
            and set(head.args) == set(body.args)
            and all(isinstance(arg, Symbol) for arg in head.args)
        ):
            columns = tuple(stored.value.columns)
            return stored.value, tuple(
                columns[body.args.index(arg)] for arg in head.args
            )
    raise ValueError(
        f"{symbol.name} is not a stored relation, nor a renaming of one"
    )


class Axis(NamedTuple):
    """The entity of a sort spanning *columns* of a predicate, by
    position."""

    sort: str
    columns: Tuple[int, ...]


class ViewDeclaration(NamedTuple):
    """A view of a predicate over its *rows* and *columns* entities, with
    the value of its argument at position *value*, if any."""

    rows: Axis
    columns: Axis
    value: Optional[int] = None


def build_sparse_views(
    symbol_table, declarations: Mapping[str, ViewDeclaration], keep=None,
    indexed: Sequence[str] = (),
) -> dict:
    """
    Build the declared views of the predicates of *symbol_table*.

    One :class:`EntityIndex` per sort numbers the entities of the sort in
    every view of the declarations, so that views can be multiplied
    along a shared sort. The columns of each entity are read once, for
    its index and its view.

    Parameters
    ----------
    symbol_table : SymbolTable
        The symbols of a program, among which the predicates, each stored
        or a renaming of a stored one (see :func:`stored_relation`).
    declarations : mapping
        :class:`ViewDeclaration` by predicate name.
    keep : callable, optional
        ``keep(frames, parameters, compute)``, which returns the arrays,
        by name, *compute()* gives, found from the tables *frames* and
        the text *parameters* only: kept (memory-mapped from a file, as
        an engine's build keeps them) or computed. The views are then
        those of the arrays it returns.
    indexed : sequence of str
        The predicates whose views are also kept by their column entities
        (:class:`ColumnIndex`), as the view's ``by_columns``: kept, with
        *keep*, apart from the views, whose arrays they leave as they are.

    Returns
    -------
    dict
        The :class:`SparseView` of each predicate, cached as
        :func:`sparse_view` does.
    """
    stored = {
        name: stored_relation(symbol_table, Symbol(name))
        for name in declarations
    }
    sorts = {}
    for declaration in declarations.values():
        for axis in declaration[:2]:
            sorts.setdefault(axis.sort, len(sorts))
    frames = [
        relation.as_pandas_dataframe() for relation, _ in stored.values()
    ]
    if keep is None or any(frame is None for frame in frames):
        indices, views = _built_views(stored, declarations)
    else:
        def state():
            arrays = {}
            indices, views = _built_views(stored, declarations)
            for sort, index in indices.items():
                for name, array in index.state().items():
                    arrays[f"sort{sorts[sort]}.{name}"] = array
            for name, view in views.items():
                for part, array in view.state().items():
                    arrays[f"view.{name}.{part}"] = array
            return arrays

        arrays = keep(frames, repr(sorted(
            (name, tuple(declaration), tuple(stored[name][1]))
            for name, declaration in declarations.items()
        )), state)
        indices = {
            sort: EntityIndex.from_state({
                name.split(".", 1)[1]: array for name, array in arrays.items()
                if name.startswith(f"sort{number}.")
            })
            for sort, number in sorts.items()
        }
        views = {}
        for name, (rows, columns, value) in declarations.items():
            positions = stored[name][1]
            views[name] = SparseView.from_state(
                {
                    part: arrays[f"view.{name}.{part}"]
                    for part in ("indptr", "indices", "data", "weighted")
                },
                [positions[c] for c in rows.columns],
                [positions[c] for c in columns.columns],
                indices[rows.sort], indices[columns.sort],
                None if value is None else positions[value],
            )
    for name, view in views.items():
        views[name] = _registered(
            stored[name][0],
            (view.rows, view.columns, view.value, view.row_index,
             view.column_index),
            lambda view=view: view,
        )
        views[name].weighted()  # the matrix products take, made once
    for name in indexed:
        view = views[name]
        frame = stored[name][0].as_pandas_dataframe()
        if keep is None or frame is None:
            view.by_columns = ColumnIndex.of(view)
        else:
            view.by_columns = ColumnIndex.from_state(keep(
                [frame], repr(("by columns", name, tuple(declarations[name]),
                               tuple(stored[name][1]))),
                lambda view=view: ColumnIndex.of(view).state(),
            ))
    # Hashing Arrow strings leaves their buffers in Arrow's memory pool,
    # which keeps them from the system (0.8 GiB for activated of
    # neurostore_iit) unless told otherwise.
    if "pyarrow" in sys.modules:
        sys.modules["pyarrow"].default_memory_pool().release_unused()
    return views


def _built_views(stored, declarations):
    """Return the indices, by sort, and the views, by predicate, of the
    *declarations* of the *stored* relations, built from their rows."""
    columns_of_sort = {}
    for name, declaration in declarations.items():
        relation, positions = stored[name]
        frame = relation.as_pandas_dataframe()
        for side, axis in enumerate(declaration[:2]):
            columns = [positions[c] for c in axis.columns]
            columns_of_sort.setdefault(axis.sort, {})[name, side] = (
                pd.DataFrame(columns=range(len(columns))) if frame is None
                else frame[columns]
            )
    indices, codes = {}, {}
    for sort, frames in columns_of_sort.items():
        indices[sort], numbers = EntityIndex.numbered(list(frames.values()))
        codes.update(zip(frames, numbers))
    views = {}
    for name, (rows, columns, value) in declarations.items():
        relation, positions = stored[name]
        views[name] = SparseView(
            relation,
            [positions[c] for c in rows.columns],
            [positions[c] for c in columns.columns],
            indices[rows.sort], indices[columns.sort],
            None if value is None else positions[value],
            (codes[name, 0], codes[name, 1]),
        )
    return indices, views
