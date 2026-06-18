"""Polars backend for NeuroLang's Relational Algebra Set (RAS) system.

Uses pl.LazyFrame internally to defer materialization, allowing Polars'
query optimizer to handle chained operations efficiently.
"""

import builtins
from collections import OrderedDict, namedtuple
from typing import (
    Callable,
    Dict,
    Iterable,
    Optional,
    Set as SetType,
    Tuple,
    Union,
)

import polars as pl
import numpy as np
import pandas as pd

from . import abstract as abc

NA = None
_SENTINEL = "__ras_dee__"


class RelationalAlgebraStringExpression(str):
    def __repr__(self):
        return "{}{{ {} }}".format(self.__class__.__name__, super().__repr__())


def _make_unnamed(iterable):
    if isinstance(iterable, (pl.LazyFrame, RelationalAlgebraFrozenSet)):
        return None
    if isinstance(iterable, set):
        iterable = list(iterable)
    if isinstance(iterable, np.ndarray):
        if iterable.ndim == 1:
            iterable = list(iterable)
        elif iterable.ndim == 2:
            col_names = [str(i) for i in range(iterable.shape[1])]
            return pl.DataFrame(iterable, schema=col_names).lazy()
    if isinstance(iterable, pd.DataFrame):
        try:
            return pl.from_pandas(iterable).lazy()
        except ImportError:
            pass
        data = [tuple(row) for row in iterable.to_numpy()]
        col_names = [str(c) for c in range(len(iterable.columns))]
        return _make_dataframe_from_rows(data, col_names)
    data = list(iterable)
    if len(data) == 0:
        return pl.LazyFrame({})
    first = data[0]
    if isinstance(first, (tuple, list)):
        arity = len(first)
    else:
        arity = 1
        data = [(v,) for v in data]
    col_names = [str(i) for i in range(arity)]
    return _make_dataframe_from_rows(data, col_names)


def _make_dataframe_from_rows(data, col_names):
    """Build a Polars DataFrame from row-oriented data, using pl.Object dtype
    for columns containing non-primitive values (namedtuples) to prevent Polars
    from silently converting them to Struct type (which yields dicts on read-back).

    Returns a LazyFrame.
    """
    if len(data) == 0:
        return pl.DataFrame({col: pl.Series(col, [], dtype=pl.Int64) for col in col_names}).lazy()
    # Handle scalar values (e.g., single int passed directly as iterable)
    try:
        first = data[0]
    except (IndexError, TypeError):
        data = list(data)
        if len(data) == 0:
            return pl.DataFrame({col: pl.Series(col, [], dtype=pl.Int64) for col in col_names}).lazy()
        first = data[0]
    if not isinstance(first, (tuple, list)):
        # Scalars: wrap each element in a tuple for columnar construction
        data = [(v,) for v in data]
    if len(col_names) == 0:
        # DEE case: 0-column relation with N rows (e.g., data=[()])
        return pl.DataFrame([() for _ in data], schema=[], orient="row").lazy()
    series_list = []
    for col_idx, col_name in enumerate(col_names):
        values = [row[col_idx] for row in data]
        if values and _is_non_primitive(values[0]):
            series_list.append(pl.Series(col_name, values, dtype=pl.Object))
        else:
            series_list.append(pl.Series(col_name, values))
    return pl.DataFrame(series_list).lazy()


def _detect_schema_overrides(first_element, col_names):
    """Detect columns containing non-primitive values that need pl.Object dtype
    to prevent Polars from converting them to Struct type.
    """
    schema_overrides = {}
    if isinstance(first_element, (tuple, list)):
        for idx, val in enumerate(first_element):
            if idx < len(col_names) and _is_non_primitive(val):
                schema_overrides[col_names[idx]] = pl.Object
    return schema_overrides


def _is_non_primitive(val):
    """Check if a value should use pl.Object dtype to preserve its Python type."""
    return hasattr(val, '_fields') or type(val).__name__ == 'namedtuple'


# ---------------------------------------------------------------------------
# Helper utils that work on either LazyFrame or DataFrame
# ---------------------------------------------------------------------------

def _real_cols(df):
    if df is None:
        return []
    names = _lf_names(df) if isinstance(df, pl.LazyFrame) else df.columns
    return [c for c in names if c != _SENTINEL]


def _strip_sentinel(df):
    if df is None:
        return None
    if isinstance(df, pl.LazyFrame):
        names = _lf_names(df)
    else:
        names = df.columns
    if _SENTINEL in names:
        return df.select([c for c in names if c != _SENTINEL])
    return df


def _lf_names(lf: pl.LazyFrame) -> list:
    """Get column names from a LazyFrame without PerformanceWarning."""
    return lf.collect_schema().names()


def _concat_safe(dfs):
    """Concatenate LazyFrames, handling Null-typed columns."""
    non_null = [df for df in dfs if df is not None and len(_lf_names(df)) > 0]
    if len(non_null) == 0:
        return None
    if len(non_null) == 1:
        return non_null[0]
    # Unify column order across all LazyFrames
    all_col_sets = [set(_lf_names(df)) for df in non_null]
    if len(all_col_sets) > 1:
        common_cols = all_col_sets[0]
        for cs in all_col_sets[1:]:
            common_cols &= cs
        if common_cols:
            col_order = [c for c in _lf_names(non_null[0]) if c in common_cols]
            for i, df in enumerate(non_null):
                names = _lf_names(df)
                existing = [c for c in col_order if c in names]
                non_null[i] = df.select(*[pl.col(c) for c in existing])
    # Check for Null-typed columns that need casting
    common_cols = set(_lf_names(non_null[0]))
    for df in non_null[1:]:
        common_cols &= set(_lf_names(df))
    col_schemas = [df.collect_schema() for df in non_null]
    for col in common_cols:
        target_type = None
        for cs in col_schemas:
            dt = cs.get(col)
            if dt is not None and dt != pl.Null:
                target_type = dt
                break
        if target_type is not None:
            for i, df in enumerate(non_null):
                if col_schemas[i].get(col) == pl.Null:
                    non_null[i] = df.with_columns(pl.lit(None, dtype=target_type).alias(col))
    return pl.concat(non_null, how="vertical")


def _has_sentinel(df):
    if df is None:
        return False
    try:
        names = _lf_names(df) if isinstance(df, pl.LazyFrame) else df.columns
    except Exception:
        names = df.columns
    return len(names) == 1 and names[0] == _SENTINEL


def _dee_df():
    return pl.LazyFrame({_SENTINEL: [0]})


def _dum_df():
    return pl.DataFrame({}).select([]).lazy()


class RelationalAlgebraFrozenSet(abc.RelationalAlgebraFrozenSet):

    def __init__(self, iterable=None):
        self._container: Optional[pl.LazyFrame] = None
        self._might_have_duplicates = True
        if iterable is not None:
            if isinstance(iterable, RelationalAlgebraFrozenSet):
                if iterable._container is not None:
                    self._container = iterable._container
                else:
                    self._container = None
            elif isinstance(iterable, pl.LazyFrame):
                self._container = iterable
            elif isinstance(iterable, pl.DataFrame):
                self._container = iterable.lazy()
            else:
                result = _make_unnamed(iterable)
                if isinstance(result, pl.DataFrame):
                    result = result.lazy()
                self._container = result

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _eager(self) -> Optional[pl.DataFrame]:
        """Materialize the lazy container, returning None if not set."""
        if self._container is None:
            return None
        return self._container.collect()

    @staticmethod
    def _to_lazy(df):
        if df is None:
            return None
        return df.lazy()

    @property
    def might_have_duplicates(self):
        return self._might_have_duplicates

    @might_have_duplicates.setter
    def might_have_duplicates(self, value):
        self._might_have_duplicates = bool(value)

    def _drop_duplicates_if_needed(self):
        if self._might_have_duplicates:
            if self._container is not None:
                eager = self._eager()
                if eager is not None and len(eager) > 0:
                    try:
                        deduped = eager.unique(maintain_order=True)
                        self._container = deduped.lazy()
                    except Exception:
                        # Fallback for Object types that can't be row-encoded
                        rows = list(dict.fromkeys(tuple(r) for r in eager.iter_rows()))
                        if rows:
                            self._container = pl.DataFrame(
                                rows, schema=eager.columns, orient="row"
                            ).lazy()
                        else:
                            self._container = pl.LazyFrame({})
            self._might_have_duplicates = False

    @classmethod
    def create_view_from(cls, other):
        if not isinstance(other, cls):
            raise ValueError(
                "View can only be created from an object of the same class"
            )
        output = cls()
        output._container = other._container
        output._might_have_duplicates = other._might_have_duplicates
        return output

    @classmethod
    def dee(cls):
        output = cls()
        output._container = _dee_df()
        output._might_have_duplicates = False
        return output

    @classmethod
    def dum(cls):
        output = cls()
        output._container = _dum_df()
        output._might_have_duplicates = False
        return output

    def is_empty(self):
        if self._container is None:
            return True
        if _has_sentinel(self._container):
            return False
        eager = self._eager()
        return eager is None or len(eager) == 0

    def is_dee(self):
        if _has_sentinel(self._container):
            return True
        return self.arity == 0 and not self.is_empty()

    def is_dum(self):
        if self._container is None:
            return True
        if not _has_sentinel(self._container) and self.arity == 0:
            eager = self._eager()
            if eager is not None and len(eager) == 0:
                return True
        return False

    def __contains__(self, element):
        element = self._normalise_element(element)
        if self.is_empty() or self.is_dee() or len(element) != self.arity:
            return False
        if self._container is None:
            return False
        try:
            eager = self._eager()
            if eager is None:
                return False
            mask = None
            for i, val in enumerate(element):
                col_name = str(i)
                col = eager[:, col_name]
                if mask is None:
                    mask = (col == val)
                else:
                    mask = mask & (col == val)
            return mask.any()
        except NotImplementedError:
            # Object type comparision not supported by Polars, fall back
            for row in eager.iter_rows():
                if tuple(row) == tuple(element):
                    return True
            return False

    @staticmethod
    def _normalise_element(element):
        if isinstance(element, tuple):
            pass
        elif hasattr(element, "__iter__"):
            element = tuple(element)
        else:
            element = (element,)
        return element

    def __iter__(self):
        if self.is_empty():
            return iter([])
        elif self.is_dee():
            return iter([tuple()])
        else:
            self._drop_duplicates_if_needed()
            if self._container is None:
                return iter([])
            return self.itervalues()

    def fetch_one(self):
        if self.is_dee():
            return tuple()
        eager = self._eager()
        if eager is None or len(eager) == 0:
            raise StopIteration("Cannot fetch_one from empty set")
        row = eager.row(0)
        return tuple(row)

    def __len__(self):
        if self._container is None:
            return 0
        self._drop_duplicates_if_needed()
        eager = self._eager()
        if eager is None:
            return 0
        return len(eager)

    @property
    def arity(self):
        if self._container is None:
            return 0
        return len(_real_cols(self._container))

    @property
    def columns(self):
        if self._container is not None:
            return tuple(_real_cols(self._container))
        return tuple()

    def as_numpy_array(self):
        self._drop_duplicates_if_needed()
        eager = self._eager()
        c = _strip_sentinel(eager) if eager is not None else None
        if c is None:
            return np.array([])
        return c.to_numpy()

    def as_pandas_dataframe(self):
        self._drop_duplicates_if_needed()
        eager = self._eager()
        c = _strip_sentinel(eager) if eager is not None else None
        if c is None:
            import pandas as pd
            return pd.DataFrame()
        try:
            return c.to_pandas()
        except ImportError:
            import pandas as pd
            return pd.DataFrame(c.to_numpy(), columns=c.columns)

    def _empty_set_same_structure(self):
        return type(self)()

    def projection(self, *columns):
        if self.is_empty():
            return self._empty_set_same_structure()
        if len(columns) == 0:
            return self.dee()
        str_cols = list(dict.fromkeys(str(c) for c in columns))
        c = _strip_sentinel(self._container)
        if c is None:
            return self._empty_set_same_structure()
        for col in str_cols:
            c_names = _lf_names(c)
            if col not in c_names:
                raise KeyError(
                    f"Column '{col}' not found in {c_names}"
                )
        new_c = c.select(*[pl.col(col) for col in str_cols])
        new_c_names = _lf_names(new_c)
        new_c = new_c.rename(
            dict(zip(new_c_names, [str(i) for i in range(len(columns))]))
        )
        output = self._empty_set_same_structure()
        output._container = new_c
        if len(columns) == self.arity:
            output._might_have_duplicates = self._might_have_duplicates
        return output

    def _selection_dict(self, select_criteria):
        """Build a boolean series for dict-based selection. Works on eager DataFrame."""
        it = iter(select_criteria.items())
        col, value = next(it)
        col_name = str(col)
        eager = self._eager()
        if eager is None:
            # return something that filters to empty
            return pl.Series([], dtype=pl.Boolean)
        if callable(value):
            col_series = eager[:, col_name]
            mask = col_series.map_elements(value, return_dtype=pl.Boolean)
        else:
            mask = eager[:, col_name] == value
        for col, value in it:
            col_name = str(col)
            if callable(value):
                col_series = eager[:, col_name]
                sub_mask = col_series.map_elements(value, return_dtype=pl.Boolean)
            else:
                sub_mask = eager[:, col_name] == value
            mask = mask & sub_mask
        return mask

    def selection(
        self,
        select_criteria: Union[
            Callable,
            RelationalAlgebraStringExpression,
            Dict[int, Union[int, Callable]],
        ],
    ):
        if self.is_empty() or self._container is None:
            return self._empty_set_same_structure()
        c = self._container
        if _has_sentinel(c):
            c = _strip_sentinel(c)

        if callable(select_criteria):
            # Need eager for callable map_rows
            eager = self._eager()
            if eager is None:
                return self._empty_set_same_structure()
            if _has_sentinel(eager):
                eager = _strip_sentinel(eager)
            mask = eager.map_rows(
                lambda row: select_criteria(tuple(row))
            ).to_series()
            mask = mask.cast(pl.Boolean)
            new_container = eager.filter(mask)
        elif isinstance(select_criteria, RelationalAlgebraStringExpression):
            expr_str = str(select_criteria)
            try:
                pl_expr = _string_expr_to_polars(expr_str)
                new_container = (
                    c.with_columns(pl_expr.alias("__mask__"))
                    .filter(pl.col("__mask__").cast(pl.Boolean))
                    .select(*[pl.col(col) for col in _lf_names(c)])
                )
            except Exception:
                # Fall back to eager row-by-row
                eager = self._eager()
                if _has_sentinel(eager):
                    eager = _strip_sentinel(eager)
                col_names = list(c.columns)
                mask = eager.map_rows(
                    lambda row: _eval_row_with_builtins(
                        expr_str, col_names, tuple(row)
                    )
                ).to_series()
                mask = mask.cast(pl.Boolean)
                new_container = eager.filter(mask)
        else:
            # Dict-based selection
            eager = self._eager()
            if eager is None:
                return self._empty_set_same_structure()
            if _has_sentinel(eager):
                eager = _strip_sentinel(eager)
            self._container = eager.lazy()
            mask = self._selection_dict(select_criteria)
            new_container = eager.filter(mask)

        output = self._empty_set_same_structure()
        output._container = self._to_lazy(new_container) if isinstance(new_container, pl.DataFrame) else new_container
        output._might_have_duplicates = self._might_have_duplicates
        return output

    def selection_columns(self, select_criteria: Dict[int, int]):
        if self.is_empty() or self._container is None:
            return self._empty_set_same_structure()
        c = self._container
        if _has_sentinel(c):
            c = _strip_sentinel(c)
        it = iter(select_criteria.items())
        col1, col2 = next(it)
        condition = pl.col(str(col1)) == pl.col(str(col2))
        for col1, col2 in it:
            condition = condition & (pl.col(str(col1)) == pl.col(str(col2)))

        new_container = c.filter(condition)
        output = self._empty_set_same_structure()
        output._container = new_container
        output._might_have_duplicates = self._might_have_duplicates
        return output

    def equijoin(self, other, join_indices):
        res = self._dee_dum_product(other)
        if res is not None:
            return res
        if self.is_empty() or other.is_empty():
            return self._empty_set_same_structure()
        self._drop_duplicates_if_needed()
        other._drop_duplicates_if_needed()
        sc = self._container
        oc = other._container
        if sc is None or oc is None:
            return self._empty_set_same_structure()

        # Rename right columns to shifted indices (self.arity, ...)
        right_cols = [str(i) for i in range(other.arity)]
        shifted_cols = [str(self.arity + i) for i in range(other.arity)]
        right_rename = dict(zip(right_cols, shifted_cols))
        right_lf = oc.rename(right_rename)

        left_on = [str(l_idx) for l_idx, r_idx in join_indices]
        right_on = [str(self.arity + r_idx) for l_idx, r_idx in join_indices]

        left_lf = sc

        tmp_col = "__tmp_join__"
        left_lf = left_lf.with_columns(pl.lit(1).alias(tmp_col))
        right_lf = right_lf.with_columns(pl.lit(1).alias(tmp_col))

        merged = left_lf.join(right_lf, on=tmp_col, how="inner")

        for lc, rc in zip(left_on, right_on):
            merged = merged.filter(pl.col(lc) == pl.col(rc))

        merged = merged.drop(tmp_col)

        num_cols = len(_lf_names(merged))
        merged = merged.rename(
            dict(zip(_lf_names(merged), [str(i) for i in range(num_cols)]))
        )

        output = self._empty_set_same_structure()
        output._container = merged
        output._might_have_duplicates = (
            self._might_have_duplicates | other._might_have_duplicates
        )
        return output

    def cross_product(self, other):
        res = self._dee_dum_product(other)
        if res is not None:
            return res
        if self.is_empty() or other.is_empty():
            return self._empty_set_same_structure()
        self._drop_duplicates_if_needed()
        other._drop_duplicates_if_needed()
        sc = self._container
        oc = other._container
        if sc is None or oc is None:
            return self._empty_set_same_structure()

        right_cols = [str(i) for i in range(other.arity)]
        shifted_cols = [str(self.arity + i) for i in range(other.arity)]
        right_rename = dict(zip(right_cols, shifted_cols))
        right_lf = oc.rename(right_rename)

        merged = sc.join(right_lf, how="cross")

        total_cols = self.arity + other.arity
        merged = merged.rename(
            dict(zip(_lf_names(merged), [str(i) for i in range(total_cols)]))
        )

        output = self._empty_set_same_structure()
        output._container = merged
        output._might_have_duplicates = (
            self._might_have_duplicates | other._might_have_duplicates
        )
        return output

    def copy(self):
        output = self._empty_set_same_structure()
        output._might_have_duplicates = self._might_have_duplicates
        if self._container is not None:
            output._container = self._container
        return output

    def __repr__(self):
        if self.is_empty():
            return "{}"
        eager = self._eager()
        return repr(eager)

    def __sub__(self, other):
        if isinstance(other, RelationalAlgebraFrozenSet):
            if (
                self.arity > 0 and other.arity > 0
            ) and self.arity != other.arity:
                raise ValueError(
                    "Relational algebra set operators can only"
                    " be used on sets with same columns."
                )
            if self.is_empty() or other.is_empty():
                return self.copy()
            if self.is_dee():
                if other.is_dee():
                    return self.dum()
                return self.dee()
        return super().__sub__(other)

    def __or__(self, other):
        if self is other:
            return self.copy()
        elif isinstance(other, RelationalAlgebraFrozenSet):
            res = self._dee_dum_sum(other)
            if res is not None:
                return res
            if self._container is None and other._container is None:
                return self._empty_set_same_structure()
            if self._container is None:
                return other.copy()
            if other._container is None:
                return self.copy()
            sc = _strip_sentinel(self._container)
            oc = _strip_sentinel(other._container)
            merged = _concat_safe([sc, oc])
            if merged is not None:
                merged = merged.unique(maintain_order=True)
            output = self._empty_set_same_structure()
            output._container = merged
            return output
        else:
            return super().__or__(other)

    def __and__(self, other):
        if self is other:
            return self.copy()
        if isinstance(other, RelationalAlgebraFrozenSet):
            res = self._dee_dum_product(other)
            if res is not None:
                return res
            sc = _strip_sentinel(self._container)
            oc = _strip_sentinel(other._container)
            if sc is None or oc is None:
                return self._empty_set_same_structure()
            merged = sc.join(oc, how="inner", on=_lf_names(sc))
            output = self._empty_set_same_structure()
            output._container = merged
            output._might_have_duplicates = (
                self._might_have_duplicates | other._might_have_duplicates
            )
            return output
        else:
            return super().__and__(other)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            if self.is_empty() and other.is_empty():
                return True
            if self.arity != other.arity:
                return False
            if self.arity == 0 and other.arity == 0:
                return self.is_dee() and other.is_dee()
            s_eager = self._eager()
            o_eager = other._eager()
            s = _strip_sentinel(s_eager)
            o = _strip_sentinel(o_eager)
            if s is None and o is None:
                return True
            if s is None or o is None:
                return False
            if set(s.columns) != set(o.columns):
                return False
            try:
                s_dedup = s.unique(maintain_order=True) if len(s) > 0 else s
                o_dedup = o.unique(maintain_order=True) if len(o) > 0 else o
                s_sorted = s_dedup.sort(by=list(s_dedup.columns)).select(sorted(s_dedup.columns))
                o_sorted = o_dedup.sort(by=list(o_dedup.columns)).select(sorted(o_dedup.columns))
                return s_sorted.frame_equal(o_sorted)
            except Exception:
                s_rows = sorted(set(tuple(r) for r in s.iter_rows()))
                o_rows = sorted(set(tuple(r) for r in o.iter_rows()))
                return s_rows == o_rows
        else:
            return super().__eq__(other)

    def groupby(self, columns):
        if self.is_empty() or self._container is None:
            return
        if not isinstance(columns, Iterable):
            columns = [columns]
        col_names = [str(c) for c in columns]
        c = _strip_sentinel(self._container)
        if c is None:
            return
        # LazyGroupBy is not iterable, so collect first
        eager = c.collect()
        for g_id, group_df in eager.group_by(col_names, maintain_order=True):
            group_set = self._empty_set_same_structure()
            group_set._container = group_df.lazy()
            if isinstance(g_id, tuple) and len(g_id) == 1:
                g_id = g_id[0]
            yield g_id, group_set

    def itervalues(self):
        if self.is_empty() or self._container is None:
            return iter([])
        c = _strip_sentinel(self._container)
        if c is None:
            return iter([])
        eager = c.collect()
        return iter(eager.iter_rows())

    def __hash__(self):
        if self._container is None:
            return hash((tuple(), None))
        self._drop_duplicates_if_needed()
        eager = self._eager()
        if eager is None:
            return hash((tuple(), None))
        c = _strip_sentinel(eager)
        if c is None:
            return hash((tuple(), None))
        col_key = tuple(c.columns)
        data_bytes = c.to_numpy().tobytes()
        return hash((col_key, data_bytes))


class NamedRelationalAlgebraFrozenSet(
    RelationalAlgebraFrozenSet, abc.NamedRelationalAlgebraFrozenSet
):

    _columns: Tuple[str, ...]

    def __init__(self, columns, iterable=None):
        if isinstance(columns, NamedRelationalAlgebraFrozenSet):
            iterable = columns
            columns = columns.columns
        # Convert column names to strings for Polars compatibility
        columns = tuple(str(c) for c in columns)
        self._check_for_duplicated_columns(columns)
        self._columns = tuple(columns)
        self._might_have_duplicates = True

        if iterable is None:
            iterable = []

        if isinstance(iterable, NamedRelationalAlgebraFrozenSet):
            self._initialize_from_named_ra_set(iterable)
            self._might_have_duplicates = iterable._might_have_duplicates
        elif isinstance(iterable, RelationalAlgebraFrozenSet):
            self._initialize_from_unnamed_ra_set(iterable)
            self._might_have_duplicates = iterable._might_have_duplicates
        elif isinstance(iterable, pl.LazyFrame):
            self._container = iterable
            self._container = self._container.rename(
                dict(zip(_lf_names(self._container), self._columns))
            )
        elif isinstance(iterable, pl.DataFrame):
            self._container = iterable.lazy()
            self._container = self._container.rename(
                dict(zip([str(i) for i in range(len(self._columns))], self._columns))
            )
        else:
            self._initialize_from_iterable(list(iterable))

    def _initialize_from_iterable(self, iterable):
        columns = [str(c) for c in self._columns]
        if len(iterable) == 0 and len(columns) > 0:
            self._container = pl.DataFrame(
                {col: pl.Series(col, [], dtype=pl.Int64) for col in columns}
            ).lazy()
        elif len(iterable) > 0:
            if len(columns) == 0:
                # DEE: 0 columns, non-empty (tuple() entries)
                self._container = _dee_df()
            else:
                self._container = _make_dataframe_from_rows(iterable, columns)
        else:
            self._container = pl.DataFrame(
                {col: pl.Series(col, [], dtype=pl.Int64) for col in columns}
            ).lazy()

    def _initialize_from_named_ra_set(self, other):
        if not other.is_dum() and self.arity != other.arity:
            raise ValueError("Relations must have the same arity")

        if not other.is_empty() and other._container is not None:
            oc = _strip_sentinel(other._container)
            if oc is not None:
                self._container = oc.select(*[pl.col(c) for c in other.columns])
            else:
                self._container = None
        else:
            col_list = list(self._columns)
            if col_list:
                self._container = pl.DataFrame(
                    {col: pl.Series(col, [], dtype=pl.Int64) for col in col_list}
                ).lazy()
            else:
                self._container = pl.LazyFrame()

    def _initialize_from_unnamed_ra_set(self, other):
        if other.is_empty() and not other.is_dee():
            col_list = list(self._columns)
            if col_list:
                self._container = pl.DataFrame(
                    {col: pl.Series(col, [], dtype=pl.Int64) for col in col_list}
                ).lazy()
            else:
                self._container = pl.LazyFrame()
        elif other.is_dee():
            self._container = _dee_df()
        elif other._container is None:
            data = list(other)
            if len(data) > 0:
                self._container = pl.DataFrame(
                    data, schema=self._columns, orient="row"
                ).lazy()
            else:
                self._container = pl.LazyFrame()
        else:
            if self.arity != other.arity:
                raise ValueError("Relations must have the same arity")
            oc = _strip_sentinel(other._container)
            if oc is not None:
                self._container = oc
                self._container = self._container.rename(
                    dict(zip(_lf_names(self._container), self._columns))
                )
            else:
                self._container = None

    @staticmethod
    def _check_for_duplicated_columns(columns):
        if len(set(columns)) != len(columns):
            columns_list = list(columns)
            dup_cols = set(c for c in columns_list if columns_list.count(c) > 1)
            raise ValueError(
                "Duplicated column names are not allowed. "
                f"Found the following duplicated columns: {dup_cols}"
            )

    @classmethod
    def create_view_from(cls, other):
        if not isinstance(other, cls):
            raise ValueError(
                "View can only be created from an object of the same class"
            )
        output = cls(columns=tuple())
        output._container = other._container
        output._columns = other._columns
        output._might_have_duplicates = other._might_have_duplicates
        return output

    def _empty_set_same_structure(self):
        return type(self)(self.columns)

    @classmethod
    def dee(cls):
        output = cls(())
        output._container = _dee_df()
        output._might_have_duplicates = False
        return output

    @classmethod
    def dum(cls):
        output = cls(())
        output._container = _dum_df()
        output._might_have_duplicates = False
        return output

    def _light_init_same_structure(
        self, container, might_have_duplicates=True, columns=None,
    ):
        if columns is None:
            columns = self.columns
        output = type(self)(columns)
        output._container = container
        output._might_have_duplicates = might_have_duplicates
        return output

    @property
    def columns(self):
        return self._columns

    @property
    def arity(self):
        return len(self._columns)

    def __contains__(self, element):
        if isinstance(element, dict) and len(element) == self.arity:
            element = tuple(element[c] for c in self._columns)
        return super().__contains__(element)

    def projection(self, *columns):
        if self.is_empty():
            return type(self)(columns)
        if len(columns) == 0:
            return self.dee()
        if self.arity == 0:
            return self
        if self._container is None:
            return type(self)(columns)
        col_list = list(columns)
        c = _strip_sentinel(self._container)
        if c is None:
            return type(self)(columns)
        new_container = c.select(*[pl.col(col) for col in col_list])
        return self._light_init_same_structure(
            new_container, might_have_duplicates=True, columns=columns
        )

    def projection_to_unnamed(self, *columns):
        unnamed_self = self.to_unnamed()
        named_columns = list(self.columns)
        column_indices = tuple(named_columns.index(c) for c in columns)
        return unnamed_self.projection(*column_indices)

    def selection(
        self,
        select_criteria: Union[
            Callable,
            RelationalAlgebraStringExpression,
            Dict[int, Union[int, Callable]],
        ],
    ):
        if callable(select_criteria):
            col_list = list(self.columns)
            Row = self._make_named_row_class(col_list)
            if self.is_empty() or self._container is None:
                return self._empty_set_same_structure()
            # Need eager for callable map_rows
            eager = self._eager()
            if eager is None:
                return self._empty_set_same_structure()
            c = _strip_sentinel(eager)
            if c is None:
                return self._empty_set_same_structure()
            mask = c.map_rows(
                lambda row: select_criteria(Row(*row))
            ).to_series()
            mask = mask.cast(pl.Boolean)
            new_container = c.filter(mask)
            return self._light_init_same_structure(
                self._to_lazy(new_container),
                might_have_duplicates=self._might_have_duplicates,
            )
        return super().selection(select_criteria)

    def equijoin(self, other, join_indices):
        raise NotImplementedError()

    def naturaljoin(self, other):
        res = self._dee_dum_product(other)
        if res is not None:
            return res
        on = [c for c in self.columns if c in other.columns]

        if len(on) == 0:
            return self.cross_product(other)

        new_columns = self.columns + tuple(
            c for c in other.columns if c not in self.columns
        )

        self._drop_duplicates_if_needed()
        other._drop_duplicates_if_needed()

        sc = _strip_sentinel(self._container)
        oc = _strip_sentinel(other._container)
        if sc is None or oc is None:
            return type(self)(new_columns)

        sc, oc = self._unify_join_types(sc, oc, on)
        new_container = sc.join(oc, on=on, how="inner")
        new_container = new_container.select(*[pl.col(c) for c in new_columns])

        return self._light_init_same_structure(
            new_container,
            might_have_duplicates=(
                self._might_have_duplicates | other._might_have_duplicates
            ),
            columns=new_columns,
        )

    @staticmethod
    def _unify_join_types(sc, oc, on):
        if sc is None or oc is None:
            return sc, oc
        sc_schema = sc.collect_schema()
        oc_schema = oc.collect_schema()
        for col in on:
            if col not in sc_schema or col not in oc_schema:
                continue
            s_dtype = sc_schema[col]
            o_dtype = oc_schema[col]
            if s_dtype != o_dtype:
                if s_dtype == pl.Null and o_dtype != pl.Null:
                    sc = sc.with_columns(pl.col(col).cast(o_dtype))
                elif o_dtype == pl.Null and s_dtype != pl.Null:
                    oc = oc.with_columns(pl.col(col).cast(s_dtype))
                elif s_dtype.is_numeric() and o_dtype.is_numeric():
                    target = s_dtype if s_dtype != pl.Null else o_dtype
                    if s_dtype != target:
                        sc = sc.with_columns(pl.col(col).cast(target))
                    if o_dtype != target:
                        oc = oc.with_columns(pl.col(col).cast(target))
        return sc, oc

    def left_naturaljoin(self, other):
        if self.is_dee():
            return self

        on = [c for c in self.columns if c in other.columns]

        self._drop_duplicates_if_needed()
        other._drop_duplicates_if_needed()

        sc = _strip_sentinel(self._container)
        oc = _strip_sentinel(other._container)

        if sc is None:
            return self.copy()

        if len(on) == 0:
            if other.is_empty() or oc is None:
                new_container = sc
                for c in other.columns:
                    new_container = new_container.with_columns(pl.lit(None).alias(c))
            else:
                return self.cross_product(other)
        else:
            if oc is None:
                new_container = sc
                for c in other.columns:
                    if c not in self.columns:
                        new_container = new_container.with_columns(pl.lit(None).alias(c))
            else:
                sc, oc = self._unify_join_types(sc, oc, on)
                new_container = sc.join(oc, on=on, how="left")

        new_columns = self.columns + tuple(
            c for c in other.columns if c not in self.columns
        )
        if new_container is not None:
            new_container = new_container.select(*[pl.col(c) for c in new_columns])

        return self._light_init_same_structure(
            new_container,
            might_have_duplicates=(
                self._might_have_duplicates | other._might_have_duplicates
            ),
            columns=new_columns,
        )

    def replace_null(self, column, value):
        if self._container is None:
            return self.copy()
        c = _strip_sentinel(self._container)
        if c is None:
            return self.copy()
        new_container = c.with_columns(pl.col(column).fill_null(value).alias(column))
        return self._light_init_same_structure(
            new_container,
            might_have_duplicates=self._might_have_duplicates,
            columns=self.columns,
        )

    def explode(self, src_column, dst_columns):
        if self._container is None:
            return self.copy()
        c = _strip_sentinel(self._container)
        if c is None:
            return self.copy()

        # Check if column is Object type (e.g., frozenset)
        c_schema = c.collect_schema()
        dt = c_schema.get(src_column)
        if dt == pl.Object or dt is None:
            # Fall back to pure Python approach since Polars can't explode Object dtype
            eager = c.collect()
            rows = list(eager.iter_rows())
            col_names = list(eager.columns)
            src_idx = col_names.index(src_column)

            is_multi = isinstance(dst_columns, tuple)

            # Determine final column names
            if is_multi:
                final_col_names = col_names + list(dst_columns)
            elif dst_columns == src_column:
                final_col_names = col_names
            else:
                final_col_names = col_names + [dst_columns]

            exploded_rows = []
            for row in rows:
                val = row[src_idx]
                items = sorted(val) if hasattr(val, '__iter__') and not isinstance(val, (str, bytes)) else [val]
                if not items:
                    items = [None]
                for item in items:
                    new_row = list(row) + [None] * (len(final_col_names) - len(row))
                    if is_multi:
                        item_tuple = tuple(item) if hasattr(item, '__iter__') and not isinstance(item, (str, bytes)) else (item,)
                        for i, dst_col in enumerate(dst_columns):
                            dst_idx = final_col_names.index(dst_col)
                            new_row[dst_idx] = item_tuple[i] if i < len(item_tuple) else None
                    else:
                        dst_idx = final_col_names.index(dst_columns)
                        new_row[dst_idx] = item
                    exploded_rows.append(tuple(new_row))

            new_df = pl.DataFrame(exploded_rows, schema=final_col_names, orient="row")
            return self._light_init_same_structure(
                new_df.lazy(), might_have_duplicates=True, columns=tuple(final_col_names),
            )

        # Native Polars explode for non-Object types
        if dst_columns == src_column:
            new_container = c.explode(src_column)
            new_columns = self.columns
        elif not isinstance(dst_columns, tuple):
            c = c.with_columns(pl.col(src_column).alias(dst_columns))
            new_container = c.explode(dst_columns)
            new_columns = self.columns + (dst_columns,)
        else:
            c = c.with_columns(pl.col(src_column).alias(dst_columns[-1]))
            new_container = c.explode(dst_columns[-1])
            n_new = len(dst_columns)
            for i in range(n_new):
                new_container = new_container.with_columns(
                    pl.col(dst_columns[-1]).list.get(i).alias(dst_columns[i])
                )
            new_columns = self.columns + dst_columns

        col_list = list(new_columns)
        existing_cols = [c for c in col_list if c in _lf_names(new_container)]
        new_container = new_container.select(*[pl.col(c) for c in existing_cols])

        return self._light_init_same_structure(
            new_container, might_have_duplicates=True, columns=new_columns,
        )

    def cross_product(self, other):
        res = self._dee_dum_product(other)
        if res is not None:
            return res.copy()
        common_cols = set(self.columns) & set(other.columns)
        if len(common_cols) > 0:
            raise ValueError("Cross product with common columns is not valid")

        new_columns = self.columns + other.columns
        if self.is_empty() or other.is_empty():
            return type(self)(new_columns)
        sc = _strip_sentinel(self._container)
        oc = _strip_sentinel(other._container)
        if sc is None or oc is None:
            return type(self)(new_columns)

        new_container = sc.join(oc, how="cross")
        new_container = new_container.select(*[pl.col(c) for c in new_columns])

        return self._light_init_same_structure(
            new_container,
            might_have_duplicates=(
                self._might_have_duplicates | other._might_have_duplicates
            ),
            columns=new_columns,
        )

    def rename_column(self, src, dst):
        if self.is_dum():
            return self
        if src not in self._columns:
            raise ValueError(f"{src} not in columns")
        if src == dst:
            return self
        if dst in self._columns:
            raise ValueError(f"{dst} cannot be in the columns")
        src_idx = self._columns.index(src)
        new_columns = (
            self._columns[:src_idx] + (dst,) + self._columns[src_idx + 1:]
        )

        if self._container is not None:
            c = _strip_sentinel(self._container)
            if c is not None:
                new_container = c.rename({src: dst})
            else:
                new_container = self._container
        else:
            new_container = self._container

        return self._light_init_same_structure(
            new_container,
            might_have_duplicates=self._might_have_duplicates,
            columns=new_columns,
        )

    def rename_columns(self, renames):
        if self.is_dum():
            return self
        self._check_for_duplicated_columns(renames.values())
        if not set(renames).issubset(self.columns):
            not_found_cols = set(c for c in renames if c not in self._columns)
            raise ValueError(f"Cannot rename non-existing columns: {not_found_cols}")
        new_columns = tuple(renames.get(col, col) for col in self._columns)
        if self._container is not None:
            c = _strip_sentinel(self._container)
            if c is not None:
                c_names = _lf_names(c)
                filtered_renames = {k: v for k, v in renames.items() if k in c_names}
                new_container = c.rename(filtered_renames)
            else:
                new_container = self._container
        else:
            new_container = self._container
        return self._light_init_same_structure(
            new_container,
            might_have_duplicates=self._might_have_duplicates,
            columns=new_columns,
        )

    def __eq__(self, other):
        if not isinstance(other, NamedRelationalAlgebraFrozenSet):
            return super().__eq__(other)
        if self._container is None and other._container is None:
            return self._columns == other._columns
        if self._container is None or other._container is None:
            return False
        s_eager = self._eager()
        o_eager = other._eager()
        scont = _strip_sentinel(s_eager)
        ocont = _strip_sentinel(o_eager)
        if scont is None and ocont is None:
            return True
        if scont is None or ocont is None:
            return False
        if set(scont.columns) != set(ocont.columns):
            return False
        if len(scont) == 0 and len(ocont) == 0:
            return True
        if len(scont.columns) == 0 and len(ocont.columns) == 0:
            return len(scont) > 0 and len(ocont) > 0
        # Compare as sorted column sets (order-independent)
        try:
            s_sorted = scont.sort(by=list(scont.columns)).select(sorted(scont.columns))
            o_sorted = ocont.sort(by=list(ocont.columns)).select(sorted(ocont.columns))
            s_rows = set(tuple(r) for r in s_sorted.iter_rows())
            o_rows = set(tuple(r) for r in o_sorted.iter_rows())
        except Exception:
            s_rows = set(tuple(r) for r in scont.iter_rows())
            o_rows = set(tuple(r) for r in ocont.iter_rows())
        return s_rows == o_rows

    def groupby(self, columns):
        if self.is_empty() or self._container is None:
            return
        col_names = list(columns)
        c = _strip_sentinel(self._container)
        if c is None:
            return
        # LazyGroupBy not iterable — collect first
        eager = c.collect()
        for g_id, group_df in eager.group_by(col_names, maintain_order=True):
            group_set = self._light_init_same_structure(
                group_df.lazy(),
                might_have_duplicates=self._might_have_duplicates,
                columns=self.columns,
            )
            if isinstance(g_id, tuple) and len(g_id) == 1:
                g_id = g_id[0]
            yield g_id, group_set

    def aggregate(self, group_columns, aggregate_function):
        group_columns = list(group_columns)
        if len(set(group_columns)) < len(group_columns):
            raise ValueError("Cannot group on repeated columns")
        if self.is_dee():
            raise ValueError(
                "Aggregation on non-empty sets with arity == 0 is unsupported."
            )
        if self.is_empty():
            if isinstance(aggregate_function, dict):
                agg_columns = list(aggregate_function.keys())
            elif isinstance(aggregate_function, (tuple, list)):
                agg_columns = list(col for col, _, _ in aggregate_function)
            return NamedRelationalAlgebraFrozenSet(columns=group_columns + agg_columns)
        self._drop_duplicates_if_needed()

        aggs, aggs_multi_column = self._classify_aggregations(
            group_columns, aggregate_function
        )

        c = _strip_sentinel(self._container)
        if c is None:
            return NamedRelationalAlgebraFrozenSet(
                columns=group_columns + list(aggs.keys()) + list(aggs_multi_column.keys())
            )

        # single-column aggregations
        agg_exprs = []
        eager_aggs = []  # (dst, src, fun) for functions _map_agg_function couldn't handle
        for dst, (src, fun) in aggs.items():
            polars_fun = _map_agg_function(fun)
            if polars_fun is not None:
                expr = polars_fun(src)
                agg_exprs.append(expr.alias(dst))
            else:
                eager_aggs.append((dst, src, fun))

        if len(group_columns) > 0:
            result = c.group_by(group_columns, maintain_order=True).agg(agg_exprs)
        else:
            if len(agg_exprs) > 0:
                result = c.select([e for e in agg_exprs])
            else:
                result = c.select(pl.lit(1).alias('__dummy__'))
                result = result.select([])

        # multi-column aggregations — need eager for these
        if len(aggs_multi_column) > 0 or len(eager_aggs) > 0:
            eager_result = result.collect()
            eager_c = c.collect()
            all_cols = list(eager_c.columns)

            if len(aggs_multi_column) > 0:
                if len(group_columns) == 0:
                    values_dict = {col: eager_c[col].to_list() for col in all_cols}
                    for dst, fun in aggs_multi_column.items():
                        agg_val = fun(_AggRowProxy(values_dict))
                        eager_result = eager_result.with_columns(pl.Series(name=dst, values=[agg_val]))
                else:
                    for dst, fun in aggs_multi_column.items():
                        groups = eager_c.group_by(group_columns, maintain_order=True)
                        result_rows = []
                        for keys, group_df in groups:
                            group_dict = {col: group_df[col].to_list() for col in group_df.columns}
                            result_rows.append(fun(_AggRowProxy(group_dict)))
                        eager_result = eager_result.with_columns(
                            pl.Series(name=dst, values=result_rows)
                        )

            # eager single-column aggregations for functions _map_agg_function couldn't handle
            if len(eager_aggs) > 0:
                if len(group_columns) == 0:
                    values_dict = {col: eager_c[col].to_list() for col in all_cols}
                    for dst, src, fun in eager_aggs:
                        agg_val = fun(values_dict[src])
                        eager_result = eager_result.with_columns(pl.Series(name=dst, values=[agg_val]))
                else:
                    groups = eager_c.group_by(group_columns, maintain_order=True)
                    group_agg_values = {dst: [] for dst, _, _ in eager_aggs}
                    for keys, group_df in groups:
                        group_dict = {col: group_df[col].to_list() for col in group_df.columns}
                        for dst, src, fun in eager_aggs:
                            agg_val = fun(group_dict[src])
                            group_agg_values[dst].append(agg_val)
                    for dst, values_list in group_agg_values.items():
                        eager_result = eager_result.with_columns(pl.Series(name=dst, values=values_list))

            result = eager_result.lazy()

        # Reorder to match original column order when possible
        all_agg_cols = list(aggs.keys()) + list(aggs_multi_column.keys())
        all_dst_columns = group_columns + all_agg_cols
        original_cols = list(self._columns)
        if original_cols:
            ordered = [c for c in original_cols if c in all_dst_columns]
            for c in all_dst_columns:
                if c not in ordered:
                    ordered.append(c)
            all_dst_columns = ordered

        result = result.select(*[pl.col(c) for c in all_dst_columns])

        return self._light_init_same_structure(
            result, might_have_duplicates=False, columns=all_dst_columns,
        )

    def _classify_aggregations(self, group_columns, aggregate_function):
        aggs = OrderedDict()
        aggs_multi_columns = OrderedDict()
        if isinstance(aggregate_function, dict):
            for k, v in aggregate_function.items():
                if k in group_columns:
                    raise ValueError(
                        f"Destination column {k} can't be part of the grouping"
                    )
                if isinstance(v, RelationalAlgebraStringExpression):
                    aggs[k] = (k, v)
                elif k in self._columns:
                    aggs[k] = (k, v)
                else:
                    aggs_multi_columns[k] = v
        elif isinstance(aggregate_function, (tuple, list)):
            for dst, src, fun in aggregate_function:
                if dst in group_columns:
                    raise ValueError(
                        f"Destination column {dst} can't be part of the grouping"
                    )
                if src in self.columns:
                    aggs[dst] = (src, fun)
                elif src is None or all(s in self.columns for s in src):
                    aggs_multi_columns[dst] = fun
                else:
                    raise ValueError(f"Source column {src} not in columns")
        else:
            raise ValueError(
                "Unsupported aggregate_function: {} of type {}".format(
                    aggregate_function, type(aggregate_function)
                )
            )
        return aggs, aggs_multi_columns

    def extended_projection(self, eval_expressions):
        proj_columns = list(eval_expressions.keys())
        if self.is_empty():
            return NamedRelationalAlgebraFrozenSet(columns=proj_columns, iterable=[])
        if self._container is None:
            return NamedRelationalAlgebraFrozenSet(columns=proj_columns, iterable=[])

        c = _strip_sentinel(self._container)
        if c is None:
            return NamedRelationalAlgebraFrozenSet(columns=proj_columns, iterable=[])

        new_container = c
        seen_pure_columns: SetType[abc.RelationalAlgebraColumn] = set()

        for dst_column, operation in eval_expressions.items():
            if isinstance(operation, RelationalAlgebraStringExpression):
                op_str = str(operation)
                if op_str == str(dst_column):
                    pass
                else:
                    try:
                        polars_expr = _string_expr_to_polars(op_str)
                        if not isinstance(polars_expr, pl.Expr):
                            raise ValueError("Not a Polars expression")
                        new_container = new_container.with_columns(
                            polars_expr.alias(dst_column)
                        )
                    except Exception:
                        # Fall back to eager row-by-row
                        eager = new_container.collect()
                        c_names = _lf_names(c)
                        col_names = c_names
                        values = []
                        for row in eager.iter_rows():
                            values.append(
                                _eval_row_with_builtins(
                                    op_str, col_names, row
                                )
                            )
                        eager = eager.with_columns(
                            pl.Series(name=dst_column, values=values)
                        )
                        new_container = eager.lazy()
            elif isinstance(operation, abc.RelationalAlgebraColumn):
                seen_pure_columns.add(operation)
                new_container = new_container.with_columns(
                    pl.col(str(operation)).alias(dst_column)
                )
            elif callable(operation):
                # Need eager for callable
                eager = new_container.collect()
                col_names = _lf_names(c)
                Row = self._make_named_row_class(col_names)
                values = [
                    operation(Row(*(
                        _DictWrapper(v) if isinstance(v, dict) else v
                        for v in row
                    ))) for row in eager.iter_rows()
                ]
                eager = eager.with_columns(
                    pl.Series(name=dst_column, values=values)
                )
                new_container = eager.lazy()
            else:
                new_container = new_container.with_columns(
                    pl.lit(operation).alias(dst_column)
                )

        new_container = new_container.select(*[pl.col(c) for c in proj_columns])
        might_have_duplicates = not (
            (len(seen_pure_columns) == len(self.columns))
            and not self._might_have_duplicates
        )
        return self._light_init_same_structure(
            new_container,
            might_have_duplicates=might_have_duplicates,
            columns=proj_columns,
        )

    @staticmethod
    def _make_named_row_class(col_names):
        has_numeric = any(
            not name.isidentifier() for name in col_names
        )
        has_underscore_prefix = any(
            name.startswith('_') and name.isidentifier() for name in col_names
        )
        if has_numeric or has_underscore_prefix:
            Row = namedtuple("_Row", col_names, rename=True)
        else:
            Row = namedtuple("_Row", col_names)
        return Row

    def __iter__(self):
        self._drop_duplicates_if_needed()
        if self.is_dee():
            yield from [tuple()]
            return
        if self._container is None:
            return
        col_list = list(self.columns)
        c = _strip_sentinel(self._container)
        if c is None:
            return
        eager = c.select(*[pl.col(col) for col in col_list]).collect()
        Row = self._make_named_row_class(col_list)
        for row in eager.iter_rows():
            yield Row(*row)

    def fetch_one(self):
        if self.is_dee():
            return tuple()
        eager = self._eager()
        if eager is None or len(eager) == 0:
            raise StopIteration("Cannot fetch_one from empty set")
        col_list = list(self.columns)
        c = _strip_sentinel(eager)
        if c is None:
            raise StopIteration("Cannot fetch_one from empty set")
        eager2 = c.select(*[pl.col(c) for c in col_list])
        Row = self._make_named_row_class(col_list)
        return Row(*eager2.row(0))

    def to_unnamed(self):
        if self._container is None:
            output = RelationalAlgebraFrozenSet()
            output._might_have_duplicates = self._might_have_duplicates
            return output
        col_list = list(self.columns)
        c = _strip_sentinel(self._container)
        if c is None:
            output = RelationalAlgebraFrozenSet()
            output._might_have_duplicates = self._might_have_duplicates
            return output
        container = c.select(*[pl.col(col) for col in col_list])
        c_names = _lf_names(container)
        container = container.rename(
            dict(zip(c_names, [str(i) for i in range(len(c_names))]))
        )
        output = RelationalAlgebraFrozenSet()
        output._container = container
        output._might_have_duplicates = self._might_have_duplicates
        return output

    def __sub__(self, other):
        if not isinstance(other, NamedRelationalAlgebraFrozenSet):
            return super().__sub__(other)
        s_eager = self._eager()
        o_eager = other._eager()
        scont = _strip_sentinel(s_eager)
        ocont = _strip_sentinel(o_eager)
        if scont is not None and ocont is not None:
            if (self.arity > 0 and other.arity > 0) and (
                set(scont.columns) != set(ocont.columns)
            ):
                raise ValueError(
                    "Difference defined only for sets with the same columns"
                )
        if self.is_empty() or other.is_empty():
            return self.copy()
        if self.is_dee():
            if other.is_dee():
                return self.dum()
            return self.dee()
        self._drop_duplicates_if_needed()
        other._drop_duplicates_if_needed()

        if scont is None or ocont is None:
            return self.copy()

        try:
            new_df = scont.join(ocont, how="anti", on=list(scont.columns))
            new_lf = new_df.lazy()
        except BaseException:
            # Polars anti-join panics on Object-type columns, use Python fallback
            s_rows = set(tuple(r) for r in scont.iter_rows())
            o_rows = set(tuple(r) for r in ocont.iter_rows())
            diff_rows = s_rows - o_rows
            if diff_rows:
                new_lf = pl.DataFrame(
                    list(diff_rows), schema=scont.columns, orient="row"
                ).lazy()
            else:
                new_lf = pl.LazyFrame({})

        return self._light_init_same_structure(
            new_lf, might_have_duplicates=False,
        )

    def __or__(self, other):
        if not isinstance(other, NamedRelationalAlgebraFrozenSet):
            return super().__or__(other)
        res = self._dee_dum_sum(other)
        if res is not None:
            return res
        if set(self.columns) != set(other.columns):
            raise ValueError("Union defined only for sets with the same columns")
        s_eager = self._eager()
        o_eager = other._eager()
        scont = _strip_sentinel(s_eager)
        ocont = _strip_sentinel(o_eager)
        if scont is None and ocont is None:
            return self.copy()
        if scont is None:
            return other.copy()
        if ocont is None:
            return self.copy()

        s_unified = self._to_lazy(scont) if isinstance(scont, pl.DataFrame) else scont
        o_unified = self._to_lazy(ocont) if isinstance(ocont, pl.DataFrame) else ocont
        s_unified, o_unified = self._unify_join_types(
            s_unified,
            o_unified,
            list(self.columns),
        )
        new_container = _concat_safe([s_unified, o_unified])
        if new_container is not None:
            new_container = new_container.unique(maintain_order=True)

        return self._light_init_same_structure(
            new_container, might_have_duplicates=True,
        )

    def __and__(self, other):
        if not isinstance(other, NamedRelationalAlgebraFrozenSet):
            return super().__and__(other)
        res = self._dee_dum_product(other)
        if res is not None:
            return res
        if set(self.columns) != set(other.columns):
            raise ValueError(
                "Intersection defined only for sets with the same columns"
            )
        if self.is_empty():
            return self.copy()
        self._drop_duplicates_if_needed()
        other._drop_duplicates_if_needed()

        s_eager = self._eager()
        o_eager = other._eager()
        scont = _strip_sentinel(s_eager)
        ocont = _strip_sentinel(o_eager)
        if scont is None or ocont is None:
            return self._empty_set_same_structure()

        s_unified = self._to_lazy(scont) if isinstance(scont, pl.DataFrame) else scont
        o_unified = self._to_lazy(ocont) if isinstance(ocont, pl.DataFrame) else ocont
        s_unified, o_unified = self._unify_join_types(
            s_unified,
            o_unified,
            list(self.columns),
        )
        new_container = s_unified.join(o_unified, on=_lf_names(s_unified), how="inner")

        return self._light_init_same_structure(
            new_container,
            might_have_duplicates=(
                self._might_have_duplicates & other._might_have_duplicates
            ),
        )

    def __lt__(self, other):
        raise NotImplementedError()

    def __le__(self, other):
        raise NotImplementedError()

    def __gt__(self, other):
        raise NotImplementedError()

    def __ge__(self, other):
        raise NotImplementedError()


class RelationalAlgebraSet(
    RelationalAlgebraFrozenSet, abc.RelationalAlgebraSet
):
    def add(self, value):
        value = self._normalise_element(value)
        if self.is_empty() or self._container is None:
            self._container = _make_unnamed([value])
        else:
            new_row = _make_unnamed([value])
            eager = self._eager()
            if eager is not None and new_row is not None:
                new_eager = new_row.collect() if isinstance(new_row, pl.LazyFrame) else new_row
                self._container = pl.concat([eager, new_eager], how="vertical").lazy()
            elif eager is not None:
                self._container = eager.lazy()

    def discard(self, value):
        if self.is_empty() or self._container is None:
            return
        try:
            value = self._normalise_element(value)
            eager = self._eager()
            if eager is None:
                return
            mask = None
            for i, val in enumerate(value):
                col = eager[:, str(i)]
                if mask is None:
                    mask = (col == val)
                else:
                    mask = mask & (col == val)
            if mask.any():
                self._container = eager.filter(~mask).lazy()
        except KeyError:
            pass

    def __ior__(self, other):
        if isinstance(other, RelationalAlgebraSet):
            if other.is_empty() or other.arity == 0:
                return self
            if self.is_empty() or self._container is None:
                if other._container is not None:
                    self._container = other._container
                self._might_have_duplicates = other._might_have_duplicates
                return self
            if other.arity != self.arity:
                raise ValueError("Operation only valid for sets with the same arity")
            self_eager = self._eager()
            other_eager = other._eager()
            sc = _strip_sentinel(self_eager)
            oc = _strip_sentinel(other_eager)
            if sc is not None and oc is not None:
                new_eager = pl.concat([sc, oc], how="vertical")
                new_eager = new_eager.unique(maintain_order=True)
                self._container = new_eager.lazy()
            self._might_have_duplicates = True
            return self
        else:
            return super().__ior__(other)

    def __isub__(self, other):
        if self.is_empty():
            return self
        if isinstance(other, RelationalAlgebraSet):
            if self.is_dee() and other.is_dee():
                self._container = _dum_df()
                return self
            if other.is_empty() or other.arity == 0:
                if self.is_empty() or self.arity == 0:
                    self._container = _dum_df()
                return self
            elif other.arity != self.arity:
                raise ValueError("Operation only valid for sets with the same arity")
            else:
                self_eager = self._eager()
                other_eager = other._eager()
                sc = _strip_sentinel(self_eager)
                oc = _strip_sentinel(other_eager)
                if sc is not None and oc is not None:
                    on_cols = list(sc.columns)
                    new_eager = sc.join(oc, on=on_cols, how="anti")
                    self._container = new_eager.lazy()
            return self
        else:
            return super().__isub__(other)


def _string_expr_to_polars(expr_str: str) -> pl.Expr:
    try:
        return eval(expr_str, {"__builtins__": builtins, "pl": pl, "col": pl.col})
    except NameError:
        raise NameError(
            f"Cannot convert '{expr_str}' to a polars expression"
        )


def _eval_row_with_builtins(op_str, col_names, row):
    env = dict(zip(col_names, row))
    globals_dict = {
        "__builtins__": builtins,
        "np": np,
        "numpy": np,
        "abs": abs,
        "log": np.log,
        "log10": np.log10,
        "log2": np.log2,
        "exp": np.exp,
        "sqrt": np.sqrt,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "min": min,
        "max": max,
        "sum": sum,
        "float": float,
        "int": int,
        "str": str,
        "len": len,
        "bool": bool,
    }
    return eval(op_str, globals_dict, env)


class _AggRowProxy:
    def __init__(self, d):
        self._values = {}
        for k, v in d.items():
            self._values[k] = v
            setattr(self, k, v)

    def __repr__(self):
        return f"_AggRowProxy({self._values})"


class _DictWrapper:
    """Wraps a dict to support attribute access (foo.bar) for nested field access."""
    def __init__(self, d):
        self._d = d

    def __getattr__(self, name):
        try:
            return self._d[name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def __repr__(self):
        return repr(self._d)

    def __getitem__(self, key):
        return self._d[key]


def _map_agg_function(fun):
    mapping = {
        sum: lambda col: pl.sum(col),
        "sum": lambda col: pl.sum(col),
        "count": lambda col: pl.len(),
        "mean": lambda col: pl.mean(col),
        "std": lambda col: pl.std(col),
        "var": lambda col: pl.var(col),
        "min": lambda col: pl.min(col),
        "max": lambda col: pl.max(col),
        "first": lambda col: pl.first(col),
        "last": lambda col: pl.last(col),
        "median": lambda col: pl.median(col),
    }
    if fun in mapping:
        return mapping[fun]
    if fun is max:
        return lambda col: pl.max(col)
    if fun is min:
        return lambda col: pl.min(col)
    if fun is len:
        return lambda col: pl.len()
    # Check for numpy functions used in BuiltinAggregationMixin constants
    try:
        import numpy as np
        if fun is np.max:
            return lambda col: pl.max(col)
        if fun is np.min:
            return lambda col: pl.min(col)
        if fun is np.sum:
            return lambda col: pl.sum(col)
        if fun is np.mean:
            return lambda col: pl.mean(col)
        if fun is np.std:
            return lambda col: pl.std(col)
    except ImportError:
        pass
    # Check for bound methods from BuiltinAggregationMixin (e.g., function_count)
    import types
    if isinstance(fun, types.MethodType):
        func_name = getattr(fun, '__name__', None)
        if func_name == 'function_count':
            return lambda col: pl.len()
    # For unknown functions, return None to signal eager fallback
    return None
