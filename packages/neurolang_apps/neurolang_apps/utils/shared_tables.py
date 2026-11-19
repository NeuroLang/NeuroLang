"""
One copy of the tables that engines built in one process have in common.

The REPL server's builder keeps every engine it builds, and engines of a
family read the same data: ``neurostore_iit`` holds 7.6 GB of tables
identical to those of ``neurostore``, each its own copy. :func:`share`,
called on each engine the builder builds, gives a relation whose table is
identical to one of an engine built before (same columns, types and rows)
the data of that table: a shallow copy of it, whose arrays are those of the
other (pandas' copy-on-write copies them only if one of the two is written
to, so neither sees the other change).

The tables are found by their shape, column names and types, then compared
row by row (:meth:`pandas.DataFrame.equals`); the pool holds them by weak
reference, so that an engine dropped takes its tables with it. Tables
mapped from a file (:class:`~neurolang_apps.utils.engine_cache.DerivedTables`)
are shared by the system already, and not compared. Only the pandas
backend's relations are shared.
"""

import weakref
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

#: The tables met so far, by their shape, column names and types.
_POOL: Dict[Tuple, List[weakref.ref]] = {}

#: Tables of fewer rows are not worth comparing.
MIN_ROWS = 10_000


def _relations(nl):
    """Yield the relations of the engine *nl* holding a pandas table."""
    for value in list(nl.program_ir.symbol_table.values()):
        relation = getattr(value, "value", None)
        if isinstance(getattr(relation, "_container", None), pd.DataFrame):
            yield relation


def _mapped(frame: pd.DataFrame) -> bool:
    """Whether *frame* is a view of a file (its first column, if numbers,
    cannot be written to)."""
    column = frame.iloc[:, 0]
    if column.dtype.kind not in "iufb":
        return False
    return not np.asarray(column.array).flags.writeable


def _key(frame: pd.DataFrame) -> Tuple:
    return (
        frame.shape, tuple(str(c) for c in frame.columns),
        tuple(str(t) for t in frame.dtypes),
    )


def share(nl) -> int:
    """
    Give the relations of the engine *nl* the data of the identical tables
    of the engines given before; return how many bytes they no longer hold
    a copy of (their arrays, not counting the strings they point to).
    """
    saved = 0
    for relation in _relations(nl):
        frame = relation._container
        if getattr(relation, "_shared_table", None) is frame:
            continue  # shared already
        if len(frame) < MIN_ROWS or frame.shape[1] == 0 or _mapped(frame):
            continue
        refs = _POOL.setdefault(_key(frame), [])
        refs[:] = [ref for ref in refs if ref() is not None]
        for ref in refs:
            other = ref()
            if other is frame:
                break
            if other.equals(frame):
                relation._container = other.copy(deep=False)
                relation._shared_table = relation._container
                saved += int(frame.memory_usage(index=False).sum())
                break
        else:
            refs.append(weakref.ref(frame))
    return saved
