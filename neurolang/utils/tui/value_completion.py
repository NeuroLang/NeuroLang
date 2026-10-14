"""
Completion of the values inside quotes and of the variables of a tuple.

Inside a quoted string, the values of the column it fills are offered:
after the SQUALL verb ``mentions '``, those of its object (a term), and
in Datalog ``mentions(s, '``, those of the argument. They are read from
a stored relation: the plainest one holding the entities of the
column's declared sort (``term`` for a term), or else the relation of
the predicate itself. :class:`ColumnValues` reads each column once.

In SQUALL, inside the tuple following a noun, ``every Voxel (?i; ``,
the variables named after its next columns are offered.
"""

import re
from typing import NamedTuple, Optional

from prompt_toolkit.completion import Completion

from neurolang.datalog.wrapped_collections import WrappedRelationalAlgebraSet
from neurolang.expressions import Symbol

from . import squall_sorts

# The kinds of predicates whose values are read without solving anything.
VALUE_KINDS = frozenset({"stored", "loaded"})

_TUPLE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()'\"]*)\Z")


class ColumnValues:
    """
    The distinct string values of the columns of an engine's relations.

    Each column is read on first use and kept as long as its relation
    is the same set. Relations of more than *limit* rows are not read.
    """

    def __init__(self, nl, limit: int = 1_000_000):
        """Start with no values read from the engine *nl*."""
        self._nl = nl
        self._limit = limit
        self._cache = {}

    def __call__(self, name: str, position: int) -> tuple:
        """Return the sorted string values of the column *position*."""
        rows = self._relation(name)
        if rows is None:
            return ()
        key = (name, position)
        cached = self._cache.get(key)
        if cached is None or cached[0] is not rows:
            cached = (rows, self._read(name, position))
            self._cache[key] = cached
        return cached[1]

    def _relation(self, name: str):
        """Return the set of the stored relation *name*, or None."""
        try:
            value = self._nl.program_ir.symbol_table[Symbol(name)]
        except KeyError:
            return None
        rows = getattr(value, "value", None)
        return rows if isinstance(rows, WrappedRelationalAlgebraSet) else None

    def _read(self, name: str, position: int) -> tuple:
        """Return the sorted string values of a column of *name*."""
        rows = self._relation(name)
        if rows is None or len(rows) > self._limit:
            return ()
        frame = rows.as_pandas_dataframe()
        if position >= frame.shape[1]:
            return ()
        values = frame.iloc[:, position].unique()
        return tuple(sorted(v for v in values if isinstance(v, str)))


class Quote(NamedTuple):
    """An open quoted string: its *quote* and where it starts."""

    quote: str
    start: int


def open_quote(text: str) -> Optional[Quote]:
    """Return the quoted string *text* ends inside of, if any."""
    quote = None
    for position, char in enumerate(text):
        if quote is None and char in "'\"":
            quote = Quote(char, position)
        elif quote is not None and char == quote.quote:
            quote = None
    return quote


def _argument_at(info, column: int):
    """Return the declared argument of *info* spanning *column*, if any."""
    start = 0
    for argument in info.arguments or ():
        if start <= column < start + len(argument.columns):
            return argument
        start += len(argument.columns)
    return None


def _sort_relation(sort: str, catalog: dict) -> Optional[str]:
    """Return the plainest stored relation of the entities of *sort*."""
    nouns = [
        info.name for info in catalog.values()
        if info.kind in VALUE_KINDS and squall_sorts.sort_of(info) == sort
    ]
    return min(nouns, key=lambda name: (len(name), name)) if nouns else None


def value_source(info, column: int, catalog: dict):
    """
    Return the relation and column holding the values of *column*.

    The result is ``(name, position, meta)`` or None. They are those of
    the relation of the column's sort, if it has one column, or else of
    *info*'s own relation, if stored.
    """
    argument = _argument_at(info, column)
    if argument is not None and len(argument.columns) == 1:
        noun = _sort_relation(argument.sort, catalog)
        if noun is not None:
            return noun, 0, argument.sort
    if info.kind in VALUE_KINDS:
        columns = info.columns or ()
        meta = columns[column] if column < len(columns) else info.name
        return info.name, column, meta
    return None


def squall_quoted_column(before: str, catalog: dict):
    """
    Return the predicate and column a SQUALL string after *before* fills.

    After a verb (``mentions '``), it is the first column of the verb's
    object. The result is ``(info, column)`` or None.
    """
    words = squall_sorts.sentence_words(before)
    info = catalog.get(words[-1].lower()) if words else None
    arguments = getattr(info, "arguments", None) or ()
    if len(arguments) != 2:
        return None
    return info, len(arguments[0].columns)


def value_completions(values, typed: str, quote: str, meta: str, limit=100):
    """
    Yield the *values* starting with *typed*, closing the *quote*.

    Values holding the *quote* cannot be written inside it. At most
    *limit* values are offered.
    """
    count = 0
    for value in values:
        if count == limit:
            return
        if value.startswith(typed) and quote not in value:
            count += 1
            yield Completion(
                value + quote, start_position=-len(typed), display=value,
                display_meta=meta,
            )


def tuple_completions(text: str, catalog: dict) -> Optional[list]:
    """
    Return the variables of the next columns of the tuple *text* ends.

    After ``every Voxel (?i; ``: the variable of the next column,
    ``?j``, and those of all the remaining ones, ``?j; ?k)``. Return
    None if *text* does not end inside the tuple of a noun of *catalog*
    with known columns.
    """
    match = _TUPLE.search(text)
    info = catalog.get(match.group(1).lower()) if match else None
    columns = getattr(info, "columns", None) or ()
    if not columns:
        return None
    parts = match.group(2).split(";")
    typed = parts[-1].lstrip()
    rest = [f"?{column}" for column in columns[len(parts) - 1:]]
    words = [rest[0], "; ".join(rest) + ")"] if rest else []
    meta = f"column {len(parts)} of {info.name}"
    return [
        Completion(word, start_position=-len(typed), display_meta=meta)
        for word in words if word.startswith(typed)
    ]
