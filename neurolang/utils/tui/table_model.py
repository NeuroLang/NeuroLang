"""
The state of the interactive table viewer, without any display.

A :class:`TableModel` holds a result frame and the viewer's cursor,
window, sort and search. It never formats the whole frame: sorting and
searching work on factorised columns, so that they stay fast on
millions of rows, and their results are cached.
"""

import threading
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

from .results import column_kind, format_value, is_probability

SEPARATOR_WIDTH = 3
_MASK_CACHE_SIZE = 64


def clamp(value: int, low: int, high: int) -> int:
    """Return *value* bounded to [*low*, *high*], *low* winning if empty."""
    return max(low, min(value, high))


def _factorize(values: pd.Series):
    """
    Return the codes and the distinct values of *values*.

    Missing values get the code -1. Unhashable values are factorised by
    their text.
    """
    try:
        return pd.factorize(values)
    except TypeError:
        return pd.factorize(values.map(str))


def _sorted_codes(values: pd.Series):
    """
    Return codes of *values* increasing with the values, and their count.

    Missing values get the code -1. Values that cannot be compared with
    each other are ordered by their text.
    """
    codes, uniques = _factorize(values)
    try:
        ranks = np.argsort(np.argsort(np.asarray(uniques), kind="stable"))
    except TypeError:
        texts = np.array([str(u) for u in uniques], dtype=object)
        ranks = np.argsort(np.argsort(texts, kind="stable"))
    return np.where(codes >= 0, np.append(ranks, -1)[codes], -1), len(ranks)


def _descending(array: np.ndarray) -> np.ndarray:
    """Return a key of the numbers *array* sorting them in reverse order."""
    if array.dtype.kind == "f" or not len(array):
        return -array
    return array.max() - array


def sort_order(values: pd.Series, ascending: bool = True) -> np.ndarray:
    """
    Return the positions of *values* in sorted order.

    The sort is stable and missing values come last in both directions.
    Numbers are sorted directly, other values through their codes.
    """
    array = values.to_numpy()
    if array.dtype.kind in "iuf":
        return np.argsort(
            array if ascending else _descending(array), kind="stable"
        )
    codes, count = _sorted_codes(values)
    key = codes if ascending else count - 1 - codes
    key = np.where(codes >= 0, key, count)
    return np.argsort(key, kind="stable")


class TextIndex(NamedTuple):
    """
    The shown texts of a column, lower-cased and factorised.

    ``codes`` gives for each row the position of its text in ``texts``,
    or -1 for a missing value.
    """

    codes: np.ndarray
    texts: pd.Series

    def contains(self, query: str) -> np.ndarray:
        """Return which rows have a text containing the lower-case *query*."""
        found = self.texts.str.contains(query, regex=False).to_numpy(bool)
        return np.append(found, False)[self.codes]


def text_index(values: pd.Series) -> TextIndex:
    """
    Return the index of the texts shown for *values*.

    The texts are those of :func:`~.results.format_value`, so that a
    search matches what the table shows.
    """
    codes, uniques = _factorize(values)
    texts = [format_value(value).lower() for value in list(uniques)]
    text_codes, distinct = pd.factorize(pd.Series(texts, dtype=object))
    lookup = np.append(text_codes, -1)
    return TextIndex(lookup[codes], pd.Series(distinct, dtype=object))


def columns_from(widths, left: int, available: int) -> list:
    """
    Return the columns from *left* whose *widths* fit in *available* cells.

    Each column takes its width plus a separator; the *left* column is
    always included.
    """
    shown, used = [], 0
    for col in range(left, len(widths)):
        used += widths[col] + SEPARATOR_WIDTH
        if used > available and shown:
            break
        shown.append(col)
    return shown


class TableModel:
    """
    The cursor, window, sort and search of a viewed result frame.

    Rows are counted in the order of the current sort (the *view*); the
    cursor is the cell (``row``, ``col``) and the window shows the
    ``height`` rows from ``top`` and the columns from ``left``.
    """

    def __init__(self, frame: pd.DataFrame, height: int = 20):
        """Set up a view of *frame* with a window of *height* rows."""
        self.frame = frame
        self.row = self.col = self.top = self.left = 0
        self.height = max(1, height)
        self.sort: Optional[tuple] = None
        self.query = ""
        self.matches = np.empty(0, dtype=np.intp)
        self._orders: dict = {}
        self._texts: dict = {}
        self._masks: dict = {}
        self._kinds: dict = {}
        self._probabilities: dict = {}
        self._texts_lock = threading.Lock()
        self.search_ready = False

    @property
    def n_rows(self) -> int:
        """Return the number of rows of the frame."""
        return len(self.frame)

    @property
    def n_cols(self) -> int:
        """Return the number of columns of the frame."""
        return len(self.frame.columns)

    def name(self, col: int) -> str:
        """Return the name of the column *col*."""
        return str(self.frame.columns[col])

    def kind(self, col: int) -> str:
        """Return the type name of the column *col* (see column_kind)."""
        if col not in self._kinds:
            self._kinds[col] = column_kind(self.frame.iloc[:, col])
        return self._kinds[col]

    def is_probability(self, col: int) -> bool:
        """Return whether the column *col* holds probabilities."""
        if col not in self._probabilities:
            self._probabilities[col] = is_probability(
                self.name(col), self.frame.iloc[:, col], self.kind(col)
            )
        return self._probabilities[col]

    @property
    def order(self) -> Optional[np.ndarray]:
        """Return the frame positions in view order, None when unsorted."""
        if self.sort is None:
            return None
        if self.sort not in self._orders:
            col, ascending = self.sort
            self._orders[self.sort] = sort_order(
                self.frame.iloc[:, col], ascending
            )
        return self._orders[self.sort]

    def positions(self, start: int, stop: int) -> np.ndarray:
        """Return the frame positions of the view rows *start* to *stop*."""
        stop = min(stop, self.n_rows)
        order = self.order
        if order is None:
            return np.arange(start, max(start, stop))
        return order[start:stop]

    def window(self) -> pd.DataFrame:
        """Return the rows of the window, in view order."""
        rows = self.positions(self.top, self.top + self.height)
        return self.frame.iloc[rows]

    def set_height(self, height: int) -> None:
        """Set the number of rows of the window, keeping the cursor in it."""
        self.height = max(1, height)
        self._scroll()

    def _scroll(self) -> None:
        """Clamp the cursor and scroll the window to show it."""
        self.row = clamp(self.row, 0, self.n_rows - 1)
        self.col = clamp(self.col, 0, self.n_cols - 1)
        if self.row < self.top:
            self.top = self.row
        elif self.row >= self.top + self.height:
            self.top = self.row - self.height + 1
        self.top = clamp(self.top, 0, self.n_rows - self.height)
        self.left = min(self.left, self.col)

    def move(self, rows: int = 0, cols: int = 0) -> None:
        """Move the cursor by *rows* and *cols*, within the frame."""
        self.row += rows
        self.col += cols
        self._scroll()

    def page(self, pages: int) -> None:
        """Move the cursor and the window by *pages* windows."""
        self.top += pages * self.height
        self.move(rows=pages * self.height)

    def goto(self, row: Optional[int] = None, col: Optional[int] = None):
        """
        Move the cursor to *row* and *col*.

        Negative ones count from the end, as in Python sequences.
        """
        if row is not None:
            self.row = row % max(self.n_rows, 1)
        if col is not None:
            self.col = col % max(self.n_cols, 1)
        self._scroll()

    def visible_columns(self, widths, available: int) -> list:
        """
        Return the columns shown in *available* cells, scrolling to the cursor.

        *widths* are the widths of all the columns. The window is scrolled
        right until the cursor column fits.
        """
        self.left = min(self.left, self.col)
        shown = columns_from(widths, self.left, available)
        while self.col not in shown:
            self.left += 1
            shown = columns_from(widths, self.left, available)
        return shown

    def cycle_sort(self) -> None:
        """
        Sort on the cursor column: ascending, then descending, then none.

        The cursor goes back to the first row and the matches of the
        search follow the new order.
        """
        if self.sort is None or self.sort[0] != self.col:
            self.sort = (self.col, True)
        elif self.sort[1]:
            self.sort = (self.col, False)
        else:
            self.sort = None
        self.row = self.top = 0
        self.search(self.query)

    def _text_index(self, col: int) -> TextIndex:
        """Return the cached text index of the column *col*."""
        with self._texts_lock:
            if col not in self._texts:
                self._texts[col] = text_index(self.frame.iloc[:, col])
            return self._texts[col]

    def prepare_search(self) -> None:
        """
        Build the text indexes of all the columns ahead of a search.

        It can run in a background thread, as the first search on a
        large frame would otherwise wait for them.
        """
        for col in range(self.n_cols):
            self._text_index(col)
        self.search_ready = True

    def column_masks(self, query: str) -> list:
        """
        Return, for each column, which frame rows contain *query*.

        The search ignores the case; the masks are cached.
        """
        query = query.lower()
        if query not in self._masks:
            if len(self._masks) >= _MASK_CACHE_SIZE:
                self._masks.clear()
            self._masks[query] = [
                self._text_index(col).contains(query)
                for col in range(self.n_cols)
            ]
        return self._masks[query]

    def search(self, query: str) -> None:
        """Set the searched text and find the view rows containing it."""
        self.query = query
        self.matches = np.empty(0, dtype=np.intp)
        if not query or not self.n_cols:
            return
        found = np.logical_or.reduce(self.column_masks(query))
        order = self.order
        if order is not None:
            found = found[order]
        self.matches = np.flatnonzero(found)

    def _goto_match(self, index: int) -> None:
        """Move the cursor to the match *index*, on a matching column."""
        self.row = int(self.matches[index % len(self.matches)])
        position = self.positions(self.row, self.row + 1)[0]
        masks = self.column_masks(self.query)
        if not masks[self.col][position]:
            self.col = next(
                col for col, mask in enumerate(masks) if mask[position]
            )
        self._scroll()

    def jump_from(self, row: int) -> bool:
        """Move to the first match at or after *row*, wrapping around."""
        if not len(self.matches):
            return False
        self._goto_match(int(np.searchsorted(self.matches, row, "left")))
        return True

    def next_match(self, forward: bool = True) -> bool:
        """Move to the next (or previous) match, wrapping around."""
        if not len(self.matches):
            return False
        if forward:
            index = np.searchsorted(self.matches, self.row, "right")
        else:
            index = np.searchsorted(self.matches, self.row, "left") - 1
        self._goto_match(int(index))
        return True

    def match_number(self) -> int:
        """Return the 1-based number of the match at the cursor, or 0."""
        index = int(np.searchsorted(self.matches, self.row, "left"))
        if index < len(self.matches) and self.matches[index] == self.row:
            return index + 1
        return 0

    def search_status(self) -> str:
        """Return the search part of the status, e.g. ``/pain match 3/516``."""
        if not self.query:
            return ""
        count = len(self.matches)
        if not count:
            return f"/{self.query} no match"
        number = self.match_number()
        if number:
            return f"/{self.query} match {number}/{count}"
        return f"/{self.query} {count} matches"

    def status(self) -> str:
        """
        Return the status line of the view.

        For instance ``rows 1–38 of 3228 · col 3/3 tfidf (float) ·
        sort tfidf↓ · /pain match 3/516``.
        """
        if not self.n_rows:
            parts = ["no rows"]
        else:
            stop = min(self.top + self.height, self.n_rows)
            parts = [f"rows {self.top + 1}–{stop} of {self.n_rows}"]
        if self.n_cols:
            parts.append(
                f"col {self.col + 1}/{self.n_cols} {self.name(self.col)} "
                f"({self.kind(self.col)})"
            )
        if self.sort is not None:
            arrow = "↑" if self.sort[1] else "↓"
            parts.append(f"sort {self.name(self.sort[0])}{arrow}")
        parts.append(self.search_status())
        return " · ".join(part for part in parts if part)
