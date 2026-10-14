"""
Paged display of query results as tables.

Numbers are right-aligned and floats shown with 4 significant digits;
the header gives the type of each column under its name. Columns are
capped in width, with an ellipsis, and the columns that do not fit the
console are left out and counted below the table. On a terminal, the
rows are numbered and a probability column gets an inline bar; other
output (piped, batch) stays plain.
"""

import math
import re
from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd
from rich import box
from rich.cells import cell_len
from rich.table import Table
from rich.text import Text

DEFAULT_PAGE_SIZE = 20
MAX_COLUMN_WIDTH = 40
NARROW_COLUMN_WIDTH = 16
BAR_WIDTH = 5
HIDDEN_COLUMNS_HINT = ".browse to see all"
BROWSE_HINT = ".browse to explore"

_BAR_EIGHTHS = " ▏▎▍▌▋▊▉"
_PROBABILITY_NAME = re.compile("prob", re.IGNORECASE)
_INFERRED_KINDS = {
    "string": "str",
    "integer": "int",
    "floating": "float",
    "mixed-integer-float": "float",
    "decimal": "float",
    "boolean": "bool",
    "bytes": "bytes",
}
_NUMERIC_KINDS = ("int", "float")


@dataclass
class ResultBlock:
    """
    The rows of a query result and how many of them have been shown.

    The title names the ``obtain`` of a SQUALL program with several.
    """

    frame: pd.DataFrame
    title: str = ""
    shown: int = 0

    @property
    def remaining(self) -> int:
        """Return the number of rows not shown yet."""
        return len(self.frame) - self.shown


def column_kind(values: pd.Series) -> str:
    """
    Return the short name of the type of *values*.

    It is ``int``, ``float``, ``bool`` or ``str`` for the usual columns;
    the type of an object column is inferred from its values.
    """
    dtype = values.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    if pd.api.types.is_integer_dtype(dtype):
        return "int"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(
        dtype
    ):
        inferred = pd.api.types.infer_dtype(values, skipna=True)
        return _INFERRED_KINDS.get(inferred, "object")
    return str(dtype)


def format_value(value) -> str:
    """
    Return the text of a table cell holding *value*.

    Floats get at most 4 significant digits, but their integer part is
    kept up to 1e15; missing values are blank and anything else is shown
    with :func:`str`.
    """
    if value is None:
        return ""
    if not isinstance(value, float):
        return str(value)
    if math.isnan(value):
        return ""
    if 1e4 <= abs(value) < 1e15:
        return f"{value:.0f}"
    return f"{value:.4g}"


def probability_bar(value: float, width: int = BAR_WIDTH) -> str:
    """
    Return the bar of *value* on a track of *width* cells.

    *value* is in [0, 1]; the bar is drawn with eighths of a block and
    takes up to *width* cells.
    """
    eighths = round(value * width * 8)
    full, rest = divmod(eighths, 8)
    return "█" * full + (_BAR_EIGHTHS[rest] if rest else "")


def is_probability(name, values: pd.Series, kind: str) -> bool:
    """
    Return whether the column *name* holds probabilities.

    It does if its name contains ``prob`` and its *values*, of the
    numeric *kind*, are all in [0, 1].
    """
    if kind not in _NUMERIC_KINDS or not _PROBABILITY_NAME.search(str(name)):
        return False
    present = values.dropna()
    return (
        not present.empty
        and bool(present.min() >= 0) and bool(present.max() <= 1)
    )


class ColumnView(NamedTuple):
    """The header, cells and layout of a column of a shown table."""

    header: Text
    cells: list
    numeric: bool
    width: int


def _with_bars(cells: list, values: list) -> list:
    """
    Return the *cells* of probability *values* followed by their bar.

    The rest of the track of the bar is dotted, so that right-aligned
    cells keep their bars aligned.
    """
    bars = [probability_bar(value) if text else "" for text, value
            in zip(cells, values)]
    return [
        Text.assemble(
            text, " ", (bar, "cyan"), ("·" * (BAR_WIDTH - len(bar)), "dim")
        ) if text else Text("")
        for text, bar in zip(cells, bars)
    ]


def column_view(name, values: pd.Series, terminal: bool) -> ColumnView:
    """
    Return how the column *name* holding *values* is shown.

    On a *terminal*, a probability column gets an inline bar after each
    value. The width is the one the column needs, at most
    :data:`MAX_COLUMN_WIDTH`.
    """
    kind = column_kind(values)
    raw = values.tolist()
    cells = [format_value(value) for value in raw]
    if terminal and is_probability(name, values, kind):
        cells = _with_bars(cells, raw)
    header = Text(str(name))
    header.append("\n" + kind, style="dim")
    widths = [cell_len(str(name)), cell_len(kind)]
    widths += [cell_len(str(cell)) for cell in cells]
    width = min(MAX_COLUMN_WIDTH, max(widths))
    return ColumnView(header, cells, kind in _NUMERIC_KINDS, width)


def columns_fitting(widths, available) -> int:
    """
    Return how many of the columns of *widths* fit in *available* cells.

    Each column takes its width plus its padding and border, and the
    table one more border; the first column is always kept.
    """
    used = 1
    for count, width in enumerate(widths):
        used += width + 3
        if used > available:
            return max(count, 1)
    return len(widths)


def narrowed(widths, available) -> list:
    """
    Return *widths* with the widest narrowed to fit in *available* cells.

    The widest columns lose one cell at a time, down to
    :data:`NARROW_COLUMN_WIDTH`, until the table fits.
    """
    widths = list(widths)
    excess = sum(widths) + 3 * len(widths) + 1 - available
    while excess > 0 and widths:
        widest = max(range(len(widths)), key=widths.__getitem__)
        if widths[widest] <= NARROW_COLUMN_WIDTH:
            break
        widths[widest] -= 1
        excess -= 1
    return widths


def fit_widths(widths, available) -> list:
    """
    Return the widths of the columns of *widths* shown in *available* cells.

    The widest columns are narrowed (see :func:`narrowed`) and only the
    columns then fitting are kept.
    """
    widths = narrowed(widths, available)
    return widths[:columns_fitting(widths, available)]


def _add_row_numbers(table: Table, start: int, count: int) -> list:
    """
    Add the dim row-number column to *table* and return its cells.

    The *count* rows are numbered from 1 after the *start* first ones.
    """
    table.add_column(
        "#", style="dim", header_style="dim", justify="right", no_wrap=True,
    )
    return [str(n) for n in range(start + 1, start + count + 1)]


def render_rows(
    df: pd.DataFrame, start: int = 0, width=None, terminal: bool = False,
):
    """
    Return a rich Table of the rows of *df* and the number of hidden columns.

    *start* is the position of the first row of *df* in the result, for
    the row numbers shown on a *terminal*. Only the columns fitting in
    *width* cells (default: no limit) are shown.
    """
    views = [column_view(name, df[name], terminal) for name in df.columns]
    table = Table(box=box.ROUNDED, header_style="bold cyan")
    columns = []
    available = width or math.inf
    if terminal:
        columns.append(_add_row_numbers(table, start, len(df)))
        available -= len(str(start + len(df))) + 3
    widths = fit_widths([view.width for view in views], available)
    for view, column_width in zip(views, widths):
        table.add_column(
            view.header, justify="right" if view.numeric else "left",
            no_wrap=True, overflow="ellipsis", max_width=column_width,
        )
        columns.append(view.cells)
    for row in zip(*columns):
        table.add_row(*row)
    return table, len(views) - len(widths)


def rows_table(
    df: pd.DataFrame, start: int = 0, width=None, terminal: bool = False,
) -> Table:
    """Return a rich Table with the rows of *df* (see :func:`render_rows`)."""
    return render_rows(df, start, width, terminal)[0]


def hidden_columns_note(hidden: int) -> str:
    """
    Return the note on *hidden* columns of a caption.

    For instance ``+2 columns · .browse to see all``, or an empty
    string when no column is hidden.
    """
    if not hidden:
        return ""
    columns = "1 column" if hidden == 1 else f"{hidden} columns"
    return f"+{columns} · {HIDDEN_COLUMNS_HINT}"


def page_caption(total: int, start: int, stop: int) -> str:
    """
    Return the one-line caption of the rows *start* to *stop* of *total*.

    For instance ``3228 rows · showing 1–20 · .more for next · .browse
    to explore``.
    """
    rows = "1 row" if total == 1 else f"{total} rows"
    if start == 0 and stop >= total:
        return rows
    if stop < total:
        return (
            f"{rows} · showing {start + 1}–{stop} · .more for next · "
            + BROWSE_HINT
        )
    return f"{rows} · showing {start + 1}–{stop} · end"


def _print_caption(console, caption: str) -> None:
    """Print *caption* dimmed on one line, if it is not empty."""
    if caption:
        console.print(
            caption, style="dim", markup=False, no_wrap=True,
            overflow="ellipsis",
        )


def print_next_page(console, block: ResultBlock, page_size: int) -> None:
    """Print the next *page_size* rows of *block* with their caption."""
    total = len(block.frame)
    if total == 0:
        console.print("[dim](empty)[/dim]")
        return
    start = block.shown
    stop = min(start + page_size, total)
    table, hidden = render_rows(
        block.frame.iloc[start:stop], start, console.width,
        console.is_terminal,
    )
    _print_table(console, table, hidden)
    _print_caption(console, page_caption(total, start, stop))
    block.shown = stop


def _print_table(console, table: Table, hidden: int) -> None:
    """Print *table* and the note on its *hidden* columns, if any."""
    console.print(table)
    _print_caption(console, hidden_columns_note(hidden))


def print_all(console, block: ResultBlock) -> None:
    """Print all the rows of *block*, through a pager on a terminal."""
    table, hidden = render_rows(
        block.frame, 0, console.width, console.is_terminal
    )
    if console.is_terminal:
        with console.pager(styles=True):
            _print_table(console, table, hidden)
    else:
        _print_table(console, table, hidden)
    block.shown = len(block.frame)
