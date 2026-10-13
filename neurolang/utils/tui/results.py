"""Paged display of query results as tables."""

from dataclasses import dataclass

import pandas as pd
from rich import box
from rich.table import Table

DEFAULT_PAGE_SIZE = 20


@dataclass
class ResultBlock:
    """The rows of a query result and how many of them have been shown."""

    frame: pd.DataFrame
    title: str = ""
    shown: int = 0

    @property
    def remaining(self) -> int:
        """Return the number of rows not shown yet."""
        return len(self.frame) - self.shown


def rows_table(df: pd.DataFrame, title: str = "") -> Table:
    """Return a rich Table with all the rows of *df*."""
    table = Table(
        title=title or None,
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold magenta",
    )
    for col in df.columns:
        table.add_column(str(col), overflow="fold")
    for row in df.itertuples(index=False):
        table.add_row(*["" if v is None else str(v) for v in row])
    return table


def page_caption(total: int, start: int, stop: int) -> str:
    """
    Return the one-line caption of the rows *start* to *stop* of *total*.

    For instance ``3228 rows · showing 1–20 · .more for next``.
    """
    rows = "1 row" if total == 1 else f"{total} rows"
    if start == 0 and stop >= total:
        return rows
    after = ".more for next" if stop < total else "end"
    return f"{rows} · showing {start + 1}–{stop} · {after}"


def print_next_page(console, block: ResultBlock, page_size: int) -> None:
    """Print the next *page_size* rows of *block* with their caption."""
    total = len(block.frame)
    if total == 0:
        console.print("[dim](empty)[/dim]")
        return
    start = block.shown
    stop = min(start + page_size, total)
    console.print(rows_table(block.frame.iloc[start:stop], block.title))
    console.print(
        page_caption(total, start, stop), style="dim", markup=False,
        no_wrap=True, overflow="ellipsis",
    )
    block.shown = stop


def print_all(console, block: ResultBlock) -> None:
    """Print all the rows of *block*, through a pager on a terminal."""
    table = rows_table(block.frame, block.title)
    if console.is_terminal:
        with console.pager(styles=True):
            console.print(table)
    else:
        console.print(table)
    block.shown = len(block.frame)
