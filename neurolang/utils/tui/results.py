"""Display of query results as tables."""

import pandas as pd
from rich import box
from rich.table import Table

MAX_TABLE_ROWS = 20


def _df_to_rich_table(df: pd.DataFrame, title: str = "") -> Table:
    """Render *df* as a rich Table, capped at MAX_TABLE_ROWS rows."""
    table = Table(
        title=title,
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold magenta",
        show_edge=True,
    )
    for col in df.columns:
        table.add_column(str(col), overflow="fold")
    total = len(df)
    for row in df.head(MAX_TABLE_ROWS).itertuples(index=False):
        table.add_row(*[str(v) if v is not None else "" for v in row])
    if total > MAX_TABLE_ROWS:
        table.caption = (
            f"[dim]{total} rows, showing first {MAX_TABLE_ROWS}[/dim]"
        )
    return table
