"""Conversion of query results to DataFrames and their display as tables."""

import pandas as pd
from rich import box
from rich.table import Table

MAX_TABLE_ROWS = 20


def _rename_unnamed_columns(df, column_names):
    """Replace auto-generated column names with *column_names* when safe."""
    if column_names is not None and len(column_names) == len(df.columns):
        unnamed = all(
            isinstance(c, int)
            or (isinstance(c, str) and c.startswith("c"))
            for c in df.columns
        )
        if unnamed:
            df.columns = column_names
    return df


def _format_as_df(result, column_names=None):
    """Convert a query result to a pandas DataFrame, or None."""
    if result is None:
        return None
    if isinstance(result, bool):
        return pd.DataFrame({"result": ["true" if result else "false"]})
    if hasattr(result, "value") and hasattr(result.value, "unwrap"):
        inner = result.value.unwrap()
        if hasattr(inner, "as_pandas_dataframe"):
            return _rename_unnamed_columns(
                inner.as_pandas_dataframe(), column_names
            )
        result = inner
    if hasattr(result, "columns") and len(result.columns) > 0:
        df = pd.DataFrame(iter(result), columns=result.columns)
        return _rename_unnamed_columns(df, column_names)
    rows = list(iter(result))
    if not rows:
        return pd.DataFrame()
    arity = result.arity if hasattr(result, "arity") else len(rows[0])
    df = pd.DataFrame(rows, columns=[f"c{i}" for i in range(arity)])
    return _rename_unnamed_columns(df, column_names)


def _sort(df, sort_specs):
    """Sort *df* by the ``col[:asc|desc]`` *sort_specs* it has columns for."""
    sort_by = []
    for spec in sort_specs:
        col, _, direction = spec.partition(":")
        sort_by.append((col, direction.lower() != "desc"))
    valid = [(c, a) for c, a in sort_by if c in df.columns]
    if not valid:
        return df
    cols, asc = zip(*valid)
    return df.sort_values(by=list(cols), ascending=list(asc))


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
