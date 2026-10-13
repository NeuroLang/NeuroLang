"""Tests for the core of the interactive REPL."""

import pandas as pd
import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

from neurolang import expressions as ir  # noqa: E402
from neurolang.datalog import WrappedRelationalAlgebraSet  # noqa: E402
from neurolang.utils.relational_algebra_set.pandas import (  # noqa: E402
    NamedRelationalAlgebraFrozenSet,
    RelationalAlgebraSet,
)
from neurolang.utils.tui import InteractiveTuiApp  # noqa: E402
from neurolang.utils.tui.results import (  # noqa: E402
    _df_to_rich_table,
    _format_as_df,
    _rename_unnamed_columns,
)


class TestDfToRichTable:
    def test_basic_table(self):
        df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        table = _df_to_rich_table(df, title="Test")
        assert table.title == "Test"
        assert [c.header for c in table.columns] == ["x", "y"]

    def test_empty_dataframe(self):
        df = pd.DataFrame({"x": pd.Series([], dtype=int)})
        table = _df_to_rich_table(df)
        assert [c.header for c in table.columns] == ["x"]

    def test_multiple_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2.5], "c": ["x"]})
        table = _df_to_rich_table(df)
        assert [c.header for c in table.columns] == ["a", "b", "c"]


class TestRenameUnnamedColumns:
    def test_renames_int_columns(self):
        df = pd.DataFrame({0: [1, 2], 1: [3, 4]})
        result = _rename_unnamed_columns(df, ["x", "y"])
        assert list(result.columns) == ["x", "y"]

    def test_preserves_named_columns(self):
        df = pd.DataFrame({"name": [1], "value": [2]})
        result = _rename_unnamed_columns(df, ["x", "y"])
        assert list(result.columns) == ["name", "value"]

    def test_ignores_wrong_length(self):
        df = pd.DataFrame({0: [1]})
        result = _rename_unnamed_columns(df, ["a", "b", "c"])
        assert list(result.columns) == [0]


class TestFormatAsDf:
    def test_none_returns_none(self):
        assert _format_as_df(None) is None

    def test_true_returns_df(self):
        assert _format_as_df(True).iloc[0, 0] == "true"

    def test_empty_result(self):
        assert _format_as_df(RelationalAlgebraSet()).empty

    def test_named_set(self):
        nras = NamedRelationalAlgebraFrozenSet(
            columns=("x", "y"), iterable=[(1, "a")]
        )
        df = _format_as_df(nras)
        assert list(df.columns) == ["x", "y"]
        assert df.iloc[0, 0] == 1

    def test_wrapped_constant(self):
        inner = RelationalAlgebraSet()
        inner.add((42,))
        result = ir.Constant(WrappedRelationalAlgebraSet(inner))
        df = _format_as_df(result, column_names=["val"])
        assert df.iloc[0, 0] == 42
        assert "val" in df.columns


class TestInteractiveTuiApp:
    def test_datalog_prompt(self, tmp_path):
        app = InteractiveTuiApp(
            engine_name="neurosynth", history_path=str(tmp_path / "h")
        )
        assert app._get_prompt() == "nl(neurosynth:DL)> "

    def test_squall_prompt(self, tmp_path):
        app = InteractiveTuiApp(
            engine_name="destrieux", squall_mode=True,
            history_path=str(tmp_path / "h"),
        )
        assert app._get_prompt() == "nl(destrieux:SQ)> "

    def test_mode_switch(self, tmp_path):
        app = InteractiveTuiApp(history_path=str(tmp_path / "h"))
        app._handle_dot_command(".mode squall")
        assert app.squall_mode
        app._handle_dot_command(".mode datalog")
        assert not app.squall_mode
