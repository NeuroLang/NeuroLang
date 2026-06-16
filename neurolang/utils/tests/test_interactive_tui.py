"""Tests for the interactive TUI module."""

import numpy as np
import pandas as pd
import pytest

from neurolang.utils.interactive_tui import (
    _df_to_rich_table,
    _format_as_df,
    _rename_unnamed_columns,
    _result_to_nifti,
    InteractiveTuiApp,
    NeuroLangReplCompleter,
    DOT_COMMANDS,
    SQUALL_KEYWORDS,
)
from neurolang.utils.interactive_parsing import LarkCompleter
from neurolang.frontend.datalog.standard_syntax import COMPILED_GRAMMAR
from neurolang.datalog import WrappedRelationalAlgebraSet
from neurolang.utils.relational_algebra_set.pandas import (
    RelationalAlgebraSet,
    NamedRelationalAlgebraFrozenSet,
)
from neurolang import expressions as ir


# ---------------------------------------------------------------------------
# NeuroLangReplCompleter
# ---------------------------------------------------------------------------


class TestNeuroLangReplCompleter:
    @pytest.fixture
    def completer(self):
        try:
            lark = LarkCompleter(COMPILED_GRAMMAR)
        except Exception:
            lark = None
        return NeuroLangReplCompleter(lark_completer=lark)

    def test_dot_command_completions(self, completer):
        """Dot commands should complete partially typed commands."""
        # Mock a document-like object
        class FakeDoc:
            text_before_cursor = ".h"

        completions = list(completer.get_completions(FakeDoc(), None))
        assert len(completions) > 0
        # Should complete to .help
        assert any(c.text == ".help" for c in completions)

    def test_dot_command_partial_engines(self, completer):
        class FakeDoc:
            text_before_cursor = ".eng"

        completions = list(completer.get_completions(FakeDoc(), None))
        assert any(c.text == ".engines" for c in completions)

    def test_dot_command_empty_dot(self, completer):
        """Typing just '.' should suggest all dot commands."""
        class FakeDoc:
            text_before_cursor = "."

        completions = list(completer.get_completions(FakeDoc(), None))
        for cmd in DOT_COMMANDS:
            assert any(c.text == cmd for c in completions), (
                f"Missing dot command {cmd!r} in completions"
            )

    def test_non_dot_does_not_crash(self, completer):
        """Non-dot input should call the Lark completer without crashing."""
        class FakeDoc:
            text_before_cursor = "ans(x) :- term"

        # Just ensure no exception is raised
        completions = list(completer.get_completions(FakeDoc(), None))
        # May or may not return completions depending on Lark grammar state,
        # but should not crash
        assert isinstance(completions, list)

    def test_squall_keywords(self):
        """Completer yields SQUALL keywords when no lark_completer is set."""
        completer = NeuroLangReplCompleter(squall_keywords=SQUALL_KEYWORDS)

        class FakeDoc:
            text_before_cursor = "ob"

        completions = list(completer.get_completions(FakeDoc(), None))
        texts = [c.text for c in completions]
        assert "obtain" in texts

    def test_engine_predicates(self):
        """Completer yields engine predicate names."""
        completer = NeuroLangReplCompleter(
            engine_predicates={"term", "study", "peak"}
        )

        class FakeDoc:
            text_before_cursor = "st"

        completions = list(completer.get_completions(FakeDoc(), None))
        texts = [c.text for c in completions]
        assert "study" in texts
        assert "term" not in texts  # doesn't start with "st"

    def test_predicates_skipped_for_non_identifiers(self):
        """Predicate completion should not fire for non-identifier prefixes."""
        completer = NeuroLangReplCompleter(
            engine_predicates={"term", "study"}
        )

        class FakeDoc:
            text_before_cursor = "12"

        completions = list(completer.get_completions(FakeDoc(), None))
        texts = [c.text for c in completions]
        assert "term" not in texts
        assert "study" not in texts

    def test_mode_switch_rebuilds_completer(self):
        """InteractiveTuiApp._rebuild_completer switches between LALR and SQUALL."""
        app = InteractiveTuiApp(engine_name="neurosynth", squall_mode=False)
        assert app._completer is not None
        # Datalog mode should have a lark completer
        assert app._completer._lark is not None

        # Switch to SQUALL
        app._cmd_mode("squall")
        assert app.squall_mode is True
        # In SQUALL mode, lark should be None
        assert app._completer._lark is None

        # Switch back to Datalog
        app._cmd_mode("datalog")
        assert app.squall_mode is False
        assert app._completer._lark is not None


# ---------------------------------------------------------------------------
# _df_to_rich_table
# ---------------------------------------------------------------------------


class TestDfToRichTable:
    def test_basic_table(self):
        df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        table = _df_to_rich_table(df, title="Test")
        assert table.title == "Test"
        # Rich Table stores columns; check they exist
        assert len(table.columns) == 2
        assert table.columns[0].header == "x"
        assert table.columns[1].header == "y"

    def test_empty_dataframe(self):
        df = pd.DataFrame({"x": pd.Series([], dtype=int)})
        table = _df_to_rich_table(df)
        assert len(table.columns) == 1
        assert table.columns[0].header == "x"

    def test_single_row(self):
        df = pd.DataFrame({"col": [42]})
        table = _df_to_rich_table(df)
        assert len(table.columns) == 1
        assert table.columns[0].header == "col"

    def test_multiple_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2.5], "c": ["x"]})
        table = _df_to_rich_table(df)
        assert len(table.columns) == 3
        headers = [c.header for c in table.columns]
        assert headers == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# _rename_unnamed_columns
# ---------------------------------------------------------------------------


class TestRenameUnnamedColumns:
    def test_renames_int_columns(self):
        df = pd.DataFrame({0: [1, 2], 1: [3, 4]})
        result = _rename_unnamed_columns(df, ["x", "y"])
        assert list(result.columns) == ["x", "y"]

    def test_renames_c0_style_columns(self):
        df = pd.DataFrame({"c0": [1], "c1": [2]})
        result = _rename_unnamed_columns(df, ["a", "b"])
        assert list(result.columns) == ["a", "b"]

    def test_preserves_named_columns(self):
        df = pd.DataFrame({"name": [1], "value": [2]})
        result = _rename_unnamed_columns(df, ["x", "y"])
        assert list(result.columns) == ["name", "value"]

    def test_ignores_wrong_length(self):
        df = pd.DataFrame({0: [1]})
        result = _rename_unnamed_columns(df, ["a", "b", "c"])
        assert list(result.columns) == [0]

    def test_handles_mixed_named_and_unnamed(self):
        """When only some columns are named, rename is not applied."""
        df = pd.DataFrame({0: [1], "label": ["x"]})
        result = _rename_unnamed_columns(df, ["a", "b"])
        assert list(result.columns) == [0, "label"]

    def test_no_column_names_none(self):
        df = pd.DataFrame({0: [1]})
        result = _rename_unnamed_columns(df, None)
        assert list(result.columns) == [0]


# ---------------------------------------------------------------------------
# _format_as_df
# ---------------------------------------------------------------------------


class TestFormatAsDf:
    def test_none_returns_none(self):
        assert _format_as_df(None) is None

    def test_true_returns_df(self):
        df = _format_as_df(True)
        assert df is not None
        assert df.iloc[0, 0] == "true"

    def test_false_returns_df(self):
        df = _format_as_df(False)
        assert df is not None
        assert df.iloc[0, 0] == "false"

    def test_empty_result(self):
        """An empty RA set should produce an empty DataFrame."""
        ras = RelationalAlgebraSet()
        df = _format_as_df(ras)
        assert df is not None
        assert df.empty

    def test_nonempty_unnamed_set(self):
        ras = RelationalAlgebraSet()
        ras.add((1, "a"))
        ras.add((2, "b"))
        df = _format_as_df(ras)
        assert df is not None
        assert len(df) == 2
        # Unnamed sets get integer column indices from RangeIndex
        assert list(df.columns) == [0, 1]

    def test_named_set(self):
        nras = NamedRelationalAlgebraFrozenSet(
            columns=("x", "y"), iterable=[(1, "a")]
        )
        df = _format_as_df(nras)
        assert df is not None
        assert list(df.columns) == ["x", "y"]
        assert df.iloc[0, 0] == 1

    def test_wrapped_constant(self):
        """Test with a Constant wrapping a WrappedRelationalAlgebraSet."""
        inner = RelationalAlgebraSet()
        inner.add((42,))
        wrapped = WrappedRelationalAlgebraSet(inner)
        result = ir.Constant(wrapped)
        df = _format_as_df(result, column_names=["val"])
        assert df is not None
        assert "val" in df.columns
        assert df.iloc[0, 0] == 42


# ---------------------------------------------------------------------------
# _result_to_nifti
# ---------------------------------------------------------------------------


class TestResultToNifti:
    def test_basic_nifti_from_ij_columns(self):
        """Using (i, j, k) columns should produce a valid Nifti1Image."""
        df = pd.DataFrame({
            "i": [10, 20],
            "j": [30, 40],
            "k": [50, 60],
            "value": [1.0, 2.0],
        })
        img = _result_to_nifti(df)
        import nibabel as nib
        assert isinstance(img, nib.Nifti1Image)
        assert img.shape == (256, 256, 256)
        # Check values at coordinate positions
        data = img.get_fdata()
        assert data[10, 30, 50] == 1.0
        assert data[20, 40, 60] == 2.0

    def test_nifti_without_value_column(self):
        """Without a value column, all voxels should be 1.0."""
        df = pd.DataFrame({
            "i": [5],
            "j": [5],
            "k": [5],
        })
        img = _result_to_nifti(df)
        data = img.get_fdata()
        assert data[5, 5, 5] == 1.0

    def test_nifti_clips_out_of_bounds(self):
        """Coordinates outside volume should be silently ignored."""
        df = pd.DataFrame({
            "i": [1000],
            "j": [1000],
            "k": [1000],
            "value": [42.0],
        })
        img = _result_to_nifti(df)
        data = img.get_fdata()
        # Should not crash, value should be 0 (not set)
        assert data[255, 255, 255] == 0.0

    def test_nifti_raises_with_no_coords(self):
        """No coordinate columns should raise."""
        df = pd.DataFrame({"label": ["a"], "value": [1.0]})
        with pytest.raises(
            ValueError, match="Cannot determine voxel coordinates"
        ):
            _result_to_nifti(df)

    def test_nifti_uses_fallback_numeric_cols(self):
        """Fallback to first 3 numeric columns when no i/j/k found."""
        df = pd.DataFrame({
            "x": [0],
            "y": [0],
            "z": [0],
            "val": [5.0],
        })
        img = _result_to_nifti(df)
        data = img.get_fdata()
        assert data[0, 0, 0] == 5.0


# ---------------------------------------------------------------------------
# InteractiveTuiApp._get_prompt
# ---------------------------------------------------------------------------


class TestInteractiveTuiAppPrompt:
    def test_datalog_prompt(self):
        app = InteractiveTuiApp(engine_name="neurosynth", squall_mode=False)
        prompt = app._get_prompt()
        assert "DL" in prompt
        assert "neurosynth" in prompt

    def test_squall_prompt(self):
        app = InteractiveTuiApp(engine_name="destrieux", squall_mode=True)
        prompt = app._get_prompt()
        assert "SQ" in prompt
        assert "destrieux" in prompt

    def test_prompt_format(self):
        app = InteractiveTuiApp(engine_name="test_engine")
        prompt = app._get_prompt()
        assert prompt.endswith("> ")
        assert "nl(" in prompt
        assert ")" in prompt
