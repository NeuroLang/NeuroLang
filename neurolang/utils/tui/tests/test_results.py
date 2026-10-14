"""Tests for the display of query results as tables."""

import io

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rich")

from rich.console import Console  # noqa: E402

from neurolang.utils.tui.results import (  # noqa: E402
    MAX_COLUMN_WIDTH,
    NARROW_COLUMN_WIDTH,
    ResultBlock,
    column_kind,
    fit_widths,
    format_value,
    hidden_columns_note,
    is_probability,
    page_caption,
    print_all,
    print_next_page,
    probability_bar,
    render_rows,
)


def _print_page(df, width=80, terminal=False, page_size=20, shown=0):
    console = Console(
        file=io.StringIO(), width=width, force_terminal=terminal,
        color_system=None,
    )
    block = ResultBlock(df, shown=shown)
    print_next_page(console, block, page_size)
    return console.file.getvalue()


@pytest.mark.parametrize("value, text", [
    (0.30000000000000004, "0.3"),
    (0.0123456, "0.01235"),
    (2 / 3, "0.6667"),
    (1e-7, "1e-07"),
    (12345.678, "12346"),
    (-42.5, "-42.5"),
    (3.0, "3"),
    (1e20, "1e+20"),
    (float("nan"), ""),
    (None, ""),
    (7, "7"),
    (10022492, "10022492"),
    ("pain", "pain"),
    (True, "True"),
])
def test_format_value(value, text):
    assert format_value(value) == text


@pytest.mark.parametrize("values, kind", [
    ([1, 2], "int"),
    ([1.5, 2], "float"),
    ([True, False], "bool"),
    (["a", "b"], "str"),
    (pd.Series(["a", None], dtype=object), "str"),
    (pd.Series([1, 2], dtype=object), "int"),
    (pd.Series([1, "a"], dtype=object), "object"),
    (pd.Series([], dtype=float), "float"),
    (pd.Series([1], dtype="int64[pyarrow]"), "int"),
])
def test_column_kind(values, kind):
    assert column_kind(pd.Series(values)) == kind


@pytest.mark.parametrize("value, bar", [
    (0, ""),
    (1, "█████"),
    (0.5, "██▌"),
    (2 / 3, "███▍"),
    (0.01, ""),
    (0.03, "▏"),
])
def test_probability_bar(value, bar):
    assert probability_bar(value) == bar


@pytest.mark.parametrize("name, values, expected", [
    ("PROB", [0.2, 1.0], True),
    ("probability", [0, 1], True),
    ("prob", [0.2, np.nan], True),
    ("PROB", [0.2, 1.5], False),
    ("PROB", [-0.1, 0.5], False),
    ("PROB", [np.nan], False),
    ("tfidf", [0.2, 0.3], False),
])
def test_is_probability(name, values, expected):
    series = pd.Series(values)
    assert is_probability(name, series, column_kind(series)) is expected


def test_is_probability_needs_numbers():
    series = pd.Series(["0.2"])
    assert not is_probability("prob", series, column_kind(series))


def test_numbers_are_right_aligned_and_text_left_aligned():
    df = pd.DataFrame({"n": [1, 100], "t": ["a", "bbb"]})
    out = _print_page(df)
    assert "│   1 │ a   │" in out
    assert "│ 100 │ bbb │" in out


def test_header_shows_type_under_name():
    df = pd.DataFrame({"study": [1], "w": [0.5], "term": ["x"]})
    lines = _print_page(df).splitlines()
    assert lines[1] == "│ study │     w │ term │"
    assert lines[2] == "│   int │ float │ str  │"


def test_plain_output_has_no_row_numbers_nor_bars():
    df = pd.DataFrame({"PROB": [0.5, 1.0]})
    out = _print_page(df)
    assert "#" not in out
    assert "█" not in out and "▌" not in out
    assert out == _print_page(df)


def test_terminal_output_numbers_rows_across_pages():
    df = pd.DataFrame({"x": list("abcdefghij")})
    out = _print_page(df, terminal=True, page_size=3, shown=6)
    rows = [line for line in out.splitlines() if "│" in line][2:]
    assert rows == ["│ 7 │ g   │", "│ 8 │ h   │", "│ 9 │ i   │"]
    assert "│ #" in out


def test_terminal_output_draws_probability_bars():
    df = pd.DataFrame({"x": [1, 2], "PROB": [0.5, 1.0]})
    out = _print_page(df, terminal=True)
    assert "│ 0.5 ██▌·· │" in out
    assert "│   1 █████ │" in out


def test_bars_only_for_probability_columns():
    df = pd.DataFrame({"tfidf": [0.5, 1.0]})
    assert "█" not in _print_page(df, terminal=True)


def test_long_values_are_cut_with_ellipsis_on_one_line():
    df = pd.DataFrame({"t": ["word " * 30, "short"]})
    out = _print_page(df, width=200)
    lines = [line for line in out.splitlines() if "word" in line]
    assert len(lines) == 1
    assert lines[0].endswith("… │")
    assert len(lines[0]) == MAX_COLUMN_WIDTH + 4


def test_wide_columns_are_narrowed_to_fit():
    df = pd.DataFrame({"a": ["x" * 40], "b": ["y" * 40]})
    out = _print_page(df, width=60)
    assert all(len(line) <= 60 for line in out.splitlines())
    assert "+" not in out
    assert "xxx…" in out and "yyy…" in out


def test_columns_not_fitting_are_counted_in_caption():
    df = pd.DataFrame({f"c{i}": ["v" * 20] for i in range(10)})
    out = _print_page(df, width=80)
    lines = out.splitlines()
    assert all(len(line) <= 80 for line in lines)
    assert lines[1].split() == ["│", "c0", "│", "c1", "│", "c2", "│", "c3",
                                "│"]
    assert lines[-2:] == ["+6 columns · .browse to see all", "1 row"]


def test_first_column_is_always_shown():
    df = pd.DataFrame({"a": ["x" * 40], "b": [1]})
    table, hidden = render_rows(df, width=12)
    assert [str(c.header) for c in table.columns] == ["a\nstr"]
    assert hidden == 1


def test_render_rows_without_width_shows_every_column():
    df = pd.DataFrame({f"c{i}": ["v" * 40] for i in range(10)})
    table, hidden = render_rows(df)
    assert hidden == 0
    assert len(table.columns) == 10


@pytest.mark.parametrize("widths, available, fitted", [
    ([5, 5], 100, [5, 5]),
    ([40, 5], 40, [28, 5]),
    ([40, 40, 40], 70, [20, 20, 20]),
    ([40, 40, 40, 40], 70, [NARROW_COLUMN_WIDTH] * 3),
    ([40], 10, [NARROW_COLUMN_WIDTH]),
    ([], 10, []),
])
def test_fit_widths(widths, available, fitted):
    assert fit_widths(widths, available) == fitted


@pytest.mark.parametrize("hidden, note", [
    (0, ""),
    (1, "+1 column · .browse to see all"),
    (3, "+3 columns · .browse to see all"),
])
def test_hidden_columns_note(hidden, note):
    assert hidden_columns_note(hidden) == note


def test_print_all_notes_hidden_columns():
    df = pd.DataFrame({f"c{i}": ["v" * 20] * 30 for i in range(10)})
    console = Console(file=io.StringIO(), width=80)
    block = ResultBlock(df)
    print_all(console, block)
    out = console.file.getvalue()
    assert out.count("v" * 15 + "…") == 30 * 4
    assert out.splitlines()[-1] == "+6 columns · .browse to see all"
    assert block.remaining == 0


@pytest.mark.parametrize("args, caption", [
    ((1, 0, 1), "1 row"),
    ((45, 0, 45), "45 rows"),
    ((45, 20, 40),
     "45 rows · showing 21–40 · .more for next · .browse to explore"),
    ((45, 40, 45), "45 rows · showing 41–45 · end"),
])
def test_page_caption(args, caption):
    assert page_caption(*args) == caption
