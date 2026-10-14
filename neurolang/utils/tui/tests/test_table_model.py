"""Tests for the state of the interactive table viewer."""

import numpy as np
import pandas as pd
import pytest

from neurolang.utils.tui.table_model import (
    TableModel,
    clamp,
    columns_from,
    sort_order,
    text_index,
)


@pytest.fixture
def frame():
    """Return 10 rows of studies, terms and weights."""
    return pd.DataFrame({
        "study": [5, 3, 9, 1, 7, 3, 8, 2, 6, 4],
        "term": ["pain", "memory", "Pain relief", "vision", "chronic pain",
                 "memory", "faces", "pain", "motor", "reward"],
        "w": [0.5, 0.25, np.nan, 0.125, 0.75, 0.3, 0.9, 0.05, 0.6, 0.45],
    })


@pytest.fixture
def model(frame):
    """Return a model of *frame* with a window of 4 rows."""
    return TableModel(frame, height=4)


@pytest.mark.parametrize("value, low, high, expected", [
    (5, 0, 9, 5), (-1, 0, 9, 0), (12, 0, 9, 9), (3, 0, -1, 0),
])
def test_clamp(value, low, high, expected):
    assert clamp(value, low, high) == expected


@pytest.mark.parametrize("values, ascending, expected", [
    ([3, 1, 2, 1], True, [1, 3, 2, 0]),
    ([3, 1, 2, 1], False, [0, 2, 1, 3]),
    ([0.5, np.nan, 0.1], True, [2, 0, 1]),
    ([0.5, np.nan, 0.1], False, [0, 2, 1]),
    (["b", None, "a", "b"], True, [2, 0, 3, 1]),
    (["b", None, "a", "b"], False, [0, 3, 2, 1]),
    ([1, "a", 2.5], True, [0, 2, 1]),
    (np.array([3, 1], dtype=np.uint8), False, [0, 1]),
    ([], True, []),
])
def test_sort_order(values, ascending, expected):
    series = pd.Series(values, dtype=None if len(values) else float)
    assert list(sort_order(series, ascending)) == expected


def test_text_index_matches_shown_texts():
    index = text_index(pd.Series([0.30000000000000004, 0.3, None, 12.0]))
    assert list(index.contains("0.3")) == [True, True, False, False]
    assert list(index.contains("12")) == [False, False, False, True]
    assert list(index.contains("0000")) == [False] * 4


def test_text_index_ignores_case_of_values():
    index = text_index(pd.Series(["Pain", "PAIN relief", "memory"]))
    assert list(index.contains("pain")) == [True, True, False]


def test_text_index_of_unhashable_values():
    index = text_index(pd.Series([[1, 2], [3]], dtype=object))
    assert list(index.contains("3")) == [False, True]


@pytest.mark.parametrize("widths, left, available, expected", [
    ([5, 5, 5], 0, 100, [0, 1, 2]),
    ([5, 5, 5], 0, 16, [0, 1]),
    ([5, 5, 5], 1, 16, [1, 2]),
    ([40, 5], 0, 10, [0]),
    ([], 0, 10, []),
])
def test_columns_from(widths, left, available, expected):
    assert columns_from(widths, left, available) == expected


class TestCursor:
    def test_moves_are_clamped(self, model):
        model.move(rows=-3, cols=-1)
        assert (model.row, model.col) == (0, 0)
        model.move(rows=50, cols=7)
        assert (model.row, model.col) == (9, 2)

    def test_window_follows_cursor(self, model):
        model.move(rows=5)
        assert (model.row, model.top) == (5, 2)
        model.move(rows=-4)
        assert (model.row, model.top) == (1, 1)
        assert list(model.window()["study"]) == [3, 9, 1, 7]

    def test_pages(self, model):
        model.page(1)
        assert (model.row, model.top) == (4, 4)
        model.page(1)
        assert (model.row, model.top) == (8, 6)
        model.page(-1)
        assert (model.row, model.top) == (4, 2)
        model.page(-5)
        assert (model.row, model.top) == (0, 0)

    def test_goto_counts_negative_from_the_end(self, model):
        model.goto(row=-1, col=-1)
        assert (model.row, model.col, model.top) == (9, 2, 6)
        model.goto(row=0)
        assert (model.row, model.top) == (0, 0)

    def test_set_height_keeps_cursor_visible(self, model):
        model.goto(row=9)
        model.set_height(2)
        assert model.top == 8
        model.set_height(20)
        assert model.top == 0
        model.set_height(0)
        assert model.height == 1

    def test_empty_frame(self):
        model = TableModel(pd.DataFrame({"x": pd.Series([], dtype=int)}))
        model.move(rows=3, cols=2)
        model.page(1)
        assert (model.row, model.col, model.top) == (0, 0, 0)
        assert model.window().empty
        assert model.status() == "no rows · col 1/1 x (int)"

    def test_visible_columns_scroll_to_cursor(self, model):
        widths = [10, 10, 10]
        assert model.visible_columns(widths, 30) == [0, 1]
        model.goto(col=2)
        assert model.visible_columns(widths, 30) == [1, 2]
        assert model.left == 1
        model.goto(col=0)
        assert model.visible_columns(widths, 30) == [0, 1]
        assert model.left == 0


class TestSort:
    def test_cycle_asc_desc_none(self, model):
        model.goto(row=5)
        model.cycle_sort()
        assert model.sort == (0, True)
        assert model.row == 0
        assert list(model.window()["study"]) == [1, 2, 3, 3]
        model.cycle_sort()
        assert model.sort == (0, False)
        assert list(model.window()["study"]) == [9, 8, 7, 6]
        model.cycle_sort()
        assert model.sort is None
        assert list(model.window()["study"]) == [5, 3, 9, 1]

    def test_other_column_starts_ascending(self, model):
        model.cycle_sort()
        model.move(cols=2)
        model.cycle_sort()
        assert model.sort == (2, True)
        model.goto(row=-1)
        assert model.window()["term"].iloc[-1] == "Pain relief"

    def test_sorted_orders_are_cached(self, model):
        model.cycle_sort()
        order = model.order
        model.cycle_sort()
        model.cycle_sort()
        model.cycle_sort()
        assert model.order is order

    def test_status_shows_sort(self, model):
        model.move(cols=2)
        model.cycle_sort()
        model.cycle_sort()
        assert "sort w↓" in model.status()


class TestSearch:
    def test_search_ignores_case_across_columns(self, model):
        model.search("PAIN")
        assert list(model.matches) == [0, 2, 4, 7]
        model.search("0.3")
        assert list(model.matches) == [5]

    def test_next_and_previous_wrap_around(self, model):
        model.search("pain")
        rows = []
        for _ in range(5):
            model.next_match()
            rows.append(model.row)
        assert rows == [2, 4, 7, 0, 2]
        model.next_match(forward=False)
        model.next_match(forward=False)
        assert model.row == 7

    def test_match_goes_to_matching_column(self, model):
        model.search("faces")
        model.next_match()
        assert (model.row, model.col) == (6, 1)
        model.search("6")
        model.next_match()
        assert (model.row, model.col) == (8, 0)

    def test_match_keeps_cursor_column_if_it_matches(self, model):
        model.goto(col=2)
        model.search("5")
        model.next_match()
        assert (model.row, model.col) == (1, 2)

    def test_jump_from_is_inclusive(self, model):
        model.search("memory")
        assert model.jump_from(1)
        assert model.row == 1
        assert model.jump_from(6)
        assert model.row == 1

    def test_no_match(self, model):
        model.goto(row=3)
        model.search("zebra")
        assert not model.next_match()
        assert not model.jump_from(0)
        assert model.row == 3
        assert model.status().endswith("/zebra no match")

    def test_empty_search_clears(self, model):
        model.search("pain")
        model.search("")
        assert len(model.matches) == 0
        assert model.status() == "rows 1–4 of 10 · col 1/3 study (int)"

    def test_matches_follow_sort(self, model):
        model.search("pain")
        model.cycle_sort()
        assert list(model.matches) == [1, 5, 7, 9]
        assert list(model.window()["study"])[1] == 2

    def test_status_counts_matches(self, model):
        model.goto(row=1)
        model.search("pain")
        assert model.status().endswith("/pain 4 matches")
        model.next_match()
        assert model.status() == (
            "rows 1–4 of 10 · col 2/3 term (str) · /pain match 2/4"
        )

    def test_masks_are_cached(self, model):
        masks = model.column_masks("pain")
        assert model.column_masks("PAIN") is masks

    def test_prepare_search_builds_indexes(self, model):
        model.prepare_search()
        assert sorted(model._texts) == [0, 1, 2]


def test_kind_and_probability_are_per_column():
    model = TableModel(pd.DataFrame({"t": ["a"], "PROB": [0.5]}))
    assert [model.kind(0), model.kind(1)] == ["str", "float"]
    assert [model.is_probability(0), model.is_probability(1)] == [
        False, True
    ]
