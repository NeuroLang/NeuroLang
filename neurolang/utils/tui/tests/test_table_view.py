"""Tests for the full-screen table viewer and the .browse command."""

import io

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

from prompt_toolkit.input import create_pipe_input  # noqa: E402
from prompt_toolkit.output import DummyOutput  # noqa: E402
from rich.console import Console  # noqa: E402

from neurolang.frontend import NeurolangPDL  # noqa: E402
from neurolang.utils.tui import InteractiveTuiApp, TuiSettings  # noqa: E402
from neurolang.utils.tui import commands, table_view  # noqa: E402
from neurolang.utils.tui.results import ResultBlock  # noqa: E402
from neurolang.utils.tui.table_model import TableModel  # noqa: E402
from neurolang.utils.tui.table_view import (  # noqa: E402
    TableViewer,
    cell_fragments,
    fit_cell,
    table_fragments,
)

ESCAPE = "\x1b"


@pytest.fixture
def frame():
    """Return 100 rows of studies, terms and probabilities."""
    terms = ["pain", "memory", "faces", "motor"]
    return pd.DataFrame({
        "study": np.arange(100, 0, -1),
        "term": [terms[i % 4] for i in range(100)],
        "PROB": np.linspace(0, 1, 100),
    })


def run_viewer(frame, keys, title=""):
    """Run a viewer of *frame* on the typed *keys* and return it."""
    with create_pipe_input() as pipe:
        viewer = TableViewer(frame, title, input=pipe, output=DummyOutput())
        pipe.send_text(keys)
        viewer.run()
    return viewer


def _text(fragments):
    return "".join(text for _, text in fragments)


@pytest.mark.parametrize("text, width, right, expected", [
    ("ab", 4, False, "ab  "),
    ("ab", 4, True, "  ab"),
    ("abcdef", 4, False, "abc…"),
    ("", 2, True, "  "),
])
def test_fit_cell(text, width, right, expected):
    assert fit_cell(text, width, right) == expected


def test_cell_fragments_with_bar():
    fragments = cell_fragments("0.5", 0.5, 10, "", True, True)
    assert _text(fragments) == " 0.5 ██▌··"
    assert fragments[1] == (" class:bar", "██▌")


def test_cell_fragments_missing_probability():
    assert _text(cell_fragments("", np.nan, 6, "", True, True)) == " " * 6


def test_table_fragments_draws_window_with_frozen_row_numbers(frame):
    model = TableModel(frame, height=3)
    lines = _text(table_fragments(model, {}, 80)).splitlines()
    assert lines[0] == "    │ study │ term   │         PROB"
    assert lines[1].split() == ["#", "│", "int", "│", "str", "│", "float"]
    assert lines[3] == "  1 │   100 │ pain   │      0 ·····"
    assert lines[5] == "  3 │    98 │ faces  │ 0.0202 ▏····"
    assert len(lines) == 6


def test_table_fragments_scroll_columns_to_cursor(frame):
    model = TableModel(frame, height=2)
    model.goto(col=2)
    lines = _text(table_fragments(model, {}, 22)).splitlines()
    assert "PROB" in lines[0] and "study" not in lines[0]
    assert lines[3].startswith("  1 │ ")


def test_current_cell_and_matches_are_styled(frame):
    model = TableModel(frame, height=3)
    model.search("mem")
    model.next_match()
    styles = {text.strip(): style
              for style, text in table_fragments(model, {}, 80)}
    assert "class:cell.current" in styles["memory"]
    assert "class:match" in styles["memory"]
    assert "class:row.current" in styles["99"]


def test_widths_never_shrink_while_scrolling():
    frame = pd.DataFrame({"t": ["a" * 30] + ["b"] * 9})
    model = TableModel(frame, height=2)
    widths = {}
    table_fragments(model, widths, 80)
    model.goto(row=-1)
    lines = _text(table_fragments(model, widths, 80)).splitlines()
    assert lines[-1] == "10 │ b" + " " * 29
    assert widths == {0: 30}


def test_q_exits_unchanged(frame):
    viewer = run_viewer(frame, "q")
    assert (viewer.model.row, viewer.model.col) == (0, 0)


def test_escape_exits(frame):
    viewer = run_viewer(frame, "jj" + ESCAPE)
    assert viewer.model.row == 2


def test_navigation_keys(frame):
    viewer = run_viewer(frame, "jjjkl" + "\x1b[B" + "\x1b[C" + "q")
    assert (viewer.model.row, viewer.model.col) == (3, 2)


@pytest.mark.parametrize("keys, row", [
    ("G", 99), ("Gg", 0), ("\x1b[6~", 36), ("\x1b[6~\x1b[5~", 0),
    ("\x06\x06", 72), ("\x1b[F", 99), ("\x1b[F\x1b[H", 0),
])
def test_paging_keys(frame, keys, row):
    viewer = run_viewer(frame, keys + "q")
    assert viewer.model.row == row


def test_first_and_last_column(frame):
    viewer = run_viewer(frame, "$q")
    assert viewer.model.col == 2
    assert run_viewer(frame, "$0q").model.col == 0


def test_window_fits_terminal(frame):
    viewer = run_viewer(frame, "q")
    assert viewer.model.height == 40 - 4


def test_s_cycles_sort(frame):
    viewer = run_viewer(frame, "sq")
    assert viewer.model.sort == (0, True)
    assert viewer.model.window()["study"].iloc[0] == 1
    assert run_viewer(frame, "ssq").model.sort == (0, False)
    assert run_viewer(frame, "sssq").model.sort is None


def test_search_jumps_and_n_wraps(frame):
    viewer = run_viewer(frame, "/MEM\rnq")
    model = viewer.model
    assert model.query == "MEM"
    assert len(model.matches) == 25
    assert (model.row, model.col) == (5, 1)
    assert "/MEM match 2/25" in model.status()
    assert not viewer.searching
    assert run_viewer(frame, "/mem\rNq").model.row == 97


def test_search_is_incremental(frame):
    viewer = run_viewer(frame, "/fa\x7f\x7fmo" + ESCAPE + "q")
    assert viewer.model.query == ""


def test_escape_cancels_search_and_restores_cursor(frame):
    viewer = run_viewer(frame, "jj/motor" + ESCAPE + "q")
    assert viewer.model.query == ""
    assert viewer.model.row == 2


def test_keys_are_typed_in_search(frame):
    viewer = run_viewer(frame, "/qs?\rq")
    assert viewer.model.query == "qs?"
    assert viewer.model.sort is None


def test_help_overlay(frame):
    viewer = run_viewer(frame, "?jqq")
    assert viewer.model.row == 0
    assert viewer.show_help is False
    viewer = run_viewer(frame, "?" + ESCAPE + "jq")
    assert viewer.model.row == 1
    viewer = run_viewer(frame, "??jq")
    assert viewer.model.row == 1


def test_status_bar(frame):
    viewer = run_viewer(frame, "lsq", title="ans")
    assert viewer._status_text() == (
        " ans · rows 1–36 of 100 · col 2/3 term (str) · sort term↑"
        " · ? help · q quit"
    )


def _repl(results):
    console = Console(file=io.StringIO(), width=100)
    app = InteractiveTuiApp(NeurolangPDL(), TuiSettings(), console=console,
                            err_console=console)
    app.results = results
    return app


def test_browse_without_terminal_prints_all(frame):
    app = _repl([ResultBlock(frame)])
    app._handle_dot_command(".browse")
    out = app.console.file.getvalue()
    assert ".browse needs a terminal" in out
    assert "│ motor  │" in out
    assert app.results[0].remaining == 0


def test_browse_opens_viewer_on_terminal(frame, monkeypatch):
    browsed = []
    monkeypatch.setattr(commands, "can_browse", lambda console: True)
    monkeypatch.setattr(
        commands, "browse", lambda df, title: browsed.append((df, title))
    )
    blocks = [ResultBlock(frame, "first"), ResultBlock(frame[:2], "second")]
    app = _repl(blocks)
    app._handle_dot_command(".browse")
    app._handle_dot_command(".browse 2")
    assert [(len(df), title) for df, title in browsed] == [
        (100, "first"), (2, "second")
    ]


@pytest.mark.parametrize("arg", ["3", "0", "x"])
def test_browse_wrong_number(frame, arg):
    app = _repl([ResultBlock(frame)])
    app._handle_dot_command(f".browse {arg}")
    assert "Give a result number from 1 to 1" in app.console.file.getvalue()


def test_browse_without_result():
    app = _repl([])
    app._handle_dot_command(".browse")
    assert "No result to browse." in app.console.file.getvalue()


def test_help_lists_browse():
    assert ".browse [N]" in commands.help_text()


def test_can_browse_needs_terminals(monkeypatch):
    console = Console(file=io.StringIO(), force_terminal=True)
    monkeypatch.setattr(table_view.sys, "stdin", io.StringIO())
    assert not table_view.can_browse(console)


def test_status_tells_search_indexing(frame):
    viewer = TableViewer(frame, output=DummyOutput())
    viewer.searching = True
    assert "indexing for search…" in viewer._status_text()
    viewer.model.prepare_search()
    assert "indexing" not in viewer._status_text()
