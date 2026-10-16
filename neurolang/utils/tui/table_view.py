"""
Full-screen viewer of a query result, opened with ``.browse``.

The viewer is a prompt_toolkit application over a
:class:`~.table_model.TableModel`: it draws only the rows of the window,
with the formatting of the result tables, and maps keys to the model.
"""

import sys
import threading

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (
    ConditionalContainer,
    Float,
    FloatContainer,
    HSplit,
    Layout,
    Window,
)
from prompt_toolkit.layout.controls import (
    BufferControl,
    FormattedTextControl,
)
from prompt_toolkit.layout.processors import BeforeInput
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame
from rich.cells import cell_len, set_cell_size

from .results import (
    BAR_WIDTH,
    MAX_COLUMN_WIDTH,
    format_value,
    probability_bar,
)
from .table_model import TableModel

HEADER_LINES = 3
STATUS_HINT = "? help · q quit"

STYLE = Style.from_dict({
    "header": "bold ansicyan",
    "type": "ansibrightblack",
    "rule": "ansibrightblack",
    "rownum": "ansibrightblack",
    "row.current": "bg:#303030",
    "cell.current": "reverse",
    "match": "bold ansiyellow",
    "bar": "ansicyan",
    "track": "ansibrightblack",
    "status": "reverse",
    "help": "bg:#202020",
})

HELP_TEXT = """\
↑ ↓ ← →  h j k l   move the cursor
PgUp PgDn  Ctrl+B Ctrl+F  page up / down
g Home  G End      first / last row
0 $                first / last column
s                  sort on the column: asc, desc, none
/                  search all columns (Enter keeps, Esc cancels)
n N                next / previous match
?                  show or hide this help
q Esc              back to the prompt"""

_ACTIONS = {
    ("up",): lambda model: model.move(rows=-1),
    ("k",): lambda model: model.move(rows=-1),
    ("down",): lambda model: model.move(rows=1),
    ("j",): lambda model: model.move(rows=1),
    ("left",): lambda model: model.move(cols=-1),
    ("h",): lambda model: model.move(cols=-1),
    ("right",): lambda model: model.move(cols=1),
    ("l",): lambda model: model.move(cols=1),
    ("pageup",): lambda model: model.page(-1),
    ("c-b",): lambda model: model.page(-1),
    ("pagedown",): lambda model: model.page(1),
    ("c-f",): lambda model: model.page(1),
    ("g",): lambda model: model.goto(row=0),
    ("home",): lambda model: model.goto(row=0),
    ("G",): lambda model: model.goto(row=-1),
    ("end",): lambda model: model.goto(row=-1),
    ("0",): lambda model: model.goto(col=0),
    ("$",): lambda model: model.goto(col=-1),
    ("s",): lambda model: model.cycle_sort(),
    ("n",): lambda model: model.next_match(),
    ("N",): lambda model: model.next_match(forward=False),
}


def fit_cell(text: str, width: int, right: bool) -> str:
    """
    Return *text* padded or cut with an ellipsis to *width* cells.

    It is right-aligned if *right*.
    """
    if cell_len(text) > width:
        return set_cell_size(text, width - 1) + "…"
    padding = " " * (width - cell_len(text))
    return padding + text if right else text + padding


def cell_fragments(
    text: str, value, width: int, style: str, right: bool, bar: bool,
):
    """
    Return the fragments of a cell of *width* cells showing *text*.

    The text is right-aligned if *right*, and a probability *value* gets
    its *bar* after it; *style* is the style of the whole cell.
    """
    if not bar or not text:
        return [(style, fit_cell(text, width, right))]
    filled = probability_bar(value)
    return [
        (style, fit_cell(text, width - BAR_WIDTH - 1, True) + " "),
        (style + " class:bar", filled),
        (style + " class:track", "·" * (BAR_WIDTH - len(filled))),
    ]


class TableGrid:
    """
    The texts of the window of a model, ready to be drawn.

    The width of each column is the widest seen so far in *widths*, so
    that columns do not jump while scrolling.
    """

    def __init__(self, model: TableModel, widths: dict):
        """Format the window of *model*, updating the column *widths*."""
        self.model = model
        window = model.window()
        self.values = [window.iloc[:, c].tolist() for c in range(model.n_cols)]
        self.texts = [[format_value(v) for v in col] for col in self.values]
        self.widths = [
            self._width(col, widths) for col in range(model.n_cols)
        ]
        self.right = [
            model.kind(col) in ("int", "float") for col in range(model.n_cols)
        ]

    def header(self, col: int) -> str:
        """Return the header of *col*, with its sort arrow if sorted."""
        sort = self.model.sort
        if sort is None or sort[0] != col:
            return self.model.name(col)
        return self.model.name(col) + ("↑" if sort[1] else "↓")

    def _width(self, col: int, widths: dict) -> int:
        """Return the width of *col* and record it in *widths*."""
        extra = BAR_WIDTH + 1 if self.model.is_probability(col) else 0
        width = max(
            [widths.get(col, 0), cell_len(self.header(col)),
             cell_len(self.model.kind(col))]
            + [cell_len(text) + extra for text in self.texts[col]]
        )
        widths[col] = min(width, MAX_COLUMN_WIDTH)
        return widths[col]


def _header_lines(grid: TableGrid, columns, number_width: int) -> list:
    """Return the fragments of the names, types and rule lines."""
    names = [("class:rownum", " " * number_width)]
    kinds = [("class:rownum", "#".rjust(number_width))]
    rule = [("class:rule", "─" * number_width)]
    for col in columns:
        right, width = grid.right[col], grid.widths[col]
        names += [("class:rule", " │ "),
                  ("class:header", fit_cell(grid.header(col), width, right))]
        kinds += [("class:rule", " │ "),
                  ("class:type", fit_cell(grid.model.kind(col), width, right))]
        rule.append(("class:rule", "─┼─" + "─" * width))
    return [names, kinds, rule]


def _row_line(grid: TableGrid, i: int, columns, number_width: int) -> list:
    """Return the fragments of the *i*-th row of the window."""
    model = grid.model
    row = model.top + i
    current = row == model.row
    line_style = "class:row.current" if current else ""
    fragments = [(line_style + " class:rownum",
                  str(row + 1).rjust(number_width))]
    query = model.query.lower()
    for col in columns:
        text = grid.texts[col][i]
        style = line_style
        if query and query in text.lower():
            style += " class:match"
        if current and col == model.col:
            style += " class:cell.current"
        fragments.append((line_style + " class:rule", " │ "))
        fragments += cell_fragments(
            text, grid.values[col][i], grid.widths[col], style,
            grid.right[col], model.is_probability(col),
        )
    return fragments


def table_fragments(model: TableModel, widths: dict, width: int) -> list:
    """
    Return the formatted text of the window of *model* in *width* cells.

    The row-number column stays on the left and the other columns
    scroll to keep the cursor column visible.
    """
    if not model.n_cols:
        return [("", "(no columns)")]
    grid = TableGrid(model, widths)
    number_width = len(str(max(model.n_rows, 1)))
    columns = model.visible_columns(grid.widths, width - number_width)
    lines = _header_lines(grid, columns, number_width)
    lines += [
        _row_line(grid, i, columns, number_width)
        for i in range(len(grid.texts[0]))
    ]
    fragments = []
    for line in lines:
        fragments += line + [("", "\n")]
    return fragments[:-1]


class TableViewer:
    """
    The full-screen viewer of a result frame.

    *input* and *output* are passed to the prompt_toolkit application,
    e.g. a pipe input and a dummy output in tests.
    """

    def __init__(self, frame, title: str = "", input=None, output=None):
        """Set up the viewer of *frame*, whose result is named *title*."""
        self.model = TableModel(frame)
        self.title = title
        self.searching = False
        self.show_help = False
        self._widths: dict = {}
        self._before_search = (0, 0, "")
        self.search_buffer = Buffer(
            multiline=False, on_text_changed=self._search_changed
        )
        self.app = Application(
            layout=self._layout(), key_bindings=self._bindings(),
            style=STYLE, full_screen=True, input=input, output=output,
        )
        self.app.ttimeoutlen = 0.05

    def _layout(self) -> Layout:
        """Return the layout: table, search line, status bar and help."""
        self._table = Window(
            FormattedTextControl(self._table_text, focusable=True),
            wrap_lines=False,
        )
        self._search = Window(
            BufferControl(self.search_buffer,
                          input_processors=[BeforeInput("/")]),
            height=1,
        )
        body = HSplit([
            self._table,
            ConditionalContainer(
                self._search, filter=Condition(lambda: self.searching)
            ),
            Window(FormattedTextControl(self._status_text), height=1,
                   style="class:status"),
        ])
        help_box = Frame(
            Window(FormattedTextControl(HELP_TEXT), style="class:help"),
            title="Keys",
        )
        return Layout(
            FloatContainer(body, floats=[Float(ConditionalContainer(
                help_box, filter=Condition(lambda: self.show_help)
            ))]),
            focused_element=self._table,
        )

    def _table_text(self) -> list:
        """Return the formatted window, sized to the terminal."""
        size = get_app().output.get_size()
        reserved = HEADER_LINES + 1 + (1 if self.searching else 0)
        self.model.set_height(size.rows - reserved)
        return table_fragments(self.model, self._widths, size.columns)

    def _status_text(self) -> str:
        """
        Return the status bar: title, model status and key hint.

        While searching, it tells if the search indexes are still being
        built.
        """
        indexing = self.searching and not self.model.search_ready
        parts = [
            self.title, self.model.status(),
            "indexing for search…" if indexing else "", STATUS_HINT,
        ]
        return " " + " · ".join(part for part in parts if part)

    def _bindings(self) -> KeyBindings:
        """Return the key bindings of the table, the search and the help."""
        bindings = KeyBindings()
        browsing = Condition(lambda: not (self.searching or self.show_help))
        for keys, action in _ACTIONS.items():
            bindings.add(*keys, filter=browsing)(self._handler(action))
        bindings.add("/", filter=browsing)(lambda event: self.start_search())
        bindings.add("?", filter=~Condition(lambda: self.searching))(
            lambda event: self.toggle_help()
        )
        bindings.add("q", filter=~Condition(lambda: self.searching))(
            lambda event: self.escape()
        )
        bindings.add("escape", eager=True)(lambda event: self.escape())
        bindings.add("c-c")(lambda event: event.app.exit())
        bindings.add("enter", filter=Condition(lambda: self.searching))(
            lambda event: self.end_search()
        )
        return bindings

    def _handler(self, action):
        """Return a key handler applying *action* to the model."""
        def handle(event):
            action(self.model)
        return handle

    def toggle_help(self) -> None:
        """Show or hide the help overlay."""
        self.show_help = not self.show_help

    def escape(self) -> None:
        """Close the help, cancel the search, or leave the viewer."""
        if self.show_help:
            self.show_help = False
        elif self.searching:
            self.cancel_search()
        else:
            self.app.exit()

    def start_search(self) -> None:
        """Open the search line, remembering the cursor to come back to."""
        model = self.model
        self._before_search = (model.row, model.col, model.query)
        self.searching = True
        self.search_buffer.reset()
        self.app.layout.focus(self._search)

    def _search_changed(self, buffer: Buffer) -> None:
        """Search the typed text, from the row where the search started."""
        row, col, _ = self._before_search
        self.model.search(buffer.text)
        if not self.model.jump_from(row):
            self.model.row, self.model.col = row, col

    def end_search(self) -> None:
        """Close the search line, keeping the search and the cursor."""
        self.searching = False
        self.app.layout.focus(self._table)

    def cancel_search(self) -> None:
        """Close the search line, restoring the cursor and the last search."""
        row, col, query = self._before_search
        self.end_search()
        self.model.search(query)
        self.model.row, self.model.col = row, col
        self.model.move()

    def run(self) -> None:
        """
        Run the viewer until it is closed.

        The text indexes of the search are built meanwhile in a
        background thread.
        """
        threading.Thread(target=self._prepare_search, daemon=True).start()
        self.app.run()

    def _prepare_search(self) -> None:
        """Build the search indexes, then redraw the status."""
        self.model.prepare_search()
        self.app.invalidate()


def can_browse(console) -> bool:
    """Return whether the viewer can run: input and output are terminals."""
    return console.is_terminal and sys.stdin.isatty()


def browse(frame, title: str = "", **kwargs) -> None:
    """Browse *frame* in the full-screen viewer until it is closed."""
    TableViewer(frame, title, **kwargs).run()
