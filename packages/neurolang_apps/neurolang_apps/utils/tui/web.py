"""
The interactive REPL driven from a web page.

:class:`WebRepl` runs what a page sends as the REPL runs what is typed
at its prompt, with the same commands and the same output: each
submission's output is rendered as it is on a terminal, by ``rich``, and
returned as HTML for the page to show. It is what the in-browser REPL
(``doc/_static/repl``) runs, on Pyodide, but works anywhere.

A submission's ``.view`` pages are returned, to be shown by the page,
rather than saved in a file and opened in a browser. So are its results,
as data (:func:`table_payload`) for the page to show as a table to sort,
filter and page through, in place of the text table of the terminal.

The LLM assistant (``.ask``, ``.llm``) works with the visitor's own API
key: the page gives it to :meth:`WebRepl.set_llm`, which keeps it in the
memory of the session, and the worker sends it to the provider and to
no one else.
"""

import io
import json
import math
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from unittest import mock

import pandas as pd
from rich.console import Console
from rich.text import Text

from . import nifti
from .app import InteractiveTuiApp, TuiSettings
from .commands import QuitRepl
from .results import ResultBlock, cell_text

# The most rows of a result sent to the page.
MAX_ROWS = 200_000

# What the output is wrapped in: a block as wide as the terminal it
# shows, in the page's monospace font.
HTML_FORMAT = (
    '<pre class="repl-output" style="font-family: inherit">'
    "<code>{code}</code></pre>"
)

# The precomputed analyses of the meta-analysis engines: ``activated``
# materialises every voxel within 3 mm of a reported peak, which takes
# gigabytes, more than a browser tab is given, and what is defined from it.
HEAVY_PREDICATES = frozenset({
    "activated", "p_a", "p_a_given_t", "p_any_voxel_given_t", "lift",
    "relative_risk", "uniformity", "ratios", "p_a_given_similar",
    "p_any_voxel_given_similar", "lift_llm", "relative_risk_llm",
    "uniformity_llm", "ratios_llm",
})


def lightweight_engines_yaml(directory) -> Path:
    """
    Write in *directory* the engines' YAML without the heavy analyses.

    The rules defining :data:`HEAVY_PREDICATES` and their declarations
    are left out of the shared configuration of the meta-analysis
    engines; the file is the one to give ``build_engine`` as
    ``yaml_path`` to fit a browser tab. Return its path.
    """
    import yaml

    from neurolang_apps.utils import engine_registry

    config = engine_registry._load_yaml_config()
    for engine in config["engines"].values():
        if "datalog_init" in engine:
            engine["datalog_init"] = "\n".join(
                rule for rule in engine["datalog_init"].split("\n")
                if rule.split("(")[0].strip() not in HEAVY_PREDICATES
            )
        for name in HEAVY_PREDICATES & set(engine.get("predicates") or {}):
            del engine["predicates"][name]
    path = Path(directory) / "engines.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


class _PrintedOn(io.TextIOBase):
    """A file whose text is printed on a console, to be recorded with it."""

    def __init__(self, console: Console):
        self._console = console

    def write(self, text: str) -> int:
        self._console.print(Text(text), end="", highlight=False)
        return len(text)

    def flush(self) -> None:
        pass


@dataclass
class Reply:
    """
    The answer to a submission.

    *html* is its output, *viewers* the HTML pages of the brain viewer
    the submission made (``.view``), *tables* its results (see
    :func:`table_payload`), *failed* tells whether a statement of it
    failed, and *prompt* is the query the assistant wrote, for the page
    to put in its input.
    """

    html: str
    viewers: List[str] = field(default_factory=list)
    failed: bool = False
    tables: List[dict] = field(default_factory=list)
    prompt: str = ""


# What the events of the assistant carry that the page does not show.
_PRIVATE_EVENT_KEYS = ("messages", "tools", "temperature", "start")


def _kind(column: pd.Series) -> str:
    """Return how the page sorts *column*: bool, int, float, str or other."""
    if pd.api.types.is_bool_dtype(column):
        return "bool"
    if pd.api.types.is_integer_dtype(column):
        return "int"
    if pd.api.types.is_float_dtype(column):
        return "float"
    if all(isinstance(v, str) for v in column.dropna()):
        return "str"
    return "other"


def _cell(value):
    """Return *value* as a JSON value: missing is None, a region its text."""
    if value is None or value is pd.NA:
        return None
    if hasattr(value, "item") and not hasattr(value, "voxels"):
        value = value.item()  # numpy scalars
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return value if math.isfinite(value) else str(value).replace("inf", "Infinity")
    if isinstance(value, (bool, int, str)):
        return value
    return cell_text(value)


def table_payload(
    frame: pd.DataFrame, title: str = "", max_rows: int = MAX_ROWS
) -> dict:
    """
    Return the result *frame* as JSON data for the page's table viewer.

    The ``columns`` and their ``kinds`` (bool, int, float, str or other,
    for the sort), the first *max_rows* ``rows`` of cells, the ``total``
    number of rows, whether the rows are ``truncated`` and the ``title``.
    """
    kept = frame.iloc[:max_rows]
    return {
        "title": title,
        "columns": [str(column) for column in frame.columns],
        "kinds": [_kind(frame[column]) for column in frame.columns],
        "rows": [[_cell(v) for v in row]
                 for row in kept.itertuples(index=False, name=None)],
        "total": len(frame),
        "truncated": len(frame) > len(kept),
    }


class WebRepl:
    """A REPL on the engine *nl*, which a web page submits input to."""

    def __init__(
        self, nl, engine_name: str = "", resolution=None, width: int = 100,
        max_rows: int = MAX_ROWS,
    ):
        """
        Set up the REPL on *nl*, the engine *engine_name* at *resolution*.

        The output is laid out on *width* columns, and at most *max_rows*
        rows of a result are sent to the page.
        """
        self.max_rows = max_rows
        self._tables: list = []
        command = "neurolang-query -i"
        if engine_name:
            command += f" --engine {engine_name}"
        if resolution is not None:
            command += f" --resolution {resolution:g}"
        self.command_line = command
        self.console = Console(
            file=io.StringIO(), record=True, width=width,
            color_system="truecolor", force_terminal=False,
        )
        # Errors are written to the file of the error console: the output
        # comes out in the order it was printed.
        errors = Console(
            file=_PrintedOn(self.console), width=width, color_system=None
        )
        self.app = InteractiveTuiApp(
            nl,
            TuiSettings(engine_name=engine_name, command_line=command),
            console=self.console, err_console=errors,
        )
        # The page shows the rows, in a table it can sort and filter.
        self.app.print_block = self._print_block
        try:
            # Until the visitor gives a key, the assistant asks for it.
            self.set_llm("anthropic", "")
        except ImportError:  # the llm extra is not installed
            pass

    def set_llm(
        self, provider: str, api_key: str, model: str = "",
        base_url: str = "",
    ) -> None:
        """
        Make the assistant (``.ask``, ``.llm``) use *provider* with *api_key*.

        The key is kept in memory, to be sent to the provider and to no
        one else. *model* and *base_url* default to the provider's (see
        :data:`neurolang_apps.utils.llm.browser.PROVIDERS`); a ValueError tells
        what is wrong with them.
        """
        from neurolang_apps.utils.llm.browser import browser_config

        state = self.app.assistant
        state._config = browser_config(provider, api_key, model, base_url)
        state.provider = provider

    def clear_llm(self) -> None:
        """Forget the key of the assistant."""
        provider = self.app.assistant.config().provider()
        self.set_llm(
            provider.name, "", provider.model, provider.base_url
        )

    def set_event_sink(self, send) -> None:
        """
        Send the events of the assistant to *send*, as it works.

        *send* is called with the JSON text of each event (see
        :data:`~neurolang_apps.utils.tui.assistant.AGENT_EVENTS`) while a
        submission is running, so that the page can show the steps before
        the reply. The messages of the conversation are left out of them:
        the page does not show them, and ``"type": "agent"`` tells the page
        what the message is. ``None`` stops sending.
        """
        def forward(event: dict) -> None:
            kept = {
                k: v for k, v in event.items() if k not in _PRIVATE_EVENT_KEYS
            }
            send(json.dumps({"type": "agent", **kept}, default=str))

        self.app.agent_events = forward if send else None

    def banner(self) -> str:
        """Return the HTML of the REPL's welcome: its command and banner."""
        self.console.print(
            f"$ {self.command_line}", style="dim", markup=False,
            highlight=False,
        )
        self.app._print_banner()
        return self._output()

    def submit(self, text: str) -> Reply:
        """
        Run *text*, as typed or pasted at the prompt, and return the reply.

        It may hold several statements and dot-commands, which run in
        turn; the rules and relations it defines stay in the session.
        """
        self.app.failed = False
        self._tables = []
        with _viewers_returned() as viewers:
            try:
                self.app._run_line(text)
            except QuitRepl:
                self.console.print(
                    "[dim]The session cannot end in a page: reload it to "
                    "start a new one.[/dim]"
                )
        prompt, self.app.pending_input = self.app.pending_input, ""
        return Reply(
            self._output(), viewers, self.app.failed, self._tables, prompt
        )

    def _print_block(self, block: ResultBlock) -> None:
        """Keep the rows of *block* for the page, and say how many."""
        if block.title:
            self.console.print(
                f"── {block.title} ──", style="bold magenta", markup=False
            )
        table = table_payload(block.frame, block.title, self.max_rows)
        self._tables.append(table)
        block.shown = len(block.frame)
        count = f"{table['total']} row{'s' * (table['total'] != 1)}"
        if table["truncated"]:
            count += f" · the first {len(table['rows'])} are in the table"
        self.console.print(
            count, style="dim", markup=False, highlight=False
        )

    def _output(self) -> str:
        """Return the HTML of what was printed since last asked."""
        html = self.console.export_html(
            clear=True, inline_styles=True, code_format=HTML_FORMAT
        )
        self.console.file.seek(0)
        self.console.file.truncate()
        return html


@contextmanager
def _viewers_returned():
    """
    Collect in a list the viewer pages saved in the ``with`` block.

    ``.view`` saves its page in a temporary file and opens it in a
    browser: the file is read and removed, and what the console says
    about opening it is replaced by where the page shows it.
    """
    def shown(console, html) -> None:
        console.print("The viewer is shown below.", style="dim")

    pages: list = []
    before = set(_viewer_files())
    with mock.patch.object(nifti, "_open_in_browser", shown):
        try:
            yield pages
        finally:
            for path in sorted(set(_viewer_files()) - before):
                pages.append(path.read_text(encoding="utf-8"))
                path.unlink()


def _viewer_files():
    """Return the files of the viewer pages ``.view`` saved."""
    return Path(tempfile.gettempdir()).glob("neurolang_view_*.html")
