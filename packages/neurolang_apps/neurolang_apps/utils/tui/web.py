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

import csv
import gzip
import inspect
import io
import json
import math
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, NamedTuple
from unittest import mock

import pandas as pd
from rich.console import Console
from rich.text import Text

from neurolang_apps.utils import engine_registry
from neurolang_apps.utils.cli import _declared_columns
from neurolang_apps.utils.predicate_catalog import listed, predicate_catalog

from . import nifti, query_graph
from .app import InteractiveTuiApp, TuiSettings
from .commands import COMMANDS, HELP_TOPICS, QuitRepl, engine_examples
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
        heavy = left_out(engine.get("datalog_init") or "")
        if "datalog_init" in engine:
            engine["datalog_init"] = "\n".join(
                rule for rule in engine["datalog_init"].split("\n")
                if _head(rule) not in heavy
            )
        for name in heavy & set(engine.get("predicates") or {}):
            del engine["predicates"][name]
    path = Path(directory) / "engines.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _head(rule: str) -> str:
    """Return the name of the predicate the Datalog *rule* defines."""
    return rule.split("(")[0].strip()


def left_out(rules: str) -> frozenset:
    """
    Return :data:`HEAVY_PREDICATES` and the predicates the Datalog *rules*
    (one per line) define from them, at any depth: those the lightweight
    engines leave out.
    """
    heavy = set(HEAVY_PREDICATES)
    body = {
        _head(rule): set(re.findall(r"(\w+)\s*\(", rule.partition(":-")[2]))
        for rule in rules.split("\n")
    }
    while grown := {h for h, b in body.items() if b & heavy} - heavy:
        heavy |= grown
    return frozenset(heavy)


def left_out_predicates(nl, engine_name: str) -> frozenset:
    """
    Return the predicates of the engine *engine_name* that a lightweight
    engine leaves out (:func:`left_out`) and *nl* does not define.
    """
    if not engine_name:
        return frozenset()
    config = engine_registry.get_engine_config(engine_name)
    rules = config.get("datalog_init") or ""
    engine = set(config.get("predicates") or {}) | {
        _head(rule) for rule in rules.split("\n")
    }
    return frozenset(
        name for name in left_out(rules) & engine
        if name not in nl.symbol_table
    )


def _from_an_ontology(name: str) -> bool:
    """
    Whether the predicate *name* is the IRI of a class or a property of an
    ontology, ``http://...#CAO_00137``: those of the vocabularies the engine
    names with a prefix (``rdfs:label``) are few, and relations to query.
    """
    return "://" in name


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
    failed, *prompt* is the query the assistant wrote, for the page
    to put in its input, and *mode* the language of the REPL, ``datalog``
    or ``squall``, which the assistant may have switched, *predicates*
    those of the engine and the session (see :meth:`WebRepl.predicates`),
    *ratable* whether it was a query or an answer of the assistant,
    which :meth:`WebRepl.rate_answer` can share, and *graph* the HTML page
    of the graph of the query it ran, or that ``.graph`` showed (see
    :mod:`~neurolang_apps.utils.tui.query_graph`).
    """

    html: str
    viewers: List[str] = field(default_factory=list)
    failed: bool = False
    tables: List[dict] = field(default_factory=list)
    prompt: str = ""
    mode: str = "datalog"
    predicates: List[dict] = field(default_factory=list)
    ratable: bool = False
    graph: str = ""


class Download(NamedTuple):
    """A file the page offers to save: its *name*, *mime* type and *data*."""

    name: str
    mime: str
    data: bytes


# The files of a result: the separator of its text formats, the type of each.
TEXT_FORMATS = {"csv": (",", "text/csv"), "tsv": ("\t", "text/tab-separated-values")}
NIFTI_MIME = "application/gzip"


def _file_stem(block: ResultBlock, index: int) -> str:
    """Return the name of the files of the result *block*, the *index*-th."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", block.title).strip("-").lower()
    return stem or f"result-{index + 1}"


def _text(frame: pd.DataFrame, separator: str) -> str:
    """Return all the rows of *frame* as delimited text, as the table's."""
    out = io.StringIO()
    writer = csv.writer(out, delimiter=separator, lineterminator="\n")
    writer.writerow(frame.columns)
    for row in frame.itertuples(index=False, name=None):
        writer.writerow(
            ["" if cell is None else cell for cell in map(_cell, row)]
        )
    return out.getvalue()


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


# The commands that end a terminal session, which a page cannot end.
TERMINAL_COMMANDS = (".quit", ".exit")

# The sort of the studies of the engines whose studies are those of PubMed.
STUDY_SORT = "study"
# What the page shows of a study: its metadata, the DOI first.
STUDY_FIELDS = ("doi", "title", "authors", "year", "journal")


def study_records(metadata: pd.DataFrame) -> dict:
    """
    Return the records of the studies of *metadata*, a frame with an ``id``
    column (the PubMed id) and some of :data:`STUDY_FIELDS`, by id. What a
    study lacks is not in its record.
    """
    columns = [name for name in STUDY_FIELDS if name in metadata.columns]
    records = {}
    for row in metadata[["id", *columns]].itertuples(index=False, name=None):
        cells = {
            name: _cell(value) for name, value in zip(columns, row[1:])
        }
        records[int(row[0])] = {
            name: value for name, value in cells.items()
            if value not in (None, "")
        }
    return records


def table_payload(
    frame: pd.DataFrame, title: str = "", max_rows: int = MAX_ROWS,
    sorts=None, studies=None,
) -> dict:
    """
    Return the result *frame* as JSON data for the page's table viewer.

    The ``columns`` and their ``kinds`` (bool, int, float, str or other,
    for the sort), the first *max_rows* ``rows`` of cells, the ``total``
    number of rows, whether the rows are ``truncated`` and the ``title``.
    With the *sorts* of the columns (the name of each, None if unknown),
    ``sorts``; and if some columns are of studies, and the *studies* are
    known (see :func:`study_records`), ``study_links``: those ``columns``,
    and the ``records`` of the studies the rows hold, by id.
    """
    kept = frame.iloc[:max_rows]
    payload = {
        "title": title,
        "columns": [str(column) for column in frame.columns],
        "kinds": [_kind(frame[column]) for column in frame.columns],
        "rows": [[_cell(v) for v in row]
                 for row in kept.itertuples(index=False, name=None)],
        "total": len(frame),
        "truncated": len(frame) > len(kept),
    }
    if sorts is not None:
        payload["sorts"] = list(sorts)
        links = _study_links(kept, payload["sorts"], studies)
        if links:
            payload["study_links"] = links
    return payload


def _study_ids(frame: pd.DataFrame, columns: list) -> set:
    """Return the values of the *columns* of *frame*, as the page has them."""
    return {_cell(value) for i in columns for value in frame.iloc[:, i]}


def _study_links(frame: pd.DataFrame, sorts: list, studies) -> dict:
    """Return the ``study_links`` of the *frame*, or an empty dict."""
    columns = [i for i, sort in enumerate(sorts) if sort == STUDY_SORT]
    if not (columns and studies):
        return {}
    known = _study_ids(frame, columns) & studies.keys()
    return {
        "columns": columns,
        "records": {str(study): studies[study] for study in known},
    }


def _arguments(info, described: dict) -> list:
    """
    Return the arguments of the predicate *info*, those of its ``%%``
    documentation or else those the engine declares (*described*): the
    ``columns``, ``sort`` and ``type`` of each.
    """
    if info.arguments:
        return [
            {"columns": list(argument.columns), "sort": argument.sort,
             "type": argument.type}
            for argument in info.arguments
        ]
    return [
        {"columns": list(argument["columns"]),
         "sort": argument["sort"], "type": argument["type"]}
        for argument in described.get(info.name, {}).get("arguments", [])
    ]


class WebRepl:
    """A REPL on the engine *nl*, which a web page submits input to."""

    def __init__(
        self, nl, engine_name: str = "", resolution=None, width: int = 100,
        max_rows: int = MAX_ROWS, study_metadata=None,
    ):
        """
        Set up the REPL on *nl*, the engine *engine_name* at *resolution*.

        The output is laid out on *width* columns, and at most *max_rows*
        rows of a result are sent to the page. The *study_metadata* (see
        :func:`study_records`) of the engine's studies, whose ids are
        those of PubMed, let the page link the studies of a result.
        """
        self.max_rows = max_rows
        self._studies = (
            study_records(study_metadata) if study_metadata is not None
            else None
        )
        self._tables: list = []
        self._rating = None  # where a rated answer goes (see set_feedback)
        # What a rating shares: "answer" (the assistant's last), the record
        # of the last query, or None.
        self._rated = None
        self._frames: list = []  # the results of the last submission
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
        # Those of a lightweight engine (lightweight_engines_yaml), which
        # the REPL and its assistant tell are not available.
        nl.left_out_predicates = left_out_predicates(nl, engine_name)
        # The page shows the rows, in a table it can sort and filter.
        self.app.print_block = self._print_block
        # and the graph of a query in a tab of its own.
        self._graph = ""
        self.app.show_graph = self._show_graph
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

    def set_feedback(self, url: str, doc: str, transport=None) -> None:
        """
        Let the page share a rated answer in the table ``Answers`` of the
        Grist document *doc* at *url*, anonymously (see
        :class:`~neurolang_apps.utils.llm.feedback.GristFeedbackClient`):
        without a *doc*, answers cannot be rated. *transport* replaces the
        browser's (in tests).
        """
        from neurolang_apps.utils.llm.feedback import GristFeedbackClient

        self._rating = GristFeedbackClient(
            url, doc, api_key_env="", transport=transport
        ) if doc else None

    def can_rate(self) -> bool:
        """Whether the page can send the rating of an answer."""
        return self._rating is not None

    def rate_answer(self, verdict: str, comment: str = "") -> dict:
        """
        Share the last answer of the assistant, or the last query, rated
        *verdict* (``good`` or ``bad``) with the visitor's *comment*, and
        return ``ok`` and a ``message`` to show. A query is shared with its
        first rows or its error (see
        :func:`~neurolang_apps.utils.llm.feedback.query_record`).
        """
        from neurolang_apps.utils.llm.feedback import FeedbackError

        run = self.app.assistant.last_run
        rated = self._rated
        if self._rating is None or verdict not in ("good", "bad") or not (
            isinstance(rated, dict) or (rated == "answer" and run)
        ):
            return {"ok": False, "message": "There is no answer to rate."}
        engine = self.app.settings.engine_name
        try:
            if isinstance(rated, dict):
                self._rating.share_record(dict(rated), verdict, comment)
            else:
                self._rating.share_rated(*run, engine, verdict, comment)
        except FeedbackError as error:
            return {"ok": False, "message": str(error)}
        return {"ok": True, "message": "Thank you: your rating is sent."}

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
        left_out = sorted(self.app.left_out_predicates)
        if left_out:
            self.console.print(
                "The analyses that need gigabytes are not in the browser: "
                + ", ".join(left_out), style="dim", markup=False,
                highlight=False, soft_wrap=True,
            )
        return self._output()

    def submit(self, text: str) -> Reply:
        """
        Run *text*, as typed or pasted at the prompt, and return the reply.

        It may hold several statements and dot-commands, which run in
        turn; the rules and relations it defines stay in the session.
        """
        self.app.failed = False
        self._tables = []
        self._frames = []
        self._graph = ""
        answered = self.app.assistant.last_run
        query = self.app.last_query
        with _viewers_returned() as viewers:
            try:
                self.app._run_line(text)
            except QuitRepl:
                self.console.print(
                    "[dim]The session cannot end in a page: reload it to "
                    "start a new one.[/dim]"
                )
        prompt, self.app.pending_input = self.app.pending_input, ""
        ran = self.app.last_query  # a new tuple for each query run
        if not self._graph and ran is not None and ran is not query:
            try:
                self._graph = query_graph.last_query_page(self.app)
            except Exception:  # a query that does not parse has no graph
                pass
        self._rated = self._what_is_rated(
            text, self.app.assistant.last_run is not answered
        )
        return Reply(
            self._output(), viewers, self.app.failed, self._tables, prompt,
            "squall" if self.app.settings.squall else "datalog",
            self.predicates(), self._rated is not None, self._graph,
        )

    def _show_graph(self, page: str) -> None:
        """Keep the *page* of the graph of a query for the reply."""
        self._graph = page

    def _what_is_rated(self, text: str, answered: bool):
        """
        Return what a rating of the submission *text* shares: "answer"
        when the assistant *answered* it, the record of a query, or None
        for a dot-command.
        """
        if answered:
            return "answer"
        if not text.strip() or text.strip().startswith("."):
            return None
        from neurolang_apps.utils.llm.feedback import query_record

        output = self.console.export_text(clear=False).strip()
        rows = "\n\n".join(
            f"{len(frame)} rows; columns {', '.join(map(str, frame.columns))}"
            f"\n{frame.head(10).to_string(index=False)}"
            for frame in self._frames
        )
        return query_record(
            text.strip(), rows, output if self.app.failed else "",
            self.app.settings.engine_name,
        )

    def _print_block(self, block: ResultBlock) -> None:
        """Keep the rows of *block* for the page, and say how many."""
        if block.title:
            self.console.print(
                f"── {block.title} ──", style="bold magenta", markup=False
            )
        sorts = [s.sort if s else None for s in block.sorts or ()]
        self._frames.append(block.frame)
        table = table_payload(
            block.frame, block.title, self.max_rows,
            sorts if len(sorts) == len(block.frame.columns) else None,
            self._studies,
        )
        table.update(self._downloads_of(block))
        self._tables.append(table)
        block.shown = len(block.frame)
        count = f"{table['total']} row{'s' * (table['total'] != 1)}"
        if table["truncated"]:
            count += f" · the first {len(table['rows'])} are in the table"
        self.console.print(
            count, style="dim", markup=False, highlight=False
        )

    def examples(self) -> dict:
        """Return the example queries of the engine, by language."""
        return engine_examples(
            self.app.settings.engine_name, self.app.left_out_predicates
        )

    def assistant_ready(self) -> bool:
        """
        Return whether the AI assistant can answer: its provider has its
        key, the page's or that of the environment, or needs none.
        """
        state = self.app.assistant
        try:
            state.config().provider(state.provider).api_key()
        except Exception:
            return False
        return True

    def questions(self) -> List[str]:
        """Return the example questions in English of the engine, for the
        query assistant."""
        engine = self.app.settings.engine_name
        if not engine:
            return []
        config = engine_registry.get_engine_config(engine)
        return list(config.get("questions") or [])

    def predicates(self) -> List[dict]:
        """
        Return the predicates of the engine and of the session, for the
        page to list, but for the IRIs of an ontology's own classes and
        properties that the engine does not declare: ``name``,
        ``kind`` (stored, derived, session...),
        ``signature`` as ``.predicates`` shows it, and its
        ``description``, if any, with its ``arguments``: the ``columns``,
        ``sort`` and ``type`` of each. Those of the ``%%`` documentation
        of the predicate in the program win over the engine's.
        """
        engine = self.app.settings.engine_name
        signatures = engine_registry.get_signatures(engine) if engine else {}
        described = engine_registry.get_predicates(engine) if engine else {}
        infos = [
            info for info in listed(predicate_catalog(
                self.app._nl, _declared_columns(engine),
                self.app.loaded_relations, signatures,
            ))
            if info.kind == "session" or info.name in described
            or not _from_an_ontology(info.name)
        ]
        return [
            {"name": info.name, "kind": info.kind,
             "signature": info.signature(),
             "description": info.description or described.get(
                 info.name, {}
             ).get("description", ""),
             "arguments": _arguments(info, described)}
            for info in infos
        ]

    def commands(self) -> List[dict]:
        """
        Return the dot-commands the page can run, to document them: the
        ``name``, ``usage`` and ``description``, the ``details`` of its
        function's docstring and its help ``topic`` (``.help TOPIC`` tells
        more), or an empty one.
        """
        return [
            {"name": name, "usage": command.usage,
             "description": command.help,
             "details": inspect.getdoc(command.handler) or "",
             "topic": name[1:] if name[1:] in HELP_TOPICS else ""}
            for name, command in COMMANDS.items()
            if name not in TERMINAL_COMMANDS
        ]

    def _downloads_of(self, block: ResultBlock) -> dict:
        """
        Return what the page may download of *block*: its ``block`` number
        for :meth:`download`, and the value columns of its NIfTI image,
        ``nifti`` (``None`` in the list is a mask), or else why it has
        none, ``nifti_reason``.
        """
        index = next(
            (i for i, b in enumerate(self.app.results) if b is block), 0
        )
        try:
            columns = nifti.image_columns(self.app, block)
            return {"block": index, "nifti": columns, "nifti_reason": ""}
        except Exception as error:  # nothing that fails here may hide a result
            reason = str(error) or "no NIfTI image can be made of this result"
            return {"block": index, "nifti": None, "nifti_reason": reason}

    def download(self, block: int, kind: str, value: str = "") -> Download:
        """
        Return the file of kind *kind* of the result *block*, the number the
        table gave: ``csv`` or ``tsv`` with all its rows (the table holds
        at most ``max_rows``), or a ``nifti`` image of the column *value*
        (the default one when empty), gzipped. A
        :class:`~.nifti.SaveError` tells why a result has no image.
        """
        try:
            result = self.app.results[block]
        except IndexError:
            raise ValueError("There is no such result: run a query first.")
        stem = _file_stem(result, block)
        if kind in TEXT_FORMATS:
            separator, mime = TEXT_FORMATS[kind]
            text = _text(result.frame, separator)
            return Download(f"{stem}.{kind}", mime, text.encode("utf-8"))
        if kind != "nifti":
            raise ValueError(f"Unknown kind of file {kind!r}.")
        request = nifti.SaveRequest(Path(f"{stem}.nii.gz"), value or None)
        saved = nifti.result_image(self.app, result, request)
        suffix = f"-{saved.value}" if saved.value is not None else ""
        return Download(
            f"{stem}{suffix}.nii.gz", NIFTI_MIME,
            gzip.compress(saved.image.to_bytes()),
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
