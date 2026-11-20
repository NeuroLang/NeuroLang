"""
The interactive REPL as an MCP server: ``neurolang-query --mcp``.

An MCP client, an LLM such as Claude, answers questions on the engine
as the query assistant does (see :class:`~.llm.agent.ReactAgent`), the
client's model doing the reasoning: ``ask`` gives it the assistant's
instructions for a question, with the syntax, the predicates and the
examples closest to it; the assistant's tools
(:class:`~.llm.agent.Toolbox`) find the predicates and values and check
and run queries; ``final_answer`` checks the answer as the assistant's,
then runs it in the session of the REPL (see
:class:`~.tui.web.WebRepl`) and returns the explanation, the query with
a comment on each rule, its first rows, glass brains of its images and
the links to its files: the viewer, the images, the rows, the query and
its graph. ``ask`` with ``refine`` changes an answer, or the last query
run, for a new question. ``repl`` runs what would be typed at the
prompt, statements and dot-commands, whose rules stay in the session.

The engine is built while the client starts; a tool called meanwhile
waits for it. The calls run one at a time, on the one session.
"""

import asyncio
import base64
import html
import json
import logging
import os
import re
import secrets
import shutil
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import anyio
import anyio.to_thread
import mcp.types as types
import yaml
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from neurolang.confine import CONFINE_ENV, lifted

from neurolang_apps.utils import engine_registry
from neurolang_apps.utils.llm import (
    AgentResult, AssistantConfig, Previous, ProviderConfig, make_agent,
    prompts,
)
from neurolang_apps.utils.llm.agent import Toolbox
from neurolang_apps.utils.llm.comments import commented_query, split_rules
from neurolang_apps.utils.llm.config import _assistant, config_path
from neurolang_apps.utils.llm.examples import (
    load_examples, question_id, select_examples,
)
from neurolang_apps.utils.llm import validation
from neurolang_apps.utils.llm.validation import run_preview
from neurolang_apps.utils.tui import nifti, query_graph
from neurolang_apps.utils.tui.app import AgentRun
from neurolang_apps.utils.tui.assistant import (
    _previous, _run_outcome, _without_left_out,
)
from neurolang_apps.utils.tui.statements import split_at_commands
from neurolang_apps.utils.tui.submission import submission_outline
from neurolang_apps.utils.tui.web import WebRepl

# The references of NeuroLang, at the end of each answer: the language
# (AUTHORS.rst, "Citing NeuroLang") and its coordinate-based meta-analyses.
NEUROLANG_REFERENCES = (
    "Zanitti GE, Soto Y, Iovene V, Martinez MV, Rodriguez RO, Simari GI, "
    "Wassermann D. Scalable Query Answering Under Uncertainty to "
    "Neuroscientific Ontological Knowledge: The NeuroLang Approach. "
    "Neuroinformatics 21(2):407-425 (2023).",
    "Iovene V, Zanitti GE, Wassermann D. Complex Coordinate-Based "
    "Meta-Analysis with Probabilistic Programming. Proceedings of the AAAI "
    "Conference on Artificial Intelligence 35(1):223-231 (2021).",
)

# What the server does, with --log: the build of the engine, each tool
# call with its arguments, time and outcome. Its own logger, to a file:
# the engine's would log every rewrite, and clients hide stderr.
log = logging.getLogger("neurolang_apps.mcp")

# The dot-commands ``repl`` refuses: those of the server's own assistant,
# which would run a model of the machine's configuration, the client's
# model being the assistant.
REFUSED_COMMANDS = frozenset({".ask", ".llm", ".explain", ".rate", ".refine"})

# The rows of each result the ``repl`` tool returns, by default and at most.
MAX_ROWS = 50
MAX_REPL_ROWS = 500
# The most cells of a result kept from the process its query ran in, to
# show it in the session without running it again (a whole-brain map at
# 1 mm is 7 million): a larger one is refused, not run again unlimited.
MAX_CELLS = 20_000_000
# The rows of each result an answer returns, and its glass brains at most.
ANSWER_ROWS = 20
ANSWER_IMAGES = 2
# Where the files of the answers are kept, by default, and for how many days.
RESULTS_DIR = Path("~/.cache/neurolang/mcp")
KEEP_RESULTS_DAYS = 7
LANGUAGE = "datalog"

INSTRUCTIONS = """\
NeuroLang is a probabilistic logic programming language for neuroimaging.
This server answers questions on the engine {engine} as its query assistant
does, with you doing the reasoning. Call `ask` with the question first: it
gives the instructions, the Datalog syntax and examples close to the
question; `list_predicates` gives the predicates. Then follow the
instructions with the tools (`search_values`,
`describe_predicate`, `run_query`, ...) and give the query with
`final_answer`, with a comment on each rule: it returns the explanation,
the commented query, its first rows, images and links to the interactive
viewer and the files of the result, in Markdown; show it to the user as it
is: the query with its comments, and always the links, the viewer's first,
and the references to cite NeuroLang. To change an
answer for a new question, call `ask` with `refine` set to the answer's id
(`a1`, ..., or one of an earlier session that `list_answers` gives) or to
`last`, the last query run, such as one the user changed and ran with
`repl`. `examples` gives example questions by topic, for a user who does
not know what to ask; `status` tells the engine, its grid and whether it is
built yet. `repl` runs statements and dot-commands (`.help`) in
the session: the rules sent alone stay in it."""

# In place of the predicates in the prompt of ``ask``: they would make it
# longer than what clients take of a tool's result (55,000 characters on
# neurosynth), and list_predicates gives them.
SCHEMA_POINTER = """\
(Call list_predicates for them: their columns, descriptions and sizes.)"""

MCP_NOTE = """\
Call the tools of this server, and those that do not depend on each other
(several search_values) in the same turn. final_answer runs the query and
checks it as run_query does, sending back one that fails or gives no rows:
give a query you are confident of to final_answer directly, run_query only
one you are unsure of. In final_answer give, besides the query and its
explanation, `comments`: one short line (at most 15 words) for each rule of
the query, in order, saying what it computes."""

# In place of the instructions, the syntax and the guidance after the first
# ask of a session: the client has them, and each turn reads them again.
AGAIN_NOTE = """\
The instructions, the syntax and the guidance are those the first ask gave;
call ask with full=true to have them again."""


class _NoModel:
    """The assistant's model, which is the MCP client's: never called."""

    def complete(self, *args, **kwargs):
        raise RuntimeError("The MCP client's model does the reasoning.")


@dataclass
class Answer:
    """An answer the session accepted, kept to refine it and for its files."""

    id: str
    question: str
    query: str
    commented: str
    explanation: str
    outcome: str = ""
    links: dict = field(default_factory=dict)
    example_ids: list = field(default_factory=list)


# The name of the folder of a session's answers: its start, and a token.
SESSION_FOLDER = re.compile(r"\d{8}-\d{6}-[0-9a-f]{8,}")


class ResultStore:
    """
    Where the files of the answers of a session are kept, a folder each,
    and the links to them: ``file://`` URIs, the server and its client
    being on the same machine, or under *base_url* when a server serves
    them (the session's folder's name, of *token_bytes* random bytes, is
    then part of their address). The folders of the sessions older than
    *keep_days* days are removed (none if 0).
    """

    def __init__(
        self, root: Path, keep_days: float = KEEP_RESULTS_DAYS,
        token_bytes: int = 4, base_url: str = "",
    ):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        root = Path(root).expanduser()
        self.root = root / f"{stamp}-{secrets.token_hex(token_bytes)}"
        self.base_url = base_url
        if keep_days and root.is_dir():
            self._prune(root, time.time() - keep_days * 86400)

    @staticmethod
    def _prune(root: Path, before: float) -> None:
        """Remove the folders of the sessions in *root* last changed
        before the time *before*: only those named as a session's."""
        for folder in root.iterdir():
            if (
                folder.is_dir() and SESSION_FOLDER.fullmatch(folder.name)
                and folder.stat().st_mtime < before
            ):
                shutil.rmtree(folder, ignore_errors=True)

    def folder(self, answer: str) -> Path:
        folder = self.root / answer
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def link(self, path: Path) -> str:
        if self.base_url:
            relative = path.resolve().relative_to(self.root.parent.resolve())
            return self.base_url + relative.as_posix()
        return path.resolve().as_uri()


class McpSession:
    """
    The REPL session on the engine the server builds, its tools as
    functions, run one at a time.
    """

    def __init__(
        self, engine: str, build=None, initial_program: str = "",
        repl: Optional[WebRepl] = None, results_dir: Path = RESULTS_DIR,
        config: Optional[AssistantConfig] = None, lazy: bool = False,
        keep_days: float = KEEP_RESULTS_DAYS, token_bytes: int = 4,
    ):
        """
        Build the engine with *build* in the background, from the start or,
        if *lazy*, at the first tool call, then run the *initial_program*
        in the session of the REPL on it; or serve the session *repl*, as
        given. The assistant's settings *config*, such as the time and
        memory a query may take, are the defaults if None.
        """
        self.engine = engine
        self.config = config or AssistantConfig()
        self.store = ResultStore(results_dir, keep_days, token_bytes)
        self.answers: dict = {}
        self._question = ""
        self._example_ids = []  # those of the prompt of the question
        self._prompted = False  # whether an ask gave the whole prompt
        self._toolbox: Optional[Toolbox] = None
        self._lock = threading.Lock()
        self._repl = Future()
        self._building = time.monotonic()
        #: What status tells of the server besides (its grid, its data).
        self.about: dict = {}
        self._pending = None  # the build not started yet
        if repl is not None:
            self._repl.set_result(repl)
        elif lazy:
            self._pending = (build, initial_program)
        else:
            self._begin(build, initial_program)

    def _begin(self, build, initial_program: str) -> None:
        threading.Thread(
            target=self._start, args=(build, initial_program), daemon=True,
        ).start()

    def _start(self, build, initial_program: str) -> None:
        self._building = time.monotonic()
        log.info("engine %s: building", self.engine)
        start = time.monotonic()
        try:
            with lifted():  # the engine's commands are the server's
                nl = build()
            repl = WebRepl(nl, self.engine, html=False)
            if initial_program:
                repl.submit(initial_program)
            self._repl.set_result(repl)
            log.info(
                "engine %s: built in %.1f s; the answers' files in %s",
                self.engine, time.monotonic() - start, self.store.root,
            )
        except BaseException as error:  # told to the client by each tool
            log.exception(
                "engine %s: failed after %.1f s", self.engine,
                time.monotonic() - start,
            )
            self._repl.set_exception(error)

    def _session(self) -> WebRepl:
        """The REPL, once the engine is built, or why it is not."""
        if self._pending is not None:  # lazy: built at the first call
            pending, self._pending = self._pending, None
            self._begin(*pending)
        try:
            return self._repl.result()
        except Exception as error:
            raise ToolError(
                f"The engine could not be built: {type(error).__name__}: "
                f"{error}"
            ) from error

    def _agent(self, repl):
        """The assistant on the session, for its prompt and its context."""
        app = repl.app
        return make_agent(
            app._nl, ProviderConfig("mcp", "command", "client"),
            self.config, LANGUAGE, "react", client=_NoModel(),
            engine_name=self.engine, loaded=app.loaded_relations,
            **_without_left_out(app, self.config.include_paper_examples),
        )

    def _tools(self, repl) -> Toolbox:
        """The assistant's tools of the question, or new ones if none was
        asked."""
        if self._toolbox is None:
            config = self.config
            self._toolbox = Toolbox(
                self._agent(repl).context, LANGUAGE, config.preview_rows,
                config.query_timeout, config.query_memory_mb, MAX_CELLS,
            )
        return self._toolbox

    def ask(self, question: str, refine: str = "", full: bool = False) -> str:
        """
        Return the assistant's instructions for *question*, and start its
        tools anew; with *refine*, an answer's id or ``last``, the task
        is to change that answer, or the last query run, for *question*.
        After the first ask, unless *full*, only the examples closest to
        *question* and the task: the client has the rest.
        """
        with self._lock:
            repl = self._session()
            previous = None
            if refine == "last":
                previous = _previous(repl.app)
                if previous is None:
                    raise ToolError("Nothing to refine: no query ran yet.")
            elif refine:
                record = dict(self.earlier_answers(most=1000)).get(refine)
                if record is None:
                    raise ToolError(
                        f"No answer {refine!r}: list_answers gives them."
                    )
                previous = Previous(
                    record["question"], record["query"], record["outcome"]
                )
            agent = self._agent(repl)
            self._toolbox = None
            self._question = question
            task = prompts.refinement_request(
                question, previous, LANGUAGE
            ) if previous else f"# Question\n{question}"
            if self._prompted and not full:
                examples = self._examples(agent, question)
                return "\n\n".join(filter(None, [AGAIN_NOTE, examples, task]))
            self._prompted = True
            prompt = agent.system_prompt(question, schema=SCHEMA_POINTER)
            self._example_ids = agent.example_ids
            return f"{prompt}\n\n{MCP_NOTE}\n\n{task}"

    def _save(self, answer: Answer) -> None:
        """Keep *answer* in its folder (answer.json), for a later session
        to list and refine it (list_answers)."""
        record = {key: getattr(answer, key) for key in (
            "id", "question", "query", "explanation", "outcome", "links",
            "example_ids",
        )}
        (self.store.folder(answer.id) / "answer.json").write_text(
            json.dumps(record, indent=1), encoding="utf-8"
        )

    def _index(self, answer: Answer) -> str:
        """Write the page of the answers of the session, with *answer*, and
        return its link."""
        answers = [*self.answers.values()]
        if answer.id not in self.answers:
            answers.append(answer)
        page = self.store.root / "index.html"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(_index_page(answers, self.store), encoding="utf-8")
        return self.store.link(page)

    def earlier_answers(self, most: int = 20) -> list:
        """The answers kept by the sessions of the folder of this one
        (answer.json), the latest first: (id, record), an earlier
        session's id being ``SESSION/aN``."""
        found = []
        root = self.store.root.parent
        for path in root.glob("*/a*/answer.json"):
            session = path.parent.parent.name
            if not SESSION_FOLDER.fullmatch(session):
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            ident = record["id"] if session == self.store.root.name else \
                f"{session}/{record['id']}"
            found.append((path.stat().st_mtime, ident, record))
        found.sort(key=lambda item: item[0], reverse=True)
        return [(ident, record) for _, ident, record in found[:most]]

    def list_answers(self) -> str:
        """The answers of this session and of the earlier ones, latest
        first, for refine."""
        answers = self.earlier_answers()
        if not answers:
            return "No answer yet."
        return "\n".join(
            f"- `{ident}`: {record.get('question') or '(no question)'}"
            + (f" ({record['links']['viewer']})"
               if record.get("links", {}).get("viewer") else "")
            for ident, record in answers
        )

    def status(self) -> str:
        """What the server is: its engine, grid and folders, whether its
        statements are confined, and whether the engine is built."""
        from neurolang.confine import confined_roots

        if self._repl.done():
            built = "failed" if self._repl.exception() else "built"
        elif self._pending is not None:
            built = "built at the first tool call"
        else:
            built = f"building ({time.monotonic() - self._building:.0f} s)"
        roots = confined_roots()
        lines = [
            f"Engine: {self.engine or '(none)'}, {built}.",
            *(f"{key}: {value}" for key, value in self.about.items()),
            f"Answers' files: {self.store.root}",
            "Statements: " + (
                "confined to " + ", ".join(map(str, roots))
                if roots is not None else "not confined"
            ),
            f"Answers of this session: {len(self.answers)}.",
        ]
        return "\n".join(lines)

    def examples(self, topic: str = "") -> str:
        """The questions of the assistant's bank on the engine: its topics
        with their numbers of questions, or the questions of *topic*."""
        with self._lock:
            topics = self._session().question_topics()
        if not topic:
            counts = {}
            for name in topics.values():
                counts[name] = counts.get(name, 0) + 1
            return "Topics of the example questions (call examples with " \
                "one):\n" + "\n".join(
                    f"- {name} ({count})" for name, count in counts.items()
                )
        questions = [q for q, name in topics.items() if name == topic]
        if not questions:
            return f"No topic {topic!r}: the topics are " + ", ".join(
                sorted(set(topics.values()))
            )
        return "\n".join(f"- {question}" for question in questions)

    def new_conversation(self) -> None:
        """A client starts a conversation: its first ask gives the whole
        prompt again, as it has none of it."""
        self._prompted = False

    def _examples(self, agent, question: str) -> str:
        """The examples of the assistant's bank closest to *question*, as
        its prompt has them."""
        bank = agent.examples
        if bank is None:
            bank = load_examples(
                self.engine, self.config.include_paper_examples
            )
        examples = select_examples(
            bank, question, LANGUAGE, self.config.few_shot_k
        )
        self._example_ids = [
            example.id or question_id(example.question)
            for example in examples
        ]
        if not examples:
            return ""
        return f"# Examples\n{prompts.format_examples(examples, LANGUAGE)}"

    def call(self, name: str, /, **arguments) -> str:
        """Return the text of the assistant's tool *name* on *arguments*."""
        with self._lock:
            observation, _ = self._tools(self._session()).call(
                name, arguments
            )
            return observation

    def repl(self, text: str, max_rows: int = MAX_ROWS) -> str:
        """
        Run *text* in the session and return its output and the first
        *max_rows* rows of each result, tab-separated.
        """
        refused = sorted({
            part.split()[0] for part in split_at_commands(text)
            if part.split()[0] in REFUSED_COMMANDS
        })
        if refused:
            return (
                f"FAILED\n\n{', '.join(refused)}: not here, where you are "
                "the assistant: answer with ask and the tools."
            )
        with self._lock:
            repl = self._session()
            repl.max_rows = min(max_rows, MAX_REPL_ROWS)
            parts, failed = [], False
            for part in split_at_commands(text):
                reply = self._submit(repl, part)
                parts += [reply.html.rstrip(), *map(_tsv, reply.tables)]
                if reply.failed:
                    failed = True
                    break
            self._toolbox = None  # the session may have changed
        if failed:
            parts.insert(0, "FAILED")
        return "\n\n".join(part for part in parts if part)

    def _submit(self, repl, part: str):
        """
        Submit *part*, a dot-command or a program, to the session; a query
        first runs within the assistant's limits of time and memory, in
        its own process, then is shown from the rows it gave, not run
        again (or, its rules staying in the session, run again).
        """
        app = repl.app
        if part.startswith(".") or not submission_outline(
            app._nl, part, False
        )[0]:
            return repl.submit(part)
        config = self.config
        preview = run_preview(
            app._nl, part, LANGUAGE, 0, self.engine, config.query_timeout,
            config.query_memory_mb, MAX_CELLS,
        )
        if not preview.ok or (
            preview.frames is None and preview.frame is not None
        ):
            return _Failed(
                preview.text if not preview.ok else
                f"Its result, {preview.rows} rows, is more than "
                f"{MAX_CELLS:,} cells: restrict it."
            )
        if preview.frames is not None and app.settings.scope_queries:
            app.agent_run = AgentRun(part, False, preview.frames)
        return repl.submit(part)

    def final_answer(
        self, query: str, explanation: str, comments: list
    ) -> types.CallToolResult:
        """
        Check the answer *query* as the assistant's final answer; once
        accepted, run it in the session and return what it gives, else
        why it is rejected.
        """
        rules = split_rules(query, LANGUAGE)
        if len(comments) != len(rules):
            numbered = "\n".join(
                f"{n}. {rule}" for n, rule in enumerate(rules, 1)
            )
            return _text(
                f"The answer is rejected: the query has {len(rules)} rules "
                f"and {len(comments)} comments. Give one comment for each "
                f"rule, in this order:\n{numbered}"
            )
        with self._lock:
            repl = self._session()
            tools = self._tools(repl)
            observation, final = tools.call(
                "final_answer", {"query": query, "explanation": explanation}
            )
            if final is None:
                return _text(observation)
            preview = tools.answer_preview
            if preview.frames is None and preview.frame is not None:
                return _text(
                    f"The answer is rejected: its result, {preview.rows} "
                    f"rows, is more than {MAX_CELLS:,} cells, too large to "
                    "show here. Restrict it, with a threshold on its values "
                    "or fewer columns, and call final_answer again."
                )
            commented = commented_query(
                rules, [" ".join(str(c).split()) for c in comments]
            )
            return self._accept(repl, tools, commented, explanation)

    def _accept(self, repl, tools, commented: str, explanation: str):
        """Show the accepted answer in the session, keep it, and return
        what it gives."""
        app = repl.app
        frames, orders = _by_value(tools.answer_frames)
        if frames is not None:  # shown, not run again
            app.agent_run = AgentRun(commented, False, frames)
        repl.max_rows = ANSWER_ROWS
        reply = repl.submit(commented)
        answer = Answer(
            f"a{len(self.answers) + 1}", self._question,
            commented, commented, explanation,
            example_ids=list(self._example_ids),
        )
        result = AgentResult(
            query=commented, language=LANGUAGE, explanation=explanation,
            valid=True, commented=commented, frames=frames,
            example_ids=list(self._example_ids),
        )
        app.assistant.last_run = (
            self._question, result, ProviderConfig("mcp", "command", "client"),
            "react",
        )
        app.assistant.last_ran = app.last_query
        answer.outcome = _run_outcome(app)
        self.answers[answer.id] = answer
        self._toolbox = None
        self._question = ""  # that of the next ask
        with lifted():  # the server writes the answer's files
            images = self._files(repl, answer)
        with lifted():
            answer.links["index"] = self._index(answer)
        answer.links = links = dict(sorted(
            answer.links.items(), key=lambda item: _link_rank(item[0])
        ))
        with lifted():
            self._save(answer)
        citations = list(NEUROLANG_REFERENCES)
        text = [
            f"## Answer {answer.id}" + (
                f": {answer.question}" if answer.question else ""
            ),
            f"**[Open the interactive viewer]({links['viewer']})**"
            if "viewer" in links else "",
            explanation,
            f"### Query\n```datalog\n{commented}\n```",
            *(_markdown_table(table, orders[n] if n < len(orders) else "")
              for n, table in enumerate(reply.tables)),
        ]
        if reply.failed:
            text.append(
                f"**Running it failed:**\n```\n{reply.html.strip()}\n```"
            )
        text.append("### Files\n" + "\n".join(
            f"- [{_link_title(name)}]({link})" for name, link in links.items()
        ))
        text.append("### How to cite NeuroLang\n" + "\n".join(
            f"{n}. {citation}" for n, citation in enumerate(citations, 1)
        ))
        markdown = "\n\n".join(part for part in text if part)
        return types.CallToolResult(
            content=[_content(markdown)] + images + [
                types.ResourceLink(
                    name=f"{answer.id} {name}", uri=link,
                    title=f"{answer.id}: {LINKS[_link_kind(name)][0]}",
                    mime_type=LINKS[_link_kind(name)][1],
                )
                for name, link in links.items()
            ],
            # The answer once: its text, which some clients (Claude Code)
            # give their model in place of the content, the links to its
            # files and its numbers of rows, for programs.
            structured_content={
                "id": answer.id, "markdown": markdown, "links": links,
                "rows": {
                    table["title"] or "ans": table["total"]
                    for table in reply.tables
                },
            },
        )

    def _files(self, repl, answer: Answer):
        """
        Write the files of the result of *answer* in its folder and keep
        their links: the query, its rows, its images and their viewer, and
        its graph; return the glass brains of its first images.
        """
        app = repl.app
        folder = self.store.folder(answer.id)
        links = answer.links
        # the question the answer is for heads its pages, images and query
        heading = f"{answer.id}: {answer.question}" if answer.question else ""
        query = folder / "query.dl"
        query.write_text("".join(
            f"% {_one_line(line, 1000)}\n"
            for line in (heading, answer.explanation) if line
        ) + answer.commented + "\n", encoding="utf-8")
        links["query"] = self.store.link(query)
        for n, block in enumerate(app.results):
            name = "rows" if len(app.results) == 1 else (
                f"rows-{_file_name(block.title) or n + 1}"
            )
            path = folder / f"{name}.csv"
            block.frame.to_csv(path, index=False)
            links[name] = self.store.link(path)
        images = []
        request = nifti.parse_view_args("")
        shown = app.results and nifti.last_results_layers(app, request)
        if shown:
            layers, extra, request, titles = shown
            try:
                page = nifti._viewer_page(
                    app, layers, request, heading or nifti._view_title(
                        request, layers[0][1], titles
                    ), **extra
                )
                view = folder / "view.html"
                view.write_text(page, encoding="utf-8")
                links["viewer"] = self.store.link(view)
            except Exception:  # the answer stands without its viewer
                pass
            for n, (name, path) in enumerate(layers):
                stem = "result" if len(layers) == 1 else (
                    f"result-{_file_name(name) or n + 1}"
                )
                image = folder / f"{stem}.nii.gz"
                try:
                    shutil.move(path, image)
                    links[stem] = self.store.link(image)
                    if n < ANSWER_IMAGES:
                        png = nifti.render_png(
                            nifti.read_nifti(image),
                            _image_title(heading, answer.id, name,
                                         len(layers) > 1),
                            request.threshold,
                        )
                        images.append(types.ImageContent(
                            type="image", mime_type="image/png",
                            data=_base64(png),
                        ))
                        # a file too: a client may show its model the
                        # image, not its user
                        brain = folder / ("glass-brain.png" if not n else
                                          f"glass-brain-{n + 1}.png")
                        brain.write_bytes(png)
                        links[brain.stem] = self.store.link(brain)
                except Exception:  # the answer stands without its image
                    pass
        try:
            graph = folder / "graph.html"
            graph.write_text(
                query_graph.last_query_page(app, heading or "The query"),
                encoding="utf-8",
            )
            links["graph"] = self.store.link(graph)
        except Exception:  # a query without a graph
            pass
        repl._output()  # what making them said, of files moved since
        return images


# The files of an answer, in the order they are given: their kind, title
# and type.
LINKS = {
    "viewer": ("interactive viewer", "text/html"),
    "glass": ("glass brain (PNG)", "image/png"),
    "result": ("NIfTI image", "application/gzip"),
    "rows": ("rows (CSV)", "text/csv"),
    "graph": ("graph of the query", "text/html"),
    "query": ("query (Datalog)", "text/plain"),
    "index": ("the answers of the session", "text/html"),
}


def _image_title(heading: str, answer: str, layer: str, several: bool) -> str:
    """The title of a glass brain of the image *layer* of an answer: its
    *heading*, the question, short enough to fit, and the layer's name if
    there are *several*; else the answer's id and the layer's name."""
    if not heading:
        return f"{answer}: {layer}"
    return _one_line(heading, 85) + (f" ({layer})" if several else "")


def _link_title(name: str) -> str:
    """The title of the file *name* of an answer in its list of files:
    ``rows-ans`` is "Rows (CSV): ans"."""
    kind, _, part = name.partition("-")
    if kind == "glass":  # glass-brain, glass-brain-2
        part = name[len("glass-brain"):].lstrip("-")
    title = LINKS[kind][0]
    return title[0].upper() + title[1:] + (f": {part}" if part else "")


def _index_page(answers, store) -> str:
    """The page of the answers of a session: each with its question, its
    glass brain and the links to its files, relative to the page."""
    items = []
    for answer in reversed(answers):
        folder = store.root / answer.id
        brain = folder / "glass-brain.png"
        files = [
            (name, path.name) for name, path in (
                (name, folder / path) for name, path in (
                    ("Interactive viewer", "view.html"),
                    ("NIfTI image", "result.nii.gz"),
                    ("Rows (CSV)", "rows.csv"),
                    ("Graph of the query", "graph.html"),
                    ("Query (Datalog)", "query.dl"),
                )
            ) if path.exists()
        ]
        items.append(
            f"<section><h2>{html.escape(answer.id)}: "
            f"{html.escape(answer.question or '(no question)')}</h2>"
            f"<p>{html.escape(answer.explanation)}</p>"
            + (f'<img src="{answer.id}/glass-brain.png" alt="">'
               if brain.exists() else "")
            + "<p>" + " · ".join(
                f'<a href="{answer.id}/{file}">{html.escape(name)}</a>'
                for name, file in files
            ) + "</p></section>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>NeuroLang answers</title><style>"
        "body{font:15px system-ui,sans-serif;max-width:52rem;margin:2rem auto;"
        "padding:0 1rem;color:#0f1b33}section{border-top:1px solid #d9e1ee;"
        "padding:1rem 0}h2{font-size:1.05rem;margin:0 0 .4rem}"
        "img{max-width:100%}a{color:#1a63c2}</style></head><body>"
        "<h1>The answers of this NeuroLang session</h1>"
        + "".join(items) + "</body></html>"
    )


def _by_value(frames):
    """Return the result *frames*, ``(title, frame)``, each by its value,
    highest first (its last column of numbers with decimals, a
    probability, a lift...), and the column each is ordered by ("" for
    none); None, () when there are no frames."""
    if frames is None:
        return None, ()
    ordered, orders = [], []
    for title, frame in frames:
        values = [
            column for column in frame.columns
            if getattr(frame[column].dtype, "kind", "") == "f"
        ]
        if values:
            frame = frame.sort_values(
                values[-1], ascending=False, kind="stable"
            ).reset_index(drop=True)
        ordered.append((title, frame))
        orders.append(str(values[-1]) if values else "")
    return ordered, orders


def _markdown_table(table: dict, order: str = "") -> str:
    """Return the rows of a result of the REPL as a Markdown table, after
    a heading with its name and number of rows, and the column they are
    ordered by, highest first; numbers to 4 significant digits."""
    shown, total = len(table["rows"]), table["total"]
    count = f"{total} row{'s' * (total != 1)}" + (
        f", the first {shown}" if table["truncated"] else ""
    ) + (f", by {order}, highest first" if order else "")
    title = f"### Result{': ' + table['title'] if table['title'] else ''}"
    lines = [
        f"{title} ({count})",
        "| " + " | ".join(map(_cell, table["columns"])) + " |",
        "|" + "---|" * len(table["columns"]),
    ]
    lines.extend(
        "| " + " | ".join(_cell(cell) for cell in row) + " |"
        for row in table["rows"]
    )
    return "\n".join(lines)


def _cell(value) -> str:
    """A value in a cell of a Markdown table."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _link_kind(name: str) -> str:
    """The kind of the file *name* of an answer: ``rows-ans`` is rows."""
    return name.split("-", 1)[0]


def _link_rank(name: str) -> int:
    return list(LINKS).index(_link_kind(name))


@dataclass
class _Failed:
    """The reply to a query refused before it ran in the session."""

    html: str
    tables: list = field(default_factory=list)
    failed: bool = True


def _text(text: str) -> types.CallToolResult:
    return types.CallToolResult(content=[_content(text)])


def _content(text: str) -> types.TextContent:
    return types.TextContent(type="text", text=text)


def _base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _file_name(title: str) -> str:
    """The *title* of a result, as a part of a file's name."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in title)


def _tsv(table: dict) -> str:
    """Return the rows of a result of the REPL as tab-separated text."""
    lines = [table["title"]] if table["title"] else []
    lines.append("\t".join(table["columns"]))
    lines.extend(
        "\t".join("" if cell is None else str(cell) for cell in row)
        for row in table["rows"]
    )
    shown = len(table["rows"])
    if table["truncated"]:
        lines.append(
            f"({shown} of {table['total']} rows: `.save PATH` writes them "
            "all)"
        )
    return "\n".join(lines)


async def _logged(name: str, arguments: dict, run):
    """
    Return ``run()``, the call of the tool *name* on *arguments*, logged
    with its time and outcome. It runs in a thread; if the client cancels
    the call, the query it runs, in a process of its own, is stopped, so
    that the session is free for the next call at once.
    """
    start = time.monotonic()
    try:
        result = await anyio.to_thread.run_sync(run, abandon_on_cancel=True)
    except anyio.get_cancelled_exc_class():
        stopped = validation.stop_queries()
        log.info(
            "%s %s -> cancelled after %.1f s (%d queries stopped)", name,
            _brief(arguments), time.monotonic() - start, stopped,
        )
        raise
    except Exception as error:
        log.info(
            "%s %s -> error after %.1f s: %s", name, _brief(arguments),
            time.monotonic() - start, error,
        )
        raise
    log.info(
        "%s %s -> %s (%.1f s)", name, _brief(arguments), _outcome(result),
        time.monotonic() - start,
    )
    return result


def _brief(arguments: dict) -> str:
    """The *arguments* of a tool call on one line of the log."""
    return " ".join(
        f"{key}={_one_line(str(value), 400)}"
        for key, value in arguments.items() if value not in (None, "")
    )


def _outcome(result) -> str:
    """What a tool returned, on one line of the log: an accepted answer's
    id, rows, images and folder, else the first line of its text."""
    if isinstance(result, types.CallToolResult):
        answer = result.structured_content
        if answer:
            rows = ", ".join(
                f"{title}: {total} rows"
                for title, total in answer["rows"].items()
            )
            images = sum(c.type == "image" for c in result.content)
            folder = answer["links"]["query"].rsplit("/", 1)[0]
            return (
                f"accepted {answer['id']} ({rows}; {images} images) "
                f"in {folder}"
            )
        result = result.content[0].text
    return _one_line(result.strip().split("\n", 1)[0], 200)


def _one_line(text: str, limit: int) -> str:
    text = " | ".join(line.strip() for line in text.splitlines())
    return text if len(text) <= limit else f"{text[:limit]}..."


def assistant_settings(path=None) -> AssistantConfig:
    """
    Return the assistant settings of the LLM configuration (see
    :func:`~.llm.config.config_path`), such as the time and memory a query
    may take: its ``assistant`` section, which needs no provider here, the
    client's model answering; the defaults without one.
    """
    try:
        with open(config_path(path), encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
    except (OSError, yaml.YAMLError):
        return AssistantConfig()
    return _assistant(data.get("assistant"))


class HttpBridge:
    """
    The MCP server's HTTP app (``streamable_http_app``, an ASGI app) run
    without an HTTP server of its own, for one that has its own, as the
    REPL server does: an event loop in a thread runs the app's lifespan,
    and :meth:`call` runs one HTTP request through the app.
    """

    def __init__(self, app, start_seconds: float = 30):
        self.app = app
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self._started = threading.Event()
        self._failed: Optional[BaseException] = None
        asyncio.run_coroutine_threadsafe(self._lifespan(), self.loop)
        if not self._started.wait(start_seconds) or self._failed:
            raise RuntimeError(
                f"The MCP app did not start: {self._failed or 'timed out'}"
            )

    async def _lifespan(self) -> None:
        asked = False

        async def receive():
            nonlocal asked
            if not asked:
                asked = True
                return {"type": "lifespan.startup"}
            await asyncio.Event().wait()  # until the loop ends

        async def send(message):
            if message["type"] == "lifespan.startup.complete":
                self._started.set()
            elif message["type"] == "lifespan.startup.failed":
                self._failed = RuntimeError(message.get("message", ""))
                self._started.set()

        try:
            await self.app(
                {"type": "lifespan", "asgi": {"version": "3.0"}}, receive,
                send,
            )
        except BaseException as error:  # told by __init__
            self._failed = error
            self._started.set()

    def call(self, method: str, path: str, headers: dict, body: bytes):
        """Return the app's answer to an HTTP request: (status, headers,
        body), its headers a list of (name, value)."""
        return asyncio.run_coroutine_threadsafe(
            self._call(method, path, headers, body), self.loop
        ).result()

    async def _call(self, method, path, headers, body):
        path, _, query = path.partition("?")
        scope = {
            "type": "http", "asgi": {"version": "3.0"},
            "http_version": "1.1", "method": method, "scheme": "http",
            "path": path, "raw_path": path.encode(),
            "query_string": query.encode(), "root_path": "",
            "headers": [
                (name.lower().encode("latin-1"), value.encode("latin-1"))
                for name, value in headers.items()
            ],
            "server": ("127.0.0.1", 0), "client": ("127.0.0.1", 0),
        }
        sent = False

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body,
                        "more_body": False}
            await asyncio.Event().wait()  # the client does not disconnect

        answer = {"status": 500, "headers": [], "body": b""}

        async def send(message):
            if message["type"] == "http.response.start":
                answer["status"] = int(message["status"])
                answer["headers"] = [
                    (name.decode("latin-1"), value.decode("latin-1"))
                    for name, value in message.get("headers", [])
                ]
            elif message["type"] == "http.response.body":
                answer["body"] += message.get("body", b"")

        await self.app(scope, receive, send)
        return answer["status"], answer["headers"], answer["body"]


def http_app(session: McpSession):
    """The HTTP app of the MCP server of *session*, stateless, answering in
    JSON (no event streams), for a server that authenticates its clients
    itself, as the REPL server does: its Host is not checked."""
    from mcp.server.transport_security import TransportSecuritySettings

    return server(session).streamable_http_app(
        stateless_http=True, json_response=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False
        ),
    )


def server(session: McpSession):
    """Return the MCP server of *session*, with its tools."""
    mcp = MCPServer(
        "neurolang", instructions=INSTRUCTIONS.format(engine=session.engine),
        log_level="WARNING",  # INFO logs every rewrite of the engine
    )

    @mcp.tool()
    async def ask(
        question: str, refine: str = "", full: bool = False
    ) -> str:
        """
        Start answering a question on the engine: the first call returns
        the instructions, the syntax, the guidance and examples close to the
        question; the next ones only the examples and the task (full=true:
        all again). Set refine to an answer's id (a1, ...) or to `last` (the
        last query run) to change it for the question instead.
        """
        return await _logged(
            "ask", {"question": question, "refine": refine, "full": full},
            lambda: session.ask(question, refine, full),
        )

    described = {tool["name"]: tool["description"] for tool in prompts.TOOLS}

    async def _called(name: str, **arguments) -> str:
        return await _logged(
            name, arguments, lambda: session.call(name, **arguments)
        )

    @mcp.tool(description=described["list_predicates"])
    async def list_predicates() -> str:
        return await _called("list_predicates")

    @mcp.tool(description=described["describe_predicate"])
    async def describe_predicate(name: str) -> str:
        return await _called("describe_predicate", name=name)

    @mcp.tool(description=described["search_values"])
    async def search_values(
        pattern: str, predicate: str = "", column: str = ""
    ) -> str:
        return await _called(
            "search_values", pattern=pattern,
            predicate=predicate or None, column=column or None,
        )

    @mcp.tool(description=described["check_query"])
    async def check_query(query: str) -> str:
        return await _called("check_query", query=query)

    @mcp.tool(description=described["run_query"])
    async def run_query(query: str) -> str:
        return await _called("run_query", query=query)

    @mcp.tool(
        description=described["final_answer"] + " With a comment, one "
        "short line, for each rule of the query, in order. Once accepted, "
        "returns the commented query, its rows, images and links to the "
        "viewer and the files of the result: show them to the user, and "
        "always end your reply with the links, the viewer's first, and the "
        "references to cite NeuroLang."
    )
    async def final_answer(
        query: str, explanation: str, comments: list[str]
    ) -> types.CallToolResult:
        return await _logged(
            "final_answer",
            {"query": query, "explanation": explanation,
             "comments": comments},
            lambda: session.final_answer(query, explanation, comments),
        )

    @mcp.tool()
    async def list_answers() -> str:
        """
        The answers of this session and of the earlier ones (kept 7 days),
        latest first, with their questions and viewers: their ids are what
        ask's refine takes.
        """
        return await _logged("list_answers", {}, session.list_answers)

    @mcp.tool()
    async def status() -> str:
        """
        What this server is: its engine and grid, where the files of the
        answers go, whether the statements are confined, and whether the
        engine is built yet (it takes a minute at the start).
        """
        return await _logged("status", {}, session.status)

    @mcp.tool()
    async def examples(topic: str = "") -> str:
        """
        The example questions of the assistant's bank on this engine: the
        topics, or the questions of one; for the user to pick what to ask.
        """
        return await _logged(
            "examples", {"topic": topic}, lambda: session.examples(topic)
        )

    @mcp.tool()
    async def repl(text: str, max_rows: int = MAX_ROWS) -> str:
        """
        Run text in the NeuroLang REPL session, as typed at its prompt:
        statements (rules, facts, queries) and dot-commands (`.help`).
        Returns the output and the first max_rows rows of each result,
        tab-separated.
        """
        return await _logged(
            "repl", {"text": text}, lambda: session.repl(text, max_rows)
        )

    return mcp


def build_engine(args):
    """
    Build the engine of the CLI's *args*, with its text embeddings loaded,
    as the REPL server does: the queries run in forked processes, and on
    macOS one cannot load the model, nor download it (SIGABRT, as
    ``similar_study`` did); it uses the one loaded here.
    """
    nl = engine_registry.build_engine(
        args.engine, Path(args.data_dir), args.resolution,
        use_cache=not args.no_engine_cache,
    )
    if getattr(nl, "text_embedding_indexes", None):
        from neurolang_apps.utils import text_embeddings

        text_embeddings.preload(nl)
    return nl


def confine(args) -> Optional[list]:
    """
    Confine the statements of the session, unless the CLI's *args* say
    ``--unconfined``: they read the engine's data and the answers' files
    only, and write no file (see :mod:`neurolang.confine`); what the
    server does, building the engine and writing the answers' files, is
    not. Return the directories they may read, None if not confined.
    """
    if args.unconfined:
        return None
    roots = [
        Path(args.data_dir).expanduser().resolve(),
        Path(args.results_dir or RESULTS_DIR).expanduser().resolve(),
        *(Path(d).expanduser().resolve()
          for d in getattr(args, "allow_dir", None) or ()),
    ]
    os.environ[CONFINE_ENV] = os.pathsep.join(map(str, roots))
    return roots


def main(args, initial_program: str = "") -> None:
    """Serve the REPL on the engine of the CLI's *args* over stdio."""
    def build():
        return build_engine(args)

    confine(args)
    os.umask(0o077)  # the answers' files and the log, the user's only

    if args.log:
        handler = logging.FileHandler(Path(args.log).expanduser())
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False
    session = McpSession(
        args.engine, build, initial_program,
        results_dir=Path(args.results_dir or RESULTS_DIR),
        config=assistant_settings(), lazy=args.build_on_first_call,
        keep_days=args.keep_results,
    )
    session.about = {
        "Grid": f"{getattr(args, 'resolution', '?')} mm",
        "Data": str(args.data_dir),
        **({"Log": str(args.log)} if args.log else {}),
    }
    server(session).run("stdio")
