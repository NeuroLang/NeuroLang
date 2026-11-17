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

import base64
import logging
import secrets
import shutil
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mcp.types as types
import yaml
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from neurolang_apps.utils import engine_registry
from neurolang_apps.utils.llm import (
    AgentResult, AssistantConfig, Previous, ProviderConfig, make_agent,
    prompts,
)
from neurolang_apps.utils.llm.agent import Toolbox
from neurolang_apps.utils.llm.comments import commented_query, split_rules
from neurolang_apps.utils.llm.config import _assistant, config_path
from neurolang_apps.utils.tui import nifti, query_graph
from neurolang_apps.utils.tui.app import AgentRun
from neurolang_apps.utils.tui.assistant import (
    _previous, _run_outcome, _without_left_out,
)
from neurolang_apps.utils.tui.web import WebRepl

# How to cite NeuroLang, at the end of each answer with the sources of the
# engine's data (their acknowledgements, as `.credits` gives them).
NEUROLANG_CITATION = (
    "NeuroLang: Zanitti GE, Soto Y, Iovene V, Martinez MV, Rodriguez RO, "
    "Simari GI, Wassermann D. Scalable Query Answering Under Uncertainty to "
    "Neuroscientific Ontological Knowledge: The NeuroLang Approach. "
    "Neuroinformatics 21(2):407-425 (2023)."
)

# What the server does, with --log: the build of the engine, each tool
# call with its arguments, time and outcome. Its own logger, to a file:
# the engine's would log every rewrite, and clients hide stderr.
log = logging.getLogger("neurolang_apps.mcp")

# The rows of each result the ``repl`` tool returns, by default.
MAX_ROWS = 50
# The rows of each result an answer returns, and its glass brains at most.
ANSWER_ROWS = 20
ANSWER_IMAGES = 2
# Where the files of the answers are kept, by default.
RESULTS_DIR = Path("~/.cache/neurolang/mcp")
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
viewer and the files of the result; show them to the user: the query with
its comments, and always the links, the viewer's first, and how to cite
NeuroLang and the data the answer comes from. To change an
answer for a new question, call `ask` with `refine` set to the answer's id
(`a1`, ...) or to `last`, the last query run, such as one the user changed
and ran with `repl`. `repl` runs statements and dot-commands (`.help`) in
the session: the rules sent alone stay in it."""

# In place of the predicates in the prompt of ``ask``: they would make it
# longer than what clients take of a tool's result (55,000 characters on
# neurosynth), and list_predicates gives them.
SCHEMA_POINTER = """\
(Call list_predicates for them: their columns, descriptions and sizes.)"""

MCP_NOTE = """\
Call the tools of this server. In final_answer give, besides the query and
its explanation, `comments`: one short line (at most 15 words) for each rule
of the query, in order, saying what it computes."""


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


class ResultStore:
    """
    Where the files of the answers of a session are kept, a folder each,
    and the links to them: ``file://`` URIs, the server and its client
    being on the same machine.
    """

    def __init__(self, root: Path):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.root = Path(root).expanduser() / f"{stamp}-{secrets.token_hex(4)}"

    def folder(self, answer: str) -> Path:
        folder = self.root / answer
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def link(self, path: Path) -> str:
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
        self.store = ResultStore(results_dir)
        self.answers: dict = {}
        self._question = ""
        self._toolbox: Optional[Toolbox] = None
        self._lock = threading.Lock()
        self._repl = Future()
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
        log.info("engine %s: building", self.engine)
        start = time.monotonic()
        try:
            repl = WebRepl(build(), self.engine, html=False)
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
            **_without_left_out(app),
        )

    def _tools(self, repl) -> Toolbox:
        """The assistant's tools of the question, or new ones if none was
        asked."""
        if self._toolbox is None:
            config = self.config
            self._toolbox = Toolbox(
                self._agent(repl).context, LANGUAGE, config.preview_rows,
                config.query_timeout, config.query_memory_mb,
            )
        return self._toolbox

    def ask(self, question: str, refine: str = "") -> str:
        """
        Return the assistant's instructions for *question*, and start its
        tools anew; with *refine*, an answer's id or ``last``, the task
        is to change that answer, or the last query run, for *question*.
        """
        with self._lock:
            repl = self._session()
            previous = None
            if refine == "last":
                previous = _previous(repl.app)
                if previous is None:
                    raise ToolError("Nothing to refine: no query ran yet.")
            elif refine:
                answer = self.answers.get(refine)
                if answer is None:
                    raise ToolError(
                        f"No answer {refine!r}: the answers are "
                        f"{', '.join(self.answers) or 'none yet'}."
                    )
                previous = Previous(
                    answer.question, answer.query, answer.outcome
                )
            agent = self._agent(repl)
            self._toolbox = None
            self._question = question
            task = prompts.refinement_request(
                question, previous, LANGUAGE
            ) if previous else f"# Question\n{question}"
            prompt = agent.system_prompt(question, schema=SCHEMA_POINTER)
            return f"{prompt}\n\n{MCP_NOTE}\n\n{task}"

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
        with self._lock:
            repl = self._session()
            repl.max_rows = max_rows
            reply = repl.submit(text)
            self._toolbox = None  # the session may have changed
        parts = [reply.html.rstrip(), *map(_tsv, reply.tables)]
        if reply.failed:
            parts.insert(0, "FAILED")
        return "\n\n".join(part for part in parts if part)

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
            commented = commented_query(
                rules, [" ".join(str(c).split()) for c in comments]
            )
            return self._accept(repl, tools, commented, explanation)

    def _accept(self, repl, tools, commented: str, explanation: str):
        """Show the accepted answer in the session, keep it, and return
        what it gives."""
        app = repl.app
        if tools.answer_frames is not None:  # shown, not run again
            app.agent_run = AgentRun(commented, False, tools.answer_frames)
        repl.max_rows = ANSWER_ROWS
        reply = repl.submit(commented)
        answer = Answer(
            f"a{len(self.answers) + 1}", self._question,
            commented, commented, explanation,
        )
        result = AgentResult(
            query=commented, language=LANGUAGE, explanation=explanation,
            valid=True, commented=commented, frames=tools.answer_frames,
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
        images = self._files(repl, answer)
        answer.links = links = dict(sorted(
            answer.links.items(), key=lambda item: _link_rank(item[0])
        ))
        text = [
            f"Answer {answer.id}" + (
                f" to: {answer.question}" if answer.question else ""
            ),
            f"Interactive viewer: {links['viewer']}"
            if "viewer" in links else "",
            explanation,
            f"```datalog\n{commented}\n```",
            *map(_tsv, reply.tables),
        ]
        if reply.failed:
            text.append(f"Running it failed:\n{reply.html.strip()}")
        text.append("Links to give the user:\n" + "\n".join(
            f"{name}: {link}" for name, link in links.items()
        ))
        citations = self.citations()
        text.append("To cite, if the answer is used:\n" + "\n".join(
            f"- {citation}" for citation in citations
        ))
        return types.CallToolResult(
            content=[_content("\n\n".join(part for part in text if part))]
            + images + [
                types.ResourceLink(
                    name=f"{answer.id} {name}", uri=link,
                    title=f"{answer.id}: {LINKS[_link_kind(name)][0]}",
                    mime_type=LINKS[_link_kind(name)][1],
                )
                for name, link in links.items()
            ],
            structured_content={
                "id": answer.id, "question": answer.question,
                "viewer": links.get("viewer"), "links": links,
                "query": commented, "explanation": explanation,
                # clients showing it in place of the text see the rows too
                "tables": [
                    {key: table[key] for key in (
                        "title", "columns", "rows", "total"
                    )}
                    for table in reply.tables
                ],
                "citations": citations,
            },
        )

    def citations(self) -> list:
        """How to cite NeuroLang and the sources of the engine's data."""
        engine = self.engine
        known = engine in engine_registry.list_engine_names()
        return [NEUROLANG_CITATION, *(
            engine_registry.get_acknowledgements(engine) if known else ()
        )]

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
                        images.append(types.ImageContent(
                            type="image", mime_type="image/png",
                            data=_base64(nifti.render_png(
                                nifti.read_nifti(image),
                                _image_title(heading, answer.id, name,
                                             len(layers) > 1),
                                request.threshold,
                            )),
                        ))
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
    "result": ("NIfTI image", "application/gzip"),
    "rows": ("rows (CSV)", "text/csv"),
    "graph": ("graph of the query", "text/html"),
    "query": ("query (Datalog)", "text/plain"),
}


def _image_title(heading: str, answer: str, layer: str, several: bool) -> str:
    """The title of a glass brain of the image *layer* of an answer: its
    *heading*, the question, short enough to fit, and the layer's name if
    there are *several*; else the answer's id and the layer's name."""
    if not heading:
        return f"{answer}: {layer}"
    return _one_line(heading, 85) + (f" ({layer})" if several else "")


def _link_kind(name: str) -> str:
    """The kind of the file *name* of an answer: ``rows-ans`` is rows."""
    return name.split("-", 1)[0]


def _link_rank(name: str) -> int:
    return list(LINKS).index(_link_kind(name))


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


def _logged(name: str, arguments: dict, run):
    """Return ``run()``, the call of the tool *name* on *arguments*,
    logged with its time and outcome."""
    start = time.monotonic()
    try:
        result = run()
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
                f"{table['title'] or 'ans'}: {table['total']} rows"
                for table in answer["tables"]
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


def server(session: McpSession):
    """Return the MCP server of *session*, with its tools."""
    mcp = MCPServer(
        "neurolang", instructions=INSTRUCTIONS.format(engine=session.engine),
        log_level="WARNING",  # INFO logs every rewrite of the engine
    )

    @mcp.tool()
    def ask(question: str, refine: str = "") -> str:
        """
        Start answering a question on the engine: returns the instructions,
        the syntax, the predicates and examples close to the question. Set
        refine to an answer's id (a1, ...) or to `last` (the last query
        run) to change it for the question instead.
        """
        return _logged(
            "ask", {"question": question, "refine": refine},
            lambda: session.ask(question, refine),
        )

    described = {tool["name"]: tool["description"] for tool in prompts.TOOLS}

    def _called(name: str, **arguments) -> str:
        return _logged(
            name, arguments, lambda: session.call(name, **arguments)
        )

    @mcp.tool(description=described["list_predicates"])
    def list_predicates() -> str:
        return _called("list_predicates")

    @mcp.tool(description=described["describe_predicate"])
    def describe_predicate(name: str) -> str:
        return _called("describe_predicate", name=name)

    @mcp.tool(description=described["search_values"])
    def search_values(
        pattern: str, predicate: str = "", column: str = ""
    ) -> str:
        return _called(
            "search_values", pattern=pattern,
            predicate=predicate or None, column=column or None,
        )

    @mcp.tool(description=described["check_query"])
    def check_query(query: str) -> str:
        return _called("check_query", query=query)

    @mcp.tool(description=described["run_query"])
    def run_query(query: str) -> str:
        return _called("run_query", query=query)

    @mcp.tool(
        description=described["final_answer"] + " With a comment, one "
        "short line, for each rule of the query, in order. Once accepted, "
        "returns the commented query, its rows, images and links to the "
        "viewer and the files of the result: show them to the user, and "
        "always end your reply with the links, the viewer's first, and the "
        "citations of NeuroLang and of the data."
    )
    def final_answer(
        query: str, explanation: str, comments: list[str]
    ) -> types.CallToolResult:
        return _logged(
            "final_answer",
            {"query": query, "explanation": explanation,
             "comments": comments},
            lambda: session.final_answer(query, explanation, comments),
        )

    @mcp.tool()
    def repl(text: str, max_rows: int = MAX_ROWS) -> str:
        """
        Run text in the NeuroLang REPL session, as typed at its prompt:
        statements (rules, facts, queries) and dot-commands (`.help`).
        Returns the output and the first max_rows rows of each result,
        tab-separated.
        """
        return _logged(
            "repl", {"text": text}, lambda: session.repl(text, max_rows)
        )

    return mcp


def main(args, initial_program: str = "") -> None:
    """Serve the REPL on the engine of the CLI's *args* over stdio."""
    def build():
        nl = engine_registry.build_engine(
            args.engine, Path(args.data_dir), args.resolution,
            use_cache=not args.no_engine_cache,
        )
        if getattr(nl, "text_embedding_indexes", None):
            # Loaded here, as the REPL server does: on macOS a query's
            # forked process cannot load a model (SIGABRT, similar_study).
            from neurolang_apps.utils import text_embeddings

            text_embeddings.preload(nl)
        return nl

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
    )
    server(session).run("stdio")
