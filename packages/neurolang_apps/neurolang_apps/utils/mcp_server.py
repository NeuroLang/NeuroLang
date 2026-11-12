"""
The interactive REPL as an MCP server: ``neurolang-query --mcp``.

An MCP client, an LLM such as Claude, queries the engine in a session of
the REPL (see :class:`~neurolang_apps.utils.tui.web.WebRepl`), over the
standard input and output: the ``repl`` tool runs what would be typed at
its prompt, statements and dot-commands, and the rules and relations they
define stay in the session. The tools of the LLM assistant on the engine
(see :class:`~neurolang_apps.utils.llm.agent.Toolbox`) tell it the
predicates, find the values to write as constants and check a query, and
``guide`` gives the syntax, the guidance of the engine and the example
queries closest to a question.

The engine is built while the client starts; a tool called meanwhile
waits for it. The calls run one at a time, on the one session.
"""

import threading
from concurrent.futures import Future
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from neurolang_apps.utils import engine_registry
from neurolang_apps.utils.llm import prompts
from neurolang_apps.utils.llm.agent import Toolbox
from neurolang_apps.utils.llm.context import EngineContext
from neurolang_apps.utils.llm.examples import load_examples, select_examples
from neurolang_apps.utils.tui.web import WebRepl

# The rows of each result the ``repl`` tool returns, by default.
MAX_ROWS = 50
# The examples ``guide`` returns.
GUIDE_EXAMPLES = 8

INSTRUCTIONS = """\
NeuroLang is a probabilistic logic programming language for neuroimaging.
This server is a session of its interactive REPL on the engine {engine}.
Call `guide` with the question first: it gives the syntax, how to answer
on this engine and the examples closest to the question. Then find the
predicates (`list_predicates`, `describe_predicate`) and the values to
write as constants (`search_values`), check the query (`check_query`)
and run it with `repl`, which also runs dot-commands (`.help` lists
them). The rules sent alone stay in the session; those sent with a query
are local to it. `.save` writes a result to a file and `.load` reads a
relation, with paths on the server's machine, relative to its working
directory."""


class McpSession:
    """
    The REPL session on the engine the server builds, its tools as
    functions returning text, run one at a time.
    """

    def __init__(
        self, engine: str, build, initial_program: str = "",
        squall: bool = False,
    ):
        """
        Build the engine with *build* in the background, then run the
        *initial_program* in the session of the REPL on it, in SQUALL if
        *squall*.
        """
        self.engine = engine
        self._squall = squall
        self._lock = threading.Lock()
        self._repl = Future()
        threading.Thread(
            target=self._start, args=(build, initial_program), daemon=True
        ).start()

    def _start(self, build, initial_program: str) -> None:
        try:
            repl = WebRepl(build(), self.engine, html=False)
            repl.app.settings.squall = self._squall
            if initial_program:
                repl.submit(initial_program)
            self._repl.set_result(repl)
        except BaseException as error:  # told to the client by each tool
            self._repl.set_exception(error)

    def _session(self) -> WebRepl:
        """The REPL, once the engine is built, or why it is not."""
        try:
            return self._repl.result()
        except Exception as error:
            raise ToolError(
                f"The engine could not be built: {type(error).__name__}: "
                f"{error}"
            ) from error

    def _language(self, repl) -> str:
        return "squall" if repl.app.settings.squall else "datalog"

    def _toolbox(self, repl) -> Toolbox:
        """The assistant's tools, on the relations the session loaded."""
        context = EngineContext(
            repl.app._nl, self.engine, repl.app.loaded_relations
        )
        return Toolbox(context, self._language(repl))

    def call(self, name: str, /, **arguments) -> str:
        """Return the text of the assistant's tool *name* on *arguments*."""
        with self._lock:
            repl = self._session()
            observation, _ = self._toolbox(repl).call(name, arguments)
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
        parts = [reply.html.rstrip(), *map(_tsv, reply.tables)]
        if reply.failed:
            parts.insert(0, "FAILED")
        return "\n\n".join(part for part in parts if part)

    def guide(self, question: str = "") -> str:
        """
        Return the syntax of the language of the session, the guidance
        of the engine and its examples closest to *question*.
        """
        with self._lock:
            language = self._language(self._session())
        config = (
            engine_registry.get_engine_config(self.engine)
            if self.engine in engine_registry.list_engine_names() else {}
        )
        examples = select_examples(
            load_examples(self.engine), question, language, GUIDE_EXAMPLES
        )
        name = prompts.LANGUAGE_NAMES[language]
        description = " ".join(config.get("description", "").split())
        parts = [f"The engine: {description}"] if description else []
        parts += [
            f"# {name} syntax\n{prompts.reference(language)}",
            f"# Predicates of the engine\n{prompts.SCHEMA_NOTE}",
        ]
        guidance = config.get("assistant_guidance", "")
        if guidance:
            parts.append(
                f"# How to answer on this engine\n{' '.join(guidance.split())}"
            )
        if examples:
            parts.append(
                f"# Examples\n{prompts.format_examples(examples, language)}"
            )
        return "\n\n".join(parts)


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


def server(session: McpSession):
    """Return the MCP server of *session*, with its tools."""
    mcp = MCPServer(
        "neurolang", instructions=INSTRUCTIONS.format(engine=session.engine),
        log_level="WARNING",  # INFO logs every rewrite of the engine
    )

    @mcp.tool()
    def repl(text: str, max_rows: int = MAX_ROWS) -> str:
        """
        Run text in the NeuroLang REPL session, as typed at its prompt:
        statements (rules, facts, queries) and dot-commands (`.help`).
        Returns the output and the first max_rows rows of each result,
        tab-separated; `.view` and `.graph` save their pages in files.
        """
        return session.repl(text, max_rows)

    @mcp.tool()
    def guide(question: str = "") -> str:
        """
        The syntax of the language, how to answer on the engine and the
        example queries closest to the question: read it first.
        """
        return session.guide(question)

    described = {tool["name"]: tool["description"] for tool in prompts.TOOLS}

    @mcp.tool(description=described["list_predicates"])
    def list_predicates() -> str:
        return session.call("list_predicates")

    @mcp.tool(description=described["describe_predicate"])
    def describe_predicate(name: str) -> str:
        return session.call("describe_predicate", name=name)

    @mcp.tool(description=described["search_values"])
    def search_values(
        pattern: str, predicate: str = "", column: str = ""
    ) -> str:
        return session.call(
            "search_values", pattern=pattern,
            predicate=predicate or None, column=column or None,
        )

    @mcp.tool(description=described["check_query"])
    def check_query(query: str) -> str:
        return session.call("check_query", query=query)

    return mcp


def main(args, initial_program: str = "") -> None:
    """Serve the REPL on the engine of the CLI's *args* over stdio."""
    def build():
        return engine_registry.build_engine(
            args.engine, Path(args.data_dir), args.resolution,
            use_cache=not args.no_engine_cache,
        )

    session = McpSession(args.engine, build, initial_program, args.squall)
    server(session).run("stdio")
