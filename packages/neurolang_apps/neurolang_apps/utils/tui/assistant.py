"""
The LLM query assistant in the REPL: ``.ask`` and ``.llm``.

``.ask <question>`` turns the question into a query of the session's
language with an LLM (see :mod:`neurolang.utils.llm`), printing the
steps of the agent as they run. The query is not run: it is put in the
next prompt, to be read, edited and run with Enter, each rule under a
comment line explaining it (``comment_rules``). ``.llm <question>``
does the same with the ReAct agent writing Datalog, whatever the mode
and the configured strategy. ``--steps N`` sets the most steps of the
ReAct agent for one question. ``.llm`` alone, ``.llm provider NAME``,
``.llm strategy icl|react``, ``.llm steps N`` and ``.llm reload`` show
and change the provider, the strategy and the steps of the assistant.

The assistant needs the ``llm`` extra; the REPL works without it.
"""

from dataclasses import dataclass, replace
from typing import NamedTuple, Optional

from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from neurolang_apps.utils import query_highlight

from .commands import command
from .interrupt import call_interruptibly

LLM_DEPENDENCIES = ("httpx",)
LANGUAGE_FLAGS = {"--squall": "squall", "--datalog": "datalog"}
STRATEGY_FLAGS = {"--icl": "icl", "--react": "react"}
ASK_USAGE = (
    ".ask [--squall|--datalog] [--icl|--react] [--steps N] QUESTION"
)
LLM_SETTINGS_USAGE = (
    ".llm [provider NAME|strategy icl|react|steps N|reload]"
)
LLM_USAGE = (
    ".llm [--steps N] QUESTION | .llm [provider NAME|strategy icl|react|"
    "steps N|reload]"
)
# The first words of .llm that change the settings rather than ask.
LLM_SETTINGS = ("provider", "strategy", "steps", "reload")


@dataclass
class AssistantState:
    """
    The assistant of a REPL session.

    *config_path* and *provider* are those requested on the command line
    (``--llm-config``, ``--llm-provider``); *strategy* and *steps* (the
    most steps of the ReAct agent) override the configuration's.
    *client* replaces the provider's chat client (in tests). The
    configuration is read when first needed.
    """

    config_path: Optional[str] = None
    provider: Optional[str] = None
    strategy: Optional[str] = None
    steps: Optional[int] = None
    client: object = None
    _config: object = None

    def config(self):
        """Return the configuration, reading it the first time."""
        if self._config is None:
            from neurolang_apps.utils.llm import load_config
            self._config = load_config(self.config_path)
        return self._config

    def reload(self):
        """Read the configuration again and return it."""
        self._config = None
        return self.config()


def _llm_available(app) -> bool:
    """Return True if the LLM dependencies are installed, else say so."""
    try:
        import httpx  # noqa: F401
    except ImportError:
        app.failed = True
        app.console.print(
            "The LLM assistant needs the optional dependency "
            f"{' and '.join(LLM_DEPENDENCIES)}. Install it with:\n"
            '    pip install "neurolang[llm]"', style="yellow", markup=False,
        )
        return False
    return True


class AskRequest(NamedTuple):
    """
    The options and the question of ``.ask`` or ``.llm``: the language,
    the strategy and the most steps of the agent, None when not given.
    """

    language: Optional[str]
    strategy: Optional[str]
    question: str
    steps: Optional[int] = None


def parse_steps(text: str) -> int:
    """Return the number of steps *text*, a positive integer."""
    if not text.isdigit() or int(text) < 1:
        raise ValueError(
            f"The number of steps is a positive integer, not {text!r}."
        )
    return int(text)


def parse_ask(arg: str, usage: str = ASK_USAGE) -> AskRequest:
    """
    Return the request of ``.ask`` *arg*: its options and its question.

    The options come first: ``--squall``/``--datalog``,
    ``--icl``/``--react`` and ``--steps N``. The question is the rest of
    the line, as typed.
    """
    language = strategy = steps = None
    rest = arg.strip()
    while rest.startswith("--"):
        flag, _, remainder = rest.partition(" ")
        if flag in LANGUAGE_FLAGS:
            language = LANGUAGE_FLAGS[flag]
        elif flag in STRATEGY_FLAGS:
            strategy = STRATEGY_FLAGS[flag]
        elif flag == "--steps":
            value, _, remainder = remainder.strip().partition(" ")
            steps = parse_steps(value)
        else:
            raise ValueError(f"Unknown option {flag}. Usage: {usage}")
        rest = remainder.strip()
    return AskRequest(language, strategy, rest, steps)


def _target_language(app, flag: Optional[str], config) -> str:
    """Return the language of the query: flag, configuration or mode."""
    if flag:
        return flag
    if config.assistant.language != "auto":
        return config.assistant.language
    return "squall" if app.settings.squall else "datalog"


# The rich styles of the kinds of fragments of a query, as the prompt
# highlights them (``editing.STYLE``).
QUERY_STYLES = {
    "comment": "italic white",  # ANSI's light grey, as at the prompt
    "keyword": "bold magenta",
    "operator": "bold blue",
    "predicate": "blue",
    "string": "green",
    "number": "cyan",
}


def query_text(query: str, language: str, indent: str = "") -> Text:
    """Return *query* highlighted as *language*, its lines indented."""
    text = Text()
    for n, line in enumerate(query.strip().split("\n")):
        if n:
            text.append("\n")
        text.append(indent)
        for kind, fragment in query_highlight.tokens(
            line, language == "squall"
        ):
            text.append(fragment, style=QUERY_STYLES.get(kind, ""))
    return text


def _call_text(step) -> str:
    """Return the tool call of *step* without its query: ``name(a=v)``."""
    arguments = ", ".join(
        f"{key}={value!r}" for key, value in step.arguments.items()
        if key != "query"
    )
    return f"{step.action}({arguments})"


def _step_printer(app, language: str):
    """
    Return the function printing a step of the agent, dimmed, but for
    the query of its tool call, highlighted as *language*.
    """
    def print_step(step) -> None:
        console = app.err_console
        if step.thought:
            thought = " ".join(step.thought.split())
            if len(thought) > 200:
                thought = thought[:200] + "…"
            console.print(f"  [dim italic]{escape(thought)}[/dim italic]")
        if step.action and step.action != "answer":
            observation = step.observation.strip().splitlines()
            summary = observation[0] if observation else ""
            if len(observation) > 1:
                summary += f" (+{len(observation) - 1} lines)"
            console.print(f"  [dim]→ {escape(_call_text(step))}[/dim]")
            query = step.arguments.get("query")
            if isinstance(query, str) and query.strip():
                console.print(query_text(query, language, "      "))
            console.print(f"    [dim]{escape(summary)}[/dim]")
    return print_step


def _switch_language(app, language: str) -> None:
    """Switch the REPL to *language*, if it is not its mode, and say so."""
    squall = language == "squall"
    if app.settings.squall != squall:
        app.settings.squall = squall
        app.console.print(
            f"[yellow]Switched to {'SQUALL' if squall else 'Datalog'} "
            "mode for the query.[/yellow]"
        )


def runnable_query(result) -> str:
    """
    Return the query of the agent's *result* to run: with its comment
    lines in Datalog, which ignores them, without them in SQUALL.
    """
    if result.commented and result.language == "datalog":
        return result.commented
    return result.query


def _show_result(app, result) -> None:
    """
    Print the query of the agent's *result*, each rule under a comment
    line explaining it if the model wrote them, and put it in the prompt.
    """
    if result.valid:
        app.console.print(Panel(
            query_text(result.commented or result.query, result.language),
            title=f"{result.language.upper()} query", border_style="green",
        ))
        if result.explanation:
            app.console.print(escape(result.explanation), style="dim")
    else:
        app.failed = True
        app.console.print(
            f"[red]No valid query:[/red] {escape(result.error)}"
        )
        if result.query:
            app.console.print(Panel(
                query_text(result.query, result.language),
                title="last attempt", border_style="red",
            ))
    tokens = result.usage.get("input", 0) + result.usage.get("output", 0)
    usage = f"{result.calls} LLM calls"
    if tokens:
        usage += f", {tokens} tokens"
    app.err_console.print(f"[dim]{usage}[/dim]")
    if result.query:
        _switch_language(app, result.language)
        app.pending_input = runnable_query(result)
        app.console.print(
            "[dim]The query is in the prompt: edit it or press Enter to "
            "run it.[/dim]"
        )


@command(".ask", "Write a query answering a question, with an LLM", ASK_USAGE)
def cmd_ask(app, arg: str) -> None:
    """Ask the LLM assistant for a query answering the question *arg*."""
    try:
        request = parse_ask(arg)
    except ValueError as exc:
        app.failed = True
        app.console.print(f"[yellow]{escape(str(exc))}[/yellow]")
        return
    if not request.question:
        app.failed = True
        app.console.print(f"[yellow]Usage: {escape(ASK_USAGE)}[/yellow]")
        return
    if not _llm_available(app):
        return
    _run_assistant(
        app, request.question, request.language, request.strategy,
        request.steps,
    )


def _run_assistant(
    app, question: str, language: Optional[str] = None,
    strategy: Optional[str] = None, steps: Optional[int] = None,
) -> None:
    """
    Write a query answering *question* and put it in the next prompt.

    *language*, *strategy* and *steps* (the most steps of the ReAct
    agent) override those of the session and of the configuration when
    given.
    """
    from neurolang_apps.utils.llm import LLMConfigError, LLMError, make_agent
    state = app.assistant
    try:
        config = state.config()
        provider = config.provider(state.provider)
        language = _target_language(app, language, config)
        assistant = config.assistant
        steps = steps or state.steps
        if steps:
            assistant = replace(assistant, max_steps=steps)
        agent = make_agent(
            app._nl, provider, assistant, language,
            strategy or state.strategy,
            engine_name=app.settings.engine_name,
            loaded=app.loaded_relations, client=state.client,
            on_step=_step_printer(app, language),
        )
        app.err_console.print(
            f"[dim]Asking {provider.name} ({provider.model or provider.api}"
            f", {agent.strategy})...[/dim]"
        )
        result = call_interruptibly(
            lambda: agent.run(question), app._print_stuck_hint
        )
    except KeyboardInterrupt:
        app.failed = True
        app.console.print("[yellow]Interrupted[/yellow]")
        return
    except (LLMConfigError, LLMError) as exc:
        app.failed = True
        app.console.print(f"[red]{escape(str(exc))}[/red]")
        return
    _show_result(app, result)


def _llm_status(app) -> None:
    """Print the configuration of the assistant."""
    state = app.assistant
    config = state.config()
    provider = config.provider(state.provider)
    strategy = state.strategy or config.assistant.strategy
    steps = state.steps or config.assistant.max_steps
    details = provider.base_url or " ".join(provider.command[:1])
    app.console.print(
        f"Provider: [bold]{provider.name}[/bold] — {provider.api} "
        f"{escape(details)}, model {escape(provider.model or '-')}, "
        f"tool calling {provider.tool_calling}\n"
        f"Strategy: {strategy} · steps: {steps} · language: "
        f"{config.assistant.language}\n"
        f"Providers: {', '.join(sorted(config.providers))}\n"
        f"[dim]Configuration: {config.path}[/dim]"
    )


@command(
    ".llm",
    "Write a Datalog query with the ReAct agent (whatever the mode and "
    "strategy), or show or change the LLM settings",
    LLM_USAGE,
)
def cmd_llm(app, arg: str) -> None:
    """
    Ask the ReAct agent for a Datalog query answering the question *arg*,
    or show the assistant's configuration, or change it.

    *arg* changes the settings when its first word is one of
    :data:`LLM_SETTINGS`; any other text is a question.
    """
    if not _llm_available(app):
        return
    action, _, value = arg.strip().partition(" ")
    if action and action not in LLM_SETTINGS:
        try:
            request = parse_ask(arg, LLM_USAGE)
        except ValueError as exc:
            app.failed = True
            app.console.print(f"[yellow]{escape(str(exc))}[/yellow]")
            return
        if request.language or request.strategy or not request.question:
            app.failed = True
            app.console.print(f"[yellow]Usage: {escape(LLM_USAGE)}[/yellow]")
            return
        _run_assistant(
            app, request.question, "datalog", "react", request.steps
        )
        return
    from neurolang_apps.utils.llm import LLMConfigError
    value = value.strip()
    state = app.assistant
    try:
        if action == "provider" and value:
            state.config().provider(value)
            state.provider = value
        elif action == "strategy" and value in STRATEGY_FLAGS.values():
            state.strategy = value
        elif action == "steps" and value:
            state.steps = parse_steps(value)
        elif action == "reload":
            state.reload()
        elif action:
            app.failed = True
            app.console.print(
                f"[yellow]Usage: {escape(LLM_SETTINGS_USAGE)}, or "
                ".llm QUESTION[/yellow]"
            )
            return
        _llm_status(app)
    except (LLMConfigError, ValueError) as exc:
        app.failed = True
        app.console.print(f"[red]{escape(str(exc))}[/red]")
