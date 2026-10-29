"""
The LLM query assistant in the REPL: ``.ask`` and ``.llm``.

``.ask <question>`` turns the question into a query of the session's
language with an LLM (see :mod:`neurolang_apps.utils.llm`), printing the
steps of the agent as they run. The query is not run: it is put in the
next prompt, to be read, edited and run with Enter, each rule under a
comment line explaining it (``comment_rules``). ``.llm <question>``
does the same with the ReAct agent writing Datalog, whatever the mode
and the configured strategy. ``--steps N`` sets the most steps of the
ReAct agent for one question. ``.llm`` alone, ``.llm provider NAME``,
``.llm strategy icl|react``, ``.llm steps N`` and ``.llm reload`` show
and change the provider, the strategy and the steps of the assistant.
When sharing is on (see :mod:`neurolang_apps.utils.llm.feedback`), each
answer is posted to a feedback server, and ``.rate`` and ``.refine``
rate and correct the last one there.

The assistant needs the ``llm`` extra; the REPL works without it.
"""

from dataclasses import dataclass, replace
from typing import NamedTuple, Optional

from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from neurolang_apps.utils import query_highlight

from .commands import HELP_TOPICS, command
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
    _feedback: object = None
    # The last answer, for .rate and .refine: its id on the feedback
    # server, its query and the REPL's last query when it was asked.
    last_answer: Optional[dict] = None

    def config(self):
        """Return the configuration, reading it the first time."""
        if self._config is None:
            from neurolang_apps.utils.llm import load_config
            self._config = load_config(self.config_path)
        return self._config

    def feedback(self, say):
        """
        Return the client of the feedback server, None if sharing is
        off (see :mod:`neurolang_apps.utils.llm.feedback`).
        """
        if self._feedback is None:
            from neurolang_apps.utils.llm.feedback import feedback_client
            self._feedback = feedback_client(self.config(), say)
        return self._feedback

    def reload(self):
        """Read the configuration again and return it."""
        self._config = self._feedback = None
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


# The events the assistant sends to ``app.agent_events`` while it works,
# dicts with an ``event`` key: ``ask`` (question, provider, model,
# strategy), then for each request to the model ``request`` (call) and
# ``reply`` (call, text, tool_calls, usage, seconds or error; the model has
# answered and its tool is about to run), and for each step of the agent
# ``step`` (thought, action, arguments, observation); ``done`` (valid,
# query, explanation, error, calls, tokens, seconds, verified) ends them.
AGENT_EVENTS = ("ask", "request", "reply", "step", "done")


def _step_event(step) -> dict:
    """Return the ``step`` event of the finished *step* of the agent."""
    return {
        "event": "step", "thought": step.thought, "action": step.action,
        "arguments": step.arguments, "observation": step.observation,
    }


def _also_sending(on_step, events):
    """Return *on_step* sending the ``step`` event of each step too."""
    def sending(step) -> None:
        on_step(step)
        events(_step_event(step))
    return sending


def _done_event(result) -> dict:
    """Return the ``done`` event of the *result* of the agent."""
    return {
        "event": "done", "valid": result.valid, "query": result.query,
        "explanation": result.explanation, "error": result.error,
        "calls": result.calls, "tokens": sum(result.usage.values()),
        "seconds": result.seconds, "verified": result.verified,
    }


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


def _show_verification(app, result) -> None:
    """Print the query read back in English and the verdict of its check."""
    if not result.back_translation:
        return
    app.console.print(
        f"[dim]Reads back as: {escape(result.back_translation)}[/dim]"
    )
    if result.verified is False:
        app.console.print(
            "[yellow]The check found that the query may not answer the "
            f"question: {escape(result.verification)}[/yellow]"
        )


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
        _show_verification(app, result)
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
        client, on_step = state.client, _step_printer(app, language)
        events = app.agent_events
        if events:
            from neurolang_apps.utils.llm import make_client
            from neurolang_apps.utils.llm.trace import TracingClient

            client = TracingClient(
                client or make_client(provider, assistant.timeout), events
            )
            on_step = _also_sending(on_step, events)
        agent = make_agent(
            app._nl, provider, assistant, language,
            strategy or state.strategy,
            engine_name=app.settings.engine_name,
            loaded=app.loaded_relations, client=client, on_step=on_step,
        )
        app.err_console.print(
            f"[dim]Asking {provider.name} ({provider.model or provider.api}"
            f", {agent.strategy})...[/dim]"
        )
        if events:
            events({
                "event": "ask", "question": question,
                "provider": provider.name,
                "model": provider.model or provider.api,
                "strategy": agent.strategy,
            })
        result = call_interruptibly(
            lambda: agent.run(question), app._print_stuck_hint
        )
        if events:
            events(_done_event(result))
    except KeyboardInterrupt:
        app.failed = True
        app.console.print("[yellow]Interrupted[/yellow]")
        return
    except (LLMConfigError, LLMError) as exc:
        app.failed = True
        app.console.print(f"[red]{escape(str(exc))}[/red]")
        return
    _show_result(app, result)
    feedback = state.feedback(_say(app))
    if feedback:
        answer = feedback.share(
            question, result, provider, agent.strategy,
            app.settings.engine_name,
        )
        state.last_answer = {
            "id": answer, "query": result.query, "ran": app.last_query,
        }
        if answer:
            app.err_console.print(
                f"[dim]Shared as {answer}: rate it with .rate good|bad|"
                "unsure \\[comment], correct it with .refine.[/dim]"
            )


def _say(app):
    """Return the function printing a message of the feedback client."""
    return lambda text: app.err_console.print(
        text, style="yellow", markup=False
    )


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


FEEDBACK_HELP = """[bold cyan]Rating the assistant's answers[/bold cyan]

When sharing is on, each answer of .ask and .llm is posted to a feedback
server, where people rate and correct the answers to improve the
assistant. Turn it on in the LLM configuration (llm.yaml):

  feedback:
    enabled: true
    url: https://feedback.example.org
    user: your name                   (optional)

or with the environment variable NEUROLANG_LLM_FEEDBACK_URL.

  .rate good|bad|unsure [COMMENT]     rate the last answer
  .rate ID good|bad|unsure [COMMENT]  rate the answer ID
  .refine                             put the last answer's query in the
                                      prompt: edit it and run it
  .refine submit [COMMENT]            check the query you last ran and
                                      share it as the correct query
"""
HELP_TOPICS["feedback"] = FEEDBACK_HELP
VERDICTS = {
    "good": "correct", "bad": "incorrect", "unsure": "unsure",
    "correct": "correct", "incorrect": "incorrect",
}
RATE_USAGE = ".rate [ID] good|bad|unsure [COMMENT]"
REFINE_USAGE = ".refine [submit [COMMENT]]"


def _feedback_or_explain(app):
    """Return the feedback client, or say how to turn sharing on."""
    from neurolang_apps.utils.llm import LLMConfigError
    try:
        feedback = app.assistant.feedback(_say(app))
    except LLMConfigError as exc:
        feedback, message = None, str(exc)
    else:
        message = (
            "Sharing the answers is off: turn it on with feedback in the "
            "LLM configuration or NEUROLANG_LLM_FEEDBACK_URL (.help "
            "feedback)."
        )
    if feedback is None:
        app.failed = True
        app.console.print(message, style="yellow", markup=False)
    return feedback


def _last_answer_id(app):
    """Return the id of the last shared answer, or say there is none."""
    answer = app.assistant.last_answer
    if answer and answer["id"]:
        return answer["id"]
    app.failed = True
    app.console.print(
        "The last answer was not shared." if answer else
        "No answer to rate: ask a question with .ask or .llm first.",
        style="yellow", markup=False,
    )
    return None


def _post_rating(app, feedback, answer_id, verdict, comment, *corrected):
    """Post a rating and say it, or the error."""
    from neurolang_apps.utils.llm.feedback import FeedbackError
    try:
        feedback.rate(answer_id, verdict, comment, *corrected)
    except FeedbackError as exc:
        app.failed = True
        app.console.print(f"[red]{escape(str(exc))}[/red]")
        return
    app.console.print(f"Thank you: answer {answer_id} rated {verdict}.")


@command(".rate", "Rate the last answer of the assistant", RATE_USAGE)
def cmd_rate(app, arg: str) -> None:
    """Rate the last answer, or the answer ID, on the feedback server."""
    if not _llm_available(app):
        return
    words = arg.split(None, 1)
    answer_id = None
    if words and words[0].lower() not in VERDICTS:
        answer_id, words = words[0], words[1:] and words[1].split(None, 1)
    if not words or words[0].lower() not in VERDICTS:
        app.failed = True
        app.console.print(f"[yellow]Usage: {escape(RATE_USAGE)}[/yellow]")
        return
    feedback = _feedback_or_explain(app)
    answer_id = feedback and (answer_id or _last_answer_id(app))
    if answer_id:
        comment = words[1] if len(words) > 1 else ""
        _post_rating(
            app, feedback, answer_id, VERDICTS[words[0].lower()], comment
        )


@command(
    ".refine", "Correct the query of the last answer of the assistant",
    REFINE_USAGE,
)
def cmd_refine(app, arg: str) -> None:
    """
    Put the last answer's query in the prompt or, with ``submit``,
    check the query last run and share it as the correct query.
    """
    if not _llm_available(app):
        return
    action, _, comment = arg.strip().partition(" ")
    if action not in ("", "submit"):
        app.failed = True
        app.console.print(f"[yellow]Usage: {escape(REFINE_USAGE)}[/yellow]")
        return
    feedback = _feedback_or_explain(app)
    answer_id = feedback and _last_answer_id(app)
    if not answer_id:
        return
    answer = app.assistant.last_answer
    if not action:
        app.pending_input = answer["query"]
        app.console.print(
            "The query is in the prompt: correct it, run it with Enter, "
            "then share it with .refine submit [comment].", markup=False,
        )
        return
    if app.last_query is answer["ran"] or (
        app.last_query[0].split() == answer["query"].split()
    ):
        app.failed = True
        app.console.print(
            "Run the corrected query first: .refine puts the answer's "
            "query in the prompt.", style="yellow", markup=False,
        )
        return
    from neurolang_apps.utils.llm.validation import check_query, run_preview
    program, squall = app.last_query
    language = "squall" if squall else "datalog"
    check = check_query(app._nl, program, language)
    if not check.ok:
        app.failed = True
        app.console.print(
            f"[red]The query fails, it is not shared:[/red] "
            f"{escape(check.message)}"
        )
        return
    preview = run_preview(
        app._nl, program, language,
        app.assistant.config().assistant.preview_rows,
        app.settings.engine_name,
    )
    _post_rating(
        app, feedback, answer_id, "incorrect", comment.strip(), program,
        preview.text,
    )
