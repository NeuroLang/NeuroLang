"""
Agents translating a question into a NeuroLang query with an LLM.

Two strategies are offered, following the text-to-SQL and text-to-SPARQL
literature:

* :class:`IclAgent`, in-context learning: the prompt holds the syntax of
  the language, the predicates of the engine and the examples closest to
  the question; the query the model writes is checked on the engine and,
  when it fails, the error is sent back for a correction (validation and
  reprompting). With several samples, the query most candidates agree on
  is kept (self-consistency).
* :class:`ReactAgent`, ReAct: the model reasons and calls tools to look
  at the predicates and their values, check and run its query, until it
  gives its final answer. The tools are called with the provider's
  native tool calls or with a Thought / Action / Observation text
  protocol that any chat model can follow.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

from neurolang.utils import engine_registry

from . import prompts
from .config import AssistantConfig, ProviderConfig
from .context import EngineContext
from .examples import load_examples, select_examples
from .providers import make_client
from .validation import check_query, run_preview

MAX_OBSERVATION = 2000
_FENCE = re.compile(r"```[ \t]*([A-Za-z]*)[ \t]*\n(.*?)```", re.DOTALL)
_ACTION = re.compile(
    r"^\s*Action\s*:\s*([A-Za-z_]+)", re.MULTILINE | re.IGNORECASE
)
_ACTION_INPUT = re.compile(
    r"^\s*Action[ _]Input\s*:", re.MULTILINE | re.IGNORECASE
)
_THOUGHT = re.compile(
    r"^\s*Thought\s*:\s*(.*?)(?=^\s*Action\s*:|\Z)",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)
_FINAL = re.compile(r"^\s*Final\s+Answer\s*:", re.MULTILINE | re.IGNORECASE)
_OBSERVATION = re.compile(r"^\s*Observation\s*:", re.MULTILINE)
# A keyword line as models write it, emphasis included: **Action:** x
_KEYWORD_LINE = re.compile(
    r"^[\s*_`#>-]*(Thought|Action[ _]Input|Action|Final[ _]Answer)"
    r"[*_`\s]*:[*_`\s]*(.*)$",
    re.IGNORECASE,
)


def _normalise_keywords(text: str) -> str:
    """Return *text* with its keyword lines without markdown emphasis."""
    lines = []
    for line in text.splitlines():
        match = _KEYWORD_LINE.match(line)
        if match:
            keyword = match.group(1).replace("_", " ")
            value = match.group(2).strip().strip("`*_").strip()
            line = f"{keyword}: {value}"
        lines.append(line)
    return "\n".join(lines)


@dataclass
class AgentStep:
    """
    A step of an agent: a tool call or a query attempt.

    *action* is the tool called (``query`` for an ICL attempt) with its
    *arguments*, and *observation* what it returned.
    """

    thought: str = ""
    action: str = ""
    arguments: dict = field(default_factory=dict)
    observation: str = ""


@dataclass
class AgentResult:
    """
    The outcome of an agent: its query, if *valid*, or its *error*.

    *usage* counts the tokens sent (``input``) and received
    (``output``), and *calls* the requests to the model.
    """

    query: str = ""
    language: str = "squall"
    explanation: str = ""
    valid: bool = False
    error: str = ""
    steps: list = field(default_factory=list)
    usage: Counter = field(default_factory=Counter)
    calls: int = 0


def truncate(text: str, limit: int = MAX_OBSERVATION) -> str:
    """Return *text* cut to *limit* characters, telling that it was cut."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… ({len(text) - limit} more characters)"


def extract_query(text: str, language: str) -> Optional[str]:
    """
    Return the query of the model's reply *text*, if any.

    It is the first fenced code block, preferring one labelled with
    *language*.
    """
    blocks = _FENCE.findall(text or "")
    for label, body in blocks:
        if label.lower() == language:
            return body.strip()
    for label, body in blocks:
        if label.lower() in ("", "text", "prolog", "sql", language):
            return body.strip()
    return None


def _json_object(text: str):
    """Return the first JSON object in *text*, or raise ValueError."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError("not a JSON object")
    return value


@dataclass
class TextAction:
    """An action parsed from a reply of the text protocol."""

    thought: str
    name: str = ""
    arguments: dict = field(default_factory=dict)
    error: str = ""


def parse_text_action(text: str, language: str) -> TextAction:
    """
    Parse the Thought / Action / Action Input of a text protocol reply.

    A reply hallucinating its own ``Observation:`` is cut there. A
    ``Final Answer:`` or a lone fenced query is read as a call of
    ``final_answer``. On a malformed reply, *error* tells the model
    how to write its action.
    """
    observed = _OBSERVATION.search(text)
    if observed:
        text = text[:observed.start()]
    text = _normalise_keywords(text)
    thought = _THOUGHT.search(text)
    thought = thought.group(1).strip() if thought else ""
    action = _ACTION.search(text)
    if action is None:
        final = _FINAL.search(text)
        rest = text[final.end():] if final else text
        try:
            arguments = _json_object(rest) if final else None
        except ValueError:
            arguments = None
        if arguments is None:
            query = extract_query(rest, language)
            if query is not None:
                arguments = {"query": query, "explanation": ""}
        if arguments is not None:
            return TextAction(thought, "final_answer", arguments)
        return TextAction(
            thought or text.strip(),
            error="Your reply has no action. Write `Action: <tool>` and "
            "`Action Input: <JSON object>`, e.g.\nAction: check_query\n"
            'Action Input: {"query": "..."}',
        )
    name = action.group(1)
    arguments_start = _ACTION_INPUT.search(text, action.end())
    if arguments_start is None:
        # Action: tool({...}) or Action: tool {...} on one line
        line_end = text.find("\n", action.end())
        rest = text[action.end():] if line_end < 0 else text[
            action.end():line_end
        ]
        if "{" in rest:
            try:
                return TextAction(thought, name, _json_object(rest))
            except ValueError:
                pass
        return TextAction(thought, name, {})
    try:
        arguments = _json_object(text[arguments_start.end():])
    except ValueError:
        return TextAction(
            thought, name,
            error="The Action Input is not a JSON object. Write it on one "
            'line, e.g. Action Input: {"query": "obtain every Study."}',
        )
    return TextAction(thought, name, arguments)


def _gave_rows(observation: str) -> bool:
    """Whether the observation of run_query shows rows."""
    return bool(re.match(r"(\w+: )?[1-9]\d* rows", observation))


class Toolbox:
    """
    The tools of the ReAct agent on an engine.

    :meth:`call` returns the observation of a tool call and, for an
    accepted ``final_answer``, the query and its explanation.
    """

    def __init__(
        self, context: EngineContext, language: str, preview_rows: int = 10
    ):
        self.context = context
        self.language = language
        self.preview_rows = preview_rows

    def call(self, name: str, arguments: dict):
        """Run the tool *name* with *arguments*: (observation, final)."""
        handler = getattr(self, f"tool_{name}", None)
        if handler is None:
            tools = ", ".join(tool["name"] for tool in prompts.TOOLS)
            return f"Unknown tool {name!r}. The tools are: {tools}.", None
        if "_invalid" in arguments:
            return "The arguments are not a JSON object.", None
        try:
            return handler(**arguments)
        except TypeError as exc:
            return f"Wrong arguments for {name}: {exc}", None
        except Exception as exc:
            return f"Error: {type(exc).__name__}: {exc}", None

    def tool_list_predicates(self):
        return self.context.schema_summary(), None

    def tool_describe_predicate(self, name: str):
        return self.context.describe(name), None

    def tool_search_values(self, pattern: str, predicate=None, column=None):
        return self.context.search_values(pattern, predicate, column), None

    def tool_check_query(self, query: str):
        result = check_query(self.context.nl, query, self.language)
        if not result.ok:
            return f"The query fails:\n{result.message}", None
        text = "The query is valid."
        if result.datalog:
            text += f" Its Datalog translation:\n{result.datalog}"
        return text, None

    def tool_run_query(self, query: str):
        check = check_query(self.context.nl, query, self.language)
        if not check.ok:
            return f"The query fails:\n{check.message}", None
        preview = run_preview(
            self.context.nl, query, self.language, self.preview_rows,
            self.context.engine_name if self.context.declared else "",
        )
        return preview.text, None

    def tool_final_answer(self, query: str, explanation: str = ""):
        result = check_query(self.context.nl, query, self.language)
        if not result.ok:
            return (
                f"The answer is rejected: the query fails:\n{result.message}"
                "\nCorrect it and call final_answer again."
            ), None
        return "Accepted.", (query.strip(), explanation.strip())


class Agent:
    """
    Base of the agents: the engine, the model and the prompt.

    *client* is the chat client of *provider* (by default made from its
    configuration), *on_step* is called with each :class:`AgentStep` as
    it ends, and the examples whose question is *exclude_question* are
    left out of the prompt (to evaluate on the example bank).

    The examples of the prompt are chosen among *examples*, a list of
    :class:`~.examples.Example`, by default the engine's bank, and the
    engine is described by *description*, by default the description
    of its configuration.
    """

    strategy = ""

    def __init__(
        self, nl, provider: ProviderConfig, config: AssistantConfig,
        language: str, engine_name: str = "", client=None, loaded=None,
        on_step: Optional[Callable] = None, exclude_question: str = "",
        examples: Optional[list] = None, description: Optional[str] = None,
    ):
        self.nl = nl
        self.provider = provider
        self.config = config
        self.language = language
        self.engine_name = engine_name
        self.client = client or make_client(provider, config.timeout)
        self.context = EngineContext(nl, engine_name, loaded)
        self.on_step = on_step or (lambda step: None)
        self.exclude_question = exclude_question
        self.examples = examples
        self.description = description

    def system_prompt(self, question: str, text_tools: bool = False) -> str:
        """Return the system prompt for *question*."""
        bank = self.examples
        if bank is None:
            bank = load_examples(self.engine_name)
        examples = select_examples(
            bank, question, self.language, self.config.few_shot_k,
            exclude=self.exclude_question,
        )
        description = self.description
        if description is None:
            description = ""
            if self.engine_name in engine_registry.list_engine_names():
                description = engine_registry.get_engine_config(
                    self.engine_name
                ).get("description", "")
        return prompts.system_prompt(
            self.language, self.context.schema_summary(), examples,
            self.strategy, text_tools, description,
        )

    def _complete(self, result: AgentResult, messages, tools=None,
                  temperature=None):
        """Return the model's reply to *messages*, counting its usage."""
        reply = self.client.complete(
            messages, tools,
            self.config.temperature if temperature is None else temperature,
        )
        result.calls += 1
        result.usage.update(reply.usage)
        return reply

    def _step(self, result: AgentResult, step: AgentStep) -> None:
        """Record *step* in *result* and report it."""
        result.steps.append(step)
        self.on_step(step)

    def run(self, question: str) -> AgentResult:
        """Return the query answering *question*."""
        raise NotImplementedError


class IclAgent(Agent):
    """In-context learning, then validation and repair of the query."""

    strategy = "icl"

    def run(self, question: str) -> AgentResult:
        samples = max(1, self.config.samples)
        if samples == 1:
            return self._attempt(question, self.config.temperature)
        temperature = max(self.config.temperature, 0.7)
        candidates = [
            self._attempt(question, temperature) for _ in range(samples)
        ]
        return self._vote(candidates)

    def _attempt(self, question: str, temperature: float) -> AgentResult:
        """Ask for a query and correct it until it passes the checks."""
        result = AgentResult(language=self.language)
        messages = [
            {"role": "system", "content": self.system_prompt(question)},
            {"role": "user", "content": question},
        ]
        for _ in range(self.config.max_repairs + 1):
            reply = self._complete(result, messages, temperature=temperature)
            query = extract_query(reply.text, self.language)
            if query is None:
                result.error = (
                    reply.text.strip() or "The model wrote no query."
                )
                self._step(result, AgentStep(reply.text.strip(), "answer"))
                return result
            check = check_query(self.nl, query, self.language)
            self._step(result, AgentStep(
                "", "query", {"query": query},
                "valid" if check.ok else check.message,
            ))
            result.query = query
            result.explanation = _FENCE.sub("", reply.text).strip()
            if check.ok:
                result.valid = True
                result.error = ""
                return result
            result.error = check.message
            messages += [
                {"role": "assistant", "content": reply.text},
                {"role": "user", "content": prompts.repair_prompt(
                    query, check.message, self.language
                )},
            ]
        return result

    def _fingerprint(self, query: str):
        """Return what identifies the result of *query*, for the vote."""
        preview = run_preview(self.nl, query, self.language, 0)
        if preview.ok and preview.frame is not None:
            rows = preview.frame.astype(str).itertuples(index=False)
            return ("rows", frozenset(map(tuple, rows)))
        return ("text", " ".join(query.split()))

    def _vote(self, candidates: list) -> AgentResult:
        """Return the valid candidate whose result most candidates give."""
        valid = [c for c in candidates if c.valid]
        chosen = candidates[0]
        if valid:
            fingerprints = [self._fingerprint(c.query) for c in valid]
            counts = Counter(fingerprints)
            chosen = max(
                zip(valid, fingerprints), key=lambda pair: counts[pair[1]]
            )[0]
        for candidate in candidates:
            if candidate is not chosen:
                chosen.usage.update(candidate.usage)
                chosen.calls += candidate.calls
        return chosen


class ReactAgent(Agent):
    """ReAct: reasoning and tool calls until a checked final answer."""

    strategy = "react"

    @property
    def native(self) -> bool:
        """Whether the provider's native tool calls are used."""
        return self.provider.tool_calling == "native"

    def run(self, question: str) -> AgentResult:
        result = AgentResult(language=self.language)
        toolbox = Toolbox(
            self.context, self.language, self.config.preview_rows
        )
        messages = [
            {"role": "system", "content": self.system_prompt(
                question, text_tools=not self.native
            )},
            {"role": "user", "content": question},
        ]
        turn = self._native_turn if self.native else self._text_turn
        for _ in range(self.config.max_steps):
            final = turn(result, messages, toolbox)
            if final is not None:
                result.query, result.explanation = final
                result.valid = True
                return result
        ran = next((
            s for s in reversed(result.steps)
            if s.action == "run_query" and _gave_rows(s.observation)
        ), None)
        if ran is not None:
            # Out of steps, the last query that gave rows is the answer.
            result.query = ran.arguments["query"]
            result.valid = True
            result.explanation = (
                f"The last query that gave rows, after "
                f"{self.config.max_steps} steps."
            )
            return result
        last = next(
            (s for s in reversed(result.steps) if "query" in s.arguments), None
        )
        if last is not None:
            result.query = last.arguments["query"]
        result.error = (
            f"No valid query after {self.config.max_steps} steps."
        )
        return result

    def _call(self, result, toolbox, thought, name, arguments):
        """Call a tool, record the step and return (observation, final)."""
        observation, final = toolbox.call(name, arguments)
        observation = truncate(observation)
        self._step(result, AgentStep(thought, name, arguments, observation))
        return observation, final

    def _native_turn(self, result, messages, toolbox):
        """Run one turn with the provider's tool calls."""
        reply = self._complete(result, messages, prompts.TOOLS)
        if not reply.tool_calls:
            query = extract_query(reply.text, self.language)
            messages.append({"role": "assistant", "content": reply.text})
            if query is not None:
                observation, final = self._call(
                    result, toolbox, reply.text, "final_answer",
                    {"query": query, "explanation": ""},
                )
                if final is not None:
                    return final
                messages.append({"role": "user", "content": observation})
                return None
            messages.append({
                "role": "user",
                "content": "Call a tool, or final_answer with your query.",
            })
            self._step(result, AgentStep(reply.text.strip(), "answer"))
            return None
        messages.append({
            "role": "assistant", "content": reply.text,
            "tool_calls": reply.tool_calls,
        })
        accepted = None
        for call in reply.tool_calls:
            observation, final = self._call(
                result, toolbox, reply.text.strip(), call.name, call.arguments
            )
            messages.append({
                "role": "tool", "tool_call_id": call.id, "name": call.name,
                "content": observation,
            })
            accepted = accepted or final
        return accepted

    def _text_turn(self, result, messages, toolbox):
        """Run one turn of the Thought / Action / Observation protocol."""
        reply = self._complete(result, messages)
        action = parse_text_action(reply.text, self.language)
        observed = _OBSERVATION.search(reply.text)
        content = reply.text[:observed.start()] if observed else reply.text
        messages.append({"role": "assistant", "content": content.strip()})
        if action.error:
            self._step(result, AgentStep(action.thought, "", {}, action.error))
            messages.append(
                {"role": "user", "content": prompts.observation(action.error)}
            )
            return None
        observation, final = self._call(
            result, toolbox, action.thought, action.name, action.arguments
        )
        if final is None:
            messages.append(
                {"role": "user", "content": prompts.observation(observation)}
            )
        return final


AGENTS = {"icl": IclAgent, "react": ReactAgent}


def make_agent(
    nl, provider: ProviderConfig, config: AssistantConfig, language: str,
    strategy: Optional[str] = None, **options,
) -> Agent:
    """Return the agent of *strategy* (default: the configuration's)."""
    return AGENTS[strategy or config.strategy](
        nl, provider, config, language, **options
    )
