"""
The assistant on Claude Code, through the Claude Agent SDK.

:class:`ClaudeCodeAgent` is the ReAct agent with Claude Code running the
loop: one conversation per question, the tools of the engine
(:class:`~.agent.Toolbox`) offered as in-process MCP tools, so that they
run in this process, on the engine of the REPL, with native tool calls and
none of Claude Code's own tools. :class:`ClaudeCodeClient` is the same for
a single request (the comments on the rules, the back-translation check,
the ICL strategy). Both run the ``claude`` CLI as :mod:`.claude_cli`
says: with the user's own sign-in (``auth: login``) or an API key.

The SDK is asynchronous: each conversation runs in a thread of its own,
with its own event loop, and the steps of the agent are given to
``on_step`` in the calling thread, as they happen.

It needs the extra ``claude`` (``claude-agent-sdk``).
"""

import asyncio
import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import claude_agent_sdk as sdk
from claude_agent_sdk import query as sdk_query

from . import claude_cli, prompts
from .agent import AgentResult, AgentStep, ReactAgent, truncate
from .config import ProviderConfig
from .providers import LLMError, Reply, render_conversation

SERVER = "neurolang"


@dataclass
class Outcome:
    """What a conversation said: its *text*, the *usage* of its tokens (as
    the other clients report it) and its *turns*."""

    text: str = ""
    usage: dict = field(default_factory=dict)
    turns: int = 0


def run_in_thread(coroutine, items: Optional[queue.Queue] = None,
                  deliver=None):
    """
    Run *coroutine* in a thread of its own and return its result, raising
    what it raised. The *items* it puts in the queue are given to *deliver*
    here, as they come.
    """
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(asyncio.run, coroutine)
        while items is not None:
            try:
                deliver(items.get(timeout=0.05))
            except queue.Empty:
                if future.done():
                    break
        while items is not None and not items.empty():
            deliver(items.get_nowait())
        return future.result()


async def read(stream) -> Outcome:
    """Read the messages of a conversation: its text, usage and turns."""
    outcome, said = Outcome(), []
    async for message in stream:
        if isinstance(message, sdk.AssistantMessage):
            said.append("".join(
                block.text for block in message.content
                if isinstance(block, sdk.TextBlock)
            ))
        elif isinstance(message, sdk.ResultMessage):
            if message.is_error:
                raise LLMError(
                    f"Claude Code: {message.result or message.subtype}"
                )
            usage = message.usage or {}
            cached = usage.get("cache_read_input_tokens") or 0
            outcome.usage = {
                "input": (usage.get("input_tokens") or 0) + cached
                + (usage.get("cache_creation_input_tokens") or 0),
                "output": usage.get("output_tokens") or 0,
                "cached": cached,
            }
            outcome.turns = message.num_turns
            outcome.text = message.result or ""
    outcome.text = "".join(said).strip() or outcome.text
    return outcome


def failing(coroutine):
    """Return *coroutine* raising LLMError for what the SDK raises."""
    async def guarded():
        try:
            return await coroutine
        except sdk.CLINotFoundError as exc:
            raise LLMError(
                "the claude CLI was not found: install Claude Code and "
                f"sign in with `claude` ({exc})."
            ) from exc
        except sdk.ClaudeSDKError as exc:
            raise LLMError(f"Claude Code: {exc}") from exc
        except asyncio.TimeoutError as exc:
            raise LLMError("Claude Code: the request timed out.") from exc
    return guarded()


def options_for(provider: ProviderConfig, cli: claude_cli.CliSettings,
                system: str, **options) -> "sdk.ClaudeAgentOptions":
    """Return the SDK options running *provider*'s CLI with the system
    prompt *system*, Claude Code's own tools and settings off."""
    return sdk.ClaudeAgentOptions(
        system_prompt=system, tools=[], setting_sources=[],
        model=provider.model, cli_path=cli.path, env=cli.env,
        extra_args={"no-session-persistence": None}, **options,
    )


class ClaudeCodeClient:
    """
    Chat client of Claude Code: one request, with no tools, a conversation
    of one turn (see :func:`~.providers.render_conversation`).
    """

    def __init__(self, provider: ProviderConfig, timeout: float = 60.0,
                 transport=None):
        self.provider = provider
        self.timeout = timeout

    def complete(self, messages: list, tools: Optional[list] = None,
                 temperature: float = 0.0) -> Reply:
        """Return Claude Code's reply to the conversation *messages*."""
        system = "\n\n".join(
            m["content"] for m in messages if m["role"] == "system"
        )
        prompt = render_conversation(messages, with_system=False)
        self.provider.api_key()  # refuses a login on a shared server

        async def ask(options):
            return await asyncio.wait_for(
                read(sdk_query(prompt=prompt, options=options)), self.timeout
            )

        with claude_cli.settings(self.provider) as cli:
            options = options_for(
                self.provider, cli, system, max_turns=1, mcp_servers={}
            )
            outcome = run_in_thread(failing(ask(options)))
        return Reply(outcome.text, usage=outcome.usage)

    def close(self) -> None:
        """Nothing to close."""


class Bridge:
    """
    The tools of the engine for one conversation: runs a call on the
    :class:`~.agent.Toolbox`, records its step and, for a final answer,
    checks it as the ReAct agent does.
    """

    def __init__(self, agent: "ClaudeCodeAgent", result: AgentResult,
                 toolbox, question: str):
        self.agent, self.result = agent, result
        self.toolbox, self.question = toolbox, question
        self.rejections = 0
        self.accepted = False

    def call(self, name: str, arguments: dict) -> str:
        """Return the observation of the tool *name* called with
        *arguments*."""
        agent, result = self.agent, self.result
        observation, final = self.toolbox.call(name, arguments)
        observation = truncate(observation)
        agent._step(result, AgentStep("", name, arguments, observation))
        if final is None or self.accepted:
            return observation
        rounds = agent.config.verify_rounds
        if rounds > 0 and self.rejections < rounds:
            rejection = agent._verify(
                result, self.question, final[0], self.toolbox.answer_rows
            )
            if rejection is not None:
                self.rejections += 1
                return rejection
        result.query, result.explanation = final
        result.frames = self.toolbox.answer_frames
        result.valid = self.accepted = True
        return observation


class ClaudeCodeAgent(ReactAgent):
    """
    The ReAct agent with Claude Code running the loop (see the module).

    A final answer is checked, and verified as the ReAct agent's is; the
    conversation is over when the model stops, or after ``max_steps``
    turns, when the ReAct agent's rule for running out of steps applies.
    """

    def _attempt(self, question: str, temperature: float) -> AgentResult:
        if self.provider.api != "claude-code":  # a fallback provider
            return super()._attempt(question, temperature)
        self.provider.api_key()  # refuses a login on a shared server
        result = AgentResult(language=self.language)
        toolbox = self.toolbox()
        self.bridge = Bridge(self, result, toolbox, question)
        system = self.system_prompt(question, text_tools=False)
        prompt = self._user_message(question)
        steps, on_step = queue.Queue(), self.on_step
        self.on_step = steps.put
        try:
            with claude_cli.settings(self.provider) as cli:
                options = options_for(
                    self.provider, cli, system,
                    max_turns=self.config.max_steps,
                    mcp_servers={SERVER: self.server()},
                    allowed_tools=[
                        f"mcp__{SERVER}__{tool['name']}"
                        for tool in prompts.TOOLS
                    ],
                )
                outcome = run_in_thread(
                    failing(self.converse(prompt, options)), steps, on_step
                )
        finally:
            self.on_step = on_step
        result.usage.update(outcome.usage)
        result.calls += outcome.turns
        if self.bridge.accepted:
            return result
        return self._out_of_steps(result)

    def toolbox(self):
        """Return the tools of the engine for a conversation."""
        from .agent import Toolbox

        return Toolbox(
            self.context, self.language, self.config.preview_rows,
            self.config.query_timeout, self.config.query_memory_mb,
        )

    def server(self):
        """Return the tools of the bridge as an in-process MCP server."""
        bridge = self.bridge

        def make(spec):
            @sdk.tool(spec["name"], spec["description"], spec["parameters"])
            async def handler(arguments):
                text = await asyncio.to_thread(
                    bridge.call, spec["name"], arguments
                )
                return {"content": [{"type": "text", "text": text}]}
            return handler

        return sdk.create_sdk_mcp_server(
            SERVER, tools=[make(spec) for spec in prompts.TOOLS]
        )

    async def converse(self, prompt: str, options) -> Outcome:
        """Run the conversation, for the time the steps of the agent may
        take at most."""
        seconds = self.config.timeout * max(self.config.max_steps, 1)
        return await asyncio.wait_for(
            read(sdk_query(prompt=prompt, options=options)), seconds
        )
