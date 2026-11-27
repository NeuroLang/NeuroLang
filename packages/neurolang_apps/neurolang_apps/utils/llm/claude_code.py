"""
The assistant on Claude Code, through the Claude Agent SDK.

:class:`ClaudeCodeAgent` is the ReAct agent with Claude Code running the
loop: one conversation per question, the tools of the engine
(:class:`~.agent.Toolbox`) offered as in-process MCP tools, so that they
run in this process, on the engine of the REPL, with native tool calls and
none of Claude Code's own tools. :class:`ClaudeCodeClient` is the same for
a single request (the comments on the rules, the back-translation check,
the ICL strategy). Both run the ``claude`` CLI with the user's own sign-in
(see :mod:`.claude_cli`).

The SDK is asynchronous: each conversation runs in its own event loop, in
the calling thread, where the tools of the agent run in a thread of
theirs.

It needs the extra ``claude`` (``claude-agent-sdk``).
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional

import claude_agent_sdk as sdk
from claude_agent_sdk import query as sdk_query

from . import claude_cli, prompts
from .agent import AgentResult, AgentStep, ReactAgent, Toolbox, truncate
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


async def read(stream, seconds: float) -> Outcome:
    """Read the messages of a conversation, for *seconds* at most: its
    text, usage and turns. What the SDK raises is an LLMError."""
    async def messages():
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

    try:
        return await asyncio.wait_for(messages(), seconds)
    except sdk.CLINotFoundError as exc:
        raise LLMError(
            "the claude CLI was not found: install Claude Code and sign in "
            f"with `claude` ({exc})."
        ) from exc
    except sdk.ClaudeSDKError as exc:
        raise LLMError(f"Claude Code: {exc}") from exc
    except asyncio.TimeoutError as exc:
        raise LLMError("Claude Code: the request timed out.") from exc


def options_for(provider: ProviderConfig, cli_path: str, system: str,
                **options) -> "sdk.ClaudeAgentOptions":
    """Return the SDK options running the CLI at *cli_path* for *provider*
    with the system prompt *system*, Claude Code's own tools and settings
    off."""
    return sdk.ClaudeAgentOptions(
        system_prompt=system, tools=[], setting_sources=[],
        model=provider.model, cli_path=cli_path,
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
        with claude_cli.settings(self.provider) as path:
            options = options_for(
                self.provider, path, system, max_turns=1, mcp_servers={}
            )
            outcome = asyncio.run(read(
                sdk_query(prompt=prompt, options=options), self.timeout
            ))
        return Reply(outcome.text, usage=outcome.usage)

    def close(self) -> None:
        """Nothing to close."""


class Bridge:
    """
    The tools of the engine for one conversation: runs a call on the
    :class:`~.agent.Toolbox`, records its step and, for a final answer,
    accepts or rejects it as the ReAct agent does.
    """

    def __init__(self, agent: "ClaudeCodeAgent", result: AgentResult,
                 question: str):
        self.agent, self.result, self.question = agent, result, question
        self.toolbox = Toolbox(
            agent.context, agent.language, agent.config.preview_rows,
            agent.config.query_timeout, agent.config.query_memory_mb,
        )
        self.rejections = 0
        self.accepted = False

    def call(self, name: str, arguments: dict) -> str:
        """Return the observation of the tool *name* called with
        *arguments*."""
        observation, final = self.toolbox.call(name, arguments)
        observation = truncate(observation)
        self.agent._step(
            self.result, AgentStep("", name, arguments, observation)
        )
        if final is None or self.accepted:
            return observation
        rejection = self.agent._accept(
            self.result, self.question, final, self.toolbox, self.rejections
        )
        if rejection is not None:
            self.rejections += 1
            return rejection
        self.accepted = True
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
        result = AgentResult(language=self.language)
        self.bridge = Bridge(self, result, question)
        system = self.system_prompt(question, text_tools=False)
        with claude_cli.settings(self.provider) as path:
            options = options_for(
                self.provider, path, system,
                max_turns=self.config.max_steps,
                mcp_servers={SERVER: self.server()},
                allowed_tools=[
                    f"mcp__{SERVER}__{tool['name']}" for tool in prompts.TOOLS
                ],
            )
            outcome = asyncio.run(read(
                sdk_query(
                    prompt=self._user_message(question), options=options
                ),
                # The time the steps of the agent may take at most.
                self.config.timeout * max(self.config.max_steps, 1),
            ))
        result.usage.update(outcome.usage)
        result.calls += outcome.turns
        if self.bridge.accepted:
            return result
        return self._out_of_steps(result)

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
