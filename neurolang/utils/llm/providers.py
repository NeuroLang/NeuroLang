"""
Chat clients of the LLM providers.

A client speaks either the OpenAI chat completions API or the Anthropic
messages API with httpx, or runs a local command (e.g. ``claude -p``)
that reads the conversation on its standard input. The conversation is
kept in a provider-neutral form, a list of messages::

    {"role": "system" | "user" | "assistant", "content": str}
    {"role": "assistant", "content": str, "tool_calls": [ToolCall, ...]}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str}

and tools are described as ``{"name", "description", "parameters"}``,
*parameters* being a JSON schema. Each client translates both into the
format of its API.
"""

import json
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import httpx

from .config import ProviderConfig

ANTHROPIC_VERSION = "2023-06-01"


class LLMError(Exception):
    """Raised when a provider cannot be reached or gives a bad reply."""


@dataclass
class ToolCall:
    """A call of the tool *name* with *arguments*, identified by *id*."""

    id: str
    name: str
    arguments: dict


@dataclass
class Reply:
    """The reply of a model: its text, tool calls and token usage."""

    text: str = ""
    tool_calls: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)


def _parse_arguments(arguments) -> dict:
    """Return the tool call *arguments*, a JSON string or a mapping."""
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {"_invalid": arguments}
    return parsed if isinstance(parsed, dict) else {"_invalid": arguments}


class ChatClient:
    """
    Base of the chat clients: HTTP session, errors and request loop.

    Subclasses define the path of the API, its payload and the parsing
    of its replies. *transport* replaces the network, e.g. by an
    ``httpx.MockTransport`` in tests.
    """

    path = ""

    def __init__(
        self, provider: ProviderConfig, timeout: float = 60.0,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        self.provider = provider
        self._http = httpx.Client(
            base_url=provider.base_url,
            headers={**self._auth_headers(), **provider.headers},
            timeout=timeout,
            transport=transport,
        )

    def _auth_headers(self) -> dict:
        """Return the headers authenticating the requests."""
        raise NotImplementedError

    def _payload(self, messages, tools, temperature) -> dict:
        """Return the JSON body of a request."""
        raise NotImplementedError

    def _reply(self, data: dict) -> Reply:
        """Return the reply in the JSON body *data* of a response."""
        raise NotImplementedError

    def complete(
        self, messages: list, tools: Optional[list] = None,
        temperature: float = 0.0,
    ) -> Reply:
        """Return the model's reply to the conversation *messages*."""
        name = self.provider.name
        try:
            response = self._http.post(
                self.path, json=self._payload(messages, tools, temperature)
            )
        except httpx.TimeoutException as exc:
            raise LLMError(f"{name}: the request timed out.") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"{name}: cannot reach the API ({exc}).") from exc
        if response.status_code >= 400:
            raise LLMError(
                f"{name}: HTTP {response.status_code}: "
                f"{response.text.strip()[:300]}"
            )
        try:
            return self._reply(response.json())
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                f"{name}: unexpected reply: {response.text.strip()[:300]}"
            ) from exc

    def close(self) -> None:
        """Close the HTTP session."""
        self._http.close()


class OpenAIChatClient(ChatClient):
    """Client of the OpenAI chat completions API and its look-alikes."""

    path = "chat/completions"

    def _auth_headers(self) -> dict:
        key = self.provider.api_key()
        return {"Authorization": f"Bearer {key}"} if key else {}

    @staticmethod
    def _message(message: dict) -> dict:
        """Return *message* in the format of the API."""
        if message["role"] == "tool":
            return {
                "role": "tool",
                "tool_call_id": message["tool_call_id"],
                "content": message["content"],
            }
        if message.get("tool_calls"):
            return {
                "role": "assistant",
                "content": message.get("content") or None,
                "tool_calls": [
                    {
                        "id": call.id, "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in message["tool_calls"]
                ],
            }
        return {"role": message["role"], "content": message["content"]}

    def _payload(self, messages, tools, temperature) -> dict:
        payload = {
            "model": self.provider.model,
            "messages": [self._message(m) for m in messages],
            "temperature": temperature,
            "max_tokens": self.provider.max_tokens,
        }
        if tools:
            payload["tools"] = [
                {"type": "function", "function": tool} for tool in tools
            ]
        return payload

    def _reply(self, data: dict) -> Reply:
        message = data["choices"][0]["message"]
        calls = [
            ToolCall(
                call.get("id") or f"call_{i}",
                call["function"]["name"],
                _parse_arguments(call["function"].get("arguments")),
            )
            for i, call in enumerate(message.get("tool_calls") or [])
        ]
        usage = data.get("usage") or {}
        return Reply(
            message.get("content") or "", calls,
            {
                "input": usage.get("prompt_tokens", 0),
                "output": usage.get("completion_tokens", 0),
            },
        )


class AnthropicChatClient(ChatClient):
    """Client of the Anthropic messages API."""

    path = "messages"

    def _auth_headers(self) -> dict:
        headers = {"anthropic-version": ANTHROPIC_VERSION}
        key = self.provider.api_key()
        if key:
            headers["x-api-key"] = key
        return headers

    @staticmethod
    def _messages(messages) -> list:
        """
        Return the non-system *messages* in the format of the API.

        Tool results are sent in a user message, those of consecutive
        tool messages in the same one.
        """
        converted = []
        for message in messages:
            role = message["role"]
            if role == "system":
                continue
            if role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": message["tool_call_id"],
                    "content": message["content"],
                }
                previous = converted[-1] if converted else None
                if previous and previous.get("_tool_results"):
                    previous["content"].append(block)
                else:
                    converted.append(
                        {"role": "user", "content": [block],
                         "_tool_results": True}
                    )
                continue
            if role == "assistant" and message.get("tool_calls"):
                content = []
                if message.get("content"):
                    content.append(
                        {"type": "text", "text": message["content"]}
                    )
                content.extend(
                    {
                        "type": "tool_use", "id": call.id,
                        "name": call.name, "input": call.arguments,
                    }
                    for call in message["tool_calls"]
                )
                converted.append({"role": "assistant", "content": content})
                continue
            converted.append({"role": role, "content": message["content"]})
        for message in converted:
            message.pop("_tool_results", None)
        return converted

    def _payload(self, messages, tools, temperature) -> dict:
        system = "\n\n".join(
            m["content"] for m in messages if m["role"] == "system"
        )
        payload = {
            "model": self.provider.model,
            "messages": self._messages(messages),
            "temperature": temperature,
            "max_tokens": self.provider.max_tokens,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["parameters"],
                }
                for tool in tools
            ]
        return payload

    def _reply(self, data: dict) -> Reply:
        text, calls = [], []
        for block in data["content"]:
            if block["type"] == "text":
                text.append(block["text"])
            elif block["type"] == "tool_use":
                calls.append(
                    ToolCall(
                        block["id"], block["name"],
                        _parse_arguments(block.get("input")),
                    )
                )
        usage = data.get("usage") or {}
        return Reply(
            "".join(text), calls,
            {
                "input": usage.get("input_tokens", 0),
                "output": usage.get("output_tokens", 0),
            },
        )


SYSTEM_PLACEHOLDER = "{system}"


def render_conversation(messages, with_system: bool) -> str:
    """
    Return the conversation *messages* as plain text, for a command.

    The system prompt comes first when *with_system* is set; the turns
    follow, headed ``USER:`` and ``ASSISTANT:``, and the text ends by
    asking for the next assistant turn.
    """
    parts = []
    turns = [m for m in messages if m["role"] != "system"]
    if with_system:
        parts.extend(m["content"] for m in messages if m["role"] == "system")
    if len(turns) == 1:
        parts.append(turns[0]["content"])
        return "\n\n".join(parts)
    for message in turns:
        role = "ASSISTANT" if message["role"] == "assistant" else "USER"
        parts.append(f"{role}:\n{message['content']}")
    parts.append(
        "Write only the next ASSISTANT message, without the ASSISTANT: "
        "header."
    )
    return "\n\n".join(parts)


class CommandChatClient:
    """
    Client running a local command for each request.

    The conversation is written as text on the standard input of the
    provider's *command* (see :func:`render_conversation`) and its
    standard output is the reply. ``{system}`` in the arguments of the
    command is replaced by the system prompt, which is otherwise written
    first on its input. Only the text tool calling protocol is possible.
    """

    def __init__(self, provider: ProviderConfig, timeout: float = 60.0,
                 transport=None):
        self.provider = provider
        self.timeout = timeout

    def complete(
        self, messages: list, tools: Optional[list] = None,
        temperature: float = 0.0,
    ) -> Reply:
        """Return the command's reply to the conversation *messages*."""
        system = "\n\n".join(
            m["content"] for m in messages if m["role"] == "system"
        )
        command = self.provider.command
        placeholder = any(SYSTEM_PLACEHOLDER in arg for arg in command)
        args = [
            arg.replace(SYSTEM_PLACEHOLDER, system)
            for arg in self.provider.command
        ]
        name = self.provider.name
        try:
            completed = subprocess.run(
                args, input=render_conversation(messages, not placeholder),
                capture_output=True, text=True, timeout=self.timeout,
            )
        except FileNotFoundError as exc:
            raise LLMError(f"{name}: command not found: {args[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise LLMError(f"{name}: the command timed out.") from exc
        if completed.returncode != 0:
            raise LLMError(
                f"{name}: the command failed with status "
                f"{completed.returncode}: "
                f"{(completed.stderr or completed.stdout).strip()[:300]}"
            )
        return Reply(completed.stdout.strip())

    def close(self) -> None:
        """Nothing to close."""


CLIENTS = {
    "openai": OpenAIChatClient,
    "anthropic": AnthropicChatClient,
    "command": CommandChatClient,
}


def make_client(
    provider: ProviderConfig, timeout: float = 60.0, transport=None
):
    """Return the chat client of *provider*."""
    return CLIENTS[provider.api](provider, timeout, transport)
