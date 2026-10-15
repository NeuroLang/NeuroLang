"""
Tracing the requests the assistant sends to its model, and its replies.

:class:`TracingClient` wraps a chat client and reports each request and
reply as an event, a dict whose ``event`` key is:

* ``request``: the ``call`` number, the ``messages`` added to the
  conversation since the previous request (from index ``start``; all of
  them when the conversation is a new one), the ``tools`` offered and
  the ``temperature``;
* ``reply``: the ``call`` number, ``text``, ``tool_calls``, ``usage``
  and ``seconds``, or ``error`` when the request failed.

:class:`Printer` prints these events, e.g. for ``neurolang-query --ask
QUESTION -vv``.
"""

import json
import re
import sys
import time
from typing import Callable

from neurolang.utils import query_highlight

_ACTION_INPUT = re.compile(r"Action[ _]Input\s*:", re.IGNORECASE)


class TracingClient:
    """The chat *client* reporting its requests and replies to *emit*."""

    def __init__(self, client, emit: Callable[[dict], None]):
        self.client = client
        self.emit = emit
        self.calls = 0
        self._previous = []

    def complete(self, messages, tools=None, temperature=0.0):
        """Return the reply of the client, reporting the exchange."""
        self.calls += 1
        call = self.calls
        start = len(self._previous)
        if messages[:start] != self._previous:
            start = 0
        self._previous = list(messages)
        self.emit({
            "event": "request", "call": call, "start": start,
            "messages": messages[start:],
            "tools": [_tool_name(tool) for tool in tools or ()],
            "temperature": temperature,
        })
        began = time.perf_counter()
        try:
            reply = self.client.complete(messages, tools, temperature)
        except Exception as exc:
            self.emit({
                "event": "reply", "call": call, "error": str(exc),
                "seconds": time.perf_counter() - began,
            })
            raise
        self.emit({
            "event": "reply", "call": call, "text": reply.text,
            "tool_calls": [
                {"id": c.id, "name": c.name, "arguments": c.arguments}
                for c in reply.tool_calls
            ],
            "usage": dict(reply.usage), "seconds": time.perf_counter() - began,
        })
        return reply

    def close(self) -> None:
        """Close the wrapped client."""
        self.client.close()


def _tool_name(tool: dict) -> str:
    """Return the name of *tool*, in the OpenAI or Anthropic format."""
    return tool.get("name") or tool.get("function", {}).get("name", "?")


class Printer:
    """
    Prints the events of a trace on *stream* (default: standard error).

    Messages are cut to *limit* characters, and system prompts to
    *system_limit*, unless *full*; *hint* tells how to see them whole.
    The queries the model writes in its tool calls are shown on their
    lines, highlighted as *language* on a terminal. Other kinds of
    events are printed by the methods ``_<event>`` of subclasses.
    """

    def __init__(self, stream=None, full: bool = False, limit: int = 4000,
                 system_limit: int = 800, hint: str = "",
                 language: str = "datalog"):
        self.stream = stream or sys.stderr
        self.full = full
        self.limit = limit
        self.system_limit = system_limit
        self.hint = hint
        self.language = language
        self.color = query_highlight.use_color(self.stream)

    def show(self, event: dict) -> None:
        """Print *event*."""
        show = getattr(self, f"_{event.get('event')}", None)
        if show:
            show(event)
            self.stream.flush()

    def note(self, text: str) -> None:
        """Print *text*, a remark of the printer rather than an event."""
        self._write(self._paint(text, "2"))
        self.stream.flush()

    def _request(self, event):
        tools = ", ".join(event["tools"]) or "none"
        self._rule("─", f"call {event['call']} → "
                        f"{len(event['messages'])} new message(s) from "
                        f"#{event['start']}, tools: {tools}", "36")
        for message in event["messages"]:
            self._message(message)

    def _reply(self, event):
        if "error" in event:
            self._rule("─", f"call {event['call']} ← failed after "
                            f"{event['seconds']:.1f} s", "31")
            self._write(event["error"])
            return
        usage = ", ".join(f"{k} {v}" for k, v in event["usage"].items())
        self._rule("─", f"call {event['call']} ← {event['seconds']:.1f} s"
                        + (f", {usage}" if usage else ""), "32")
        if event["text"]:
            self._reply_text(event["text"])
        for call in event["tool_calls"]:
            self._write(f"tool call {call['name']}:")
            self._write("\n".join(argument_lines(
                call["arguments"], self.language, self.color, "  "
            )))

    def _reply_text(self, text: str):
        """Print a reply, with the arguments of its text action laid out."""
        from .agent import parse_text_action

        start = _ACTION_INPUT.search(text)
        action = parse_text_action(text, self.language) if start else None
        if action is None or "query" not in action.arguments:
            self._write(self._cut(text, self.limit))
            return
        self._write(self._cut(text[:start.end()].rstrip(), self.limit))
        self._write("\n".join(argument_lines(
            action.arguments, self.language, self.color, "  "
        )))

    def _message(self, message: dict):
        role = message.get("role", "?")
        limit = self.system_limit if role == "system" else self.limit
        self._write(self._paint(f"[{role}]", "33"))
        self._write(self._cut(message_text(message), limit))

    def _cut(self, text: str, limit: int) -> str:
        if self.full or len(text) <= limit:
            return text
        hint = f", {self.hint}" if self.hint else ""
        return f"{text[:limit]}\n… ({len(text) - limit} more characters{hint})"

    def _rule(self, char: str, title: str, color: str):
        self._write(self._paint(f"{char * 3} {title} {char * 3}", color))

    def _paint(self, text: str, color: str) -> str:
        return f"\033[{color}m{text}\033[0m" if self.color else text

    def _write(self, text: str):
        print(text, file=self.stream)


def argument_lines(arguments: dict, language: str, color: bool,
                   indent: str = "") -> list:
    """
    Return the lines showing the *arguments* of a tool call: its query on
    its own lines, highlighted as *language* if *color*, then the others.
    """
    lines = []
    query = arguments.get("query")
    if isinstance(query, str):
        lines += query_lines(query, language, color, indent)
    for name, value in arguments.items():
        if name != "query":
            value = value if isinstance(value, str) else json.dumps(
                value, ensure_ascii=False
            )
            lines.append(f"{indent}{name}: {value}")
    return lines


def query_lines(query: str, language: str, color: bool,
                indent: str = "") -> list:
    """Return the lines of *query*, highlighted as *language* if *color*."""
    text = query.strip()
    if color:
        text = query_highlight.ansi(text, language)
    return [indent + line for line in text.split("\n")]


def message_text(message: dict) -> str:
    """Return the text of *message*, with its tool calls or results."""
    content = message.get("content")
    if isinstance(content, str):
        text = content
    elif content is None:
        text = ""
    else:
        text = json.dumps(content, indent=1, ensure_ascii=False)
    extra = {
        key: value for key, value in message.items()
        if key not in ("role", "content")
    }
    if extra:
        text += ("\n" if text else "") + json.dumps(
            extra, ensure_ascii=False, default=str
        )
    return text
