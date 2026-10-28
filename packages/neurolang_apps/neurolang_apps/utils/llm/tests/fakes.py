"""A fake chat model and test settings of the LLM assistant tests."""

from ..config import AssistantConfig, ProviderConfig
from ..providers import Reply, ToolCall


class ScriptedClient:
    """
    A chat client replaying canned replies, recording the requests.

    Each reply is a string (text), a :class:`Reply` or a list of
    ``(name, arguments)`` tool calls.
    """

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def complete(self, messages, tools=None, temperature=0.0):
        self.requests.append(
            {"messages": [dict(m) for m in messages], "tools": tools,
             "temperature": temperature}
        )
        if not self.replies:
            raise AssertionError("the model was asked more than scripted")
        reply = self.replies.pop(0)
        if isinstance(reply, Reply):
            return reply
        if isinstance(reply, str):
            return Reply(reply, usage={"input": 10, "output": 5})
        calls = [
            ToolCall(f"call_{i}", name, arguments)
            for i, (name, arguments) in enumerate(reply)
        ]
        return Reply("", calls, {"input": 10, "output": 5})


def provider(tool_calling="text", api="openai"):
    """Return a provider configuration for the tests."""
    return ProviderConfig(
        "fake", api, "fake-model", "http://llm.test/v1",
        tool_calling=tool_calling,
    )


def assistant_config(**options):
    """
    Return assistant settings for the tests: without the request for the
    comments of the rules, unless *options* ask for it.
    """
    return AssistantConfig(**{
        "max_steps": 4, "max_repairs": 2, "few_shot_k": 2,
        "comment_rules": False, **options,
    })
