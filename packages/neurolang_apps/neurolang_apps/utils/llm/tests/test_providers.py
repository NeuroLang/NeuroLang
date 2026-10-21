import json
import sys

import httpx
import pytest

from ..config import ProviderConfig
from ..providers import (
    LLMError,
    ToolCall,
    make_client,
    render_conversation,
)

TOOLS = [{
    "name": "check_query", "description": "Check a query.",
    "parameters": {
        "type": "object", "properties": {"query": {"type": "string"}},
    },
}]

CONVERSATION = [
    {"role": "system", "content": "You write queries."},
    {"role": "user", "content": "Which studies mention pain?"},
    {"role": "assistant", "content": "Let me check.", "tool_calls": [
        ToolCall("c1", "check_query", {"query": "ans(s) :- study(s)"}),
        ToolCall("c2", "check_query", {"query": "ans(s) :- x(s)"}),
    ]},
    {"role": "tool", "tool_call_id": "c1", "name": "check_query",
     "content": "valid"},
    {"role": "tool", "tool_call_id": "c2", "name": "check_query",
     "content": "invalid"},
]


def client(api, handler, monkeypatch, **options):
    """Return a client of *api* whose requests *handler* answers."""
    monkeypatch.setenv("TEST_LLM_KEY", "secret")
    provider = ProviderConfig(
        "test", api, "model-x", "https://llm.test/v1",
        api_key_env="TEST_LLM_KEY", headers={"X-Title": "NeuroLang"},
        **options,
    )
    return make_client(provider, transport=httpx.MockTransport(handler))


def test_openai_request_and_tool_calls(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": None, "tool_calls": [{
                "id": "c3", "type": "function", "function": {
                    "name": "run_query",
                    "arguments": '{"query": "ans(s) :- study(s)"}',
                },
            }]}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        })

    reply = client("openai", handler, monkeypatch).complete(
        CONVERSATION, TOOLS, 0.2
    )
    assert seen["url"] == "https://llm.test/v1/chat/completions"
    assert seen["headers"]["authorization"] == "Bearer secret"
    assert seen["headers"]["x-title"] == "NeuroLang"
    body = seen["body"]
    assert body["model"] == "model-x" and body["temperature"] == 0.2
    assert body["messages"][0] == {
        "role": "system", "content": "You write queries."
    }
    call = body["messages"][2]["tool_calls"][0]
    assert call["function"]["name"] == "check_query"
    assert json.loads(call["function"]["arguments"]) == {
        "query": "ans(s) :- study(s)"
    }
    assert body["messages"][3] == {
        "role": "tool", "tool_call_id": "c1", "content": "valid"
    }
    assert body["tools"][0]["function"]["name"] == "check_query"
    assert reply.text == ""
    assert reply.tool_calls == [
        ToolCall("c3", "run_query", {"query": "ans(s) :- study(s)"})
    ]
    assert reply.usage == {"input": 12, "output": 3}


def test_anthropic_request_and_tool_calls(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "content": [
                {"type": "text", "text": "Running it."},
                {"type": "tool_use", "id": "t1", "name": "run_query",
                 "input": {"query": "obtain every Study."}},
            ],
            "usage": {"input_tokens": 20, "output_tokens": 4},
        })

    reply = client("anthropic", handler, monkeypatch).complete(
        CONVERSATION, TOOLS
    )
    assert seen["url"] == "https://llm.test/v1/messages"
    assert seen["headers"]["x-api-key"] == "secret"
    assert seen["headers"]["anthropic-version"]
    body = seen["body"]
    assert body["system"] == "You write queries."
    assert [m["role"] for m in body["messages"]] == [
        "user", "assistant", "user"
    ]
    assistant = body["messages"][1]["content"]
    assert assistant[0] == {"type": "text", "text": "Let me check."}
    assert assistant[1]["type"] == "tool_use"
    results = body["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["c1", "c2"]
    assert body["tools"][0]["input_schema"]["type"] == "object"
    assert reply.text == "Running it."
    assert reply.tool_calls[0].arguments == {"query": "obtain every Study."}
    assert reply.usage == {"input": 20, "output": 4}


def test_invalid_tool_arguments_are_kept(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {
            "content": "", "tool_calls": [{"id": "c", "function": {
                "name": "run_query", "arguments": "{not json",
            }}],
        }}]})

    reply = client("openai", handler, monkeypatch).complete(CONVERSATION)
    assert reply.tool_calls[0].arguments == {"_invalid": "{not json"}


@pytest.mark.parametrize("response, message", [
    (httpx.Response(401, text="bad key"), "HTTP 401: bad key"),
    (httpx.Response(200, json={"nothing": 1}), "unexpected reply"),
    (httpx.Response(200, text="<html>"), "unexpected reply"),
])
def test_errors(monkeypatch, response, message):
    chat = client("openai", lambda request: response, monkeypatch)
    with pytest.raises(LLMError, match=message):
        chat.complete(CONVERSATION)


def test_connection_errors(monkeypatch):
    def unreachable(request):
        raise httpx.ConnectError("refused")

    def slow(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(LLMError, match="cannot reach"):
        client("openai", unreachable, monkeypatch).complete(CONVERSATION)
    with pytest.raises(LLMError, match="timed out"):
        client("anthropic", slow, monkeypatch).complete(CONVERSATION)


def test_render_conversation():
    text = render_conversation(CONVERSATION[:2], with_system=True)
    assert text == "You write queries.\n\nWhich studies mention pain?"
    messages = [
        CONVERSATION[0], CONVERSATION[1],
        {"role": "assistant", "content": "Action: list_predicates"},
        {"role": "user", "content": "Observation: study(c0)"},
    ]
    text = render_conversation(messages, with_system=False)
    assert "You write queries." not in text
    assert "ASSISTANT:\nAction: list_predicates" in text
    assert "USER:\nObservation: study(c0)" in text
    assert text.endswith("without the ASSISTANT: header.")


def command_provider(script, *extra):
    """Return a provider running the Python *script*."""
    return ProviderConfig(
        "cmd", "command", "", command=(sys.executable, "-c", script, *extra),
    )


def test_command_client_reads_stdin_and_substitutes_the_system_prompt():
    script = (
        "import sys; print('system=' + sys.argv[1]); "
        "print('input=' + sys.stdin.read())"
    )
    chat = make_client(command_provider(script, "{system}"))
    reply = chat.complete(CONVERSATION[:2])
    assert reply.text == (
        "system=You write queries.\ninput=Which studies mention pain?"
    )


def test_command_client_without_placeholder_sends_the_system_prompt():
    chat = make_client(command_provider(
        "import sys; print(sys.stdin.read())"
    ))
    assert chat.complete(CONVERSATION[:2]).text.startswith(
        "You write queries."
    )


def test_command_client_errors():
    failing = make_client(command_provider(
        "import sys; sys.stderr.write('no login'); sys.exit(3)"
    ))
    with pytest.raises(LLMError, match="status 3: no login"):
        failing.complete(CONVERSATION[:2])
    missing = make_client(ProviderConfig(
        "cmd", "command", "", command=("neurolang-no-such-command",)
    ))
    with pytest.raises(LLMError, match="command not found"):
        missing.complete(CONVERSATION[:2])


def test_command_client_retries_os_errors(monkeypatch):
    from .. import providers

    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        raise OSError(8, "Exec format error")

    monkeypatch.setattr(providers.subprocess, "run", flaky)
    monkeypatch.setattr(providers, "COMMAND_RETRY_SECONDS", 0)
    chat = make_client(command_provider("print(1)"))
    with pytest.raises(LLMError, match="Exec format error"):
        chat.complete(CONVERSATION[:2])
    assert len(calls) == providers.COMMAND_ATTEMPTS


def test_rate_limits_are_retried(monkeypatch):
    from .. import providers

    replies = [
        httpx.Response(429, headers={"retry-after": "0"}, text="slow down"),
        httpx.Response(503, text="busy"),
        httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}]}
        ),
    ]
    waits = []
    monkeypatch.setattr(providers.time, "sleep", waits.append)
    chat = client("openai", lambda request: replies.pop(0), monkeypatch)
    assert chat.complete(CONVERSATION).text == "ok"
    assert waits == [1.0, 4.0]


def test_rate_limit_error_after_the_last_attempt(monkeypatch):
    from .. import providers

    monkeypatch.setattr(providers.time, "sleep", lambda seconds: None)
    chat = client(
        "openai", lambda request: httpx.Response(429, text="limit"),
        monkeypatch,
    )
    with pytest.raises(LLMError, match="HTTP 429: limit"):
        chat.complete(CONVERSATION)


def test_requests_are_paced(monkeypatch):
    from .. import providers

    waits = []
    monkeypatch.setattr(providers.time, "sleep", waits.append)
    chat = client(
        "openai",
        lambda request: httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}]}
        ),
        monkeypatch, requests_per_minute=30,
    )
    chat.complete(CONVERSATION)
    chat.complete(CONVERSATION)
    assert len(waits) == 1 and 1.9 < waits[0] <= 2.0
