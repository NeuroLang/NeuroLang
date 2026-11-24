"""
The Claude Code backend: its provider, the ``claude`` CLI it runs and the
agent and client built on the Agent SDK.

The SDK's ``query`` and the ``claude`` executable are faked: nothing here
reaches the network nor uses a sign-in.
"""

import contextlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

sdk = pytest.importorskip("claude_agent_sdk")

from .. import claude_cli, claude_code  # noqa: E402
from ..agent import IclAgent, ReactAgent, make_agent  # noqa: E402
from ..config import (  # noqa: E402
    EXAMPLE_CONFIG,
    MULTI_USER_ENV,
    LLMConfigError,
    ProviderConfig,
    load_config,
    parse_config,
)
from ..providers import LLMError, make_client  # noqa: E402
from .fakes import ScriptedClient, assistant_config  # noqa: E402

PAIN = "ans(s) :- mentions(s, 'pain')"


def provider(auth="login", **options):
    """Return a Claude Code provider for the tests."""
    entry = {"api_key_env": "FAKE_KEY"} if auth == "key" else {}
    return ProviderConfig(
        "claude-agent", "claude-code", "sonnet", tool_calling="native",
        auth=auth, **{**entry, **options},
    )


def parsed(**entry):
    """Return the provider of a configuration with the *entry*."""
    providers = {"c": {"api": "claude-code", **entry}}
    return parse_config({"providers": providers}).provider("c")


# --- the provider -----------------------------------------------------------


def test_claude_code_provider_defaults():
    p = parsed(model="sonnet")
    assert (p.api, p.auth, p.tool_calling, p.api_key_env) == (
        "claude-code", "login", "native", None
    )
    k = parsed(model="sonnet", auth="key", api_key_env="ANTHROPIC_API_KEY")
    assert (k.auth, k.api_key_env) == ("key", "ANTHROPIC_API_KEY")


@pytest.mark.parametrize("entry, message", [
    ({}, "lacks model"),
    ({"model": "s", "tool_calling": "text"}, "only tool_calling: native"),
    ({"model": "s", "auth": "oauth"}, "auth"),
    ({"model": "s", "auth": "key"}, "needs api_key_env"),
    ({"model": "s", "auth": "login", "api_key_env": "K"}, "no api_key_env"),
    ({"model": "s", "base_url": "http://x"}, "base_url"),
])
def test_claude_code_provider_is_checked(entry, message):
    with pytest.raises(LLMConfigError, match=message):
        parsed(**entry)


def test_the_example_configuration_has_the_claude_agents():
    config = load_config(EXAMPLE_CONFIG)
    assert config.providers["claude-agent"].auth == "login"
    assert config.providers["claude-agent-key"].auth == "key"
    assert config.providers["claude-agent-key"].api_key_env


def test_a_login_needs_no_key(monkeypatch):
    monkeypatch.delenv(MULTI_USER_ENV, raising=False)
    assert provider("login").api_key() is None


def test_a_login_is_refused_on_a_shared_server(monkeypatch):
    monkeypatch.setenv(MULTI_USER_ENV, "1")
    with pytest.raises(LLMConfigError, match="shared server"):
        provider("login").api_key()
    monkeypatch.setenv("FAKE_KEY", "sk-test")
    assert provider("key").api_key() == "sk-test"


def test_a_key_is_read_from_the_environment(monkeypatch):
    monkeypatch.delenv("FAKE_KEY", raising=False)
    with pytest.raises(LLMConfigError, match="FAKE_KEY"):
        provider("key").api_key()


# --- the claude CLI ---------------------------------------------------------


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    """
    A ``claude`` on the PATH: ``claude auth status`` prints the JSON in
    ``status.json`` and any other call records its environment in
    ``env.txt``. Return the folder; set ``status`` to change the sign-in.
    """
    folder = tmp_path / "bin"
    folder.mkdir()
    script = folder / "claude"
    script.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = auth ]; then cat "$FAKE_CLAUDE_DIR/status.json"; '
        "exit 0; fi\n"
        'env > "$FAKE_CLAUDE_DIR/env.txt"\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("FAKE_CLAUDE_DIR", str(folder))
    monkeypatch.setenv("PATH", f"{folder}{os.pathsep}{os.environ['PATH']}")

    class FakeClaude:
        def __truediv__(self, name):
            return folder / name

        def sign_in(self, method="claude.ai", logged_in=True,
                    email="me@x.org"):
            (folder / "status.json").write_text(json.dumps(
                {"loggedIn": logged_in, "authMethod": method, "email": email}
            ))

    fake = FakeClaude()
    fake.sign_in()
    return fake


def recorded_env(folder):
    """Return the environment the fake ``claude`` ran with."""
    lines = (folder / "env.txt").read_text().splitlines()
    return dict(line.split("=", 1) for line in lines if "=" in line)


def test_status_reads_the_sign_in(fake_claude):
    status = claude_cli.status()
    assert status.logged_in and status.method == "claude.ai"
    assert status.email == "me@x.org"


def test_status_runs_claude_without_the_keys(
        fake_claude, monkeypatch):
    (fake_claude / "claude").write_text(
        "#!/bin/sh\n"
        'env > "$FAKE_CLAUDE_DIR/env.txt"\n'
        'cat "$FAKE_CLAUDE_DIR/status.json"\n'
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "t")
    claude_cli.status()
    env = recorded_env(fake_claude)
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env


def test_a_login_through_an_api_key_is_no_sign_in(fake_claude):
    fake_claude.sign_in(method="api_key")
    with pytest.raises(LLMError, match="not signed in"):
        claude_cli.require_login()


def test_not_logged_in(fake_claude):
    fake_claude.sign_in(logged_in=False)
    with pytest.raises(LLMError, match="run `claude`"):
        claude_cli.require_login()


def test_no_claude_executable(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(LLMError, match="claude.*not found"):
        claude_cli.require_login()


def test_a_login_runs_the_cli_without_the_keys(fake_claude, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "t")
    with claude_cli.settings(provider("login")) as cli:
        assert cli.env == {}
        subprocess.run([cli.path, "-p"], check=True)
    env = recorded_env(fake_claude)
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env


def test_a_login_needs_a_subscription(fake_claude):
    fake_claude.sign_in(method="api_key")
    with pytest.raises(LLMError, match="not signed in"):
        with claude_cli.settings(provider("login")):
            pass


def test_a_key_runs_the_cli_with_the_key_and_a_private_configuration(
        fake_claude, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "t")
    fake_claude.sign_in(method="api_key")  # a key needs no sign-in
    with claude_cli.settings(provider("key")) as cli:
        config = Path(cli.env["CLAUDE_CONFIG_DIR"])
        assert cli.env["ANTHROPIC_API_KEY"] == "sk-test"
        assert config.is_dir()
        subprocess.run(
            [cli.path, "-p"], check=True, env={**os.environ, **cli.env}
        )
    env = recorded_env(fake_claude)
    assert env["ANTHROPIC_API_KEY"] == "sk-test"
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    assert not config.exists()


def test_the_wrapper_is_removed(fake_claude):
    with claude_cli.settings(provider("login")) as cli:
        wrapper = Path(cli.path)
        assert wrapper.exists()
    assert not wrapper.exists()


# --- the SDK conversation ---------------------------------------------------


def result_message(**options):
    """Return the ResultMessage ending a conversation."""
    return sdk.ResultMessage(**{
        "subtype": "success", "duration_ms": 1, "duration_api_ms": 1,
        "is_error": False, "num_turns": 1, "session_id": "s",
        "usage": {"input_tokens": 100, "output_tokens": 20,
                  "cache_read_input_tokens": 50},
        "result": "", **options,
    })


@pytest.fixture
def no_cli(monkeypatch):
    """The conversations run without the claude CLI, recording how it
    would run."""
    seen = {}

    @contextlib.contextmanager
    def settings(p):
        seen["provider"] = p
        yield claude_cli.CliSettings("/fake/claude", {"FAKE": "1"})

    monkeypatch.setattr(claude_code.claude_cli, "settings", settings)
    return seen


def scripted(monkeypatch, run, seen=None):
    """
    Replace the SDK's ``query`` by *run*, an async generator function of the
    prompt and the options, recording both in *seen*.
    """
    seen = {} if seen is None else seen

    def query(prompt, options):
        seen["prompt"], seen["options"] = prompt, options
        return run(prompt, options)

    monkeypatch.setattr(claude_code, "sdk_query", query)
    return seen


def text_reply(text, **options):
    async def run(prompt, options_):
        yield sdk.AssistantMessage(
            content=[sdk.TextBlock(text)], model="m"
        )
        yield result_message(result=text, **options)
    return run


def agent(nl, auth="login", **options):
    instance = claude_code.ClaudeCodeAgent(
        nl, provider(auth), assistant_config(**options), "datalog",
        client=ScriptedClient([]),
    )
    return instance


def tool_calls(instance, calls):
    """A conversation making the tool *calls*, as the SDK would."""
    async def run(prompt, options):
        for name, arguments in calls:
            instance.bridge.call(name, arguments)
            yield sdk.AssistantMessage(content=[
                sdk.ToolUseBlock("id", name, arguments)
            ], model="m")
        yield result_message()
    return run


def test_the_agent_is_made_for_the_react_strategy(nl, no_cli):
    made = make_agent(
        nl, provider(), assistant_config(strategy="react"), "datalog",
        client=ScriptedClient([]),
    )
    assert isinstance(made, claude_code.ClaudeCodeAgent)
    assert isinstance(made, ReactAgent)
    made = make_agent(
        nl, provider(), assistant_config(strategy="icl"), "datalog",
        client=ScriptedClient([]),
    )
    assert isinstance(made, IclAgent)


def test_the_agent_answers_with_the_tools_of_the_engine(
        nl, no_cli, monkeypatch):
    react = agent(nl)
    steps = []
    react.on_step = steps.append
    seen = scripted(monkeypatch, tool_calls(react, [
        ("describe_predicate", {"name": "mentions"}),
        ("final_answer", {"query": PAIN, "explanation": "studies on pain"}),
    ]))
    result = react.run("Which studies mention pain?")
    assert result.valid and result.query == PAIN
    assert result.explanation == "studies on pain"
    assert [s.action for s in steps] == ["describe_predicate", "final_answer"]
    assert steps[1].observation == "Accepted."
    assert result.frames is not None
    # The usage of the conversation, as the other clients report it.
    assert result.usage["input"] == 150 and result.usage["output"] == 20
    assert result.usage["cached"] == 50
    assert "Which studies mention pain?" in seen["prompt"]


def test_the_agent_uses_the_engine_tools_alone(nl, no_cli, monkeypatch):
    react = agent(nl, max_steps=7)
    seen = scripted(monkeypatch, tool_calls(react, [
        ("final_answer", {"query": PAIN})
    ]))
    react.run("Which studies mention pain?")
    options = seen["options"]
    assert options.tools == []
    assert options.setting_sources == []
    assert options.max_turns == 7
    assert options.model == "sonnet"
    assert options.cli_path == "/fake/claude"
    assert options.env == {"FAKE": "1"}
    assert set(options.mcp_servers) == {"neurolang"}
    assert "mcp__neurolang__final_answer" in options.allowed_tools
    assert all(t.startswith("mcp__neurolang__") for t in options.allowed_tools)
    assert "Datalog" in options.system_prompt or "datalog" in (
        options.system_prompt
    )
    assert no_cli["provider"].auth == "login"


def test_out_of_steps_the_last_query_with_rows_answers(
        nl, no_cli, monkeypatch):
    react = agent(nl)
    scripted(monkeypatch, tool_calls(react, [
        ("run_query", {"query": PAIN}),
    ]))
    result = react.run("Which studies mention pain?")
    assert result.valid and result.out_of_steps and result.query == PAIN


def test_without_a_query_the_agent_fails(nl, no_cli, monkeypatch):
    react = agent(nl)
    scripted(monkeypatch, text_reply("I cannot tell."))
    result = react.run("Which studies mention pain?")
    assert not result.valid and result.error


def test_a_rejected_verification_goes_back_to_the_model(
        nl, no_cli, monkeypatch):
    react = agent(nl, verify_rounds=1)
    react.client = ScriptedClient([
        "The studies that mention memory.",
        "VERDICT: no\nIt is about memory, not pain.",
    ])
    observations = []

    async def run(prompt, options):
        for _ in range(2):
            observations.append(react.bridge.call(
                "final_answer", {"query": PAIN, "explanation": "pain"}
            ))
        yield result_message()

    scripted(monkeypatch, run)
    result = react.run("Which studies mention pain?")
    assert observations[0] != "Accepted."
    assert "memory" in observations[0]
    assert observations[1] == "Accepted."
    assert result.valid and result.query == PAIN


def test_a_conversation_failing_is_an_llm_error(nl, no_cli, monkeypatch):
    react = agent(nl)

    async def run(prompt, options):
        raise sdk.CLINotFoundError("no claude")
        yield

    scripted(monkeypatch, run)
    with pytest.raises(LLMError, match="claude"):
        react.run("Which studies mention pain?")


def test_an_error_result_is_an_llm_error(nl, no_cli, monkeypatch):
    react = agent(nl)
    scripted(monkeypatch, text_reply(
        "Invalid API key · Fix external API key", is_error=True
    ))
    with pytest.raises(LLMError, match="Invalid API key"):
        react.run("Which studies mention pain?")


# --- the one-shot client ----------------------------------------------------


def test_the_client_makes_one_request_without_tools(no_cli, monkeypatch):
    seen = scripted(monkeypatch, text_reply("pong"))
    client = make_client(provider())
    assert isinstance(client, claude_code.ClaudeCodeClient)
    reply = client.complete([
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "ping"},
    ])
    assert reply.text == "pong"
    assert reply.usage == {"input": 150, "output": 20, "cached": 50}
    assert seen["prompt"] == "ping"
    options = seen["options"]
    assert options.system_prompt == "Be brief."
    assert options.tools == [] and options.max_turns == 1
    assert options.mcp_servers == {}
    assert options.setting_sources == []


def test_the_client_reads_a_conversation(no_cli, monkeypatch):
    seen = scripted(monkeypatch, text_reply("ok"))
    make_client(provider()).complete([
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ])
    assert "USER:\none" in seen["prompt"]
    assert "ASSISTANT:\ntwo" in seen["prompt"]


def test_the_client_refuses_a_login_on_a_shared_server(
        no_cli, monkeypatch):
    monkeypatch.setenv(MULTI_USER_ENV, "1")
    scripted(monkeypatch, text_reply("pong"))
    with pytest.raises(LLMConfigError, match="shared server"):
        make_client(provider("login")).complete(
            [{"role": "user", "content": "ping"}]
        )


@pytest.mark.slow
def test_the_real_cli_answers():
    if shutil.which("claude") is None:
        pytest.skip("the claude CLI is not installed")
    try:
        claude_cli.require_login()
    except LLMError as exc:
        pytest.skip(str(exc))
    reply = make_client(provider()).complete([
        {"role": "user", "content": "Reply with the single word: pong"}
    ])
    assert "pong" in reply.text.lower()
