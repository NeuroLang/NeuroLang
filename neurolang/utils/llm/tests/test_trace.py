"""Tests of the tracing of the requests to the model, and of --ask -v."""

import io

import pytest

from neurolang.utils import cli, engine_registry
from neurolang.utils.llm import agent as agent_module
from neurolang.utils.llm.trace import Printer, TracingClient

from .fakes import ScriptedClient

SEARCH = (
    'Thought: find the term\nAction: search_values\nAction Input: '
    '{"pattern": "pain"}'
)
ANSWER = (
    'Thought: done\nAction: final_answer\nAction Input: '
    '{"query": "ans(s) :- mentions(s, \'pain\')", "explanation": "x"}'
)


def test_requests_hold_the_new_messages_only():
    events = []
    client = TracingClient(ScriptedClient(["a", "b", "c"]), events.append)
    system = {"role": "system", "content": "You write queries."}
    first = [system, {"role": "user", "content": "Which studies?"}]
    client.complete(first)
    client.complete(first + [
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "Observation: 3 rows"},
    ])
    client.complete([system, {"role": "user", "content": "Again?"}])
    requests = [e for e in events if e["event"] == "request"]
    assert [(r["start"], len(r["messages"])) for r in requests] == [
        (0, 2), (2, 2), (0, 2)
    ]
    replies = [e for e in events if e["event"] == "reply"]
    assert [r["text"] for r in replies] == ["a", "b", "c"]
    assert replies[0]["usage"] == {"input": 10, "output": 5}


def test_failed_requests_are_reported():
    events = []
    client = TracingClient(ScriptedClient([]), events.append)
    with pytest.raises(AssertionError):
        client.complete([{"role": "user", "content": "q"}])
    assert "asked more than scripted" in events[-1]["error"]


def test_printer():
    stream = io.StringIO()
    printer = Printer(stream, limit=20, system_limit=10, hint="--full")
    client = TracingClient(
        ScriptedClient(["Thought: look\n" + "x" * 50]), printer.show
    )
    client.complete([
        {"role": "system", "content": "s" * 100},
        {"role": "user", "content": "Which studies?"},
    ])
    text = stream.getvalue()
    assert "call 1 → 2 new message(s) from #0, tools: none" in text
    assert "[system]" in text and "90 more characters, --full" in text
    assert "call 1 ←" in text and "input 10, output 5" in text
    assert "Thought: look" in text


@pytest.fixture
def ask(nl, tmp_path, monkeypatch):
    """Run ``neurolang-query --ask`` on *nl* with scripted replies."""
    config = tmp_path / "llm.yaml"
    config.write_text(
        "default_provider: fake\n"
        "providers:\n"
        "  fake: {api: openai, base_url: 'http://llm.test/v1', "
        "model: m, tool_calling: text}\n"
        "assistant: {strategy: react, few_shot_k: 0}\n"
    )
    monkeypatch.setattr(
        engine_registry, "build_engine", lambda *args, **kwargs: nl
    )

    def run(*options, replies=(SEARCH, ANSWER)):
        client = ScriptedClient(replies)
        monkeypatch.setattr(
            agent_module, "make_client", lambda *args, **kwargs: client
        )
        monkeypatch.setattr(
            "neurolang.utils.llm.make_client", lambda *args, **kwargs: client
        )
        cli.main([
            "--ask", "Which studies mention pain?", "--llm-config",
            str(config), *options,
        ])
    return run


def test_ask_prints_a_line_per_step(ask, capsys):
    ask()
    out, err = capsys.readouterr()
    assert out.strip() == "ans(s) :- mentions(s, 'pain')"
    assert "  → search_values: Facts with values containing 'pain'" in err
    assert "thought" not in err


def test_ask_verbose_prints_the_steps(ask, capsys):
    ask("-v")
    out, err = capsys.readouterr()
    assert out.strip() == "ans(s) :- mentions(s, 'pain')"
    assert "  thought: find the term" in err
    assert "  → search_values(pattern='pain')" in err
    assert "    mentions(1, 'pain')" in err
    assert "[system]" not in err


def test_ask_very_verbose_prints_the_messages(ask, capsys):
    ask("-vv")
    out, err = capsys.readouterr()
    assert out.strip() == "ans(s) :- mentions(s, 'pain')"
    assert "─── call 1 → 2 new message(s) from #0" in err
    assert "[system]" in err and "Work in steps" in err
    assert "[user]\nWhich studies mention pain?" in err
    assert "─── call 2 → 2 new message(s) from #2" in err
    assert "Observation: Facts with values containing 'pain'" in err
    assert "more characters" not in err
