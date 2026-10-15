"""Tests of the LLM query assistant in the REPL: ``.ask`` and ``.llm``."""

import io

import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")
pytest.importorskip("httpx")

from rich.console import Console  # noqa: E402

from neurolang.utils.llm.config import parse_config  # noqa: E402
from neurolang.utils.llm.tests.fakes import ScriptedClient  # noqa: E402
from neurolang.utils.tui import InteractiveTuiApp, TuiSettings  # noqa: E402
from neurolang.utils.tui.assistant import parse_ask  # noqa: E402
from neurolang.utils.tui.commands import help_text  # noqa: E402

CONFIG = {
    "providers": {
        "first": {"base_url": "http://llm.test/v1", "model": "m1"},
        "second": {"base_url": "http://llm.test/v1", "model": "m2",
                   "api": "anthropic", "tool_calling": "native"},
    },
    "assistant": {"strategy": "icl", "max_repairs": 1},
}
PAIN = "obtain every Study that mentions 'pain'."


@pytest.fixture
def app():
    """A REPL on a small engine whose assistant answers with a script."""
    from neurolang.frontend import NeurolangPDL

    nl = NeurolangPDL()
    nl.add_tuple_set([(1,), (2,)], name="study")
    nl.add_tuple_set([(1, "pain"), (2, "fear")], name="mentions")
    console = Console(file=io.StringIO(), width=100)
    repl = InteractiveTuiApp(
        nl, TuiSettings(), console=console, err_console=console
    )
    repl.assistant._config = parse_config(CONFIG)
    return repl


def output(app):
    return app.console.file.getvalue()


def script(app, *replies):
    app.assistant.client = ScriptedClient(replies)
    return app.assistant.client


@pytest.mark.parametrize("arg, expected", [
    ("Which studies?", (None, None, "Which studies?")),
    ("--squall --react what's there", ("squall", "react", "what's there")),
    ("--datalog   --icl  q", ("datalog", "icl", "q")),
])
def test_parse_ask(arg, expected):
    assert parse_ask(arg) == expected


def test_parse_ask_unknown_option():
    with pytest.raises(ValueError, match="Unknown option --fast"):
        parse_ask("--fast q")


def test_ask_puts_the_query_in_the_prompt(app):
    client = script(app, "```datalog\nans(s) :- mentions(s, 'pain')\n```\n"
                         "Studies mentioning pain.")
    app._run_line(".ask Which studies mention pain?")
    assert not app.failed
    assert app.pending_input == "ans(s) :- mentions(s, 'pain')"
    text = output(app)
    assert "DATALOG query" in text and "Studies mentioning pain." in text
    assert "1 LLM calls, 15 tokens" in text
    system = client.requests[0]["messages"][0]["content"]
    assert "Datalog syntax" in system


def test_ask_follows_the_mode_or_the_flag(app):
    script(app, f"```squall\n{PAIN}\n```")
    app.settings.squall = True
    app._run_line(".ask Which studies mention pain?")
    assert app.pending_input == PAIN
    script(app, "```datalog\nans(s) :- study(s)\n```")
    app._run_line(".ask --datalog Which studies?")
    assert not app.settings.squall
    assert "Switched to Datalog mode" in output(app)


def test_ask_with_react_strategy_prints_the_steps(app):
    script(
        app,
        'Thought: check\nAction: check_query\nAction Input: {"query": '
        '"ans(s) :- mentions(s, \'pain\')"}',
        'Action: final_answer\nAction Input: {"query": '
        '"ans(s) :- mentions(s, \'pain\')", "explanation": "pain"}',
    )
    app._run_line(".ask --react Which studies mention pain?")
    text = output(app)
    assert "→ check_query()\n      ans(s) :- mentions(s, 'pain')" in text
    assert "The query is valid." in text
    assert app.pending_input == "ans(s) :- mentions(s, 'pain')"


def test_ask_failure_keeps_the_last_attempt(app):
    wrong = "```datalog\nans(s) :- mention(s, 'pain')\n```"
    script(app, wrong, wrong)
    app._run_line(".ask Which studies mention pain?")
    assert app.failed
    text = output(app)
    assert "No valid query" in text and "last attempt" in text
    assert app.pending_input == "ans(s) :- mention(s, 'pain')"


def test_ask_reports_provider_errors(app):
    from neurolang.utils.llm import LLMError

    class Failing:
        def complete(self, *args, **kwargs):
            raise LLMError("first: HTTP 401: bad key")

    app.assistant.client = Failing()
    app._run_line(".ask Which studies?")
    assert app.failed and "HTTP 401: bad key" in output(app)


def test_ask_usage(app):
    app._run_line(".ask")
    assert "Usage: .ask" in output(app)
    app._run_line(".ask --fast q")
    assert "Unknown option --fast" in output(app)


def test_ask_without_httpx(app, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_httpx(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError(name="httpx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_httpx)
    app._run_line(".ask Which studies?")
    assert 'pip install "neurolang[llm]"' in output(app)


def test_llm_shows_and_changes_the_settings(app):
    app._run_line(".llm")
    text = output(app)
    assert "Provider: first" in text and "Strategy: icl" in text
    app._run_line(".llm provider second")
    app._run_line(".llm strategy react")
    assert app.assistant.provider == "second"
    assert app.assistant.strategy == "react"
    assert "Provider: second — anthropic" in output(app)
    app._run_line(".llm provider third")
    assert app.failed and "Unknown LLM provider 'third'" in output(app)
    app._run_line(".llm colour")
    assert "Usage: .llm" in output(app)


def test_settings_select_the_provider(app):
    settings = TuiSettings(llm_provider="second", llm_config="x.yaml")
    repl = InteractiveTuiApp(app._nl, settings, console=app.console)
    assert repl.assistant.provider == "second"
    assert repl.assistant.config_path == "x.yaml"


def test_help_lists_the_assistant_commands():
    text = help_text()
    assert ".ask" in text and ".llm" in text


def test_next_prompt_is_prefilled(app):
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    script(app, "```datalog\nans(s) :- mentions(s, 'pain')\n```")
    app._run_line(".ask Which studies mention pain?")
    with create_pipe_input() as pipe:
        pipe.send_text("\r")
        session = app._make_session(input=pipe, output=DummyOutput())
        assert app._read_and_run(session)
    assert app.pending_input == ""
    assert app.last_query == ("ans(s) :- mentions(s, 'pain')", False)
    assert app.results and list(app.results[0].frame.iloc[:, 0]) == [1]


def test_query_text_styles_the_query():
    from neurolang.utils.tui.assistant import query_text

    text = query_text("q(s) :- study(s)\nans(s) :- q(s)", "datalog", "  ")
    assert text.plain == "  q(s) :- study(s)\n  ans(s) :- q(s)"
    styles = {text.plain[span.start:span.end]: span.style
              for span in text.spans}
    assert styles[":-"] == "bold blue" and styles["study"] == "blue"
