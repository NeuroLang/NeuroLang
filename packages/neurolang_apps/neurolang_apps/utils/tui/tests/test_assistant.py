"""Tests of the LLM query assistant in the REPL: ``.ask`` and ``.llm``."""

import io
import json

import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")
pytest.importorskip("httpx")

from rich.console import Console  # noqa: E402

from neurolang_apps.utils.llm.config import parse_config  # noqa: E402
from neurolang_apps.utils.llm.tests.fakes import ScriptedClient  # noqa: E402
from neurolang_apps.utils.tui import (  # noqa: E402
    InteractiveTuiApp,
    TuiSettings,
)
from neurolang_apps.utils.tui.assistant import parse_ask  # noqa: E402
from neurolang_apps.utils.tui.commands import help_text  # noqa: E402

CONFIG = {
    "providers": {
        "first": {"base_url": "http://llm.test/v1", "model": "m1"},
        "second": {"base_url": "http://llm.test/v1", "model": "m2",
                   "api": "anthropic", "tool_calling": "native"},
    },
    "assistant": {
        "strategy": "icl", "max_repairs": 1, "comment_rules": False,
    },
}
PAIN = "obtain every Study that mentions 'pain'."


@pytest.fixture
def app(monkeypatch):
    """A REPL on a small engine whose assistant answers with a script."""
    monkeypatch.delenv("NEUROLANG_LLM_FEEDBACK_URL", raising=False)
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
    ("Which studies?", (None, None, "Which studies?", None)),
    (
        "--squall --react what's there",
        ("squall", "react", "what's there", None),
    ),
    ("--datalog   --icl  q", ("datalog", "icl", "q", None)),
    ("--steps 12 --react q", (None, "react", "q", 12)),
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
    from neurolang_apps.utils.llm import LLMError

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
    for line in (".llm strategy fast", ".llm strategy", ".llm provider"):
        app.failed = False
        app._run_line(line)
        assert app.failed and "Usage: .llm [provider NAME" in output(app)
    assert app.assistant.strategy == "react"


REACT_PAIN = (
    'Thought: run it\nAction: run_query\nAction Input: {"query": '
    '"ans(s) :- mentions(s, \'pain\')"}',
    'Action: final_answer\nAction Input: {"query": '
    '"ans(s) :- mentions(s, \'pain\')", "explanation": "pain"}',
)


def test_llm_question_writes_datalog_with_react(app):
    """
    ``.llm QUESTION`` uses ReAct and Datalog, although the configuration
    says ICL and the session is in SQUALL mode.
    """
    app.settings.squall = True
    client = script(app, *REACT_PAIN)
    app._run_line(".llm Which studies mention pain?")
    text = output(app)
    assert not app.failed
    assert "Asking first (m1, react)" in text
    assert "→ run_query()\n      ans(s) :- mentions(s, 'pain')" in text
    assert "DATALOG query" in text and "Switched to Datalog mode" in text
    assert app.pending_input == "ans(s) :- mentions(s, 'pain')"
    assert not app.settings.squall
    system = client.requests[0]["messages"][0]["content"]
    assert "Answer in Datalog only" in system and "Action Input:" in system
    assert app.assistant.strategy is None


def test_llm_one_word_question(app):
    script(app, *REACT_PAIN)
    app._run_line(".llm pain?")
    assert app.pending_input == "ans(s) :- mentions(s, 'pain')"


def test_llm_steps_of_one_question(app):
    """``.llm --steps 1`` stops the agent after one step: the query it ran
    with rows is the answer."""
    client = script(app, *REACT_PAIN)
    app._run_line(".llm --steps 1 Which studies mention pain?")
    assert len(client.requests) == 1
    assert app.pending_input == "ans(s) :- mentions(s, 'pain')"
    assert "after 1 steps" in output(app)


def test_llm_steps_setting(app):
    app._run_line(".llm steps 12")
    assert app.assistant.steps == 12 and "steps: 12" in output(app)
    for line in (".llm steps none", ".llm --steps 0 q", ".llm --steps"):
        app.failed = False
        app._run_line(line)
        assert app.failed
    assert app.assistant.steps == 12


def test_llm_puts_the_commented_query_in_the_prompt(app):
    app.assistant.config().assistant.comment_rules = True
    script(app, *REACT_PAIN, '["The studies mentioning pain"]')
    app._run_line(".llm Which studies mention pain?")
    commented = "% The studies mentioning pain\nans(s) :- mentions(s, 'pain')"
    assert app.pending_input == commented
    assert "% The studies mentioning pain" in output(app)
    app._run_line(app.pending_input)
    assert not app.failed and len(app.results[-1].frame) == 1


def test_llm_question_without_httpx(app, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_httpx(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_httpx)
    app._run_line(".llm Which studies?")
    assert app.failed and 'pip install "neurolang[llm]"' in output(app)


def test_settings_select_the_provider(app):
    settings = TuiSettings(llm_provider="second", llm_config="x.yaml")
    repl = InteractiveTuiApp(app._nl, settings, console=app.console)
    assert repl.assistant.provider == "second"
    assert repl.assistant.config_path == "x.yaml"


def test_help_lists_the_assistant_commands():
    text = help_text()
    assert ".ask" in text and ".llm" in text
    assert "Write a Datalog query with the ReAct agent" in text


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
    from neurolang_apps.utils.tui.assistant import query_text

    text = query_text("q(s) :- study(s)\nans(s) :- q(s)", "datalog", "  ")
    assert text.plain == "  q(s) :- study(s)\n  ans(s) :- q(s)"
    styles = {text.plain[span.start:span.end]: span.style
              for span in text.spans}
    assert styles[":-"] == "bold blue" and styles["study"] == "blue"


ASKED = "```datalog\nans(s) :- mentions(s, 'pain')\n```\nPain."


@pytest.fixture
def shared(app):
    """The app sharing its answers with a mocked feedback server."""
    import httpx

    from neurolang_apps.utils.llm.feedback import FeedbackClient

    posts = []

    def handle(request):
        posts.append((request.url.path, json.loads(request.content)))
        return httpx.Response(201, json={"id": f"{len(posts):032x}"})
    app.assistant._feedback = FeedbackClient(
        "http://fb.test", "ann", lambda text: app.console.print(text),
        httpx.MockTransport(handle),
    )
    return posts


def test_answers_are_shared_when_on(app, shared):
    script(app, ASKED, ASKED)
    app._run_line(".ask Which studies mention pain?")
    app._run_line(".ask Which studies mention pain again?")
    assert [path for path, _ in shared] == ["/api/answers"] * 2
    assert shared[0][1]["question"] == "Which studies mention pain?"
    assert shared[0][1]["query"] == "ans(s) :- mentions(s, 'pain')"
    text = output(app)
    assert text.count("are shared with http://fb.test") == 1
    assert f"Shared as {2:032x}" in text and "unsure [comment]" in text


def test_rating_needs_sharing_on(app):
    script(app, ASKED)
    app._run_line(".ask Which studies mention pain?")
    app._run_line(".rate good")
    assert app.failed and "NEUROLANG_LLM_FEEDBACK_URL" in output(app)
    assert app.assistant.last_answer is None
    assert ".rate" in help_text() and ".refine" in help_text()
    app._run_line(".help feedback")
    assert "[COMMENT]" in output(app) and ".rate ID" in output(app)


def test_rate_the_last_answer_or_an_id(app, shared):
    app._run_line(".rate good")
    assert app.failed and "No answer to rate" in output(app)
    script(app, ASKED)
    app._run_line(".ask Which studies mention pain?")
    app.failed = False
    app._run_line(".rate bad it misses chronic pain")
    app._run_line(f".rate {'f' * 32} unsure")
    assert not app.failed
    app._run_line(".rate great")
    assert app.failed and "Usage: .rate" in output(app)
    assert shared[1:] == [
        (f"/api/answers/{1:032x}/feedback",
         {"verdict": "incorrect", "user": "ann",
          "comment": "it misses chronic pain"}),
        (f"/api/answers/{'f' * 32}/feedback",
         {"verdict": "unsure", "user": "ann", "comment": ""}),
    ]


def test_refine_shares_the_checked_query(app, shared):
    script(app, ASKED)
    app._run_line(".ask Which studies mention pain?")
    app._run_line(".refine")
    assert app.pending_input == "ans(s) :- mentions(s, 'pain')"
    app._run_line(".refine submit")
    assert app.failed and "Run the corrected query first" in output(app)
    app._run_line("ans(s) :- mentions(s, 'fear')")
    app.failed = False
    app._run_line(".refine submit pain was wrong")
    assert not app.failed
    path, rating = shared[-1]
    assert path == f"/api/answers/{1:032x}/feedback"
    assert rating["corrected_query"] == "ans(s) :- mentions(s, 'fear')"
    assert rating["corrected_query_checked"] is True
    assert rating["comment"] == "pain was wrong"
    assert "2" in rating["corrected_rows"]


def test_refine_does_not_share_a_failing_query(app, shared):
    script(app, ASKED)
    app._run_line(".ask Which studies mention pain?")
    app._run_line("ans(s) :- mentions(s)")
    app._run_line(".refine submit")
    assert app.failed and "not shared" in output(app)
    assert len(shared) == 1


def test_feedback_end_to_end(app, tmp_path, monkeypatch):
    """.ask, .rate and .refine against a feedback server, then export."""
    import threading
    import urllib.request

    import yaml

    from neurolang_apps.utils.llm.feedback_server import (
        FeedbackStore,
        examples_yaml,
        make_server,
    )

    server = make_server(tmp_path / "fb.sqlite", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("NEUROLANG_LLM_FEEDBACK_URL", url)
    try:
        script(app, ASKED)
        app._run_line(".ask Which studies mention pain?")
        app._run_line(".rate bad")
        app._run_line(".refine")
        app._run_line("ans(s) :- mentions(s, 'fear')")
        app._run_line(".refine submit fear, not pain")
        assert not app.failed
        with urllib.request.urlopen(f"{url}/api/answers") as response:
            (record,) = json.load(response)["answers"]
        with urllib.request.urlopen(url) as response:
            assert b"Assistant answers" in response.read()
    finally:
        server.shutdown()
        server.server_close()
    assert record["question"] == "Which studies mention pain?"
    assert [r["verdict"] for r in record["feedback"]] == ["incorrect"] * 2
    store = FeedbackStore(tmp_path / "fb.sqlite")
    bank = yaml.safe_load(examples_yaml(store))
    assert bank["examples"] == [{
        "question": "Which studies mention pain?",
        "datalog": "ans(s) :- mentions(s, 'fear')",
    }]
