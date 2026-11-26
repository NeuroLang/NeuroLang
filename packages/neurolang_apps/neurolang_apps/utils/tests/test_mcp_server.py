"""Tests of the MCP server of the REPL (``neurolang-query --mcp``)."""

import asyncio
import json
import os

import pytest

pytest.importorskip("rich")
pytest.importorskip("httpx")
pytest.importorskip("mcp")

from neurolang.frontend import NeurolangPDL  # noqa: E402
from neurolang_apps.utils.cli import main  # noqa: E402
from neurolang_apps.utils.llm import AssistantConfig  # noqa: E402
from neurolang_apps.utils.mcp_server import McpSession, server  # noqa: E402
from neurolang_apps.utils.tui import nifti  # noqa: E402


def _engine():
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [("alice", 3), ("bob", 5), ("carol", 7)], name="person"
    )
    return nl


def _client(session, *calls):
    """
    Return the tools of the server of *session* as a client has them, and
    the result of each of the *calls*, (name, arguments), of its tools.
    """
    from mcp import Client

    async def run():
        async with Client(server(session)) as client:
            tools = (await client.list_tools()).tools
            return tools, [
                await client.call_tool(name, arguments)
                for name, arguments in calls
            ]

    return asyncio.run(run())


@pytest.fixture
def session(tmp_path):
    return McpSession("", _engine, results_dir=tmp_path)


class TestRepl:
    def test_a_query_gives_its_rows_tab_separated(self, session):
        text = session.repl("ans(n, a) :- person(n, a), (a > 4)")
        assert "n\ta\nbob\t5\ncarol\t7" in text
        assert not text.startswith("FAILED")

    def test_the_rules_stay_in_the_session(self, session):
        session.repl("older(n) :- person(n, a), (a > 4)")
        assert "bob\ncarol" in session.repl("ans(n) :- older(n)")

    def test_a_long_result_is_cut_and_says_how_to_have_it_all(self, session):
        text = session.repl("ans(n) :- person(n, a)", max_rows=2)
        assert "(2 of 3 rows: `.save PATH` writes them all)" in text
        rows = text.split("\nn\n")[1].splitlines()[:-1]
        assert len(rows) == 2

    def test_a_failed_statement_says_so(self, session):
        text = session.repl("ans(n) :- person(n, a), (a >)")
        assert text.startswith("FAILED")
        assert "parse error" in text

    def test_dot_commands_run(self, session):
        assert ".save" in session.repl(".help")

    def test_the_initial_program_runs_first(self):
        session = McpSession(
            "", _engine, "young(n) :- person(n, a), (a < 4)"
        )
        assert "alice" in session.repl("ans(n) :- young(n)")

    def test_an_engine_that_fails_to_build_tells_why_to_each_tool(self):
        def build():
            raise FileNotFoundError("no data in ~/neurolang_data")

        _, results = _client(
            McpSession("", build),
            ("repl", {"text": "ans(n) :- person(n, a)"}),
            ("check_query", {"query": "ans(n) :- person(n, a)"}),
        )
        for result in results:
            assert result.is_error
            assert result.content[0].text.endswith(
                "The engine could not be built: FileNotFoundError: no data "
                "in ~/neurolang_data"
            )


class TestAssistantTools:
    def test_check_query_tells_the_error(self, session):
        assert "The query is valid" in session.call(
            "check_query", query="ans(n) :- person(n, a)"
        )
        assert "fails" in session.call("check_query", query="ans(n) :-")

    def test_search_values_finds_a_relation_loaded_in_the_session(
        self, session, tmp_path
    ):
        path = tmp_path / "pets.csv"
        path.write_text("name,pet\nalice,zebrafish\n")
        session.repl(f".load {path}")
        assert "zebrafish" in session.call(
            "search_values", pattern="zebra", predicate=None, column=None
        )

    def test_describe_predicate_shows_its_rows(self, session):
        assert "alice" in session.call("describe_predicate", name="person")

    def test_the_tools_know_the_rules_of_the_session(self, session):
        session.repl("older(n) :- person(n, a), (a > 4)")
        assert "older(" in session.call("list_predicates")
        assert "carol" in session.call("describe_predicate", name="older")

    def test_ask_gives_the_instructions_syntax_and_examples(self):
        text = McpSession("neurosynth", _engine).ask("working memory")
        assert "final_answer" in text and "comments" in text
        assert "Datalog" in text and "# How to answer on this engine" in text
        assert "Call list_predicates" in text and "- person(" not in text
        assert "# Question\nworking memory" in text

    def test_the_tools_work_without_ask(self, session):
        assert "bob" in session.call(
            "run_query", query="ans(n) :- person(n, 5)"
        )


class TestServer:
    def test_it_offers_the_tools(self, session):
        tools, [result] = _client(
            session, ("repl", {"text": "ans(n) :- person(n, 5)"})
        )
        assert {tool.name for tool in tools} == {
            "ask", "repl", "list_predicates", "describe_predicate",
            "search_values", "check_query", "run_query", "final_answer",
            "list_answers", "status", "examples",
        }
        assert not result.is_error
        assert "bob" in result.content[0].text


@pytest.mark.parametrize(
    "flag", [["-i"], ["--ask", "which?"], ["--squall"]]
)
def test_mcp_cannot_be_used_with_another_mode(flag, capsys):
    with pytest.raises(SystemExit):
        main(["--mcp", *flag, "--engine", "neurosynth"])
    assert "--mcp cannot be used with" in capsys.readouterr().err


OLDER = "older(n) :- person(n, a), (a > 4)\nans(n) :- older(n)"


def _answer(session, query=OLDER, comments=("over 4", "the older")):
    return session.final_answer(query, "Who is over 4.", list(comments))


class TestFinalAnswer:
    def test_a_comment_per_rule_or_rejected_before_running(self, session):
        result = _answer(session, comments=["one"])
        text = result.content[0].text
        assert "2 rules and 1 comments" in text
        assert "1. older(n) :- person(n, a), (a > 4)\n2. ans(n) :-" in text
        assert not session.answers

    def test_an_accepted_answer_gives_all_it_has(self, session, tmp_path):
        session.ask("who is over 4?")
        result = _answer(session)
        text = result.content[0].text
        assert text.startswith(
            "## Answer a1: who is over 4?\n\nWho is over 4."
        )
        assert "### Query\n```datalog\n% over 4\nolder(n) :- person(n, a), " \
            "(a > 4)\n% the older\nans(n) :- older(n)\n```" in text
        assert "### Result (2 rows)\n| n |\n|---|\n| bob |\n| carol |" in text
        assert "### Files\n- [Rows (CSV)](file://" in text
        assert result.structured_content["markdown"] == text
        assert set(result.structured_content) == {
            "id", "markdown", "links", "rows",
        }
        assert result.structured_content["rows"] == {"ans": 2}
        links = result.structured_content["links"]
        assert {"query", "rows", "graph"} <= set(links)
        folder = next(tmp_path.glob("*/a1"))
        assert "% over 4" in (folder / "query.dl").read_text()
        assert "bob" in (folder / "rows.csv").read_text()
        assert links["rows"] == (folder / "rows.csv").resolve().as_uri()

    def test_no_rows_is_rejected_once(self, session):
        query = "ans(n) :- person(n, a), (a > 40)"
        first = session.final_answer(query, "None.", ["none"])
        assert "gives no rows" in first.content[0].text
        second = session.final_answer(query, "None.", ["none"])
        assert second.content[0].text.startswith("## Answer a1")

    def test_an_accepted_answer_is_shown_not_run_again(self, session):
        repl = session._session()
        submit, said = repl.submit, []

        def recorded(text):
            reply = submit(text)
            said.append(reply.html)
            return reply

        repl.submit = recorded
        _answer(session)
        assert "not run again" in said[-1]

    def test_an_answer_without_ask_has_no_question_of_before(self, session):
        session.ask("who is over 4?")
        _answer(session)
        result = session.final_answer(
            "ans(n) :- person(n, 3)", "Who is 3.", ["three"]
        )
        assert result.content[0].text.startswith("## Answer a2\n")

    def test_each_answer_has_its_id(self, session):
        _answer(session)
        result = session.final_answer(
            "ans(n) :- person(n, 3)", "Who is 3.", ["three"]
        )
        assert result.structured_content["id"] == "a2"
        assert set(session.answers) == {"a1", "a2"}


class TestAskAgain:
    def test_a_second_ask_gives_only_the_examples_and_the_task(
        self, session
    ):
        first = session.ask("who is over 4?")
        second = session.ask("who is under 4?")
        assert "# Datalog syntax" in first and "# Datalog syntax" not in second
        assert second.startswith("The instructions, the syntax and the ")
        assert second.endswith("# Question\nwho is under 4?")
        assert len(second) < len(first) / 2

    def test_full_gives_them_all_again(self, session):
        session.ask("who is over 4?")
        assert "# Datalog syntax" in session.ask("who?", full=True)


class TestRefine:
    def test_an_answer_is_refined_from_its_query_and_rows(self, session):
        session.ask("who is over 4?")
        _answer(session)
        text = session.ask("and over 6?", refine="a1")
        assert "# Previous question\nwho is over 4?" in text
        assert "% over 4\nolder(n)" in text
        assert "bob" in text.split("# What happened when it ran")[1]
        assert "# New question\nand over 6?" in text

    def test_last_is_the_query_the_user_ran_after(self, session):
        session.ask("who is over 4?")
        _answer(session)
        session.repl("ans(n) :- person(n, a), (a > 6)")
        text = session.ask("and their ages?", refine="last")
        assert "(a > 6)" in text.split("# What happened")[0]

    def test_an_unknown_answer_is_told(self, session):
        from mcp.server.mcpserver.exceptions import ToolError

        with pytest.raises(ToolError, match="No answer 'a9'"):
            session.ask("and?", refine="a9")


TINY_ENGINE = """\
engines:
  tiny:
    sorts:
      study: {columns: [study_id], type: int}
      voxel: {columns: [i, j, k], type: int, space: voxel}
      value: {columns: [v], type: float}
    predicates:
      reports: {arguments: [study, voxel]}
      weight: {arguments: [voxel, value]}
"""


def test_an_answer_of_voxels_has_its_glass_brain_and_viewer(
    monkeypatch, tmp_path
):
    import nibabel as nib
    import numpy as np
    from neurolang_apps.utils import engine_registry

    engines = tmp_path / "engines.yaml"
    engines.write_text(TINY_ENGINE)
    monkeypatch.setattr(engine_registry, "_ENGINES_YAML", engines)

    def build():
        nl = NeurolangPDL()
        nl.template = nib.Nifti1Image(
            np.ones((4, 4, 4), dtype=np.uint8), np.diag([2.0, 2, 2, 1])
        )
        nl.add_tuple_set(
            [(1, 2, 3, 0.25), (0, 0, 0, 0.5), (2, 2, 2, 0.75)],
            name="weight",
        )
        return nl

    session = McpSession("tiny", build, results_dir=tmp_path / "results")
    session.ask("Which voxels have a weight?")
    result = session.final_answer(
        "ans(i, j, k, v) :- weight(i, j, k, v)", "The weights.",
        ["the weight of each voxel"],
    )
    images = [c for c in result.content if c.type == "image"]
    assert len(images) == 1 and images[0].mime_type == "image/png"
    links = result.structured_content["links"]
    assert list(links) == [
        "viewer", "glass-brain", "result", "rows", "graph", "query", "index",
    ]
    assert result.content[0].text.split("\n\n")[1] == (
        f"**[Open the interactive viewer]({links['viewer']})**"
    )
    assert "| 0 | 0 | 0 | 0.5 |" in result.content[0].text
    resources = [c for c in result.content if c.type == "resource_link"]
    assert [(c.uri, c.title, c.mime_type) for c in resources[:3]] == [
        (links["viewer"], "a1: interactive viewer", "text/html"),
        (links["glass-brain"], "a1: glass brain (PNG)", "image/png"),
        (links["result"], "a1: NIfTI image", "application/gzip"),
    ]
    folder = next((tmp_path / "results").glob("*/a1"))
    assert (folder / "result.nii.gz").exists()
    assert (folder / "glass-brain.png").read_bytes()[:4] == b"\x89PNG"
    assert links["glass-brain"].endswith("a1/glass-brain.png")
    assert "| 2 | 2 | 2 | 0.75 |" in result.content[0].text.split(
        "by v, highest first")[1].splitlines()[3]
    assert "<html" in (folder / "view.html").read_text().lower()
    heading = "a1: Which voxels have a weight?"
    assert f"<h1>{heading}</h1>" in (folder / "graph.html").read_text()
    assert heading in (folder / "view.html").read_text()
    assert (folder / "query.dl").read_text().startswith(
        f"% {heading}\n% The weights.\n% the weight of each voxel\n"
    )

    def failing(*args, **kwargs):
        raise ValueError("no glass brain")

    monkeypatch.setattr(nifti, "render_png", failing)
    result = session.final_answer(
        "ans(i, j, k, v) :- weight(i, j, k, v), (v > 0.3)", "Above 0.3.",
        ["the weights above 0.3"],
    )
    assert result.structured_content["id"] == "a2"
    assert not [c for c in result.content if c.type == "image"]


def test_the_assistant_settings_need_no_provider(tmp_path):
    from neurolang_apps.utils.mcp_server import assistant_settings

    path = tmp_path / "llm.yaml"
    path.write_text("assistant:\n  query_memory_mb: 16000\n")
    assert assistant_settings(path).query_memory_mb == 16000
    assert assistant_settings(tmp_path / "none.yaml").query_memory_mb == (
        AssistantConfig().query_memory_mb
    )


def test_the_client_gets_text_images_and_structured_content(session):
    _, [result] = _client(session, ("final_answer", {
        "query": "ans(n) :- person(n, 5)", "explanation": "Bob.",
        "comments": ["the person of 5"],
    }))
    assert not result.is_error
    assert "bob" in result.content[0].text
    assert result.structured_content["id"] == "a1"


def test_a_lazy_engine_is_built_at_the_first_call(tmp_path):
    built = []

    def build():
        built.append(True)
        return _engine()

    session = McpSession("", build, results_dir=tmp_path, lazy=True)
    assert not built
    assert "bob" in session.repl("ans(n) :- person(n, 5)")
    assert built == [True]
    session.repl("ans(n) :- person(n, 3)")
    assert built == [True]


def test_the_log_tells_the_build_and_each_call(session, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="neurolang_apps.mcp"):
        _client(
            session,
            ("run_query", {"query": "ans(n) :- person(n, 5)"}),
            ("final_answer", {
                "query": "ans(n) :- person(n, 5)", "explanation": "Bob.",
                "comments": ["the person of 5"],
            }),
            ("final_answer", {
                "query": "ans(n) :- person(n, 5)", "explanation": "Bob.",
                "comments": [],
            }),
        )
    lines = [record.getMessage() for record in caplog.records]
    assert any(line.startswith("engine : built in") for line in lines)
    assert any(
        line.startswith("run_query query=ans(n) :- person(n, 5) -> ")
        for line in lines
    )
    assert any(
        "final_answer" in line and "-> accepted a1 (ans: 1 rows; 0 images)"
        in line for line in lines
    )
    assert any(
        "-> The answer is rejected: the query has 1 rules and 0 comments"
        in line for line in lines
    )


def test_a_glass_brain_is_entitled_with_the_question():
    from neurolang_apps.utils.mcp_server import _image_title

    assert _image_title("", "a1", "lift", False) == "a1: lift"
    assert _image_title("a1: Where?", "a1", "lift", False) == "a1: Where?"
    assert _image_title("a1: Where?", "a1", "p", True) == "a1: Where? (p)"
    assert _image_title("a1: " + "x" * 90, "a1", "p", False).endswith("...")


def test_an_answer_ends_with_the_references_of_neurolang(tmp_path):
    session = McpSession("neurosynth", _engine, results_dir=tmp_path)
    result = _answer(session)
    text = result.content[0].text
    references = text.split("### How to cite NeuroLang\n")[1].splitlines()
    assert [line[:16] for line in references] == [
        "1. Zanitti GE, S", "2. Iovene V, Zan",
    ]
    assert "Yarkoni" not in text  # the data's sources: .credits


def test_the_engine_is_built_with_its_text_embeddings_loaded(monkeypatch):
    """A query's forked process uses the model loaded before it: on macOS
    it could not load it (SIGABRT, similar_study)."""
    from types import SimpleNamespace

    from neurolang_apps.utils import engine_registry, mcp_server
    from neurolang_apps.utils import text_embeddings

    loaded = []
    nl = SimpleNamespace(text_embedding_indexes={"studies": object()})
    monkeypatch.setattr(
        engine_registry, "build_engine", lambda *args, **kwargs: nl
    )
    monkeypatch.setattr(text_embeddings, "preload", loaded.append)
    args = SimpleNamespace(
        engine="neurosynth", data_dir="d", resolution=2.0,
        no_engine_cache=False,
    )
    assert mcp_server.build_engine(args) is nl
    assert loaded == [nl]


class TestConfined:
    @pytest.fixture
    def confined(self, monkeypatch, tmp_path):
        from neurolang.confine import CONFINE_ENV

        data = tmp_path / "data"
        data.mkdir()
        monkeypatch.setenv(CONFINE_ENV, str(data))
        return McpSession("", _engine, results_dir=tmp_path / "results")

    def test_a_file_outside_the_data_is_not_read(self, confined, tmp_path):
        path = tmp_path / "private.csv"
        path.write_text("k,v\napi_key,secret\n")
        text = confined.repl(f".load {path} private")
        assert text.startswith("FAILED") and "secret" not in text

    def test_no_file_is_written(self, confined, tmp_path):
        confined.repl("ans(n) :- person(n, a)")
        assert confined.repl(f".save {tmp_path / 'out.csv'}").startswith(
            "FAILED"
        )
        assert not (tmp_path / "out.csv").exists()

    def test_the_answers_files_are_written(self, confined, tmp_path):
        result = _answer(confined)
        assert result.structured_content["id"] == "a1"
        assert next((tmp_path / "results").glob("*/a1/rows.csv")).exists()


@pytest.mark.parametrize("command", [
    ".llm reload", ".ask which?", ".explain", ".llm which?",
])
def test_the_servers_own_assistant_is_refused(session, command):
    text = session.repl(f"ans(n) :- person(n, a)\n{command}")
    assert text.startswith("FAILED")
    assert "you are the assistant" in text


def test_the_session_is_confined_unless_unconfined(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from neurolang.confine import CONFINE_ENV
    from neurolang_apps.utils.mcp_server import confine

    monkeypatch.setenv(CONFINE_ENV, "")  # restored after the test
    monkeypatch.delenv(CONFINE_ENV)
    args = SimpleNamespace(
        unconfined=True, data_dir=tmp_path / "d", results_dir=tmp_path / "r"
    )
    assert confine(args) is None and CONFINE_ENV not in os.environ
    args.unconfined = False
    roots = confine(args)
    assert roots == [(tmp_path / "d").resolve(), (tmp_path / "r").resolve()]
    assert os.environ[CONFINE_ENV].split(os.pathsep) == list(map(str, roots))


class TestLimits:
    def test_a_repl_query_runs_within_the_limits(self, tmp_path, monkeypatch):
        from neurolang_apps.utils import mcp_server
        from neurolang_apps.utils.llm.validation import Preview

        calls = []

        def stopped(nl, text, language, rows, engine, timeout, memory, cells):
            calls.append((text, timeout, memory, cells))
            return Preview(False, "The query ran for more than 7 s and was "
                           "stopped: restrict it.")

        monkeypatch.setattr(mcp_server, "run_preview", stopped)
        session = McpSession(
            "", _engine, results_dir=tmp_path,
            config=AssistantConfig(query_timeout=7, query_memory_mb=99),
        )
        text = session.repl("ans(n) :- person(n, a)")
        assert text.startswith("FAILED") and "was stopped" in text
        assert calls == [
            ("ans(n) :- person(n, a)", 7, 99, mcp_server.MAX_CELLS)
        ]
        text = session.repl("older(n) :- person(n, a), (a > 4)\n.predicates")
        assert "older" in text and len(calls) == 1  # no query: no preview

    def test_a_repl_query_is_shown_not_run_again(self, session):
        text = session.repl("ans(n) :- person(n, a), (a > 4)")
        assert "not run again" in text and "bob\ncarol" in text

    def test_the_rows_of_repl_are_capped(self, tmp_path):
        def build():
            nl = _engine()
            nl.add_tuple_set([(i,) for i in range(600)], name="number")
            return nl

        session = McpSession("", build, results_dir=tmp_path)
        text = session.repl("ans(i) :- number(i)", max_rows=10_000)
        assert "(500 of 600 rows" in text

    def test_an_answer_too_large_to_keep_is_refused(
        self, session, monkeypatch
    ):
        from neurolang_apps.utils import mcp_server

        monkeypatch.setattr(mcp_server, "MAX_CELLS", 1)
        result = _answer(session)
        assert "too large to show here" in result.content[0].text
        assert not session.answers


@pytest.mark.parametrize("given, built", [([], 2.0), (["-r", "1"], 1.0)])
def test_the_engine_of_mcp_is_on_a_grid_of_2_mm(given, built, monkeypatch):
    from neurolang_apps.utils import mcp_server

    seen = []
    monkeypatch.setattr(
        mcp_server, "main", lambda args, program: seen.append(args.resolution)
    )
    main(["--mcp", "--engine", "neurosynth", *given])
    assert seen == [built]


def test_the_answers_of_old_sessions_are_removed(tmp_path):
    import time

    from neurolang_apps.utils.mcp_server import ResultStore

    old, recent = tmp_path / "20261001-120000-0a1b2c3d", (
        tmp_path / "20261117-120000-0a1b2c3d"
    )
    mine = tmp_path / "notes"
    for folder in (old, recent, mine):
        folder.mkdir()
    week_ago = time.time() - 8 * 86400
    for folder in (old, mine):
        os.utime(folder, (week_ago, week_ago))
    ResultStore(tmp_path, keep_days=0)
    assert old.exists()
    ResultStore(tmp_path, keep_days=7)
    assert not old.exists() and recent.exists() and mine.exists()


def test_the_files_of_the_server_are_its_users_only(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from neurolang.confine import CONFINE_ENV
    from neurolang_apps.utils import mcp_server

    monkeypatch.setenv(CONFINE_ENV, "")
    log = tmp_path / "mcp.log"
    monkeypatch.setattr(
        mcp_server, "McpSession", lambda *a, **k: SimpleNamespace()
    )
    monkeypatch.setattr(
        mcp_server, "server",
        lambda session: SimpleNamespace(run=lambda transport: None),
    )
    args = SimpleNamespace(
        unconfined=True, data_dir=tmp_path, results_dir=tmp_path,
        log=str(log), build_on_first_call=True, keep_results=7, engine="",
    )
    umask = os.umask(0o022)
    try:
        mcp_server.main(args)
        assert os.umask(0o022) == 0o077
    finally:
        os.umask(umask)
        for handler in list(mcp_server.log.handlers):
            mcp_server.log.removeHandler(handler)
            handler.close()
        mcp_server.log.propagate = True
        mcp_server.log.setLevel(0)
    assert oct(log.stat().st_mode & 0o777) == "0o600"


def test_a_cancelled_call_stops_the_query_it_runs(monkeypatch):
    import threading

    import anyio
    from neurolang_apps.utils import mcp_server

    freed = threading.Event()

    def stop_queries():  # the query's process killed: the call returns
        freed.set()
        return 1

    monkeypatch.setattr(mcp_server.validation, "stop_queries", stop_queries)

    async def call():
        with anyio.move_on_after(0.2) as scope:
            await mcp_server._logged("run_query", {}, lambda: freed.wait(5))
        return scope.cancelled_caught

    assert anyio.run(call)
    assert freed.wait(1)


def test_stop_queries_stops_a_running_query(forking, monkeypatch):
    import threading
    import time

    import neurolang_apps
    from neurolang_apps.utils.llm import validation

    monkeypatch.setattr(neurolang_apps, "BLAS_FORK_SAFE", True)
    raised = []

    def run():
        try:
            validation._in_child(lambda: time.sleep(30), 60, None)
        except validation.QueryStopped:
            raised.append(True)

    thread = threading.Thread(target=run)
    start = time.monotonic()
    thread.start()
    while not validation._RUNNING and time.monotonic() - start < 10:
        time.sleep(0.05)
    assert validation.stop_queries() == 1
    thread.join(10)
    assert raised == [True] and time.monotonic() - start < 10
    assert not validation._RUNNING and not validation._STOPPED


def test_a_markdown_cell_is_short_and_one_line():
    from neurolang_apps.utils.mcp_server import _cell

    assert _cell(0.123456789) == "0.1235"
    assert _cell("a|b\nc") == "a\\|b c"
    assert _cell(None) == "" and _cell(7) == "7"


class TestHttp:
    HEADERS = {
        "content-type": "application/json",
        "accept": "application/json, text/event-stream",
        "host": "repl.example.org", "mcp-protocol-version": "2025-06-18",
    }

    def _rpc(self, bridge, method, params=None):
        import json

        status, headers, body = bridge.call(
            "POST", "/mcp", self.HEADERS, json.dumps({
                "jsonrpc": "2.0", "id": 1, "method": method,
                "params": params or {},
            }).encode(),
        )
        return status, json.loads(body)

    def test_the_http_app_answers_through_the_bridge(self, session):
        from neurolang_apps.utils.mcp_server import HttpBridge, http_app

        bridge = HttpBridge(http_app(session))
        status, answer = self._rpc(bridge, "initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"},
        })
        assert status == 200 and "NeuroLang" in (
            answer["result"]["instructions"]
        )
        status, answer = self._rpc(bridge, "tools/call", {
            "name": "run_query",
            "arguments": {"query": "ans(n) :- person(n, 5)"},
        })
        assert status == 200
        assert "bob" in answer["result"]["content"][0]["text"]

    def test_links_under_a_base_url(self, tmp_path):
        from neurolang_apps.utils.mcp_server import ResultStore

        store = ResultStore(tmp_path, token_bytes=16,
                            base_url="https://repl.example.org/mcp/files/")
        assert len(store.root.name.rsplit("-", 1)[1]) == 32
        path = store.folder("a1") / "view.html"
        path.write_text("x")
        assert store.link(path) == (
            "https://repl.example.org/mcp/files/"
            f"{store.root.name}/a1/view.html"
        )

    def test_a_new_conversation_gets_the_whole_prompt(self, session):
        session.ask("who?")
        session.new_conversation()
        assert "# Datalog syntax" in session.ask("who else?")


def _floats():
    nl = _engine()
    nl.add_tuple_set([("a", 0.2), ("b", 0.9), ("c", 0.5)], name="score")
    return nl


class TestUsability:
    def test_a_result_is_ordered_by_its_value(self, tmp_path):
        session = McpSession("", _floats, results_dir=tmp_path)
        text = session.final_answer(
            "ans(n, v) :- score(n, v)", "Scores.", ["each score"]
        ).content[0].text
        assert "### Result (3 rows, by v, highest first)\n| n | v |\n" \
            "|---|---|\n| b | 0.9 |\n| c | 0.5 |\n| a | 0.2 |" in text

    def test_the_answers_have_an_index_and_records(self, session, tmp_path):
        session.ask("who is over 4?")
        result = _answer(session)
        index = result.structured_content["links"]["index"]
        page = next(tmp_path.glob("*/index.html"))
        assert index == page.resolve().as_uri()
        assert "who is over 4?" in page.read_text()
        assert 'href="a1/rows.csv"' in page.read_text()
        record = json.loads(next(tmp_path.glob("*/a1/answer.json"))
                            .read_text())
        assert record["question"] == "who is over 4?"

    def test_an_earlier_sessions_answer_is_listed_and_refined(self, tmp_path):
        first = McpSession("", _engine, results_dir=tmp_path)
        first.ask("who is over 4?")
        _answer(first)
        later = McpSession("", _engine, results_dir=tmp_path)
        listed = later.list_answers()
        ident = f"{first.store.root.name}/a1"
        assert f"`{ident}`: who is over 4?" in listed
        text = later.ask("and over 6?", refine=ident)
        assert "# Previous question\nwho is over 4?" in text

    def test_status_tells_the_engine_and_the_confinement(self, session):
        session.about = {"Grid": "2.0 mm"}
        session._session()
        text = session.status()
        assert "built" in text and "Grid: 2.0 mm" in text
        assert "Statements: not confined" in text

    def test_examples_give_topics_then_questions(self):
        session = McpSession("neurosynth", _engine)
        topics = session.examples()
        assert topics.startswith("Topics of the example questions")
        topic = topics.splitlines()[1][2:].rsplit(" (", 1)[0]
        assert session.examples(topic).startswith("- ")
        assert "No topic 'nope'" in session.examples("nope")


def test_the_mcp_options_come_from_their_file(monkeypatch, tmp_path, capsys):
    config = tmp_path / "mcp.yaml"
    config.write_text("resolution: 1.0\nlog: ~/mcp.log\n"
                      "allow_dir: [/data/maps]\n")
    monkeypatch.setenv("NEUROLANG_MCP_CONFIG", str(config))
    main(["--mcp", "--engine", "neurostore", "--print-mcp-command"])
    command = capsys.readouterr().out
    assert command.startswith("claude mcp add --scope user neurolang -- ")
    assert "--engine neurostore" in command and "--resolution 1.0" in command
    home = os.path.expanduser("~")
    assert f"--data-dir {home}/neurolang_data" in command
    assert f"--log {home}/mcp.log" in command
    assert "--allow-dir /data/maps" in command
    config.write_text("nope: 1\n")
    with pytest.raises(SystemExit):
        main(["--mcp", "--print-mcp-command"])


def test_an_allowed_folder_is_read_confined(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from neurolang.confine import CONFINE_ENV
    from neurolang_apps.utils.mcp_server import confine

    monkeypatch.setenv(CONFINE_ENV, "")
    maps = tmp_path / "maps"
    maps.mkdir()
    (maps / "pets.csv").write_text("name,pet\nalice,zebrafish\n")
    confine(SimpleNamespace(
        unconfined=False, data_dir=tmp_path / "d",
        results_dir=tmp_path / "r", allow_dir=[str(maps)],
    ))
    session = McpSession("", _engine, results_dir=tmp_path / "r")
    assert "zebrafish" in session.repl(
        f".load {maps / 'pets.csv'} pets\nans(n, p) :- pets(n, p)"
    )
    text = session.repl(f".load {tmp_path / 'outside.csv'} x")
    assert text.startswith("FAILED") and "--allow-dir" in text
