"""Tests of the MCP server of the REPL (``neurolang-query --mcp``)."""

import asyncio

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
        assert text.startswith("Answer a1 to: who is over 4?")
        assert "Who is over 4." in text
        assert "% over 4\nolder(n) :- person(n, a), (a > 4)" in text
        assert "% the older\nans(n) :- older(n)" in text
        assert "n\nbob\ncarol" in text
        [table] = result.structured_content["tables"]
        assert table["columns"] == ["n"] and table["total"] == 2
        assert table["rows"] == [["bob"], ["carol"]]
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
        assert second.content[0].text.startswith("Answer a1")

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
        assert result.content[0].text.startswith("Answer a2\n")

    def test_each_answer_has_its_id(self, session):
        _answer(session)
        result = session.final_answer(
            "ans(n) :- person(n, 3)", "Who is 3.", ["three"]
        )
        assert result.structured_content["id"] == "a2"
        assert set(session.answers) == {"a1", "a2"}


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
    result = session.final_answer(
        "ans(i, j, k, v) :- weight(i, j, k, v)", "The weights.",
        ["the weight of each voxel"],
    )
    images = [c for c in result.content if c.type == "image"]
    assert len(images) == 1 and images[0].mime_type == "image/png"
    links = result.structured_content["links"]
    assert {"viewer", "result", "rows", "query"} <= set(links)
    folder = next((tmp_path / "results").glob("*/a1"))
    assert (folder / "result.nii.gz").exists()
    assert "<html" in (folder / "view.html").read_text().lower()

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
