"""Tests of the MCP server of the REPL (``neurolang-query --mcp``)."""

import asyncio

import pytest

pytest.importorskip("rich")
pytest.importorskip("httpx")
pytest.importorskip("mcp")

from neurolang.frontend import NeurolangPDL  # noqa: E402
from neurolang_apps.utils.cli import main  # noqa: E402
from neurolang_apps.utils.mcp_server import McpSession, server  # noqa: E402


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
def session():
    return McpSession("", _engine)


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

    def test_a_session_of_squall(self):
        def build():
            nl = _engine()
            nl.add_tuple_set([("bob",)], name="adult")
            return nl

        session = McpSession("", build, squall=True)
        assert "bob" in session.repl("obtain every adult.")
        assert "# SQUALL syntax" in session.guide()


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

    def test_the_guide_gives_the_syntax_and_the_examples(self):
        guide = McpSession("neurosynth", _engine).guide("working memory")
        assert "# Datalog syntax" in guide
        assert "# How to answer on this engine" in guide
        assert "# Examples" in guide


class TestServer:
    def test_it_offers_the_tools(self, session):
        tools, [result] = _client(
            session, ("repl", {"text": "ans(n) :- person(n, 5)"})
        )
        assert {tool.name for tool in tools} == {
            "repl", "guide", "list_predicates", "describe_predicate",
            "search_values", "check_query",
        }
        assert not result.is_error
        assert "bob" in result.content[0].text


@pytest.mark.parametrize("flag", [["-i"], ["--ask", "which?"]])
def test_mcp_cannot_be_used_with_another_mode(flag, capsys):
    with pytest.raises(SystemExit):
        main(["--mcp", *flag, "--engine", "neurosynth"])
    assert "--mcp cannot be used with" in capsys.readouterr().err
