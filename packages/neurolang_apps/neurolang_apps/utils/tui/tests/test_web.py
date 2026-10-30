"""Tests for the REPL driven from a web page."""

from pathlib import Path

import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

import numpy as np  # noqa: E402

from neurolang.frontend import NeurolangPDL  # noqa: E402
from neurolang_apps.utils.tui import nifti  # noqa: E402
from neurolang_apps.utils.tui.web import WebRepl  # noqa: E402
from neurolang_apps.utils.tui.tests import test_nifti  # noqa: E402

tiny = test_nifti.tiny  # the REPL on the tiny engine, with its template


@pytest.fixture
def repl():
    nl = NeurolangPDL()
    nl.add_tuple_set([("alice", 3), ("bob", 5)], name="person")
    return WebRepl(nl, engine_name="neurosynth")


def test_a_query_comes_back_as_html_with_its_rows(repl):
    reply = repl.submit("ans(x, n) :- person(x, n)")
    assert reply.tables[0]["rows"]
    assert "Query completed" in reply.html
    assert "<span" in reply.html  # styled, as on a terminal
    assert not reply.failed


def test_the_session_keeps_its_rules_between_submissions(repl):
    repl.submit("big(x) :- person(x, n) & (n > 4)")
    [table] = repl.submit("ans(x) :- big(x)").tables
    assert table["rows"] == [["bob"]]


def test_a_pasted_block_runs_as_in_the_repl(repl):
    reply = repl.submit(
        "% the people with more than 4\n"
        "big(x) :- person(x, n) & (n > 4)\n"
        "ans(x) :- big(x)\n"
        ".head 1\n"
    )
    assert any(["bob"] in table["rows"] for table in reply.tables)
    assert not reply.failed


def test_an_error_is_reported_and_marks_the_reply_failed(repl):
    reply = repl.submit("ans(x) :- person(x, n")
    assert reply.failed
    assert "error" in reply.html.lower()
    assert not repl.submit("ans(x) :- person(x, n)").failed


def test_replies_do_not_repeat_earlier_output(repl):
    repl.submit("ans(x) :- person(x, n)")
    assert "alice" not in repl.submit(".help").html


def test_the_banner_names_the_engine_and_the_session_command_line():
    nl = NeurolangPDL()
    banner = WebRepl(nl, engine_name="neurosynth", resolution=2).banner()
    assert "neurosynth" in banner
    assert "neurolang-query -i --engine neurosynth --resolution 2" in banner


def test_quit_does_not_end_the_page(repl):
    reply = repl.submit(".quit")
    assert "reload" in reply.html.lower()


def test_view_pages_are_returned_not_opened(repl, monkeypatch, tmp_path):
    monkeypatch.setattr(nifti.tempfile, "tempdir", str(tmp_path))
    repl.submit("ans(x) :- person(x, n)")
    monkeypatch.setattr(
        nifti, "last_result_images", lambda app, save: [("layer", Path("a"))]
    )
    monkeypatch.setattr(
        nifti, "_viewer_page", lambda *args, **kwargs: "<html>page</html>"
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("opened a browser")

    monkeypatch.setattr(nifti.webbrowser, "open", forbidden)
    reply = repl.submit(".view")
    assert reply.viewers == ["<html>page</html>"]
    assert "Open it in a browser" not in reply.html
    assert not list(tmp_path.glob("neurolang_view_*.html"))


class TestTables:
    """Check results come back as data for the page's table viewer."""

    def test_a_result_is_a_table_not_text(self, repl):
        reply = repl.submit("ans(x, n) :- person(x, n)")
        [table] = reply.tables
        assert table["columns"] == ["x", "n"]
        assert sorted(table["rows"]) == [["alice", 3], ["bob", 5]]
        assert table["kinds"] == ["str", "int"]
        assert table["total"] == 2 and not table["truncated"]
        assert "alice" not in reply.html
        assert "2 rows" in reply.html

    def test_the_tables_are_json(self, repl):
        import json

        reply = repl.submit("ans(x, n) :- person(x, n)")
        assert json.loads(json.dumps(reply.tables)) == reply.tables

    def test_missing_values_are_null_and_infinities_are_text(self):
        import json

        import pandas as pd

        from neurolang_apps.utils.tui.web import table_payload

        frame = pd.DataFrame(
            {"f": [1.5, float("nan"), float("inf")], "s": ["a", None, "b"]}
        )
        table = table_payload(frame)
        assert table["kinds"] == ["float", "str"]
        assert table["rows"] == [[1.5, "a"], [None, None], ["Infinity", "b"]]
        json.loads(json.dumps(table))  # a page can parse it

    def test_regions_are_shown_as_in_the_terminal(self):
        import numpy as np

        from neurolang.regions import ExplicitVBR

        region = ExplicitVBR(np.array([[0, 0, 0]]), np.eye(4))
        nl = NeurolangPDL()
        nl.add_tuple_set([("a", region)], name="area")
        [table] = WebRepl(nl).submit("ans(n, r) :- area(n, r)").tables
        assert table["kinds"] == ["str", "other"]
        assert table["rows"] == [["a", "VBR(n_voxels=1, centroid=(0.0, 0.0, 0.0) mm)"]]

    def test_a_long_result_is_cut_and_says_so(self):
        nl = NeurolangPDL()
        nl.add_tuple_set([(i,) for i in range(10)], name="n")
        reply = WebRepl(nl, max_rows=4).submit("ans(i) :- n(i)")
        [table] = reply.tables
        assert len(table["rows"]) == 4
        assert table["total"] == 10 and table["truncated"]
        assert "10 rows" in reply.html and "4" in reply.html

    def test_an_empty_result_has_no_rows(self, repl):
        [table] = repl.submit("ans(x) :- person(x, 99)").tables
        assert table["rows"] == [] and table["total"] == 0

    def test_a_boolean_or_a_rule_has_no_table(self, repl):
        assert repl.submit("ans() :- person(x, n)").tables == []
        assert repl.submit("big(x) :- person(x, n)").tables == []

    def test_sorting_again_gives_the_table_again(self, repl):
        repl.submit("ans(x, n) :- person(x, n)")
        [table] = repl.submit(".sort n:desc").tables
        assert [row[0] for row in table["rows"]] == ["bob", "alice"]

    def test_no_rows_are_left_for_more(self, repl):
        repl.submit("ans(x, n) :- person(x, n)")
        assert repl.submit(".more").tables == []


class TestDownloads:
    """The files of a result the page offers to save."""

    @pytest.fixture
    def web(self, tiny):
        return WebRepl(tiny._nl, engine_name="tiny")

    def test_a_table_tells_the_columns_of_its_image(self, web):
        [table] = web.submit("ans(i, j, k, w) :- weight(i, j, k, w)").tables
        assert table["block"] == 0
        assert table["nifti"] == ["w"] and table["nifti_reason"] == ""

    def test_a_mask_is_an_image_without_a_value_column(self, web):
        [table] = web.submit("ans(i, j, k) :- voxel(i, j, k)").tables
        assert table["nifti"] == [None]

    def test_a_result_without_voxels_has_no_image_and_says_why(self, web):
        web.app._nl.add_tuple_set([(10, 1, 2, 3)], name="seen")
        [table] = web.submit("ans(s, a, b, c) :- seen(s, a, b, c)").tables
        assert table["nifti"] is None
        assert "no voxel columns" in table["nifti_reason"]

    def test_the_image_is_the_one_save_writes(self, web, tmp_path):
        web.submit("ans(i, j, k, w) :- weight(i, j, k, w)")
        name, mime, data = web.download(0, "nifti", "w")
        assert name == "result-1-w.nii.gz" and mime == "application/gzip"
        image = nifti.read_nifti(_written(data, tmp_path / name))
        assert image.shape == (4, 4, 4)
        assert np.asanyarray(image.dataobj)[1, 2, 3] == pytest.approx(0.75)
        web.app._handle_dot_command(f".save {tmp_path / 'saved.nii.gz'} --value w")
        saved = nifti.read_nifti(tmp_path / "saved.nii.gz")
        assert np.array_equal(
            np.asanyarray(image.dataobj), np.asanyarray(saved.dataobj)
        )
        assert np.array_equal(image.affine, saved.affine)

    def test_a_mask_is_named_without_a_value(self, web, tmp_path):
        web.submit("ans(i, j, k) :- voxel(i, j, k)")
        name, _, data = web.download(0, "nifti")
        assert name == "result-1.nii.gz"
        image = nifti.read_nifti(_written(data, tmp_path / name))
        assert np.count_nonzero(np.asanyarray(image.dataobj)) == 2

    def test_all_the_rows_come_as_text(self, web):
        web.submit("ans(i, j, k, w) :- weight(i, j, k, w)")
        csv_file = web.download(0, "csv")
        assert csv_file.name == "result-1.csv"
        lines = csv_file.data.decode().splitlines()
        assert lines[0] == "i,j,k,w"
        assert sorted(lines[1:]) == ["0,0,0,0.5", "1,2,3,0.25", "1,2,3,0.75"]
        assert web.download(0, "tsv").data.decode().splitlines()[0] == (
            "i\tj\tk\tw"
        )

    def test_text_has_every_row_though_the_table_keeps_fewer(self, tiny):
        web = WebRepl(tiny._nl, engine_name="tiny", max_rows=1)
        [table] = web.submit("ans(i, j, k, w) :- weight(i, j, k, w)").tables
        assert table["truncated"] and len(table["rows"]) == 1
        assert len(web.download(0, "csv").data.decode().splitlines()) == 4

    def test_a_block_of_a_program_with_several_is_named_by_its_title(self, web):
        web.submit(".mode squall")
        reply = web.submit(
            "obtain every voxel as First. obtain every voxel as Second."
        )
        assert [t["block"] for t in reply.tables] == [0, 1]
        assert web.download(1, "csv").name == "second.csv"

    def test_what_cannot_be_downloaded_is_told(self, web):
        web.app._nl.add_tuple_set([(10, 1, 2, 3)], name="seen")
        web.submit("ans(s, a, b, c) :- seen(s, a, b, c)")
        with pytest.raises(nifti.SaveError, match="no voxel columns"):
            web.download(0, "nifti")
        with pytest.raises(ValueError, match="Unknown kind"):
            web.download(0, "pdf")
        with pytest.raises(ValueError, match="no such result"):
            web.download(3, "csv")


def _written(data: bytes, path):
    """Write the *data* of a file the page saved, and return its path."""
    path.write_bytes(data)
    return path


# The LLM assistant, with the key of the page's visitor.
SECRET = "sk-test-0123456789"
ASK = "```datalog\nans(x) :- person(x, n)\n```\nEveryone."
# The reply to the second request of .ask, for the comment of each rule.
COMMENTS = '["Everyone who has a name"]'


def test_the_providers_of_the_page_are_listed_for_its_form():
    from neurolang_apps.utils.llm.browser import (
        providers_for_form as llm_providers,
    )

    providers = {p["id"]: p for p in llm_providers()}
    assert {"anthropic", "openai", "openrouter", "custom"} <= set(providers)
    assert providers["anthropic"]["model"]
    assert providers["custom"]["base_url"] == ""
    assert not any("key" in p for p in providers.values())


def test_ask_without_a_key_tells_to_add_it_in_the_page(repl):
    reply = repl.submit(".ask Who is there?")
    assert reply.failed
    assert "API key" in reply.html and "AI assistant" in reply.html
    assert "environment variable" not in reply.html


def test_the_predicates_of_the_session_come_with_each_reply(repl):
    reply = repl.submit("adult(x) :- person(x, n), (n > 4)")
    named = {p["name"]: p for p in reply.predicates}
    assert named["person"]["kind"] == "stored"
    assert named["person"]["signature"] == "person/2"
    assert named["adult"]["kind"] == "session"
    assert reply.predicates == repl.predicates()


def test_the_examples_of_the_engine_are_by_language(repl):
    examples = repl.examples()
    assert examples["datalog"] and examples["squall"]
    assert all(isinstance(query, str) for query in examples["datalog"])


def test_the_reply_tells_the_mode_of_the_repl(repl):
    assert repl.submit("1").mode == "datalog"
    assert repl.submit(".mode squall").mode == "squall"
    assert repl.submit(".mode datalog").mode == "datalog"


def test_ask_with_a_key_puts_the_query_in_the_input(repl):
    from neurolang_apps.utils.llm.tests.fakes import ScriptedClient

    repl.set_llm("anthropic", SECRET)
    repl.app.assistant.client = ScriptedClient([ASK, COMMENTS])
    reply = repl.submit(".ask --icl Who is there?")
    assert not reply.failed
    assert reply.prompt.splitlines()[-1] == "ans(x) :- person(x, n)"
    assert "Asking anthropic" in reply.html
    assert repl.submit("1").prompt == ""


def test_the_steps_of_the_assistant_are_sent_while_it_works(repl):
    import json

    from neurolang_apps.utils.llm.tests.fakes import ScriptedClient

    repl.set_llm("anthropic", SECRET)
    repl.app.assistant.client = ScriptedClient([
        [("check_query", {"query": "ans(x) :- person(x, n)"})],
        [("final_answer", {
            "query": "ans(x) :- person(x, n)", "explanation": "Everyone.",
        })],
        "Everyone with a name.", "VERDICT: yes", COMMENTS,
    ])
    sent = []
    repl.set_event_sink(lambda text: sent.append(json.loads(text)))
    reply = repl.submit(".ask --react Who is there?")
    assert not reply.failed
    assert {event["type"] for event in sent} == {"agent"}
    kinds = [event["event"] for event in sent]
    assert kinds[0] == "ask" and kinds[-1] == "done"
    assert sent[0]["strategy"] == "react"
    assert sent[0]["language"] == "datalog"
    # The reply of the model comes before the step of its tool call.
    first = kinds.index("reply")
    assert kinds[first - 1] == "request"
    assert kinds[first + 1] == "step"
    assert sent[first]["tool_calls"][0]["name"] == "check_query"
    step = sent[first + 1]
    assert step["action"] == "check_query" and step["observation"]
    assert sent[-1]["valid"] and sent[-1]["query"].startswith("ans(x)")
    # What the page shows above the query: what it does, and how it reads back.
    assert sent[-1]["explanation"] == "Everyone."
    assert sent[-1]["back_translation"] == "Everyone with a name."
    assert sent[-1]["verified"] is True
    assert all("messages" not in event for event in sent)
    repl.set_event_sink(None)
    sent.clear()
    repl.app.assistant.client = ScriptedClient([ASK, COMMENTS])
    repl.submit(".ask --icl Who is there?")
    assert sent == []


def test_react_gives_the_result_in_the_reply_and_enter_does_not_run_it_again(
    repl, monkeypatch
):
    from neurolang_apps.utils.llm.tests.fakes import ScriptedClient

    repl.set_llm("anthropic", SECRET)
    query = "ans(x) :- person(x, n)"
    repl.app.assistant.client = ScriptedClient([
        [("final_answer", {"query": query, "explanation": "Everyone."})],
        "Everyone with a name.", "VERDICT: yes", COMMENTS,
    ])
    reply = repl.submit(".ask --react Who is there?")
    [table] = reply.tables
    assert sorted(row[0] for row in table["rows"]) == ["alice", "bob"]
    assert reply.prompt.splitlines()[-1] == query
    runs = []
    execute = repl.app._nl.execute_datalog_program
    monkeypatch.setattr(
        repl.app._nl, "execute_datalog_program",
        lambda *args, **kwargs: runs.append(args) or execute(*args, **kwargs),
    )
    again = repl.submit(reply.prompt)
    assert runs == [] and "not run again" in again.html
    assert again.tables[0]["rows"] == table["rows"]
    assert again.tables[0]["block"] == 0


def test_the_key_goes_to_the_provider_and_nowhere_else(repl, monkeypatch):
    import httpx

    seen = []

    def handler(request):
        seen.append(request)
        text = ASK if len(seen) == 1 else COMMENTS
        return httpx.Response(200, json={
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda **kw: real_client(
            **{**kw, "transport": httpx.MockTransport(handler)}
        ),
    )
    repl.set_llm("anthropic", SECRET, model="claude-test")
    reply = repl.submit(".ask --icl Who is there?")
    # The query, then the comments of its rules: both to the provider.
    assert len(seen) == 2
    for request in seen:
        assert request.url.host == "api.anthropic.com"
        assert request.headers["x-api-key"] == SECRET
        assert (
            request.headers["anthropic-dangerous-direct-browser-access"]
            == "true"
        )
        assert b"claude-test" in request.content
    assert SECRET not in reply.html
    assert SECRET not in repl.submit(".llm").html
    assert SECRET not in repr(repl.app.assistant.config().provider())


def test_a_custom_provider_takes_the_address_of_its_api(repl):
    repl.set_llm("custom", "k", model="qwen3", base_url="http://localhost:11434/v1")
    html = repl.submit(".llm").html
    assert "http://localhost:11434/v1" in html and "qwen3" in html


def test_clearing_the_key_forgets_it(repl):
    repl.set_llm("openai", SECRET)
    repl.clear_llm()
    reply = repl.submit(".ask Who is there?")
    assert reply.failed and "API key" in reply.html


def test_an_unknown_provider_is_refused(repl):
    with pytest.raises(ValueError, match="nonesuch"):
        repl.set_llm("nonesuch", SECRET)
    with pytest.raises(ValueError, match="address"):
        repl.set_llm("custom", SECRET, model="m")


# The engine of a browser's tab: the analyses that fill gigabytes left out.
def test_the_lightweight_engines_leave_out_the_analyses_that_need_gigabytes(
    tmp_path,
):
    from neurolang.utils import engine_registry
    from neurolang.utils.tui.web import lightweight_engines_yaml

    path = lightweight_engines_yaml(tmp_path)
    full = engine_registry.get_engine_config("neurosynth")
    light = engine_registry.get_engine_config("neurosynth", path)
    for heavy in ("activated", "lift", "ratios", "lift_llm", "p_a_given_t"):
        assert f"\n{heavy}(" not in "\n" + light["datalog_init"]
        assert f"\n{heavy}(" in "\n" + full["datalog_init"]
    for light_rule in ("mentions", "reports", "p_term", "bayes"):
        if f"\n{light_rule}(" in "\n" + full["datalog_init"]:
            assert f"\n{light_rule}(" in "\n" + light["datalog_init"]
    assert not {"activated", "lift", "ratios"} & set(light["predicates"])
    assert "mentions" in light["predicates"]
    assert engine_registry.list_engine_names(
        path
    ) == engine_registry.list_engine_names()
