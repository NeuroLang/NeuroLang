"""Tests for the REPL driven from a web page."""

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

import numpy as np  # noqa: E402

from neurolang.frontend import NeurolangPDL  # noqa: E402
from neurolang_apps.utils.tui import nifti  # noqa: E402
from neurolang_apps.utils.tui.web import (  # noqa: E402
    QUESTION_TOPICS, WebRepl, question_topic,
)
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


def test_the_banner_gives_the_kernel_radius_and_the_voxel_size():
    """The banner tells how far from a peak a study activates, in
    millimetres and in voxels of the engine's grid."""
    import nibabel as nib

    nl = NeurolangPDL()
    nl.template = nib.Nifti1Image(
        np.ones((2, 2, 2), dtype=np.uint8), np.diag([2.0, 2.0, 2.0, 1.0])
    )
    banner = WebRepl(nl, engine_name="neurosynth", resolution=2).banner()
    assert "Grid: 2 mm voxels" in banner
    assert "within 6 mm (3 voxels) of its peaks" in banner


def test_quit_does_not_end_the_page(repl):
    reply = repl.submit(".quit")
    assert "reload" in reply.html.lower()


def test_view_pages_are_returned_not_opened(repl, monkeypatch, tmp_path):
    monkeypatch.setattr(nifti.tempfile, "tempdir", str(tmp_path))
    repl.submit("ans(x) :- person(x, n)")
    monkeypatch.setattr(
        nifti, "last_result_images", lambda app, block, save: [("layer", Path("a"))]
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


def test_the_query_the_assistant_wrote_comes_with_its_graph(repl):
    from neurolang_apps.utils.llm.tests.fakes import ScriptedClient

    repl.set_llm("anthropic", SECRET)
    repl.app.assistant.client = ScriptedClient([
        [("check_query", {"query": "ans(x) :- person(x, n)"})],
        [("final_answer", {
            "query": "ans(x) :- person(x, n)", "explanation": "Everyone.",
        })],
        "Everyone with a name.", "VERDICT: yes", COMMENTS,
    ])
    reply = repl.submit(".ask --react Who is there?")
    assert ">person<" in reply.graph and ">ans<" in reply.graph


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
    # The tokens of the answer: of its two requests.
    assert {k: reply.tokens[k] for k in ("input", "output", "calls",
                                         "model")} == {
        "input": 2, "output": 2, "calls": 2, "model": "claude-test"}
    assert repl.submit("ans(x) :- person(x, 5)").tokens == {}
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
    # the files of its relations are those of the package, or addresses
    for relation in light["relations"].values():
        assert relation["file"].startswith("https://") or \
            Path(relation["file"]).is_file(), relation



def _text(html):
    """Return the text of the HTML *html*, without its tags."""
    import re

    return re.sub("<[^>]+>", "", html)


def test_the_rules_defined_from_a_heavy_predicate_are_left_out():
    from neurolang_apps.utils.tui.web import left_out

    heavy = left_out(
        "near(s) :- activated(i, j, k, s)\n"
        "nearer(s) :- near(s), study(s)\n"
        "mentioned(s) :- mentions(s, t)"
    )
    assert {"activated", "near", "nearer"} <= heavy
    assert not {"mentioned", "mentions", "study"} & heavy


def test_the_predicates_the_engine_leaves_out_are_named(repl):
    """The browser engine (here: one defining none of them) lacks the
    analyses that need gigabytes and what is defined from them."""
    left_out = repl.app.left_out_predicates
    assert repl.app._nl.left_out_predicates == left_out
    assert {"activated", "uniformity", "uniformity_llm", "lift"} <= left_out
    assert not {"mentions", "destrieux_activated", "person"} & left_out
    full = NeurolangPDL()
    for name in ("activated", "uniformity"):
        full.add_tuple_set([(1,)], name=name)
    defined = WebRepl(full, engine_name="neurosynth").app.left_out_predicates
    assert not {"activated", "uniformity"} & defined


def test_the_full_engine_leaves_out_nothing():
    """The rules of the analyses define them in the engine of the
    terminal, or of ``?engine=full``."""
    from unittest.mock import patch

    from neurolang_apps.utils.tests import test_engines

    with patch(
        f"{test_engines.NS_INIT}.fetch_study_metadata",
        return_value=test_engines.FAKE_METADATA,
    ):
        nl = test_engines._build_small_neurosynth()
    assert WebRepl(nl, "neurosynth").app.left_out_predicates == frozenset()


def test_the_examples_leave_out_the_missing_predicates(repl):
    examples = repl.examples()
    assert examples["datalog"] and examples["squall"]
    for query in examples["datalog"] + examples["squall"]:
        assert "uniformity" not in query and "activated(" not in query
    assert (
        "ans(study, year, title) :- mentions(study, 'pain'), "
        "study_year(study, year), study_title(study, title)"
    ) in examples["datalog"]


def test_the_banner_says_what_the_engine_leaves_out(repl):
    banner = _text(repl.banner())
    missing = next(
        line for line in banner.splitlines() if "not in the browser" in line
    )
    assert "activated" in missing and "uniformity_llm" in missing
    assert "  uniformity(" not in banner and "  activated(" not in banner
    assert "  mentions(" in banner


def test_a_missing_predicate_is_not_available_rather_than_unknown(repl):
    reply = repl.submit("ans(i, j, k, u) :- uniformity('pain', i, j, k, u)")
    text = _text(reply.html)
    assert reply.failed
    assert "uniformity is not available in the browser" in text
    assert "Undefined symbol" not in text
    reply = repl.submit("ans(x) :- nobody(x)")
    assert "predicate nobody has no facts or rules" in _text(reply.html)
    assert "not available" not in _text(reply.html)


STUDIES = [
    {"id": 9065511, "doi": "10.1523/JNEUROSCI.17-07-02512.1997",
     "title": "Environmental knowledge.", "authors": "Aguirre GK",
     "year": 1997, "journal": "J Neurosci"},
    {"id": 9084599, "doi": None, "title": "Anatomy of motor learning.",
     "authors": "Jueptner M", "year": 1997, "journal": float("nan")},
    {"id": 1, "doi": None, "title": "Not in the result", "authors": "",
     "year": 2000, "journal": ""},
]


@pytest.fixture
def studies_repl():
    import pandas as pd

    nl = NeurolangPDL()
    nl.add_tuple_set([(9065511,), (9084599,)], name="study")
    nl.add_tuple_set([(9065511, "pain"), (9084599, "motor")], name="mentions")
    return WebRepl(
        nl, engine_name="neurosynth", study_metadata=pd.DataFrame(STUDIES)
    )


class TestStudyLinks:
    def test_a_column_of_studies_tells_its_sort(self, studies_repl):
        [table] = studies_repl.submit("ans(s, t) :- mentions(s, t)").tables
        assert table["sorts"] == ["study", "term"]
        assert table["positions"] == [0, 0]

    def test_the_columns_of_a_voxel_tell_their_position(self, studies_repl):
        studies_repl.submit(
            "peak_reported(i, j, k, s) :- (i == 1), (j == 2), (k == 3), "
            "study(s)"
        )
        [table] = studies_repl.submit(
            "ans(i, j, k, s) :- peak_reported(i, j, k, s)"
        ).tables
        assert table["sorts"][:3] == ["voxel"] * 3
        assert table["positions"] == [0, 1, 2, 0]

    def test_the_records_of_the_studies_shown_come_with_the_table(
        self, studies_repl
    ):
        [table] = studies_repl.submit("ans(s) :- study(s)").tables
        links = table["study_links"]
        assert links["columns"] == [0]
        assert set(links["records"]) == {"9065511", "9084599"}
        first = links["records"]["9065511"]
        assert first["doi"] == "10.1523/JNEUROSCI.17-07-02512.1997"
        assert first["title"] == "Environmental knowledge."
        assert first["year"] == 1997
        # What the metadata lacks is left out, not null or NaN.
        assert links["records"]["9084599"] == {
            "title": "Anatomy of motor learning.", "authors": "Jueptner M",
            "year": 1997,
        }

    def test_the_links_are_json(self, studies_repl):
        import json

        reply = studies_repl.submit("ans(s) :- study(s)")
        assert json.loads(json.dumps(reply.tables)) == reply.tables

    def test_a_result_without_studies_has_no_links(self, studies_repl):
        [table] = studies_repl.submit("ans(t) :- mentions(s, t)").tables
        assert "study_links" not in table

    def test_without_metadata_there_are_no_links(self, repl):
        nl = NeurolangPDL()
        nl.add_tuple_set([(1,)], name="study")
        [table] = WebRepl(nl, engine_name="neurosynth").submit(
            "ans(s) :- study(s)"
        ).tables
        assert table["sorts"] == ["study"]
        assert "study_links" not in table


def test_the_predicates_of_the_engine_say_what_their_columns_are(repl):
    named = {p["name"]: p for p in repl.predicates()}
    assert named["person"]["arguments"] == []  # the session's have no declaration
    nl = NeurolangPDL()
    nl.add_tuple_set([(1, 2, 3, 9065511)], name="peak_reported")
    peaks = {
        p["name"]: p
        for p in WebRepl(nl, engine_name="neurosynth").predicates()
    }["peak_reported"]
    assert peaks["description"]
    assert peaks["arguments"] == [
        {"columns": ["i", "j", "k"], "sort": "voxel", "type": "int"},
        {"columns": ["study_id"], "sort": "study", "type": "int"},
    ]


class TestRatingAnAnswer:
    """The page shares a rated answer in a Grist table, anonymously."""

    @staticmethod
    def answered(repl, transport):
        from collections import Counter

        from neurolang_apps.utils.llm import AgentResult
        from neurolang_apps.utils.llm.config import ProviderConfig

        repl.set_feedback("https://grist.test", "doc1", transport)
        provider = ProviderConfig("p", "openai", "m", "http://x/v1")
        result = AgentResult(
            query="ans(s) :- person(s, n)", language="datalog", valid=True,
            usage=Counter(input=1, output=2),
        )
        repl.app.assistant.last_run = ("Who?", result, provider, "react")
        repl._rated = "answer"

    def test_without_a_document_nothing_can_be_rated(self, repl):
        assert not repl.can_rate()
        assert not repl.rate_answer("good")["ok"]

    def test_the_rating_is_a_row_with_the_answer(self, repl):
        import httpx

        requests = []

        def handle(request):
            requests.append(request)
            return httpx.Response(200, json={"records": [{"id": 3}]})

        self.answered(repl, httpx.MockTransport(handle))
        assert repl.can_rate()
        assert repl.rate_answer("bad", "wrong table")["ok"]
        (request,) = requests
        assert "Authorization" not in request.headers
        fields = json.loads(request.content)["records"][0]["fields"]
        assert fields["verdict"] == "bad" and fields["comment"] == "wrong table"
        assert fields["question"] == "Who?"
        assert fields["query"] == "ans(s) :- person(s, n)"

    def test_a_failure_says_so(self, repl):
        import httpx

        self.answered(repl, httpx.MockTransport(
            lambda request: httpx.Response(403, json={"error": "no"})
        ))
        reply = repl.rate_answer("good")
        assert not reply["ok"] and "failed" in reply["message"]

    def test_there_is_nothing_to_rate_before_an_answer(self, repl):
        import httpx

        repl.set_feedback(
            "https://grist.test", "doc1",
            httpx.MockTransport(lambda request: httpx.Response(200)),
        )
        assert not repl.rate_answer("good")["ok"]
        assert not repl.rate_answer("meh")["ok"]


class TestRatingAQuery:
    """A query typed in the page is rated as the assistant's answers are."""

    @staticmethod
    def recording(repl):
        import httpx

        requests = []

        def handle(request):
            requests.append(request)
            return httpx.Response(200, json={"records": [{"id": 4}]})

        repl.set_feedback("https://grist.test", "doc1", httpx.MockTransport(handle))
        return requests

    @staticmethod
    def fields(request):
        return json.loads(request.content)["records"][0]["fields"]

    def test_a_query_is_sent_with_its_rows(self, repl):
        requests = self.recording(repl)
        assert repl.submit("ans(x) :- person(x, n)").ratable
        assert repl.rate_answer("good")["ok"]
        fields = self.fields(requests[0])
        assert fields["question"] == "" and fields["verdict"] == "good"
        assert fields["query"] == "ans(x) :- person(x, n)"
        assert fields["valid"] is True and fields["rows"].startswith("2 rows")

    def test_a_failed_query_is_sent_with_its_error(self, repl):
        requests = self.recording(repl)
        reply = repl.submit("ans(x) <- person(x, n)")
        assert reply.failed and reply.ratable
        assert repl.rate_answer("bad", "it fails")["ok"]
        fields = self.fields(requests[0])
        assert fields["valid"] is False and "parse error" in fields["error"]
        assert fields["comment"] == "it fails"

    def test_a_dot_command_is_not_rated(self, repl):
        self.recording(repl)
        assert not repl.submit(".help").ratable
        assert not repl.rate_answer("good")["ok"]

    def test_a_question_without_an_answer_is_not_rated(self, repl):
        self.recording(repl)
        reply = repl.submit(".ask Who is there?")  # no key: no answer
        assert reply.failed and not reply.ratable


def test_the_iris_of_an_ontology_are_not_listed(repl):
    nl = repl.app._nl
    nl.add_tuple_set([("a", "b")], name="rdfs:label")
    nl.add_tuple_set(
        [("a",)], name="http://www.cognitiveatlas.org/ontology/cogat.owl#CAO_00137"
    )
    names = {p["name"] for p in repl.predicates()}
    assert {"person", "rdfs:label"} <= names
    assert not any("://" in name for name in names)


def test_the_commands_come_with_what_they_do(repl):
    commands = {c["name"]: c for c in repl.commands()}
    assert {".view", ".mode", ".help", ".ask"} <= set(commands)
    view = commands[".view"]
    assert view["usage"].startswith(".view")
    assert view["description"]
    assert view["topic"] == "view"  # .help view tells more
    assert commands[".mode"]["topic"] == ""
    assert "Switch the query language" in commands[".mode"]["description"]
    assert commands[".mode"]["details"].startswith("Switch between")


def test_the_commands_that_end_a_terminal_session_are_not_listed(repl):
    names = {c["name"] for c in repl.commands()}
    assert ".quit" not in names and ".exit" not in names


def test_the_commands_are_json(repl):
    import json

    assert json.loads(json.dumps(repl.commands())) == repl.commands()


def test_the_assistant_is_ready_with_a_key(repl):
    pytest.importorskip("httpx")
    assert not repl.assistant_ready()
    repl.set_llm("anthropic", "a-key")
    assert repl.assistant_ready()
    repl.clear_llm()
    assert not repl.assistant_ready()


def test_a_confined_session_writes_and_reads_no_file_of_the_server(
    repl, tmp_path, monkeypatch
):
    """On a server (neurolang.confine): .save, .transcript, .load and
    .view of a path reach its data only, and write nothing."""
    from neurolang.confine import CONFINE_ENV

    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv(CONFINE_ENV, str(data))
    (tmp_path / "secret.csv").write_text("a\n1\n")
    repl.submit("ans(n, a) :- person(n, a)")
    for statement, says in (
        (f".save {tmp_path / 'out.csv'}", "writes no result"),
        (f".transcript {tmp_path / 'session.md'}", "writes no transcript"),
        (f".load {tmp_path / 'secret.csv'} stolen", "No file"),
        (f".view {tmp_path / 'secret.nii.gz'}", "reads no image"),
    ):
        reply = repl.submit(statement)
        assert says in reply.html, statement
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "data", "secret.csv"]


def test_the_page_uploads_a_file_that_load_loads(repl, monkeypatch, tmp_path):
    """A file the page uploads is kept in the session's memory, and loaded
    by its name, also on a server which reads none of its files for a
    statement (neurolang.confine)."""
    import gzip
    import re

    from neurolang.confine import CONFINE_ENV
    from neurolang_apps.utils.tui import session

    monkeypatch.setenv(CONFINE_ENV, str(tmp_path))
    answer = repl.upload("my scores.csv", b"study,score\ns1,0.5\ns2,0.7\n")
    assert answer["statement"] == ".load 'my scores.csv'"
    def text(reply):
        return re.sub(r"<[^>]+>", "", reply.html)

    reply = repl.submit(answer["statement"] + " scores")
    assert "loaded scores(study, score): 2 rows" in text(reply)
    repl.upload("t.tsv.gz", gzip.compress(b"a\tb\n1\t2\n"))
    assert "loaded t(a, b): 1 row" in text(repl.submit(".load t.tsv.gz"))
    reply = repl.submit(".load other.csv")
    assert reply.failed and "No file other.csv was uploaded" in text(reply)
    assert list(tmp_path.iterdir()) == []  # nothing written on a disk
    for name, data in (("../x.csv", b"a\n1\n"), ("/etc/x.csv", b"a\n"),
                       ("notes.txt", b"a\n"), (".csv", b"a\n"),
                       ("big.csv", b"a" * (session.MOST_UPLOAD_BYTES + 1))):
        with pytest.raises(ValueError):
            repl.upload(name, data)


def test_the_page_uploads_an_image_that_image_loads(repl):
    """A NIfTI image the page uploads is loaded by .image, replacing the
    image of the session; an engine that cannot decode it gives no query."""
    answer = repl.upload("my map.nii.gz", b"\x1f\x8b")
    assert answer == {
        "name": "my map.nii.gz", "relation": "my_map",
        "statement": ".image 'my map.nii.gz' --name my_map",
        "query": "", "kind": "", "cortex_query": "",
        "threshold": nifti.IMAGE_DEFAULTS["threshold"],
        "radius": nifti.IMAGE_DEFAULTS["endpoint_radius"], "index": "",
    }


def test_the_page_uploads_a_tractogram_that_image_loads(repl, monkeypatch):
    """A tractogram the page uploads is loaded by .image, named after its
    file, and comes with the queries decoding it: of kind tractogram, the
    voxels within radius mm of the ends of its streamlines."""
    monkeypatch.setattr(nifti, "decoding_query",
                        lambda app, name, voxels="brain": (f"{voxels}", []))
    answer = repl.upload("my uf.tck", b"mrtrix tracks")
    assert answer["statement"] == ".image 'my uf.tck' --name my_uf"
    assert answer["kind"] == "tractogram"
    assert (answer["query"], answer["cortex_query"]) == ("brain", "cortex")
    assert answer["radius"] == nifti.IMAGE_DEFAULTS["endpoint_radius"]
    assert answer["index"] == ""  # an engine without tractography
    (listed,) = repl.uploads()["files"]
    assert (listed["index"], listed["indexed"]) == ("", "")


def test_the_engines_tractogram_is_listed_first_with_the_uploads(
    repl, monkeypatch
):
    """The tractogram of an engine of tractography is listed before the
    uploads, the engine's, with how it is indexed again."""
    monkeypatch.setattr(
        nifti, "tractogram_index",
        lambda app, source, ident="": {"index": f"index {ident}",
                                       "indexed": ""})
    repl.app.engine_tractograms = ["iit"]
    repl.upload("t.csv", b"a,b\n1,2\n")
    first, upload = repl.uploads()["files"]
    assert first == {"name": "iit", "size": 0, "age": 0, "loaded": ["iit"],
                     "engine": True, "index": "index iit", "indexed": ""}
    assert upload["name"] == "t.csv" and "engine" not in upload


def test_the_page_is_told_if_an_uploaded_image_is_a_mask(repl):
    """A mask is decoded as it is; a map of values once said of
    z-scores (.image --zscore): the page asks."""
    import gzip

    import nibabel as nib
    import numpy as np

    for values, kind in (([1, 1], "mask"), ([1.5, 4.0], "map")):
        data = np.zeros((4, 4, 4), dtype=np.float32)
        data[1, 1, 1], data[2, 2, 2] = values
        image = gzip.compress(nib.Nifti1Image(data, np.eye(4)).to_bytes())
        assert repl.upload("i.nii.gz", image)["kind"] == kind


def test_each_uploaded_image_is_a_relation_of_its_own(repl):
    """Two images uploaded are loaded as two relations, each of its file's
    name; the same file uploaded again replaces its own."""
    import gzip

    import nibabel as nib
    import numpy as np

    data = np.zeros((4, 4, 4), dtype=np.float32)
    data[1, 1, 1] = 1
    image = gzip.compress(nib.Nifti1Image(data, np.eye(4)).to_bytes())
    relations = []
    for name in ("first.nii.gz", "second.nii.gz", "first.nii.gz"):
        answer = repl.upload(name, image)
        relations.append(answer["relation"])
        assert answer["statement"].endswith(f"--name {answer['relation']}")
        repl.app.uploads.loaded(name, answer["relation"])
        repl.app._nl.add_tuple_set([(1, 1, 1, 1.0)], name=answer["relation"])
    assert relations == ["first", "second", "first"]


def test_the_page_gets_the_queries_decoding_the_brain_and_the_cortex(
    repl, monkeypatch
):
    """An uploaded image comes with the query decoding it, a voxel chosen
    in the brain, and the one choosing it in the cortex (Only the cortex,
    .image --cortex)."""
    import gzip

    import nibabel as nib
    import numpy as np

    monkeypatch.setattr(nifti, "decoding_query",
                        lambda app, name, voxels="brain": (f"{voxels}", []))
    data = np.zeros((4, 4, 4), dtype=np.float32)
    data[1, 1, 1] = 1
    image = gzip.compress(nib.Nifti1Image(data, np.eye(4)).to_bytes())
    answer = repl.upload("i.nii.gz", image)
    assert (answer["query"], answer["cortex_query"]) == ("brain", "cortex")


@pytest.mark.parametrize("name", ["my roi.nii.gz", "my roi.nii"])
def test_a_server_loads_shows_and_tells_an_uploaded_image(
    repl, monkeypatch, tmp_path, name
):
    """The whole path of an image on a server (neurolang.confine): the
    page uploads it, runs the statement the upload returns, and gets the
    viewer back; an engine without the peaks of studies says it does not
    decode it."""
    import gzip
    import re

    import nibabel as nib
    import numpy as np
    from neurolang.confine import CONFINE_ENV

    monkeypatch.setenv(CONFINE_ENV, str(tmp_path / "data"))
    monkeypatch.setattr(
        nifti, "_viewer_page", lambda *args, **kwargs: "<html>page</html>"
    )
    repl.app._nl.template = nib.Nifti1Image(
        np.ones((4, 4, 4), dtype=np.uint8), np.eye(4)
    )
    data = np.zeros((4, 4, 4), dtype=np.uint8)
    data[1, 1, 1] = data[2, 2, 2] = 1
    image = nib.Nifti1Image(data, np.eye(4)).to_bytes()  # .nii
    if name.endswith(".gz"):
        image = gzip.compress(image)
    answer = repl.upload(name, image)
    reply = repl.submit(answer["statement"])
    text = " ".join(re.sub(r"<[^>]+>", "", reply.html).split())
    assert not reply.failed, text
    assert f"loaded my_roi(i, j, k, value): 2 voxels from {name}, " \
        "a mask" in text
    assert "cannot decode the image: it has no activated" in text
    assert answer["query"] == answer["cortex_query"] == ""  # nor does the
    # page offer the query
    assert reply.viewers == ["<html>page</html>"]
    assert not (tmp_path / "data").exists()  # nothing written there


def test_a_reply_tells_the_uploads_of_the_session(repl):
    """Each reply carries the files the page uploaded, what they were
    loaded as and the room taken, for the page to list them."""
    from neurolang_apps.utils.tui.local_backend import reply_payload

    assert repl.submit("ans(x, n) :- person(x, n)").uploads["files"] == []
    repl.upload("my scores.csv", b"study,score\ns1,0.5\n")
    reply = repl.submit(".load 'my scores.csv' scores")
    (upload,) = reply.uploads["files"]
    assert (upload["name"], upload["size"], upload["loaded"]) == (
        "my scores.csv", 19, ["scores"])
    assert reply.uploads["used"] == 19
    assert reply.uploads["most"] == repl.app.uploads.most_all
    assert reply_payload(reply)["uploads"] == reply.uploads


def test_more_questions_are_those_of_the_bank_the_page_does_not_list(repl):
    """The REPL lists some questions first; the rest of the assistant's
    bank on the engine follows, each once."""
    first = repl.questions()
    more = repl.more_questions()
    assert first and more
    assert not set(first) & set(more)
    assert len(more) == len(set(more))


def test_question_queries_give_the_query_of_each_question(repl):
    """Picking a question fills the box of the query too: the page has
    the query of every question the REPL lists."""
    queries = repl.question_queries()
    for question in repl.questions() + repl.more_questions():
        assert queries[question]["datalog"]


@pytest.mark.parametrize("question, query, topic", [
    ("Which studies mention pain?", "ans(s) :- mentions(s, 'pain')", 0),
    ("Which gyri are activated by studies about language?",
     "ans(r) :- destrieux_activated(r, s)", 1),
    ("Which terms have a Bayes factor above 3 for the left putamen?",
     "bayes_factor(l, t, bf) :- p(l, t, bf)\n"
     "ans(term, bf) :- bayes_factor('pPUT lh', term, bf)", 2),
    ("Which regions are functionally connected at rest with the thalamus?",
     "ans(r) :- f(r)", 3),
    ("How many white-matter streamlines connect the hemispheres?",
     "ans(n) :- n = 1", 4),
    ("Which regions are hippocampal, with their full names?",
     "ans(r, n) :- tian_name(r, n)", 5),
])
def test_the_topic_of_a_question(question, query, topic):
    assert question_topic(question, query) == QUESTION_TOPICS[topic]


def test_the_topics_of_the_questions_are_known_and_grouped(repl):
    """Every question of the bank has a topic of the list, and the more
    questions come in the order of the topics."""
    topics = repl.question_topics()
    more = repl.more_questions()
    assert set(topics.values()) <= set(QUESTION_TOPICS)
    ranks = [QUESTION_TOPICS.index(topics[q]) for q in more]
    assert ranks == sorted(ranks)


def _luminance(hex_colour: str) -> float:
    """Return the relative luminance (WCAG) of a colour ``#rrggbb``."""
    def channel(value: int) -> float:
        value /= 255
        return value / 12.92 if value <= 0.03928 else (
            (value + 0.055) / 1.055
        ) ** 2.4
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


@pytest.mark.parametrize("panel", ["#070d19", "#0f1b33"])
def test_the_colours_of_the_output_are_legible_on_its_panel(repl, panel):
    # the dark panel of the output, --term in tokens.css, in either theme
    repl.app.console.print(
        "[red]failed[/red] [dim]a note[/dim] [green]saved[/green] "
        "[magenta]a path[/magenta] [cyan]1[/cyan] [yellow]warned[/yellow]"
    )
    colours = re.findall(r"color: (#[0-9a-f]{6})", repl._output())
    assert len(colours) >= 6
    for colour in colours:
        bright, dark = sorted(
            (_luminance(colour), _luminance(panel)), reverse=True
        )
        assert (bright + 0.05) / (dark + 0.05) >= 4.5, colour
