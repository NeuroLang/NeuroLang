"""The LLM is told the ``%%`` documentation of the predicates."""

from ..agent import Toolbox
from ..context import EngineContext

DOCUMENTED = """
%% pain_study(s: study)
%% The studies mentioning pain.
pain_study(s) :- mentions(s, 'pain')
%% The terms of the studies.
.document(mentions, "The terms each study mentions.")
"""


def test_schema_summary_shows_the_documentation(nl):
    nl.execute_datalog_program(DOCUMENTED)
    lines = EngineContext(nl).schema_summary().splitlines()
    assert (
        "- pain_study(s: study) — The studies mentioning pain. [rule]"
        in lines
    )
    assert (
        "- mentions(c0, c1) — The terms each study mentions. "
        "[stored, 4 rows]" in lines
    )


def test_documentation_wins_over_the_declaration(nl, monkeypatch):
    declared = {
        "pain_study": {"columns": ["study_id"], "description": "Declared"},
        "mentions": {"columns": ["id", "term"], "description": "Declared"},
    }
    monkeypatch.setattr(
        "neurolang_apps.utils.llm.context.declared_predicates",
        lambda name: declared,
    )
    nl.execute_datalog_program(DOCUMENTED)
    context = EngineContext(nl, "neurosynth")
    lines = context.schema_summary().splitlines()
    assert (
        "- pain_study(s: study) — The studies mentioning pain. [rule]"
        in lines
    )
    # documented without arguments: the declared columns stay
    assert (
        "- mentions(id, term) — The terms each study mentions. "
        "[stored, 4 rows]" in lines
    )
    assert context.columns("pain_study") == ["s"]


def test_tools_show_the_documentation(nl):
    nl.execute_datalog_program(DOCUMENTED)
    toolbox = Toolbox(EngineContext(nl), "datalog")
    listed, _ = toolbox.call("list_predicates", {})
    assert "pain_study(s: study) — The studies mentioning pain." in listed
    described, _ = toolbox.call("describe_predicate", {"name": "pain_study"})
    assert described.splitlines()[0] == (
        "pain_study(s: study) — The studies mentioning pain. [rule]"
    )
    assert "2 rows" not in described and "1 rows" in described


def test_documentation_of_a_scope_goes_with_it(nl):
    with nl.scope:
        nl.execute_datalog_program(DOCUMENTED)
        assert "pain" in EngineContext(nl).describe("pain_study")
    assert "Unknown predicate" in EngineContext(nl).describe("pain_study")


def test_schema_summary_describes_the_declared_sorts(nl, monkeypatch):
    from neurolang_apps.utils.engine_registry import Sort

    monkeypatch.setattr(
        "neurolang_apps.utils.engine_registry.get_sorts",
        lambda name, yaml_path=None: {
            "study": Sort("study", ("s",), "int", "a study of the  base"),
            "voxel": Sort("voxel", ("i", "j", "k"), "int", ""),
        },
    )
    nl.execute_datalog_program(DOCUMENTED)
    lines = EngineContext(nl, "neurosynth").schema_summary().splitlines()
    at = lines.index("Sorts of the arguments:")
    assert lines[at + 1] == "- study: a study of the base"
    assert lines[at + 2].startswith("Functions: ")


def test_the_summary_has_the_first_paragraph_describe_all(nl):
    nl.execute_datalog_program(
        "%% p(s: study)\n%% The studies.\n%%\n%% How they are found.\n"
        "p(s) :- study(s)"
    )
    context = EngineContext(nl)
    assert "- p(s: study) — The studies. [rule]" in (
        context.schema_summary().splitlines()
    )
    assert context.describe("p").splitlines()[0] == (
        "p(s: study) — The studies. How they are found. [rule]"
    )
