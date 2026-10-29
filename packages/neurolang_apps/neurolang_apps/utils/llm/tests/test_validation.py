import warnings

import pytest

from ..context import EngineContext
from ..validation import (
    _undefined_predicates,
    arity_errors,
    check_query,
    run_preview,
)


def symbols(nl):
    """Return the names of the symbols of the engine."""
    return {getattr(s, "name", s) for s in nl.symbol_table}


@pytest.mark.parametrize("query, language", [
    ("ans(s) :- mentions(s, 'pain')", "datalog"),
    ("obtain every Study that mentions 'pain'.", "squall"),
])
def test_valid_queries(nl, query, language):
    result = check_query(nl, query, language)
    assert result.ok, result.message
    if language == "squall":
        assert "mentions(s₀, 'pain')" in result.datalog


@pytest.mark.parametrize("query, language, message", [
    ("", "datalog", "empty"),
    ("ans(s) :- mentions(s, 'pain'", "datalog", "parse error"),
    ("obtain every Study that mentions 'pain'", "squall", "parse error"),
    ("q(s) :- study(s)", "datalog", "add a rule whose head is ans"),
    ("define as Q every Study.", "squall", "starting with 'obtain'"),
    ("ans(s) :- mention(s, 'pain')", "datalog",
     "predicate mention has no facts or rules — did you mean mentions?"),
    ("obtain every Study that mention 'pain'.", "squall",
     "did you mean mentions?"),
    ("ans(s) :- mentions(s)", "datalog",
     "mentions has 2 columns, not 1"),
    ("obtain every Mentions (?s).", "squall", "mentions has 2 columns"),
])
def test_invalid_queries(nl, query, language, message):
    result = check_query(nl, query, language)
    assert not result.ok
    assert message in result.message


def test_predicates_generated_by_the_rewriting_are_not_undefined():
    """
    The warnings of a dry run name the predicates the rewriting of the
    query generates (magic sets, numbered SQUALL rules): only those the
    query writes are undefined.
    """
    def warning(name):
        return warnings.WarningMessage(
            f"Symbol {name} used in rule body has no entry in the symbol "
            "table. Types cannot be inferred for this predicate.",
            UserWarning, "", 0,
        )

    caught = [warning(name) for name in (
        "magic_cooccurrence^fb_0", "roi_study_0", "mention",
    )]
    query = "define as Roi_study every Study that mention 'pain'."
    assert _undefined_predicates(caught, ["mentions"], query) == [
        "predicate mention has no facts or rules — did you mean mentions?"
    ]


def test_checks_leave_the_engine_unchanged(nl):
    before = symbols(nl)
    check_query(nl, "q(s) :- study(s)\nans(s) :- q(s)", "datalog")
    run_preview(nl, "define as Q every Study.\nobtain every Q.", "squall")
    assert symbols(nl) == before


def test_arity_errors_ignore_strings_and_functions(nl):
    assert arity_errors(nl, "ans(s) :- mentions(s, 'a, b'), exp(s)") == []


def test_run_preview(nl):
    preview = run_preview(nl, "ans(s, t) :- mentions(s, t)", "datalog", 2)
    assert preview.ok and preview.rows == 4
    assert preview.text.startswith("4 rows; columns s, t")
    assert len(preview.text.splitlines()) == 4
    squall = run_preview(
        nl, "obtain every Study that mentions 'pain'.", "squall"
    )
    assert squall.rows == 1 and "columns study" in squall.text


def test_run_preview_of_several_results(nl):
    preview = run_preview(
        nl, "obtain every Study as A. obtain every Study that mentions "
        "'fear' as B.", "squall",
    )
    assert preview.rows == 4
    assert "a: 3 rows" in preview.text and "b: 1 rows" in preview.text


def test_run_preview_error(nl):
    preview = run_preview(nl, "ans(s) :- mentions(s, 'pain'", "datalog")
    assert not preview.ok and "parse error" in preview.text


def test_schema_summary(nl):
    nl.execute_datalog_program("pain_study(s) :- mentions(s, 'pain')")
    summary = EngineContext(nl).schema_summary()
    lines = summary.splitlines()
    assert "- mentions(c0, c1) [stored, 4 rows]" in lines
    assert "- study(c0) [stored, 3 rows]" in lines
    assert "- pain_study(s) [rule]" in lines
    assert lines[-1].startswith("Functions: ") and "exp" in lines[-1]


def test_schema_summary_uses_the_engine_configuration(nl, monkeypatch):
    declared = {"mentions": {
        "columns": ["study_id", "term"], "description": "Studies and terms",
    }}
    monkeypatch.setattr(
        "neurolang_apps.utils.llm.context.declared_predicates",
        lambda name: declared,
    )
    context = EngineContext(nl, "neurosynth")
    assert (
        "- mentions(study_id, term) — Studies and terms "
        "[stored, 4 rows]" in context.schema_summary()
    )
    assert "mentions(1, 'pain')" in context.search_values(
        "pain", "mentions", "term"
    )


def test_describe_and_search(nl):
    context = EngineContext(nl)
    description = context.describe("mentions", rows=2)
    assert description.startswith("mentions(c0, c1) [stored, 4 rows]")
    assert "4 rows, e.g.:" in description
    assert "Did you mean mentions" in context.describe("mention")
    assert context.search_values("PAIN", "mentions", "c1") == (
        "Facts with values containing 'PAIN':\n"
        "mentions(1, 'pain')\nmentions(2, 'chronic pain')"
    )
    assert context.search_values("pain", "mentions", 1).splitlines()[1:] \
        == ["mentions(1, 'pain')", "mentions(2, 'chronic pain')"]
    assert context.search_values("memroy", "mentions", "c1") == (
        "No value of mentions contains 'memroy'; its values are: "
        "'chronic pain', 'fear', 'memory', 'pain'"
    )
    assert "no column 'term'" in context.search_values(
        "x", "mentions", "term"
    )
    assert "Unknown predicate 'nothing'" in context.search_values(
        "x", "nothing"
    )


def test_search_shows_the_closest_of_many_values(nl, monkeypatch):
    monkeypatch.setattr(
        "neurolang_apps.utils.llm.context.MAX_VALUES_LISTED", 2
    )
    context = EngineContext(nl)
    assert "closest are: 'memory'" in context.search_values(
        "memroy", "mentions", "c1"
    )
    assert "matches" in context.search_values("zzzzzz", "mentions", "c1")


def test_schema_summary_counts_the_classes_named_by_iri(nl):
    for i in range(3):
        nl.add_tuple_set([(1,)], name=f"http://x.org/onto#C{i}")
    summary = EngineContext(nl).schema_summary()
    assert "http://x.org/onto#C0" not in summary
    assert (
        "- 3 classes named by IRI in <http://x.org/onto#>, not listed: "
        "search_values finds their IRIs by label"
    ) in summary.splitlines()


def test_describe_shows_the_columns_as_written_in_a_query(nl, monkeypatch):
    declared = {"region": {"columns": ["label", "region"]}}
    monkeypatch.setattr(
        "neurolang_apps.utils.llm.context.declared_predicates",
        lambda name: declared,
    )
    nl.add_tuple_set(
        [(f"r{i}", frozenset({i})) for i in range(5)], name="region"
    )
    context = EngineContext(nl, "neurosynth")
    description = context.describe("region")
    lines = description.splitlines()
    assert lines[0] == "region(label, region) [stored, 5 rows]"
    assert lines[2:] == [
        f"region('r{i}', frozenset(…))" for i in range(3)
    ]
    assert context.search_values("r1", "region") == (
        "Facts with values containing 'r1':\nregion('r1', frozenset(…))"
    )


def test_search_all_stored_relations(nl):
    nl.add_tuple_set(
        [("http://x.org/ann", "Ann Lee"), ("_:b0", "fear")], name="pv:name"
    )
    found = EngineContext(nl).search_values("fear")
    assert found.splitlines() == [
        "Facts with values containing 'fear':",
        "mentions(1, 'fear')",
        "pv:name(<_:b0>, 'fear')",
    ]
    by_word = EngineContext(nl).search_values("Professor Lee")
    assert by_word.splitlines() == [
        "Facts with values containing 'Lee':",
        "pv:name(<http://x.org/ann>, 'Ann Lee')",
    ]
    assert "No stored value contains 'zzz'" in (
        EngineContext(nl).search_values("zzz")
    )


def test_rows_of_a_rule(nl):
    nl.execute_datalog_program("pain_study(s) :- mentions(s, 'pain')")
    context = EngineContext(nl)
    frame = context.frame("pain_study")
    assert list(frame.columns) == ["s"] and list(frame["s"]) == [1]
    with pytest.raises(ValueError, match="Unknown predicate"):
        context.frame("nothing")


def test_arity_errors_skip_iris_and_read_prefixed_names(nl):
    nl.add_tuple_set([("a", "b")], name="pv:name")
    text = (
        "ans(n) :- pv:name(<http://x.org/Python_(language)>, n), "
        "mentions(n, 'a (b, c)')"
    )
    assert arity_errors(nl, text) == []
    assert arity_errors(nl, "ans(n) :- pv:name(n)") == [
        "pv:name has 2 columns, not 1: use it as pv:name/2"
    ]


@pytest.mark.parametrize("query, message", [
    ("n(c) :- AGGREGATE[](mentions(s, t), study(s) @ count(s)) = c\n"
     "ans(c) :- n(c)", "takes a single atom"),
    ("n(s, c) :- AGGREGATE[s](mentions(s, t) @ count(s)) = c\n"
     "ans(s, c) :- n(s, c)", "groups by s, the variable it aggregates"),
])
def test_aggregation_misuses(nl, query, message):
    result = check_query(nl, query, "datalog")
    assert not result.ok and message in result.message


def test_aggregation_well_used(nl):
    query = (
        "n(t, c) :- AGGREGATE[t](mentions(s, t) @ count(s)) = c\n"
        "ans(t, c) :- n(t, c)"
    )
    assert check_query(nl, query, "datalog").ok


@pytest.fixture
def forking():
    """Skip the test where the system refuses to fork (no memory left)."""
    import os

    try:
        pid = os.fork()
    except (AttributeError, OSError) as exc:
        pytest.skip(f"cannot fork: {exc}")
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)


def test_previews_stop_slow_queries(forking, nl, monkeypatch):
    import time

    from .. import validation

    def slow(*args, **kwargs):
        while True:
            time.sleep(0.01)

    monkeypatch.setattr(validation, "_execute", slow)
    preview = run_preview(nl, "ans(s) :- study(s)", "datalog", timeout=0.2)
    assert not preview.ok and "more than 0 s and was stopped" in preview.text


def test_search_matches_iris_and_shows_varied_predicates(nl):
    blake = "http://dbpedia.org/resource/Quentin_Blake"
    nl.add_tuple_set(
        [(f"http://x.org/book{i}", blake) for i in range(6)],
        name="dbo:illustrator",
    )
    nl.add_tuple_set([(blake, "Quentin Blake")], name="rdfs:label")
    found = EngineContext(nl).search_values("Quentin Blake").splitlines()
    assert found[1] == f"rdfs:label(<{blake}>, 'Quentin Blake')"
    illustrated = [f for f in found if f.startswith("dbo:illustrator(")]
    assert len(illustrated) == 3
    assert found[-1] == "… and 3 more"


def test_previews_in_a_child_process(forking, nl):
    preview = run_preview(
        nl, "ans(s, t) :- mentions(s, t)", "datalog", 2, timeout=30,
        memory_mb=4096,
    )
    assert preview.ok and preview.rows == 4
    assert list(preview.frame.columns) == ["s", "t"]
    failed = run_preview(nl, "ans(s) :- mentions(s", "datalog", timeout=30)
    assert not failed.ok and "parse error" in failed.text


def test_previews_stop_queries_using_too_much_memory(forking, nl, monkeypatch):
    import time

    from .. import validation

    def greedy(*args, **kwargs):
        hoard = bytearray(400 * 2 ** 20)
        hoard[::4096] = b"x" * len(hoard[::4096])
        time.sleep(30)

    monkeypatch.setattr(validation, "_execute", greedy)
    preview = run_preview(
        nl, "ans(s) :- study(s)", "datalog", timeout=20, memory_mb=200
    )
    assert not preview.ok and "too much memory" in preview.text


def test_previews_report_a_crashed_child(forking, nl, monkeypatch):
    """A child killed by a signal (e.g. a library that cannot run after
    fork) is reported as a crash, not as a query using too much memory.

    The child is killed by SIGTERM: one that dumps core (SIGABRT, SIGSEGV)
    stays alive while the system writes the core of the large process of a
    whole test run, longer than the timeout, whatever RLIMIT_CORE says
    when the dump is piped to a program (systemd-coredump).
    """
    import os
    import signal

    from .. import validation

    monkeypatch.setattr(
        validation, "_execute",
        lambda *a, **k: os.kill(os.getpid(), signal.SIGTERM),
    )
    preview = run_preview(
        nl, "ans(s) :- study(s)", "datalog", timeout=20, memory_mb=4096
    )
    assert not preview.ok
    assert "crashed (SIGTERM)" in preview.text
    assert "too much memory" not in preview.text


def test_previews_without_fork_run_in_a_thread(nl, monkeypatch):
    """A system refusing to fork (ENOMEM) must not fail the preview."""
    import errno
    import os

    def no_memory():
        raise OSError(errno.ENOMEM, "Cannot allocate memory")

    monkeypatch.setattr(os, "fork", no_memory)
    preview = run_preview(
        nl, "ans(s, t) :- mentions(s, t)", "datalog", 2, timeout=30,
        memory_mb=4096,
    )
    assert preview.ok and preview.rows == 4
