"""Tests for the session commands of the interactive REPL."""

import gzip
import io
import re

import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

from rich.console import Console  # noqa: E402

from neurolang.expressions import Symbol  # noqa: E402
from neurolang.frontend import NeurolangPDL  # noqa: E402
from neurolang.utils.tui import InteractiveTuiApp, TuiSettings  # noqa: E402


def _output(app):
    return app.console.file.getvalue()


@pytest.fixture
def app():
    """Return a REPL whose session defined ``active``."""
    nl = NeurolangPDL()
    nl.add_tuple_set([("alice",), ("bob",)], name="person")
    nl.add_tuple_set([("alice",)], name="plays")
    nl.add_tuple_set([("bob",)], name="sings")
    console = Console(file=io.StringIO(), width=100)
    app = InteractiveTuiApp(
        nl, TuiSettings(engine_name="neurosynth"), console=console,
        err_console=console,
    )
    app._execute_and_display("active(x) :- person(x), plays(x)")
    return app


def _idb_names(app):
    return [s.name for s in app._nl.program_ir.intensional_database()]


def test_idb_lists_session_predicate(app):
    app._handle_dot_command(".idb")
    assert "active" in _output(app)


def test_clear_all_removes_session_idb(app):
    app._handle_dot_command(".clear")
    assert "removed active (1 rule)" in _output(app)
    assert "active" not in _idb_names(app)
    app._execute_and_display("again(x) :- person(x)")
    assert "again" in _idb_names(app)


def test_clear_specific_removes_predicate(app):
    app._handle_dot_command(".clear active")
    assert "removed active (1 rule)" in _output(app)
    assert "active" not in _idb_names(app)


def test_clear_rejects_engine_predicate(app):
    app._handle_dot_command(".clear person")
    assert "is part of the engine" in _output(app)
    assert "person" in [s.name for s in app._nl.program_ir.symbol_table]


def test_sets_reuses_cli_listing(app, capsys):
    app._handle_dot_command(".sets")
    out = capsys.readouterr().out
    assert "person" in out and "Total RA sets" in out


def _plain_lines(out):
    """Return the lines of *out* with their spaces collapsed."""
    return [" ".join(line.split()) for line in out.splitlines()]


def test_predicates_lists_the_session_and_loaded_ones(app, tmp_path):
    path = _write(tmp_path / "scores.csv", "Name,Score\nalice,3\n")
    app._handle_dot_command(f".load {path}")
    app._execute_and_display("pair(x, y) :- person(x), scores(y, s)")
    lines = _plain_lines(_command(app, ".predicates"))
    assert lines[:4] == [
        "Predicate Kind Columns",
        "active session active(x)",
        "pair session pair(x, y)",
        "scores loaded scores(Name, Score)",
    ]
    assert "person stored person/1" in lines


def test_engines(app, capsys):
    app._handle_dot_command(".engines")
    out = capsys.readouterr().out
    assert "Available engines:" in out
    for name in ("destrieux", "neurosynth"):
        assert f"  {name}\n" in out
    assert "Destrieux 2009 cortical atlas" in out


def test_rewritten_shows_last_query(app, capsys):
    app._execute_and_display("ans(x) :- active(x)")
    capsys.readouterr()
    app._handle_dot_command(".rewritten")
    assert "── rewritten program ──" in capsys.readouterr().out
    assert _idb_names(app).count("ans") <= 1


def test_rewritten_without_query(capsys):
    console = Console(file=io.StringIO(), width=100)
    app = InteractiveTuiApp(NeurolangPDL(), console=console)
    app._handle_dot_command(".rewritten")
    assert "No query has been run yet." in _output(app)


def test_datalog_shows_squall_translation(app, capsys):
    app._handle_dot_command(".mode squall")
    app._execute_and_display("obtain every person that plays.")
    capsys.readouterr()
    app._handle_dot_command(".datalog")
    assert "── query (obtain_0) ──" in capsys.readouterr().out


def test_datalog_prints_the_last_datalog_query(app, capsys):
    app._execute_and_display("fan(x) :- active(x)\nans(x) :- fan(x)")
    capsys.readouterr()
    out = _command(app, ".datalog") + capsys.readouterr().out
    assert "use .rewritten" not in out
    assert "── rule ──\nfan(x) :-\n    active(x)" in out
    assert "── query ──\nans(x) :-\n    fan(x)" in out


def _command(app, line):
    """Run the dot-command *line* and return what it printed."""
    app.console.file.seek(0)
    app.console.file.truncate()
    app.console.width = 1000
    app._handle_dot_command(line)
    return _output(app)


def _query(app, query):
    """Run the Datalog *query* and return its rows as a set of tuples."""
    app._execute_and_display(query)
    return set(app.results[0].frame.itertuples(index=False, name=None))


@pytest.fixture
def two_rules(app):
    """Return the REPL *app* where ``both`` has two rules."""
    app._execute_and_display("both(x) :- person(x), plays(x)")
    app._execute_and_display("both(x) :- person(x), sings(x)")
    return app


def test_idb_numbers_the_rules(two_rules):
    out = _command(two_rules, ".idb both")
    assert out.splitlines() == [
        "1  both(x) :- person(x), plays(x)",
        "2  both(x) :- person(x), sings(x)",
    ]


def test_idb_of_unknown_predicate(app):
    assert "'nope' is not an IDB predicate" in _command(app, ".idb nope")



@pytest.fixture
def conditional(app):
    """
    Return the REPL *app* where SQUALL defined the conditional ``marg``.

    NeuroLang compiles ``marg`` into three rules: one of ``marg`` and one
    of each of its helpers ``marg^cond_num`` and ``marg^cond_den``,
    whose symbols have the name ``marg`` too.
    """
    nl = app._nl
    nl.add_tuple_set(
        [("s1", "pain"), ("s1", "memory"), ("s2", "pain")], name="mentions"
    )
    nl.add_tuple_set([("pain",), ("memory",)], name="term")
    nl.add_uniform_probabilistic_choice_over_set(
        [("s1",), ("s2",)], name="selected_study"
    )
    app._handle_dot_command(".mode squall")
    app._execute_and_display(
        "define as marg with inferred probability every Term where a "
        "Selected_study mentions the Term conditioned to the "
        "Selected_study that mentions 'pain'."
    )
    return app


def test_idb_lists_a_conditional_predicate_once(conditional):
    output = _command(conditional, ".idb")
    assert output.count("marg") == 1
    assert "Total session IDB predicates: 2" in output


def test_drop_removes_the_helpers_of_a_conditional(conditional):
    _command(conditional, ".drop marg")
    assert "marg" not in _idb_names(conditional)
    assert "not an IDB predicate" in _command(conditional, ".datalog marg")


def test_datalog_of_a_squall_predicate(conditional):
    output = _command(conditional, ".datalog marg")
    assert "marg(s₀, PROB(s₀)) :-" in output
    assert " | " in output
    assert "cond_num" not in output and "cond_den" not in output


def test_datalog_of_a_datalog_predicate(app):
    output = _command(app, ".datalog active")
    assert "active(x) :-" in output
    assert "plays(x)" in output


def test_datalog_of_an_unknown_predicate(app):
    assert "'nothing' is not an IDB predicate" in _command(
        app, ".datalog nothing"
    )

def test_drop_rule_removes_exactly_it(two_rules):
    out = _command(two_rules, ".drop both 1")
    assert "removed rule 1 of both (1 rule left)" in out
    assert _command(two_rules, ".idb both").splitlines() == [
        "1  both(x) :- person(x), sings(x)"
    ]
    assert _query(two_rules, "ans(x) :- both(x)") == {("bob",)}


def test_drop_without_number_removes_all_rules(two_rules):
    assert "removed both (2 rules)" in _command(two_rules, ".drop both")
    assert "both" not in _idb_names(two_rules)
    assert "active" in _idb_names(two_rules)


def test_drop_last_rule_removes_predicate(app):
    out = _command(app, ".drop active 1")
    assert "removed rule 1 of active (no rules left: active removed)" in out
    assert "active" not in _idb_names(app)


def test_drop_session_rule_of_engine_predicate(app):
    nl = NeurolangPDL()
    nl.add_tuple_set([("alice",), ("bob",)], name="person")
    nl.add_tuple_set([("alice",)], name="plays")
    nl.add_tuple_set([("bob",)], name="sings")
    nl.execute_datalog_program("named(x) :- person(x), plays(x)")
    console = Console(file=io.StringIO(), width=100)
    app = InteractiveTuiApp(nl, console=console, err_console=console)
    app._execute_and_display("named(x) :- person(x), sings(x)")
    out = _command(app, ".drop named 1")
    assert "Rule 1 of named is part of the engine" in out
    assert "removed named (1 rule); its 1 engine rule remain" in _command(
        app, ".drop named"
    )
    assert _query(app, "ans(x) :- named(x)") == {("alice",)}


@pytest.mark.parametrize(
    "line, message",
    [
        (".drop person", "'person' is part of the engine"),
        (".drop nope", "'nope' is not defined"),
        (".drop active 2", "active has 1 rule; there is no rule 2"),
        (".drop active 0", "must be a positive integer, not '0'"),
        (".drop active x", "must be a positive integer, not 'x'"),
        (".drop", "Usage: .drop PREDICATE [N]"),
    ],
)
def test_drop_refusals(app, line, message):
    assert message in _command(app, line)
    assert "active" in _idb_names(app)
    assert "person" in [s.name for s in app._nl.program_ir.symbol_table]


def _write(path, text):
    """Write *text* to *path*, gzipped if its name ends with ``.gz``."""
    if path.name.endswith(".gz"):
        path.write_bytes(gzip.compress(text.encode()))
    else:
        path.write_text(text)
    return path


@pytest.mark.parametrize(
    "file_name, text",
    [
        ("scores.csv", "Name,Score\nalice,3\ncarol,5\n"),
        ("scores.tsv", "Name\tScore\nalice\t3\ncarol\t5\n"),
        ("scores.csv.gz", "Name,Score\nalice,3\ncarol,5\n"),
    ],
)
def test_load_and_query(app, tmp_path, file_name, text):
    path = _write(tmp_path / file_name, text)
    out = _command(app, f".load {path}")
    assert f"loaded scores(Name, Score): 2 rows from {path}" in out
    assert _query(app, "ans(x, s) :- scores(x, s), person(x)") == {
        ("alice", 3)
    }
    app._handle_dot_command(".mode squall")
    app._execute_and_display("obtain every scores.")
    assert list(app.results[0].frame.columns) == ["Name", "Score"]


@pytest.mark.parametrize(
    "text, columns",
    [
        ("term\nmemory\npain\n", ["term"]),
        ("term,weight\nmemory,1\npain,2\n", ["term", "weight"]),
        ("term,weight,n\nmemory,1,3\npain,2,4\n", ["term", "weight", "n"]),
    ],
)
def test_loaded_relation_keeps_its_header(
    app, tmp_path, capsys, text, columns
):
    path = _write(tmp_path / "myterms.csv", text)
    app._handle_dot_command(f".load {path}")
    variables = ", ".join(columns)
    app._execute_and_display(f"ans({variables}) :- myterms({variables})")
    assert list(app.results[0].frame.columns) == columns
    app._handle_dot_command(".sets")
    assert str(columns) in capsys.readouterr().out
    app._handle_dot_command(".mode squall")
    app._execute_and_display("obtain every myterms.")
    frame = app.results[0].frame
    assert list(frame.columns) == columns
    assert sorted(frame[columns[0]]) == ["memory", "pain"]


@pytest.mark.parametrize(
    "aggregate, rows",
    [
        ("count", {("a", 2), ("b", 1)}),
        ("sum", {("a", 3), ("b", 5)}),
        ("max", {("a", 2), ("b", 5)}),
    ],
)
def test_query_with_aggregation_in_head(app, aggregate, rows):
    app._nl.add_tuple_set([(1, "a"), (2, "a"), (5, "b")], name="hits")
    assert _query(app, f"ans(t, {aggregate}(s)) :- hits(s, t)") == rows
    assert list(app.results[0].frame.columns) == ["t", aggregate]
    assert not app.failed


def test_sets_shows_loaded_columns(app, tmp_path, capsys):
    path = _write(tmp_path / "scores.csv", "Name,Score\nalice,3\n")
    app._handle_dot_command(f".load {path}")
    app._handle_dot_command(".sets")
    assert "['Name', 'Score']" in capsys.readouterr().out


@pytest.mark.parametrize(
    "file_name, name",
    [
        ("My Scores-2024.CSV", "my_scores_2024"),
        ("2024.tsv.gz", "rel_2024"),
        ("a.b.csv", "a_b"),
    ],
)
def test_default_relation_name(file_name, name):
    from neurolang.utils.tui.session import default_relation_name

    assert default_relation_name(file_name) == name


def test_load_with_name_in_home(app, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    _write(tmp_path / "s.csv", "a\n1\n")
    out = _command(app, ".load ~/s.csv Ones")
    assert f"loaded Ones(a): 1 row from {tmp_path / 's.csv'}" in out


def test_load_name_clashes(app, tmp_path):
    path = _write(tmp_path / "person.csv", "a\n1\n")
    assert "'person' is already defined" in _command(app, f".load {path}")
    assert "'active' is already defined" in _command(
        app, f".load {path} active"
    )
    assert "Invalid relation name '1x'" in _command(app, f".load {path} 1x")
    _command(app, f".load {path} ones")
    out = _command(app, f".load {path} ones")
    assert "(replaces the previous ones)" in out
    other = _write(tmp_path / "ones.csv", "a,b\n1,2\n")
    assert "already loaded; to replace it" in _command(app, f".load {other}")
    out = _command(app, f".load {other} ones")
    assert "loaded ones(a, b)" in out
    assert _query(app, "ans(x, y) :- ones(x, y)") == {(1, 2)}


@pytest.mark.parametrize(
    "file_name, text, message",
    [
        ("missing.csv", None, "No such file: "),
        ("r.xlsx", "a\n1\n", "Unsupported relation file format: r.xlsx"),
        ("empty.csv", "", "the file is empty"),
        ("header.csv", "a,b\n", "the file is empty"),
        ("bad.csv", 'a,b\n1,"x\n', "Cannot load "),
    ],
)
def test_load_errors(app, tmp_path, file_name, text, message):
    path = tmp_path / file_name
    if text is not None:
        path.write_text(text)
    out = _command(app, f".load {path}")
    assert message in out
    assert len(out.strip().splitlines()) == 1
    assert "Traceback" not in out
    assert app.failed


def test_load_usage(app):
    assert "Usage: .load PATH [NAME]" in _command(app, ".load")


def test_clear_removes_loaded_relations(app, tmp_path, capsys):
    path = _write(tmp_path / "scores.csv", "Name\nalice\n")
    app._handle_dot_command(f".load {path}")
    out = _command(app, ".clear")
    assert out.splitlines() == [
        "removed active (1 rule)", "removed relation scores"
    ]
    assert "scores" not in [s.name for s in app._nl.program_ir.symbol_table]
    assert app.loaded_relations == {}
    assert "loaded scores" in _command(app, f".load {path}")


def test_drop_loaded_relation(app, tmp_path):
    path = _write(tmp_path / "scores.csv", "Name\nalice\n")
    app._handle_dot_command(f".load {path}")
    assert "is a relation, it has no rules" in _command(app, ".drop scores 1")
    assert "removed relation scores" in _command(app, ".drop scores")
    assert "scores" not in [s.name for s in app._nl.program_ir.symbol_table]
    assert "scores" not in app.loaded_relations


def test_help_lists_new_commands(app):
    out = _command(app, ".help")
    for usage in (".idb [PREDICATE]", ".drop PREDICATE [N]",
                  ".load PATH [NAME]", ".format [table|csv|json]"):
        assert usage in out


def _submit(app, program):
    """Run *program* as typed at the prompt and return what it printed."""
    app.console.file.seek(0)
    app.console.file.truncate()
    app.console.width = 1000
    app._run_line(program)
    return _output(app)


def _session_symbols(app):
    """Return the symbols of the session's scope and their values."""
    return list(app._nl.program_ir.symbol_table._symbols.items())


def _same_symbols(before, after):
    """Return whether *after* has exactly the symbols and values *before*."""
    return len(before) == len(after) and all(
        k1 is k2 and v1 is v2 for (k1, v1), (k2, v2) in zip(before, after)
    )


@pytest.mark.parametrize(
    "squall, program",
    [
        (False, "bar(x) :- person(x)\nans(x) :- bar(x, y)"),
        (False, "bar(x) :- person(x)\nbaz(x) :- bar(x, y)\nans(x) :- baz(x)"),
        (
            True,
            "define as Bar every person that plays. "
            "obtain every Bar (?x; ?y).",
        ),
        (
            True,
            "define as Bar every person that plays. "
            "obtain every Person that bars a person.",
        ),
    ],
)
def test_failed_submission_defines_nothing(app, squall, program):
    app.settings.squall = squall
    session = app._nl.program_ir.symbol_table
    before = _session_symbols(app)
    out = _submit(app, program)
    assert "nothing defined: the submission failed" in out
    assert app.failed
    assert app._nl.program_ir.symbol_table is session
    assert _same_symbols(before, _session_symbols(app))
    assert sorted(_idb_names(app)) == ["active"]
    assert "'bar' is not an IDB predicate" in _command(app, ".datalog bar")
    assert app.squall_rules == {}


def test_failed_rule_of_a_predicate_is_not_added(app):
    _submit(app, "active(x) :- sings(x)\nans(x) :- active(x, y)")
    assert _command(app, ".idb active").splitlines() == [
        "1  active(x) :- person(x), plays(x)"
    ]


def test_rule_added_to_a_predicate_replaces_its_rules(app):
    _submit(app, "active(x) :- person(x), sings(x)")
    session = app._nl.program_ir.symbol_table
    assert len(session._symbols[Symbol("active")].formulas) == 2
    assert Symbol("active") not in session.enclosing_scope
    assert len(_command(app, ".idb active").splitlines()) == 2
    assert _query(app, "ans(x) :- active(x)") == {("alice",), ("bob",)}


def test_failed_query_without_definitions_says_nothing_more(app):
    out = _submit(app, "ans(x) :- person(x, y)")
    assert app.failed
    assert "nothing defined" not in out


def test_interrupted_submission_defines_nothing(app, monkeypatch):
    nl = app._nl

    def interrupted(program):
        nl.execute_datalog_program("bar(x) :- person(x)")
        nl.program_ir.push_scope()
        raise KeyboardInterrupt()

    monkeypatch.setattr(app, "_execute", interrupted)
    session = nl.program_ir.symbol_table
    before = _session_symbols(app)
    out = _submit(app, "bar(x) :- person(x)\nans(x) :- bar(x)")
    assert "Query interrupted" in out
    assert "nothing defined: the submission failed" in out
    assert nl.program_ir.symbol_table is session
    assert _same_symbols(before, _session_symbols(app))


def test_session_works_after_a_failed_submission(app, tmp_path):
    path = _write(tmp_path / "scores.csv", "Name\nalice\n")
    app._handle_dot_command(f".load {path}")
    _submit(app, "bar(x) :- scores(x)\nans(x) :- bar(x, y)")
    assert "scores" in app.loaded_relations
    _submit(app, "bar(x) :- scores(x)")
    assert _query(app, "ans(x) :- bar(x)") == {("alice",)}
    assert "removed bar (1 rule)" in _command(app, ".drop bar")
    assert "removed relation scores" in _command(app, ".drop scores")


def test_failed_submission_is_left_out_of_the_replay_script(app, tmp_path):
    _submit(app, "bar(x) :- person(x)\nans(x) :- bar(x, y)")
    _submit(app, "baz(x) :- person(x)")
    path = tmp_path / "replay.nl"
    _command(app, f".transcript {path} --inputs")
    script = path.read_text()
    assert "baz(x) :- person(x)" in script
    assert "bar" not in script


def _messages(out):
    """Return the lines of *out* that are not part of a table."""
    return [
        line.strip() for line in out.splitlines()
        if line.strip() and line.strip()[0] not in "│╭╰├┃┏┗┡"
    ]


@pytest.mark.parametrize(
    "program, messages",
    [
        ("bar(x) :- person(x)", ["defined bar/1"]),
        ("active(x) :- person(x), sings(x)",
         ["defined active/1 (now 2 rules)"]),
        ("active(x) :- person(x), plays(x)",
         ["active/1 unchanged (1 rule)"]),
        ("two(x, y) :- person(x), plays(y)\n"
         "two(x, y) :- person(x), sings(y)",
         ["defined two/2 (2 rules)"]),
        ("a1(x) :- person(x)\na2(x) :- a1(x)",
         ["defined a1/1", "defined a2/1"]),
        ("foo(1)\nfoo(2)", ["defined foo/1 (2 facts)"]),
    ],
)
def test_definition_says_what_it_defined(app, program, messages):
    out = _submit(app, program)
    assert _messages(out) == messages
    assert not app.failed


def test_facts_added_to_a_relation(app):
    _submit(app, "foo(1)")
    assert _messages(_submit(app, "foo(2)\nfoo(3)")) == [
        "defined foo/1 (now 3 facts)"
    ]


def test_definition_with_a_query_shows_both(app):
    out = _submit(app, "bar(x) :- plays(x)\nans(x) :- bar(x)")
    messages = _messages(out)
    assert messages[:2] == ["Running query...", "defined bar/1"]
    assert messages[-1].startswith("Query completed in")
    assert _query(app, "ans(x) :- bar(x)") == {("alice",)}


def test_query_says_nothing_about_definitions(app):
    out = _submit(app, "ans(x) :- person(x)")
    assert "Running query..." in out and "Query completed in" in out
    assert "defined" not in out


def test_squall_definitions_say_what_they_defined(app):
    app.settings.squall = True
    out = _submit(app, "define as Bar every person that plays.")
    assert _messages(out) == ["defined bar/1"]
    out = _submit(app, "define as Bar every person that sings.")
    assert _messages(out) == ["defined bar/1 (now 2 rules)"]


def test_squall_conditional_is_defined_once(conditional):
    out = _output(conditional)
    assert "defined marg/2" in out
    assert out.count("defined marg") == 1
    assert "(no result)" not in out


def test_failed_facts_are_discarded(app):
    _submit(app, "person('carol')\nans(x) :- person(x, y)")
    assert app.failed
    assert _query(app, "ans(x) :- person(x)") == {("alice",), ("bob",)}


def test_facts_added_to_an_engine_relation(app, capsys):
    out = _submit(app, "person('carol')")
    assert _messages(out) == ["defined person/1 (now 3 facts)"]
    assert _query(app, "ans(x) :- person(x)") == {
        ("alice",), ("bob",), ("carol",)
    }
    capsys.readouterr()
    app._handle_dot_command(".sets")
    sets = [
        line.split()[0] for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert sets.count("person") == 1
    _command(app, ".clear")
    assert _query(app, "ans(x) :- person(x)") == {("alice",), ("bob",)}


@pytest.fixture
def likes(app):
    """Return the REPL *app* where ``likes`` is a rule of the session."""
    app._execute_and_display("likes(x, y) :- person(x), plays(y)")
    return app


@pytest.mark.parametrize(
    "squall, query",
    [
        (False, "ans(x) :- likes(x, 'alice')"),
        (True, "obtain every Person that likes 'alice'."),
    ],
)
def test_rewritten_shows_the_magic_sets_of_a_query(
    likes, capsys, squall, query
):
    likes.settings.squall = squall
    likes._execute_and_display(query)
    capsys.readouterr()
    out = _command(likes, ".rewritten") + capsys.readouterr().out
    assert "── rewritten program ──" in out
    assert "likes" in out and "plays(" in out
    assert not likes.failed


@pytest.mark.parametrize(
    "squall, program",
    [
        (False, "fan(x) :- person(x), plays(x)\nans(x) :- fan(x)"),
        (True, "define as Fan every person that plays. obtain every Fan."),
    ],
)
def test_rewritten_has_the_rules_of_the_last_query_once(
    app, capsys, squall, program
):
    app.settings.squall = squall
    app._execute_and_display(program)
    capsys.readouterr()
    out = _command(app, ".rewritten") + capsys.readouterr().out
    assert len(re.findall(r"^fan\(", out, re.MULTILINE)) == 1
    assert _command(app, ".idb fan").count("fan(") == 1


@pytest.mark.parametrize(
    "squall, query",
    [
        (False, "ans(x) :- active(x)"),
        (True, "obtain every Person that plays."),
    ],
)
def test_ra_prints_the_plans_of_the_last_query(app, capsys, squall, query):
    app.settings.squall = squall
    app._execute_and_display(query)
    rows = list(app.results[0].frame.itertuples(index=False, name=None))
    capsys.readouterr()
    out = _command(app, ".ra") + capsys.readouterr().out
    assert "── deterministic stratum ──" in out
    assert not app.settings.show_ra
    assert not app.failed
    assert list(app.results[0].frame.itertuples(index=False, name=None)) == (
        rows
    )
    assert sorted(_idb_names(app)) == ["active"]


def test_ra_without_a_query(app):
    assert "No query has been run yet." in _command(app, ".ra")
    assert not app.settings.show_ra


def test_ra_of_a_query_with_its_rules(app, capsys):
    app.settings.squall = True
    app._execute_and_display(
        "define as Fan every person that plays. obtain every Fan."
    )
    capsys.readouterr()
    out = _command(app, ".ra") + capsys.readouterr().out
    assert "fan" in out
    assert not app.failed
    assert _command(app, ".idb fan").count("fan(") == 1


def test_clear_says_what_it_removed(two_rules, capsys):
    _submit(two_rules, "foo(1)\nfoo(2)")
    _submit(two_rules, "person('carol')")
    assert _command(two_rules, ".clear").splitlines() == [
        "removed active (1 rule)",
        "removed both (2 rules)",
        "removed foo (2 facts)",
        "removed person (1 fact); its 2 engine facts remain",
    ]
    assert _command(two_rules, ".clear").splitlines() == ["nothing to clear"]
    assert _query(two_rules, "ans(x) :- person(x)") == {("alice",), ("bob",)}


def test_clear_keeps_the_engine_rules():
    nl = NeurolangPDL()
    nl.add_tuple_set([("alice",), ("bob",)], name="person")
    nl.execute_datalog_program("named(x) :- person(x)")
    console = Console(file=io.StringIO(), width=100)
    app = InteractiveTuiApp(nl, console=console, err_console=console)
    app._execute_and_display("named(x) :- person(x), person(x)")
    assert _command(app, ".clear").splitlines() == [
        "removed named (1 rule); its 1 engine rule remain"
    ]


def test_clear_lists_a_conditional_once(conditional):
    lines = _command(conditional, ".clear").splitlines()
    assert [line for line in lines if "marg" in line] == [
        "removed marg (1 rule)"
    ]
    assert "removed mentions (3 facts)" in lines
