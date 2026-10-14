"""Tests for the session commands of the interactive REPL."""

import gzip
import io

import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

from rich.console import Console  # noqa: E402

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
    assert "cleared" in _output(app)
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


def test_predicates_reuses_cli_listing(app, capsys):
    app._handle_dot_command(".predicates")
    assert "peak_reported(i, j, k, study_id)" in capsys.readouterr().out


def test_engines(app, capsys):
    app._handle_dot_command(".engines")
    assert "neurosynth" in capsys.readouterr().out


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


def test_datalog_needs_squall_query(app):
    app._handle_dot_command(".datalog")
    assert "use .rewritten for a Datalog query" in _output(app)


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
    assert "IDB predicates and loaded relations cleared" in out
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
