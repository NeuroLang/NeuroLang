"""
What the query assistant is told when it misuses EXPECT: the message of
the engine reaches the model whole, naming the construct and what to
write instead, never the internals of the walkers.
"""
import pytest

from ..agent import ReactAgent
from ..validation import check_query, run_preview
from .fakes import ScriptedClient, assistant_config, provider


@pytest.fixture
def probabilistic():
    from neurolang.frontend import NeurolangPDL

    nl = NeurolangPDL()
    nl.add_probabilistic_choice_from_tuples(
        [(0.5, 1), (0.5, 2)], name="roll"
    )
    nl.add_tuple_set([(1,)], name="r")
    return nl


INTERNALS = (
    "No match for", "Fol2Datalog", "λ", "Internal error", "Symbol{",
    "Error trying to process rule",
)

CASES = (
    ("ans(e) :- r(x), ~EXPECT[x | roll(x)] = e", "must be a conjunct"),
    ("ans(e) :- (EXPECT[x | roll(x)] = e | r(e))", "must be a conjunct"),
    ("ans(n) :- r(n), exists(e; EXPECT[x | roll(x)] = e)",
     "must be a conjunct"),
    ("ans(p) :- PROB[EXPECT[x | roll(x)] = e] = p", "inside PROB or MARG"),
    ("ans(e) :- EXPECT[x | EXPECT[y | roll(y)] = x] = e",
     "inside the event of another EXPECT"),
    ("ans(n) :- AGGREGATE[](EXPECT[x | roll(x)] = m @ sum(m)) = n",
     "inside an AGGREGATE"),
    ("ans(e) :- EXPECT[x + z | roll(x)] = e", "bind every variable of f"),
    ("ans(e) :- EXPECT[1 | roll(2)] = e", "no variable to sum over"),
)


@pytest.mark.parametrize("program, phrase", CASES)
def test_the_checked_query_says_what_to_write_instead(
    probabilistic, program, phrase
):
    result = check_query(probabilistic, program, "datalog")
    assert not result.ok
    assert phrase in result.message
    assert "EXPECT[" in result.message, "an example is given"
    assert not any(word in result.message for word in INTERNALS)


def test_a_relation_called_in_f_says_to_join_it(probabilistic):
    program = "ans(e) :- EXPECT[r(x) | roll(x)] = e"
    assert check_query(probabilistic, program, "datalog").ok
    preview = run_preview(probabilistic, program, "datalog", 5, "")
    assert not preview.ok
    assert "cannot be called inside an expression" in preview.text
    assert "score(x, v)" in preview.text
    assert not any(word in preview.text for word in INTERNALS)


def test_an_engine_without_probabilities_says_so():
    from neurolang.frontend import NeurolangDL

    nl = NeurolangDL()
    nl.add_tuple_set([(1,), (2,)], name="roll")
    result = check_query(nl, "ans(e) :- EXPECT[x | roll(x)] = e", "datalog")
    assert not result.ok
    assert "this engine has none" in result.message
    assert "AGGREGATE[](" in result.message
    assert not any(word in result.message for word in INTERNALS)


def test_the_model_corrects_an_expectation_from_the_message(probabilistic):
    bad = "ans(e) :- EXPECT[x + z | roll(x)] = e"
    good = "ans(e) :- EXPECT[x | roll(x)] = e"
    replies = [
        [("run_query", {"query": bad})],
        [("run_query", {"query": good})],
        [("final_answer", {"query": good})],
    ]
    steps = []
    react = ReactAgent(
        probabilistic, provider("native"), assistant_config(), "datalog",
        client=ScriptedClient(replies), on_step=steps.append,
    )
    result = react.run("What is the expected value of a roll?")
    assert result.valid and result.query == good
    first = steps[0].observation
    assert first.startswith("The query fails:")
    assert "bind every variable of f" in first and "'z'" in first
    assert "1.5" in steps[1].observation
