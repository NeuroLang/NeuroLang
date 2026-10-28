"""Magic sets rewriting through the frontends: the rewriting's tests are in
``neurolang/datalog/tests/test_magic_sets.py``."""
import pandas as pd
import pytest

from ...frontend import NeurolangDL
from ...frontend.probabilistic_frontend import NeurolangPDL


@pytest.fixture
def nl():
    nl = NeurolangPDL()
    studies = (
        10191322,
        10227106,
        10349031,
        10402199,
        10571235,
        10407201,
        10197540,
        10022496,
        10022494,
        9819274,
        11430815,
        11275483,
        10913505,
        10808134,
        10751455,
    )
    nl.add_uniform_probabilistic_choice_over_set(
        list(map(lambda x: (x,), studies)), name="SelectedStudy"
    )
    peaks_data = (
        (-48, -74, 34, 10349031),
        (-33, 28, 7, 10191322),
        (21, 52, 20, 10349031),
        (-42, 28, 9, 10191322),
        (42, -24, 12, 10022496),
        (-38, 15, 26, 10191322),
        (-36, 26, 9, 10191322),
        (-53, -26, -21, 10349031),
        (-42, 12, 17, 10191322),
        (0, -73, 6, 10913505),
        (10, 41, 1, 10349031),
        (-40, 22, 32, 10191322),
        (-52, -28, 52, 9819274),
        (-48, -30, 12, 10022496),
        (-44, 27, 18, 10191322),
        (-48, -24, 12, 10022496),
        (-30, -31, -2, 10197540),
        (-35, 16, 39, 10191322),
        (18, -76, -16, 10022494),
        (-48, 14, 36, 10227106),
        (-48, 32, -16, 10808134),
        (-48, 26, 27, 10402199),
        (38, -84, 0, 10751455),
        (-40, 16, 19, 10191322),
        (23, -53, -19, 10913505),
        (48, -20, 57, 10197540),
        (32, -72, -20, 10022494),
        (-35, 24, 27, 10402199),
        (-42, -26, 12, 10022496),
        (-33, 9, 29, 10191322),
        (2, 4, -2, 10349031),
        (22, 44, 32, 10349031),
        (-60, -36, 16, 10571235),
        (-53, -61, -29, 10913505),
        (-31, -18, 46, 10913505),
        (-40, 16, 19, 10191322),
        (14, -13, -10, 10197540),
        (-52, -28, 12, 10407201),
        (32, 61, -16, 10913505),
        (40, -16, 12, 10022496),
        (-58, -41, -20, 10913505),
        (26, -93, 2, 10227106),
        (-52, -16, 8, 10407201),
        (59, -19, -12, 10197540),
        (-29, -31, -2, 10197540),
        (-56, -43, 19, 10197540),
        (55, -3, -1, 10913505),
        (-49, 28, 4, 10227106),
        (-42, -26, 8, 10022496),
        (5, -39, 22, 10913505),
    )
    peaks = pd.DataFrame(peaks_data, columns=("x", "y", "z", "tfidf"))

    terms_data = (
        (10751455, "emotion", 0.052538),
        (10808134, "emotion", 0.244368),
        (10913505, "emotion", 0.059463),
        (11275483, "emotion", 0.123356),
        (11430815, "emotion", 0.065204),
        (9819274, "auditory", 0.059705),
        (10751455, "auditory", 0.052538),
        (10022494, "auditory", 0.10052),
        (10022496, "auditory", 0.28146),
        (10197540, "auditory", 0.38657),
        (10407201, "auditory", 0.16817),
        (11430815, "auditory", 0.065204),
        (10191322, "language", 0.28298),
        (10751455, "language", 0.052538),
        (10227106, "language", 0.05432),
        (10349031, "language", 0.05329),
        (10402199, "language", 0.18801),
        (10571235, "language", 0.08257),
        (11430815, "language", 0.065204),
    )
    terms = pd.DataFrame(terms_data, columns=("study", "term", "tfidf"))
    nl.add_tuple_set(peaks, name="PeakReported")
    nl.add_tuple_set(terms, name="TermInStudyTFIDF")
    nl.add_tuple_set(terms[["term"]].drop_duplicates(), name="Term")
    return nl


def test_neurosynth_query(nl):
    query = """
    TermInStudy(term, study) :: (1 / (1 + exp(-300 * (tfidf - 0.001)))) :- TermInStudyTFIDF(study, term, tfidf)
    TermAssociation(term) :- SelectedStudy(study) , TermInStudy(term, study)
    Activation(x, y, z) :- SelectedStudy(s) , PeakReported(x, y, z, s)
    ActivationGivenTerm(x, y, z, t, PROB) :- Activation(x, y, z) // TermAssociation(t)
    QueryActivation(x, y, z, p) :- ActivationGivenTerm(x, y, z, "emotion", p)
    ans(x, y, z, p) :- QueryActivation(x, y, z, p)
    """
    res = nl.execute_datalog_program(query)
    expected = (
        (-58, -41, -20, 0.2),
        (-53, -61, -29, 0.2),
        (-48, 32, -16, 0.2),
        (-31, -18, 46, 0.2),
        (0, -73, 6, 0.2),
        (5, -39, 22, 0.2),
        (23, -53, -19, 0.2),
        (32, 61, -16, 0.2),
        (38, -84, 0, 0.2),
        (55, -3, -1, 0.2),
    )
    assert res is not None
    assert len(res) == len(expected)
    p = res.as_pandas_dataframe()
    assert set(tuple(x) for x in p[["x", "y", "z"]].values) == set(
        (x, y, z) for x, y, z, _ in expected
    )


@pytest.mark.parametrize("frontend", [NeurolangDL, NeurolangPDL])
@pytest.mark.parametrize(
    "threshold,query,expected",
    [
        ("", "ans(s) :- mentions(s, 'pain'), mentions(s, 'emotion')", {1}),
        (
            ", (v > 0.03)",
            "ans(s) :- mentions(s, 'pain'), mentions(s, 'emotion')",
            {1},
        ),
        ("", "ans(s) :- mentions(s, 'pain'), ~mentions(s, 'emotion')", {2, 3}),
        (
            ", (v > 0.03)",
            "ans(s) :- mentions(s, 'pain'), ~mentions(s, 'emotion')",
            {3},
        ),
    ],
)
def test_frontend_same_idb_predicate_different_constants(
    frontend, threshold, query, expected
):
    nl = frontend()
    nl.add_tuple_set(
        [(1, "pain", 0.5), (1, "emotion", 0.5), (2, "pain", 0.01),
         (3, "pain", 0.5)],
        name="tfidf",
    )
    nl.execute_datalog_program(
        f"mentions(s, t) :- tfidf(s, t, v){threshold}"
    )
    res = nl.execute_datalog_program(query)
    assert set(res.as_pandas_dataframe().s) == expected


@pytest.mark.parametrize("frontend", [NeurolangDL, NeurolangPDL])
def test_frontend_idb_predicate_several_bound_constants(frontend):
    nl = frontend()
    nl.add_tuple_set(
        [(10, 20, 30, 1), (11, 20, 30, 1), (10, 20, 30, 2)],
        name="peak_reported",
    )
    res = nl.execute_datalog_program(
        "reports(s, i, j, k) :- peak_reported(i, j, k, s)\n"
        "ans(s) :- reports(s, 10, 20, 30)"
    )
    assert set(res.as_pandas_dataframe().s) == {1, 2}


def _similarity(text: str, study: int) -> float:
    return float(len(text) % study == 0)


@pytest.mark.parametrize("frontend", [NeurolangDL, NeurolangPDL])
@pytest.mark.parametrize("query, expected", [
    ('ans(s) :- task(_, "motor task", d) & close(d, s, v)', {1, 2}),
    # a comparison with a constant binds nothing: it stays after close
    ('ans(s) :- task(_, "motor task", d) & close(d, s, v) & (s > 1)', {2}),
])
def test_frontend_variable_bound_by_an_edb_atom_with_a_constant(
    frontend, query, expected
):
    nl = frontend()
    nl.add_tuple_set([(1,), (2,), (3,)], name="study")
    nl.add_tuple_set(
        [("t1", "motor task", "move"), ("t2", "reading task", "read it")],
        name="task",
    )
    nl.add_symbol(_similarity, name="similarity")
    res = nl.execute_datalog_program(
        "close(q, s, v) :- study(s) & (v == similarity(q, s)) & (v > .5)\n"
        + query
    )
    assert set(res.as_pandas_dataframe().s) == expected


@pytest.mark.parametrize("frontend", [NeurolangDL, NeurolangPDL])
def test_frontend_edb_atom_with_a_constant_binding_an_aggregate(frontend):
    """The value an EDB atom with a constant gives to an aggregate is not
    bound by magic sets: the query is solved as without the binding."""
    nl = frontend()
    nl.add_tuple_set([(1, 2), (1, 3), (2, 2)], name="e")
    nl.add_tuple_set([("a",), ("b",)], name="r")
    res = nl.execute_datalog_program(
        "c(count(x)) :- r(x)\n"
        "ans(n) :- e(1, n) & c(n)"
    )
    assert set(res.as_pandas_dataframe().n) == {2}


def _rank_similarity(text: str, study: int) -> float:
    return (study + 1) / 6


def test_frontend_conditional_probability_bound_through_a_rule_compared():
    """
    The text of the similarity is bound by magic sets through the
    conditional probability, also when the query compares it: the
    comparison once came before ``p`` and left its magic set unseeded.
    """
    nl = NeurolangPDL()
    studies = [(s,) for s in range(6)]
    nl.add_tuple_set(studies, name="study")
    nl.add_uniform_probabilistic_choice_over_set(studies, name="selected")
    nl.add_tuple_set([(0, "a"), (1, "a"), (3, "a"), (4, "b")], name="act")
    nl.add_tuple_set([("x", "a text")], name="definition")
    nl.add_symbol(_rank_similarity, name="similarity")
    res = nl.execute_datalog_program(
        "sim(q, s, v) :- study(s) & (v == similarity(q, s)) & (v > 0)\n"
        "close(q, s) :- sim(q, s, v) & (v > .5) & selected(s)\n"
        "active(l, s) :- act(s, l) & selected(s)\n"
        "p(q, l, PROB) :- active(l, s) // close(q, s)\n"
        "ans(l, r) :- definition('x', d) & p(d, l, r) & (r > 0)"
    )
    # close('a text', s) for s in {3, 4, 5}: a in 3, b in 4
    assert dict(res.as_pandas_dataframe().values) == pytest.approx(
        {"a": 1 / 3, "b": 1 / 3}
    )


@pytest.mark.parametrize("frontend", [NeurolangDL, NeurolangPDL])
def test_frontend_aggregate_of_a_rule_with_a_constant(frontend):
    """
    Magic sets leave aggregates free, in the query or through a rule
    using one: the constant text of the rule they aggregate binds the
    similarity.
    """
    nl = frontend()
    nl.add_tuple_set([(s,) for s in range(6)], name="study")
    nl.add_tuple_set([(s, f"t{s}") for s in range(6)], name="title")
    nl.add_symbol(_rank_similarity, name="similarity")
    nl.execute_datalog_program(
        "ssim(q, s, v) :- study(s) & (v == similarity(q, s)) & (v > 0)"
    )
    res = nl.execute_datalog_program(
        "sim(s, v) :- ssim('a text', s, v)\n"
        "top(m) :- AGGREGATE[](sim(s, v) @ max(v)) = m\n"
        "ans(t) :- sim(s, m) & top(m) & title(s, t)"
    )
    assert set(res.as_pandas_dataframe().t) == {"t5"}
    res = nl.execute_datalog_program(
        "ans(m) :- AGGREGATE[](sim(s, v) @ max(v)) = m"
    )
    assert res.as_pandas_dataframe().m.tolist() == [1.0]
    # v, bound by sim, is not passed on to the aggregate of top
    res = nl.execute_datalog_program(
        "ans(t, v) :- sim(s, v) & top(v) & title(s, t)"
    )
    assert res.as_pandas_dataframe().values.tolist() == [["t5", 1.0]]
