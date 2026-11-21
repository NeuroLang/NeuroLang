"""Magic sets rewriting through the frontends: the rewriting's tests are in
``neurolang/datalog/tests/test_magic_sets.py``."""
import numpy as np
import pandas as pd
import pytest
from nl_core.datalog import magic_sets

from ...frontend import NeurolangDL
from ...frontend import probabilistic_frontend, query_resolution_datalog
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


@pytest.mark.parametrize("program", [
    "p(q, l, PROB) :- active(l, s) // close(q, s)\n"
    "ans(l, r) :- definition('x', d) & p(d, l, r) & (r > 0)",
    # d is bound by the rule defn, whatever the head of matching
    "matching(s) :- defn('x', d) & close(d, s)\n"
    "ans(l, r) :- PROB[active(l, s) // matching(s)] = r",
])
def test_frontend_conditional_probability_bound_through_a_rule(program):
    """
    The text of the similarity is bound by magic sets through the
    conditional probability, also when the query compares it (the
    comparison once came before ``p`` and left its magic set unseeded),
    and by a rule left of the similarity, whose variables are all bound.
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
        "defn(t, d) :- definition(t, d)\n" + program
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


@pytest.mark.parametrize("program", [
    "close(s) :- similar_study('a text', s), selected_study(s)\n"
    "both(s) :- act(s, 'a'), close(s)\n"
    "ans(p) :- PROB[both(s) // close(s)] = p",
    "close(q, s) :- similar_study(q, s), selected_study(s)\n"
    "both(q, s) :- act(s, 'a'), close(q, s)\n"
    "ans(p) :- definition('x', q), PROB[both(q, s) // close(q, s)] = p",
])
def test_frontend_bound_rule_on_an_aggregate_of_a_shared_relation(program):
    """
    Magic sets make the relation of similarities and the magic predicate
    of its percentile depend on each other: they are solved together,
    before the percentile, which once failed with
    NoValidChaseClassForStratumException.
    """
    nl = NeurolangPDL()
    studies = range(6)
    nl.add_tuple_set([(s,) for s in studies], name="study")
    nl.add_uniform_probabilistic_choice_over_set(
        [(s,) for s in studies], name="selected_study"
    )
    nl.add_tuple_set([(0, "a"), (1, "a"), (3, "a"), (4, "b")], name="act")
    nl.add_tuple_set([("x", "a text")], name="definition")
    nl.add_symbol(_rank_similarity, name="similarity")
    nl.execute_datalog_program(
        "sim(q, s, v) :- study(s) & (v == similarity(q, s))\n"
        "cutoff(q, percentile(v, 50)) :- sim(q, s, v)\n"
        "similar_study(q, s) :- sim(q, s, v) & cutoff(q, c) & (v >= c)"
    )
    res = nl.execute_datalog_program(program)
    similarities = [_rank_similarity("a text", s) for s in studies]
    similar = {
        s for s, v in zip(studies, similarities)
        if v >= np.percentile(similarities, 50)
    }
    expected = len(similar & {0, 1, 3}) / len(similar)
    assert res.as_pandas_dataframe().p.tolist() == pytest.approx([expected])


@pytest.mark.parametrize("frontend", [NeurolangDL, NeurolangPDL])
def test_frontend_aggregate_read_by_two_literals_one_after_the_other(frontend):
    """
    The magic predicate of the second literal gets its bindings from the
    first, and the one of the aggregate from the second: the aggregate would
    depend on a literal that reads it, and the rewritten program is not
    stratified (once NoValidChaseClassForStratumException). The rewrite is
    refused and the query is solved as it is written.
    """
    nl = frontend()
    nl.add_tuple_set([("a", "b"), ("a", "c")], name="first")
    nl.add_tuple_set([("a", "c"), ("a", "d")], name="second")
    nl.add_tuple_set([("a", "b"), ("a", "c")], name="item")
    nl.add_tuple_set([("a", "b"), ("a", "c")], name="chosen")
    res = nl.execute_datalog_program(
        "n(t, count(l)) :- item(t, l), chosen(t, l)\n"
        "one(t, s) :- first(t, s), n(t, v)\n"
        "two(t, s) :- second(t, s), n(t, v)\n"
        "both(t, s) :- one(t, s), two(t, s)\n"
        'ans(s) :- both("a", s)'
    )
    assert set(res.as_pandas_dataframe().s) == {"c"}


@pytest.mark.parametrize("frontend", [NeurolangDL, NeurolangPDL])
def test_frontend_bound_rule_with_a_literal_sharing_no_variable(frontend):
    """
    ``other(y)`` shares no variable: the probabilistic frontend reorders the
    rule bodies to avoid cross products when they are walked, which once
    put ``r^bf(d, s)`` before the literal binding ``d`` in the adorned
    query. Its magic set was left undefined, and the answer empty.
    """
    nl = frontend()
    nl.add_tuple_set([("x", 1), ("y", 2)], name="e")
    nl.add_tuple_set([("a", "x")], name="f")
    nl.add_tuple_set([(1,), (2,)], name="other")
    res = nl.execute_datalog_program(
        "r(q, s) :- e(q, s)\n"
        "c(t, d) :- f(t, d)\n"
        "ans(d, s, y) :- c('a', d), r(d, s), other(y)"
    )
    assert set(map(tuple, res.as_pandas_dataframe().values)) == {
        ("x", 1, 1), ("x", 1, 2)
    }


@pytest.mark.parametrize("query", [
    "ans(p, pa) :- defn('x', d), PROB[act(s) // close(d, s)] = p, "
    "PROB[act(s)] = pa",
    "ans(p, pa) :- defn('x', d), PROB[act(s)] = pa, "
    "PROB[act(s) // close(d, s)] = p",
])
def test_frontend_two_probabilities_one_bound_by_the_main_rule(query):
    """
    The text ``d`` bound by the main rule is passed to the conditional
    probability next to a probability sharing no variable with it, which
    once raised UnboundVariablesError on the similarity (#1091).
    """
    nl = NeurolangPDL()
    studies = [(s,) for s in range(6)]
    nl.add_tuple_set(studies, name="study")
    nl.add_uniform_probabilistic_choice_over_set(studies, name="selected")
    nl.add_tuple_set([(0,), (1,), (3,), (4,)], name="default_mode")
    nl.add_tuple_set([("x", "a text")], name="definition")
    nl.add_symbol(_rank_similarity, name="similarity")
    res = nl.execute_datalog_program(
        "sim(q, s, v) :- study(s) & (v == similarity(q, s)) & (v > 0)\n"
        "close(q, s) :- sim(q, s, v) & (v > .5) & selected(s)\n"
        "act(s) :- default_mode(s), selected(s)\n"
        "defn(t, d) :- definition(t, d)\n" + query
    )
    # close('a text', s) for s in {3, 4, 5}, of which 3 and 4 are active
    solution = res.as_pandas_dataframe()
    assert len(solution) == 1
    assert solution[["p", "pa"]].values[0] == pytest.approx([2 / 3, 4 / 6])


@pytest.mark.parametrize("query", [
    "ans(t, l, p) :- chosen(t) & p_l_given_t(t, l, p)",
    # a stored relation of the query's rule is put first
    "ans(t, l, p) :- p_l_given_t(t, l, p) & chosen(t)",
])
def test_frontend_stored_relation_of_the_query_binds_a_probability(
    query, capsys
):
    """
    A relation of the extensional database in the query's own rule binds
    its variables, without a constant: the conditional probability of
    each region given a term is computed for the chosen terms alone (it
    was for every term, filtered after), with the probabilities of the
    query of each term.
    """
    nl = NeurolangPDL()
    studies = [(s,) for s in range(6)]
    nl.add_uniform_probabilistic_choice_over_set(studies, name="selected")
    nl.add_tuple_set(
        [(0, "a"), (1, "a"), (1, "b"), (3, "a"), (4, "b"), (5, "b")],
        name="act",
    )
    nl.add_tuple_set(
        [(0, "pain"), (1, "pain"), (3, "fear"), (4, "fear"), (5, "noise")],
        name="term",
    )
    nl.add_tuple_set([("pain",), ("fear",)], name="chosen")
    rules = (
        "active(l, s) :- act(s, l) & selected(s)\n"
        "about(t, s) :- term(s, t) & selected(s)\n"
        "p_l_given_t(t, l, PROB) :- active(l, s) // about(t, s)\n"
    )
    with nl.scope:
        nl.execute_datalog_program(
            rules + query, show_rewritten=True, dry_run=True
        )
    assert "magic_p_l_given_t^bff" in capsys.readouterr().out
    with nl.scope:
        res = nl.execute_datalog_program(rules + query)
    got = {tuple(r[:2]): r[2] for r in res.as_pandas_dataframe().values}
    expected = {}
    for t in ("pain", "fear"):
        with nl.scope:
            one = nl.execute_datalog_program(
                rules + f"ans(l, p) :- p_l_given_t('{t}', l, p)"
            )
        expected.update(
            {(t, l): p for l, p in one.as_pandas_dataframe().values}
        )
    assert got == pytest.approx(expected)
    assert {t for t, _ in got} == {"pain", "fear"}


def _lower(text: str) -> str:
    return text.lower()


def _similar_study_engine(frontend=NeurolangPDL, similarity=None):
    """
    The ``similar_study`` of the engine neurosynth, its similarity a stub:
    its text ``q`` is bound only through its head, by the rules using it.
    Studies 2 to 5 are similar to any text.
    """
    nl = frontend()
    studies = [(s,) for s in range(6)]
    nl.add_tuple_set(studies, name="study")
    nl.add_tuple_set([("t1", "Stroop Task"), ("t2", "Other")], name="name")
    nl.add_tuple_set(
        [("t1", "a text"), ("t2", "another text")], name="definition"
    )
    nl.add_tuple_set(
        [(0, "a"), (1, "a"), (3, "a"), (4, "b")], name="act"
    )
    nl.add_symbol(
        similarity or _rank_similarity, name="query_similarity"
    )
    nl.add_symbol(lambda q: .5, name="query_similarity_cutoff")
    nl.add_symbol(_lower, name="lcase")
    nl.execute_datalog_program(
        "term(x, t) :- name(x, n), (t == lcase(n))\n"
        "similar_study(q, s) :- study(s) & "
        "(v == query_similarity(q, s)) & "
        "(c == query_similarity_cutoff(q)) & (v >= c)"
    )
    return nl


@pytest.mark.parametrize("frontend", [NeurolangDL, NeurolangPDL])
@pytest.mark.parametrize("program", [
    # the text joined from a stored relation without a constant, in the
    # same body as the literal binding it (UnboundVariablesError in
    # similar_study_0: the rewrite left similar_study unbound)
    "uses(s) :- term(t, 'stroop task'), definition(t, d), "
    "similar_study(d, s)",
    # the same, the literals in another order
    "uses(s) :- similar_study(d, s), definition(t, d), "
    "term(t, 'stroop task')",
    # bound in a rule of its own
    "task_def(d) :- term(t, 'stroop task'), definition(t, d)\n"
    "uses(s) :- task_def(d), similar_study(d, s)",
    # every text of a stored relation, no constant: magic sets were refused
    "uses(s) :- definition(t, d), similar_study(d, s)",
])
def test_frontend_similar_study_bound_from_its_rule_body(frontend, program):
    """
    ``similar_study(q, s)`` is a relation over the texts ``q`` the rules
    using it bind, by a stored relation of their body as well as by a
    rule: a stored relation without a constant passes its values to a
    predicate whose rules need them.
    """
    nl = _similar_study_engine(frontend)
    res = nl.execute_datalog_program(
        program + "\nans(count(s)) :- uses(s)"
    )
    assert res.as_pandas_dataframe().values.tolist() == [[4]]


@pytest.mark.parametrize("frontend", [NeurolangDL, NeurolangPDL])
@pytest.mark.parametrize("body", [
    "task(t), definition(t, d), similar_study(d, s)",
    "definition(t, d), task(t), similar_study(d, s)",
    "similar_study(d, s), definition(t, d), task(t)",
    # the constant binds first, and restricts task
    "task(t), term(t, 'stroop task'), definition(t, d), "
    "similar_study(d, s)",
])
def test_frontend_similar_study_bound_after_the_rule_restricting_it(
    frontend, body
):
    """
    ``definition(t, d)`` binds the text of ``similar_study`` after
    ``task(t)``, a rule without a constant, restricts ``t``: the
    similarity is computed for the stroop task's text alone, not for
    every definition (``similar_study`` was seeded with every text of
    ``definition``, put first).
    """
    texts = set()

    def query_similarity(text: str, study: int) -> float:
        texts.add(text)
        return _rank_similarity(text, study)

    nl = _similar_study_engine(frontend, query_similarity)
    nl.add_tuple_set(
        [("t1", "a text"), ("t2", "another text")] +
        [(f"u{i}", f"unrelated text {i}") for i in range(20)],
        name="definition",
    )
    res = nl.execute_datalog_program(
        "task(t) :- term(t, 'stroop task')\n"
        f"uses(s) :- {body}\n"
        "ans(count(s)) :- uses(s)"
    )
    assert res.as_pandas_dataframe().values.tolist() == [[4]]
    assert texts == {"a text"}


@pytest.mark.parametrize("frontend", [NeurolangDL, NeurolangPDL])
def test_frontend_similar_study_when_magic_sets_are_unstratified(frontend):
    """
    Bound, ``u`` would be in a cycle with a negation of the rewritten
    program, and unbound too, through ``t``, shared with ``v``: the
    rewrite was refused, and ``similar_study`` solved unbound
    (UnboundVariablesError). The predicates that need bindings are
    bound, the others left unbound.
    """
    nl = _similar_study_engine(frontend)
    nl.add_tuple_set([(1, "a"), (1, "c"), (2, "c")], name="lab")
    nl.add_tuple_set([(1, 0), (1, 1), (2, 0), (2, 1)], name="ep")
    nl.add_tuple_set([(1, 4), (2, 5), (1, 1)], name="in_study")
    res = nl.execute_datalog_program(
        "t(x, r) :- lab(x, r)\n"
        "u(x) :- ep(x, 0), t(x, 'a'), ~t(x, 'b')\n"
        "v(x) :- ep(x, 0), t(x, 'c')\n"
        "b(x, 'U') :- u(x)\n"
        "b(x, 'V') :- v(x)\n"
        "end(k, x, e) :- b(x, k), ep(x, e)\n"
        "pair(k, x) :- end(k, x, 0), end(k, x, 1)\n"
        "about(s) :- term(c, 'stroop task'), definition(c, d), "
        "similar_study(d, s)\n"
        "co(k, s) :- pair(k, x), in_study(x, s), about(s)\n"
        "ans(k, count(s)) :- co(k, s)"
    )
    assert sorted(res.as_pandas_dataframe().values.tolist()) == [
        ["U", 1], ["V", 2]
    ]


@pytest.mark.parametrize("frontend", [NeurolangDL, NeurolangPDL])
def test_frontend_rewrite_not_seeded_only_when_it_binds_more(
    frontend, monkeypatch
):
    """
    Not seeding from the stored relations only binds less: when the
    seeded rewrite binds nothing, the query is solved as it is, without
    trying the rewrite again unseeded.
    """
    seeds = []

    def recording(query, datalog, seed_from_edb=True):
        seeds.append(seed_from_edb)
        return magic_sets.magic_rewrite(query, datalog, seed_from_edb)

    for module in (probabilistic_frontend, query_resolution_datalog):
        monkeypatch.setattr(module, "magic_rewrite", recording)
    nl = frontend()
    nl.add_tuple_set([("1", "a"), ("2", "b")], name="r")
    res = nl.execute_datalog_program("ans(x, y) :- r(x, y)")
    assert len(res.as_pandas_dataframe()) == 2
    assert seeds == [True]


@pytest.mark.parametrize("program", [
    "p_similar(q, PROB) :- similar_study(q, s) & selected_study(s)\n"
    "texts(d, p) :- definition(t, d), p_similar(d, p)\n"
    "ans(d, p) :- texts(d, p)",
    "p_a(l, q, PROB) :- (act(s, l) & selected_study(s)) // "
    "(similar_study(q, s) & selected_study(s))\n"
    "lift(q, l, r) :- p_a(l, q, r)\n"
    "texts(d, p) :- definition(t, d), lift(d, 'a', p)\n"
    "ans(d, p) :- texts(d, p)",
])
def test_frontend_similar_study_bound_through_probabilities(program):
    """The text joined from a stored relation without a constant reaches
    ``similar_study`` through the probabilistic queries using it."""
    nl = _similar_study_engine()
    nl.add_uniform_probabilistic_choice_over_set(
        [(s,) for s in range(6)], name="selected_study"
    )
    res = nl.execute_datalog_program(program)
    # studies 2 to 5 are similar; one of them, 3, is about a
    assert dict(res.as_pandas_dataframe().values) == pytest.approx(
        {"a text": 4 / 6 if "p_similar" in program else 1 / 4,
         "another text": 4 / 6 if "p_similar" in program else 1 / 4}
    )


def _rewrite_outcomes(monkeypatch):
    """The exception class name, or None, of each rewrite attempt."""
    from nl_core.datalog import magic_sets
    outcomes = []
    rewrite = magic_sets._magic_rewrite

    def recorded(*args):
        try:
            result = rewrite(*args)
        except Exception as error:
            outcomes.append(type(error).__name__)
            raise
        outcomes.append(None)
        return result

    monkeypatch.setattr(magic_sets, "_magic_rewrite", recorded)
    return outcomes


@pytest.mark.parametrize("frontend", [NeurolangDL, NeurolangPDL])
def test_frontend_solves_unrewritten_a_recursion_of_the_rewrite(
    frontend, monkeypatch
):
    """
    ``a`` reads ``p`` then ``q``, ``b`` ``q`` then ``p``: bound, they share
    adorned and magic predicates which depend on each other, a recursion
    the program does not have. The query is solved unrewritten, after one
    rewrite: not again without seeding from the stored relations.
    """
    nl = frontend()
    nl.add_tuple_set([(0, 0), (0, 1), (1, 2)], name="ep")
    nl.add_tuple_set([(0,), (1,)], name="eq")
    nl.execute_datalog_program(
        "p(x, y) :- ep(x, y)\n"
        "q(x) :- eq(x)\n"
        "a(x) :- p(x, y), q(y)\n"
        "b(x) :- q(x), p(x, y)\n"
        "c(x) :- a(x), b(x)"
    )
    outcomes = _rewrite_outcomes(monkeypatch)
    res = nl.execute_datalog_program("ans(y) :- c(0), ep(0, y)")
    assert outcomes == ["NewRecursionInMagicSetsRewriteError"]
    assert set(res.as_pandas_dataframe().iloc[:, 0]) == {0, 1}


def test_frontend_keeps_the_rewrite_a_similarity_needs(monkeypatch):
    """
    The similarity of a query text to a study is a relation over the
    texts the rules using it bind (``q`` of ``sim`` is bound only through
    the head), and the magic predicate of its percentile depends on it:
    the rewrite has a recursion the program does not have. Unrewritten,
    the program has no answer: the rewrite binding only ``sim`` and
    ``cutoff`` is kept, recursion and all.
    """
    nl = NeurolangPDL()
    studies = range(6)
    nl.add_tuple_set([(s,) for s in studies], name="study")
    nl.add_uniform_probabilistic_choice_over_set(
        [(s,) for s in studies], name="selected_study"
    )
    nl.add_tuple_set([(0, "a"), (1, "a"), (3, "a"), (4, "b")], name="act")
    nl.add_symbol(_rank_similarity, name="similarity")
    nl.execute_datalog_program(
        "sim(q, s, v) :- study(s) & (v == similarity(q, s))\n"
        "cutoff(q, percentile(v, 50)) :- sim(q, s, v)\n"
        "similar_study(q, s) :- sim(q, s, v) & cutoff(q, c) & (v >= c)"
    )
    outcomes = _rewrite_outcomes(monkeypatch)
    res = nl.execute_datalog_program(
        "close(s) :- similar_study('a text', s), selected_study(s)\n"
        "both(s) :- act(s, 'a'), close(s)\n"
        "ans(p) :- PROB[both(s) // close(s)] = p"
    )
    assert outcomes == ["NewRecursionInMagicSetsRewriteError", None]
    similarities = [_rank_similarity("a text", s) for s in studies]
    similar = {
        s for s, v in zip(studies, similarities)
        if v >= np.percentile(similarities, 50)
    }
    expected = len(similar & {0, 1, 3}) / len(similar)
    assert res.as_pandas_dataframe().p.tolist() == pytest.approx([expected])


@pytest.mark.parametrize("close", [
    "close(s) :- similar_study('a text', s), selected_study(s)\n",
    "close(s) :- definition('x', q), similar_study(q, s), "
    "selected_study(s)\n",
    "text(q) :- definition('x', q)\n"
    "close(s) :- text(q), similar_study(q, s), selected_study(s)\n",
])
def test_frontend_binds_a_similarity_with_a_large_relation(close):
    """
    With a stored relation of more than ``[MAGIC_SETS] seed_rows`` tuples,
    the rewrite is first tried with the bindings of the constants and the
    heads alone; the query text of ``similar_study`` is still bound,
    whether it is a constant, read from a stored relation or from a rule
    (``text``), which binds as the argument ``similar_study`` needs.
    """
    nl = NeurolangPDL()
    studies = range(6)
    nl.add_tuple_set([(s,) for s in studies], name="study")
    nl.add_uniform_probabilistic_choice_over_set(
        [(s,) for s in studies], name="selected_study"
    )
    nl.add_tuple_set([(0, "a"), (1, "a"), (3, "a"), (4, "b")], name="act")
    nl.add_tuple_set([("x", "a text")], name="definition")
    nl.add_tuple_set([(s, float(s)) for s in range(200)], name="val")
    nl.add_tuple_set([("k", -1.0), ("m", 10.0)], name="box")
    nl.add_symbol(_rank_similarity, name="similarity")
    nl.execute_datalog_program(
        "sim(q, s, v) :- study(s) & (v == similarity(q, s))\n"
        "cutoff(q, percentile(v, 50)) :- sim(q, s, v)\n"
        "similar_study(q, s) :- sim(q, s, v) & cutoff(q, c) & (v >= c)\n"
        "above(s, b) :- val(s, v), box(b, w), (v > w)"
    )
    res = nl.execute_datalog_program(
        close + "both(s) :- act(s, 'a'), close(s), above(s, 'k')\n"
        "ans(p) :- PROB[both(s) // close(s)] = p"
    )
    similarities = [_rank_similarity("a text", s) for s in studies]
    similar = {
        s for s, v in zip(studies, similarities)
        if v >= np.percentile(similarities, 50)
    }
    expected = len(similar & {0, 1, 3}) / len(similar)
    assert res.as_pandas_dataframe().p.tolist() == pytest.approx([expected])
