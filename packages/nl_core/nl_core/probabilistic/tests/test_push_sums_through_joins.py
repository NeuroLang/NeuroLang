"""
Tests for PushSumsThroughJoins: the projections of the provenance plan of a
safe query, sums of probabilities, pushed through its ordered joins,
products of probabilities, give the same probabilities.
"""

import itertools
from typing import AbstractSet

import numpy as np
import pytest

from nl_expr.expressions import Constant
from ...relational_algebra import (
    NamedRelationalAlgebraFrozenSet,
    NaturalJoin,
    Projection,
    str2columnstr_constant,
)
from ...relational_algebra_provenance import NaturalJoinInverse
from .. import small_dichotomy_theorem_based_solver as sdt
from ..query_resolution import PushSumsThroughJoins


def _relation(*columns):
    """A named relation of the *columns*, one row of zeros."""
    return Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(columns, [(0,) * len(columns)])
    )


def _project(relation, *columns):
    return Projection(
        relation, tuple(str2columnstr_constant(c) for c in columns)
    )


def test_a_projection_is_pushed_through_ordered_joins():
    """π[t]((voxel(v) ⋈ activated(v, s)) ⋈ terms(s, t)): v is summed out
    before the join with terms, s once joined with it."""
    voxel, activated, terms = (
        _relation("v"), _relation("v", "s"), _relation("s", "t")
    )
    query = _project(NaturalJoin(NaturalJoin(voxel, activated), terms), "t")
    pushed = PushSumsThroughJoins().walk(query)
    assert pushed == _project(
        NaturalJoin(
            _project(NaturalJoin(voxel, activated), "s"), terms
        ),
        "t",
    )


def test_a_projection_keeping_every_column_is_left_alone():
    left, right = _relation("v", "s"), _relation("s", "t")
    query = _project(NaturalJoin(left, right), "v", "s", "t")
    assert PushSumsThroughJoins().walk(query) is query


def test_a_division_is_not_a_product():
    """A sum does not distribute over the division of NaturalJoinInverse:
    its projection stays above it."""
    left, right = _relation("v", "s"), _relation("s")
    query = _project(NaturalJoinInverse(left, right), "s")
    assert PushSumsThroughJoins().walk(query) is query


@pytest.fixture
def without_push(monkeypatch):
    """Solve with the small dichotomy theorem solver, but without pushing
    the sums through the joins."""
    def solver(*args, **kwargs):
        kwargs["push_sums"] = False
        return original(*args, **kwargs)

    original = sdt.generate_provenance_query_solver
    monkeypatch.setattr(sdt, "generate_provenance_query_solver", solver)


def _decoding_engine(probabilities):
    """The program decoding an image: studies 10 and 11, of pain, activate
    the voxel (1, 2, 3), and 12, of vision, (0, 0, 0); each voxel of the
    image is active with its probability, and one is chosen at random."""
    from neurolang.frontend import NeurolangPDL

    nl = NeurolangPDL()
    nl.add_tuple_set(
        [(1, 2, 3, 10), (1, 2, 3, 11), (0, 0, 0, 12)], name="activated"
    )
    nl.add_tuple_set(
        [(10, "pain"), (11, "pain"), (12, "vision")], name="term_in_study"
    )
    nl.add_uniform_probabilistic_choice_over_set(
        [(10,), (11,), (12,)], name="selected_study"
    )
    nl.add_tuple_set(probabilities, name="image_probability")
    return nl


DECODING = """
image_active(i, j, k) :: p :- image_probability(i, j, k, p)
image_size(sum(o)) :- image_probability(i, j, k, p), (o == 1.0)
image_voxel(i, j, k) ^ q :- image_probability(i, j, k, p), image_size(n), (q == 1.0 / n)
image_hit(s) :- image_voxel(i, j, k), image_active(i, j, k), activated(i, j, k, s)
p_image_given_term(t, PROB) :- (image_hit(s) & selected_study(s)) // (term_in_study(s, t) & selected_study(s))
ans(t, p) :- p_image_given_term(t, p)
"""  # noqa: E501


def _probabilities(nl, program):
    result = nl.execute_datalog_program(program).as_pandas_dataframe()
    return {tuple(row[:-1]): row[-1]
            for row in result.itertuples(index=False, name=None)}


def test_the_decoding_of_an_image_by_its_voxels_as_facts():
    """P(hit | s) = 1/2 p_v over the voxel v that s activates: 0.45 for
    the studies of pain, of (1, 2, 3) of 0.9, and 0.2 for that of vision."""
    nl = _decoding_engine([(1, 2, 3, 0.9), (0, 0, 0, 0.4)])
    probabilities = _probabilities(nl, DECODING)
    assert probabilities == pytest.approx({("pain",): 0.45, ("vision",): 0.2})


def test_the_decoding_is_that_without_the_push(without_push):
    nl = _decoding_engine([(1, 2, 3, 0.9), (0, 0, 0, 0.4)])
    assert _probabilities(nl, DECODING) == pytest.approx(
        {("pain",): 0.45, ("vision",): 0.2})


def test_rows_of_a_deterministic_relation_are_not_counted_twice():
    """Two rows of d for x = 1: summed out before or after the join, the
    probability that a chosen x has a row is still 1."""
    from neurolang.frontend import NeurolangPDL

    nl = NeurolangPDL()
    nl.add_uniform_probabilistic_choice_over_set([(1,), (2,)], name="c")
    nl.add_tuple_set([(1, "a"), (1, "b"), (2, "a")], name="d")
    assert _probabilities(nl, "ans(PROB) :- c(x), d(x, y)") == \
        pytest.approx({(): 1.0})
    assert _probabilities(nl, "ans(y, PROB) :- c(x), d(x, y)") == \
        pytest.approx({("a",): 1.0, ("b",): 0.5})


RANDOM_QUERIES = (
    "ans(t, PROB) :- v(x), f(x), d(x, s), c(s), e(s, t)",
    "ans(PROB) :- v(x), f(x), d(x, s), c(s)",
    "h(s) :- v(x), f(x), d(x, s)\n"
    "ans(t, PROB) :- (h(s) & c(s)) // (e(s, t) & c(s))",
    "ans(s, PROB) :- v(x), d(x, s), e(s, t)",
)


def _random_engine(seed):
    """A choice v of a voxel x, facts f of x, a choice c of a study s, and
    deterministic d(x, s) and e(s, t), several rows for some values."""
    from neurolang.frontend import NeurolangPDL

    rng = np.random.default_rng(seed)
    nl = NeurolangPDL()
    weights = rng.random(4) + 0.1
    nl.add_probabilistic_choice_from_tuples(
        [(float(w / weights.sum()), x) for x, w in enumerate(weights)],
        name="v",
    )
    nl.add_probabilistic_facts_from_tuples(
        [(float(rng.random()), x) for x in range(4)], name="f"
    )
    studies = rng.random(5) + 0.1
    nl.add_probabilistic_choice_from_tuples(
        [(float(w / studies.sum()), s) for s, w in enumerate(studies)],
        name="c",
    )
    nl.add_tuple_set(
        [(x, s) for x, s in itertools.product(range(4), range(5))
         if rng.random() < 0.5] or [(0, 0)], name="d"
    )
    nl.add_tuple_set(
        [(s, t) for s, t in itertools.product(range(5), "ab")
         if rng.random() < 0.6] or [(0, "a")], name="e"
    )
    return nl


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("query", RANDOM_QUERIES)
def test_the_probabilities_are_those_without_the_push(
    seed, query, monkeypatch
):
    """Random programs of choices, facts and deterministic relations: the
    same probabilities with the sums pushed through the joins as without."""
    pushed = _probabilities(_random_engine(seed), query)
    original = sdt.generate_provenance_query_solver

    def solver(*args, **kwargs):
        kwargs["push_sums"] = False
        return original(*args, **kwargs)

    monkeypatch.setattr(sdt, "generate_provenance_query_solver", solver)
    unpushed = _probabilities(_random_engine(seed), query)
    assert pushed.keys() == unpushed.keys()
    for key, value in unpushed.items():
        assert np.isclose(pushed[key], value), key


def test_the_random_programs_push_sums(monkeypatch):
    """The random programs above are solved by the small dichotomy
    theorem, and some of their sums are pushed through joins: they test
    the push."""
    from .. import query_resolution

    narrowed = []
    narrow = query_resolution._project_if_narrower

    def counting(operand, operand_columns, needed):
        result = narrow(operand, operand_columns, needed)
        if result is not operand:
            narrowed.append(result)
        return result

    monkeypatch.setattr(query_resolution, "_project_if_narrower", counting)
    for query in RANDOM_QUERIES[:3]:
        before = len(narrowed)
        _probabilities(_random_engine(0), query)
        assert len(narrowed) > before, query
