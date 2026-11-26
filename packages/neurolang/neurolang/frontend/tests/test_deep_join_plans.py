"""
Queries of more than 16 independent conjuncts or disjuncts, whose plans are
trees of chunks of joins (``join_in_chunks``), against the products of the
probabilities of their independent parts.
"""
from functools import reduce

import pytest

from nl_core.probabilistic import dalvi_suciu_lift
from nl_expr.config import config
from ...frontend import NeurolangDL, NeurolangPDL

N = 40
P = {i: 0.2 + 0.015 * i for i in range(N)}


def _engine(lifted_only=True, domain=(0, 1)):
    if lifted_only:
        nl = NeurolangPDL(
            probabilistic_solvers=(dalvi_suciu_lift.solve_succ_query,),
            probabilistic_marg_solvers=(dalvi_suciu_lift.solve_marg_query,),
        )
    else:
        nl = NeurolangPDL()
    for i in range(N):
        nl.add_probabilistic_facts_from_tuples(
            [(P[i] * (1 + x) / 2, x) for x in domain], name=f"a{i}"
        )
    return nl


def _solve(nl, program):
    nl.execute_datalog_program(program)
    df = nl.solve_all()["q"].as_pandas_dataframe()
    return {tuple(row[:-1]): row[-1] for row in df.itertuples(index=False)}


def test_independent_conjunction_of_many_atoms():
    found = _solve(_engine(), "q(x, PROB) :- " + ", ".join(
        f"a{i}(x)" for i in range(N)
    ))
    for x in (0, 1):
        expected = reduce(lambda a, b: a * b, (
            P[i] * (1 + x) / 2 for i in range(N)
        ))
        assert found[(x,)] == pytest.approx(expected, rel=1e-9)


def test_independent_disjunction_of_many_atoms():
    found = _solve(_engine(), "q(x, PROB) :- " + " | ".join(
        f"a{i}(x)" for i in range(N)
    ))
    for x in (0, 1):
        expected = 1 - reduce(lambda a, b: a * b, (
            1 - P[i] * (1 + x) / 2 for i in range(N)
        ))
        assert found[(x,)] == pytest.approx(expected, rel=1e-9)



@pytest.mark.parametrize("contractions", [False, True])
def test_independent_conjunction_of_atoms_on_distinct_variables(
    contractions
):
    """The independent join of 40 components sharing no column is the
    product of their probabilities, solved with or without contractions."""
    previous = config.get("PROBABILISTIC_SOLVER", "contractions")
    config.set("PROBABILISTIC_SOLVER", "contractions", str(contractions))
    try:
        found = _solve(_engine(), "q(PROB) :- " + ", ".join(
            f"a{i}(x{i})" for i in range(N)
        ))
    finally:
        config.set("PROBABILISTIC_SOLVER", "contractions", previous)
    expected = reduce(lambda a, b: a * b, (
        1 - reduce(lambda a, b: a * b, (1 - P[i] * (1 + x) / 2 for x in (0, 1)))
        for i in range(N)
    ))
    assert found[()] == pytest.approx(expected, rel=1e-9)


def _datalog_with_hundreds_of_atoms(shape, n):
    """A one-row relation per atom, the atoms sharing no variable, or a
    centre with the variables of leaves that share one with it only."""
    nl = NeurolangDL()
    if shape == "disjoint":
        nl.add_tuple_set([(1,)], name="r")
        body = ", ".join(f"r(x{i})" for i in range(n))
    else:
        nl.add_tuple_set([tuple(range(n))], name="c")
        nl.add_tuple_set([(i,) for i in range(n)], name="l")
        body = (
            "c(" + ", ".join(f"x{i}" for i in range(n)) + "), "
            + ", ".join(f"l(x{i})" for i in range(n))
        )
    return nl, f"q(x0) :- {body}\nans(x0) :- q(x0)"


@pytest.mark.parametrize("shape", ["disjoint", "star"])
def test_conjunction_of_hundreds_of_atoms_sharing_nothing_with_the_last(shape):
    """The chain of joins of such a conjunction was as deep as its atoms
    are many, and walked with a RecursionError at about 330."""
    nl, program = _datalog_with_hundreds_of_atoms(shape, 335)

    result = nl.execute_datalog_program(program)

    assert [tuple(row) for row in result] == [(0 if shape == "star" else 1,)]
