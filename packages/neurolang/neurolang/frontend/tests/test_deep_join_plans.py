"""
Queries of more than 16 independent conjuncts or disjuncts, whose plans are
trees of chunks of joins (``join_in_chunks``), against the products of the
probabilities of their independent parts.
"""
from functools import reduce

import pytest

from nl_core.probabilistic import dalvi_suciu_lift
from ...frontend import NeurolangPDL

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

