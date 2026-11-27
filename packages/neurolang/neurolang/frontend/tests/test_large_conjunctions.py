"""
Probabilistic queries of hundreds of conjuncts, whose preparation used to
exceed the recursion limit: the logic transformations (distribution of the
conjunctions, standardisation of the quantified variables) and the
selection of the shattering recursed on the formulas of a conjunction one
after the other.
"""
import pytest

from nl_core.probabilistic import dalvi_suciu_lift
from ...frontend import NeurolangPDL

PROBABILITIES = {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6}


def _engine():
    nl = NeurolangPDL(
        probabilistic_solvers=(dalvi_suciu_lift.solve_succ_query,),
        probabilistic_marg_solvers=(dalvi_suciu_lift.solve_marg_query,),
    )
    nl.add_probabilistic_facts_from_tuples(
        [(p, x) for x, p in PROBABILITIES.items()], name="r"
    )
    return nl


def _probability(excluded, others):
    """P[∃x r(x) ∧ x ≠ c for each c of `excluded` and of `others`]."""
    nl = _engine()
    comparisons = [f"(x != {c})" for c in (*excluded, *others)]
    nl.execute_datalog_program(
        f"q(p) :- PROB[r(x), {', '.join(comparisons)}] = p"
    )
    return nl.solve_all()["q"].as_pandas_dataframe().values[0][-1]


@pytest.mark.parametrize("others", (10, 300))
def test_many_comparisons_in_a_probability(others):
    """Only the tuples other than those of x = 2 and x = 3 count: the
    probability of at least one of those left, 1 - (1 - .9)(1 - .6)."""
    probability = _probability((2, 3), range(1000, 1000 + others))
    assert probability == pytest.approx(1 - (1 - 0.9) * (1 - 0.6))


def test_many_comparisons_that_remove_every_tuple():
    nl = _engine()
    comparisons = [f"(x != {c})" for c in range(1, 5)]
    comparisons += [f"(x != {c})" for c in range(1000, 1300)]
    nl.execute_datalog_program(
        f"q(p) :- PROB[r(x), {', '.join(comparisons)}] = p"
    )
    result = nl.solve_all().get("q")
    assert result is None or len(result.as_pandas_dataframe()) == 0
