from typing import AbstractSet

from ....datalog import Fact
from ....expressions import Constant, ExpressionBlock, Symbol
from ....logic import Conjunction, Implication
from ...expressions import ProbabilisticChoice
from ....relational_algebra import NamedRelationalAlgebraFrozenSet
from ....relational_algebra_provenance import ProvenanceAlgebraSet
from ..gm_provenance_solver import solve_succ_query
from .. import testing

P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
Z = Symbol("Z")
x = Symbol("x")

a = Constant("a")
b = Constant("b")


def test_deterministic_program():
    """
    We define the program

        P(x) <- Q(x)

    And we expect the provenance set resulting from the
    marginalisation of P(x) to be

        _p_ | x
        ====|===
        1.0 | a
        1.0 | b

    """
    code = ExpressionBlock((Fact(Q(a)), Fact(Q(b)), Implication(P(x), Q(x)),))
    query_pred = P(x)
    result = solve_succ_query(query_pred, code)
    assert len(result.value) == 2
    assert set(result.value) == {("a", 1.0), ("b", 1.0)}


def test_simple_bernoulli_program():
    """
    We define the program

        P(a) : 0.7 <- T
        P(b) : 0.8 <- T

    And expect the provenance set resulting from the
    marginalisation of P(x) to be

        _p_ | x
        ====|===
        0.7 | a
        0.8 | b

    """
    code = ExpressionBlock(())
    probfacts_sets = {P: {(0.7, "a"), (0.8, "b")}}
    result = solve_succ_query(P(x), code, probfacts_sets=probfacts_sets)
    assert len(result.value) == 2
    assert set(result.value) == {(0.7, "a"), (0.8, "b")}


def test_conjunction_bernoulli_program():
    code = ExpressionBlock(
        (Implication(Z(x), Conjunction((P(x), Q(x), R(x)))),)
    )
    probfacts_sets = {
        P: {(1.0, "a"), (0.5, "b")},
        Q: {(0.9, "b"), (0.1, "c")},
        R: {(0.9, "b"), (0.1, "c")},
    }
    result = solve_succ_query(Z(x), code, probfacts_sets=probfacts_sets)
    assert len(result.value) == 1
    assert set(result.value) == {(0.5 * 0.9 * 0.9, "b")}


def test_simple_existential():
    """
    We define the following program

        P(a) : 0.2 v P(b) : 0.8 <- T
                           Q(a) <- ∃x, P(x)

    Whose equivalent graphical model is

             P(a) ----|
              ^       |
        c_P --|       |-> Q(a)
              v       |
             P(b) ----|

    We expect the following to hold

        - Pr[P(a)] = 0.2
        - Pr[P(b)] = 0.8
        - Pr[Q(a)] = 1.0

    """
    pass
