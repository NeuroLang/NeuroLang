from ....datalog import Fact
from ....expressions import Constant, Symbol
from ....logic import Conjunction, Implication, Union
from .. import testing
from ..gm_provenance_solver import solve_succ_query
from ..program import CPLogicProgram

P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
Z = Symbol("Z")
H = Symbol("H")
x = Symbol("x")
y = Symbol("y")

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
    code = Union((Fact(Q(a)), Fact(Q(b)), Implication(P(x), Q(x)),))
    cpl_program = CPLogicProgram()
    cpl_program.walk(code)
    query_pred = P(x)
    result = solve_succ_query(query_pred, cpl_program)
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
    code = Union(())
    cpl_program = CPLogicProgram()
    cpl_program.walk(code)
    cpl_program.add_probabilistic_facts_from_tuples(
        P, {(0.7, "a"), (0.8, "b")}
    )
    result = solve_succ_query(P(x), cpl_program)
    assert len(result.value) == 2
    assert set(result.value) == {(0.7, "a"), (0.8, "b")}


def test_conjunction_bernoulli_program():
    code = Union((Implication(Z(x), Conjunction((P(x), Q(x), R(x)))),))
    probfacts_sets = {
        P: {(1.0, "a"), (0.5, "b")},
        Q: {(0.9, "b"), (0.1, "c")},
        R: {(0.9, "b"), (0.1, "c")},
    }
    cpl_program = CPLogicProgram()
    cpl_program.walk(code)
    for pred_symb, pfact_set in probfacts_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(pred_symb, pfact_set)
    result = solve_succ_query(Z(x), cpl_program)
    assert len(result.value) == 1
    assert set(result.value) == {(0.5 * 0.9 * 0.9, "b")}


def test_multi_level_conjunctive_program():
    """
    We consider the program

           P(a) : 0.2 <- T
           Q(a) : 0.9 <- T
        R(a, a) : 0.1 <- T
        R(a, b) : 0.5 <- T
                 Z(x) <- P(x), Q(x)
              H(x, y) <- Z(x), R(x, y)

    And expect the prov set resulting from the
    marginalisation of H(x, y) to be

        _p_   | x  | y
        ======|====|===
        0.018 | a  | a
        0.09  | a  | b

    """
    probfacts_sets = {
        P: {(0.2, "a")},
        Q: {(0.9, "a")},
        R: {(0.1, "a", "a"), (0.5, "a", "b")},
    }
    code = Union(
        (
            Implication(Z(x), Conjunction((P(x), Q(x)))),
            Implication(H(x, y), Conjunction((Z(x), R(x, y)))),
        )
    )
    cpl_program = CPLogicProgram()
    cpl_program.walk(code)
    for pred_symb, pfact_set in probfacts_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(pred_symb, pfact_set)
    result = solve_succ_query(H(x, y), cpl_program)
    expected = testing.make_prov_set(
        [(0.2 * 0.9 * 0.1, "a", "a"), (0.2 * 0.9 * 0.5, "a", "b"),],
        ("_p", "x", "y"),
    )
    assert testing.eq_prov_relations(result, expected)


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
