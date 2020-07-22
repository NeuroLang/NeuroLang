import numpy as np
import pytest

from ....datalog import Fact
from ....expressions import Constant, Symbol
from ....logic import Conjunction, Implication, Union
from ....relational_algebra import RenameColumn
from ...expressions import PROB, ProbabilisticQuery
from .. import testing
from ..gm_provenance_solver import solve_marg_query, solve_succ_query
from ..problog_solver import solve_succ_all as problog_solve_succ_all
from ..program import CPLogicProgram

P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
Z = Symbol("Z")
H = Symbol("H")
A = Symbol("A")
B = Symbol("B")
C = Symbol("C")
x = Symbol("x")
y = Symbol("y")
z = Symbol("z")

a = Constant("a")
b = Constant("b")


def test_deterministic():
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
    expected = testing.make_prov_set([(1.0, "a"), (1.0, "b")], ("_p_", "x"))
    assert testing.eq_prov_relations(result, expected)


def test_deterministic_conjunction_varying_arity():
    code = Union(
        (
            Fact(Q(a, b)),
            Fact(P(a)),
            Fact(P(b)),
            Implication(Z(x, y), Conjunction((Q(x, y), P(x), P(y)))),
        )
    )
    cpl_program = CPLogicProgram()
    cpl_program.walk(code)
    query_pred = Z(x, y)
    result = solve_succ_query(query_pred, cpl_program)
    expected = testing.make_prov_set([(1.0, "a", "b")], ("_p_", "x", "y"))
    assert testing.eq_prov_relations(result, expected)


def test_simple_bernoulli():
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
    expected = testing.make_prov_set([(0.7, "a"), (0.8, "b")], ("_p_", "x"))
    assert testing.eq_prov_relations(result, expected)


def test_bernoulli_conjunction():
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


def test_multi_level_conjunction():
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
        ("_p_", "x", "y"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_intertwined_conjunctions_and_probfacts():
    """
    We consider the program

        P(a) : 0.8  <-  T
        C(a) : 0.5  <-  T
        C(b) : 0.9  <-  T
              A(x)  <-  B(x), C(x)
              B(x)  <-  P(x)
              Z(x)  <-  A(x), B(x), C(x)

    And expect SUCC[ Z(x) ] to yield the provenance relation

        _p_ | x
        ====|===
        0.4 | a

    """
    cpl_code = Union(
        (
            Implication(A(x), Conjunction((B(x), C(x)))),
            Implication(B(x), P(x)),
            Implication(Z(x), Conjunction((A(x), B(x), C(x)))),
        )
    )
    cpl = CPLogicProgram()
    cpl.walk(cpl_code)
    cpl.add_probabilistic_facts_from_tuples(P, {(0.8, "a")})
    cpl.add_probabilistic_facts_from_tuples(C, {(0.5, "a"), (0.9, "b")})
    result = solve_succ_query(Z(y), cpl)
    expected = testing.make_prov_set([(0.8 * 0.5, "a")], ("_p_", "y"))
    assert testing.eq_prov_relations(result, expected)


def test_simple_probchoice():
    pchoice_as_sets = {P: {(0.2, "a"), (0.8, "b")}}
    cpl_program = CPLogicProgram()
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    qpred = P(x)
    exp, result = testing.inspect_resolution(qpred, cpl_program)
    expected = testing.make_prov_set([(0.2, "a"), (0.8, "b"),], ("_p_", "x"),)
    assert testing.eq_prov_relations(result, expected)


def test_mutual_exclusivity():
    pchoice_as_sets = {P: {(0.2, "a"), (0.8, "b")}}
    pfact_sets = {Q: {(0.5, "a", "b")}}
    code = Union((Implication(Z(x, y), Conjunction((P(x), P(y), Q(x, y)))),))
    cpl_program = CPLogicProgram()
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    for pred_symb, pfact_set in pfact_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(pred_symb, pfact_set)
    cpl_program.walk(code)
    qpred = Z(x, y)
    exp, result = testing.inspect_resolution(qpred, cpl_program)
    assert isinstance(exp, RenameColumn)
    assert isinstance(exp.relation, RenameColumn)
    expected = testing.make_prov_set([], ("_p_", "x", "y"))
    assert testing.eq_prov_relations(result, expected)


def test_multiple_probchoices_mutual_exclusivity():
    pchoice_as_sets = {
        P: {(0.2, "a"), (0.8, "b")},
        Q: {(0.5, "a", "b"), (0.4, "b", "c"), (0.1, "b", "b")},
    }
    rule = Implication(Z(x, y), Conjunction((P(x), Q(y, y))))
    code = Union((rule,))
    cpl_program = CPLogicProgram()
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    cpl_program.walk(code)
    qpred = Z(x, y)
    exp, result = testing.inspect_resolution(qpred, cpl_program)
    assert isinstance(exp, RenameColumn)
    assert isinstance(exp.relation, RenameColumn)
    expected = testing.make_prov_set(
        [(0.2 * 0.1, "a", "b"), (0.8 * 0.1, "b", "b")], ("_p_", "x", "y")
    )
    assert testing.eq_prov_relations(result, expected)


@pytest.mark.skip
def test_large_probabilistic_choice():
    n = int(10e3)
    with testing.temp_seed(42):
        probs = np.random.rand(n)
    probs = probs / probs.sum()
    pchoice_as_sets = {P: {(float(prob), i) for i, prob in enumerate(probs)}}
    pfact_sets = {Q: {(0.5, 0, 0), (0.5, 0, 1)}}
    code = Union((Implication(Z(x, y), Conjunction((P(x), Q(x, y)))),))
    cpl_program = CPLogicProgram()
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    for pred_symb, pfact_set in pfact_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(pred_symb, pfact_set)
    cpl_program.walk(code)
    qpred = Z(x, y)
    result = solve_succ_query(qpred, cpl_program)
    expected = testing.make_prov_set(
        [(0.5 * probs[0], 0, 0), (0.5 * probs[0], 0, 1),], ("_p_", "x", "y")
    )
    assert testing.eq_prov_relations(result, expected)


def test_simple_existential():
    """
    We define the following program

        P(a, a) : 0.2 v P(a, b) : 0.8 <- T
                           Q(x) <- ∃y, P(x, y)

    We expect the following to hold

        - Pr[P(a, a)] = 0.2
        - Pr[P(a, b)] = 0.8
        - Pr[Q(a)] = 1.0

    """
    pchoice_as_sets = {P: {(0.2, "a", "a"), (0.8, "a", "b")}}
    code = Union((Implication(Q(x), P(x, y)),))
    cpl_program = CPLogicProgram()
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    cpl_program.walk(code)
    exp, result = testing.inspect_resolution(Q(x), cpl_program)
    expected = testing.make_prov_set([(1.0, "a")], ("_p_", "x"))
    assert testing.eq_prov_relations(result, expected)


def test_existential_in_conjunction():
    pchoice_as_sets = {
        P: {(0.2, "a", "b", "c"), (0.4, "b", "b", "c"), (0.4, "b", "a", "c")},
        Z: {(0.5, "b"), (0.5, "d")},
    }
    code = Union((Implication(Q(x), Conjunction((Z(y), P(x, y, z)))),))
    cpl_program = CPLogicProgram()
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    cpl_program.walk(code)
    exp, result = testing.inspect_resolution(Q(x), cpl_program)
    expected = testing.make_prov_set([(0.1, "a"), (0.2, "b")], ("_p_", "x"))
    assert testing.eq_prov_relations(result, expected)


def test_existential_alternative_variables():
    pchoice_as_sets = {
        P: {(0.8, "a", "b"), (0.1, "c", "d"), (0.1, "d", "e")},
    }
    pfact_sets = {
        Z: {(0.2, "a"), (0.7, "e")},
    }
    code = Union(
        (
            Fact(R(Constant[str]("a"))),
            Fact(R(Constant[str]("b"))),
            Implication(H(x), Conjunction((Z(y), P(y, x)))),
        )
    )
    cpl_program = CPLogicProgram()
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    for pred_symb, pfact_set in pfact_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(pred_symb, pfact_set)
    cpl_program.walk(code)
    qpred = H(z)
    result = solve_succ_query(qpred, cpl_program)
    expected = testing.make_prov_set([(0.2 * 0.8, "b")], ("_p_", "z"))
    assert testing.eq_prov_relations(result, expected)


def test_multilevel_existential():
    pchoice_as_sets = {
        P: {(0.5, "a", "b"), (0.5, "b", "c")},
        R: {(0.1, "a"), (0.4, "b"), (0.5, "c")},
        Q: {(0.9, "a"), (0.1, "c")},
        Z: {(0.1, "b"), (0.9, "c")},
    }
    code = Union(
        (
            Implication(H(x, y), Conjunction((R(x), Z(y)))),
            Implication(A(x), Conjunction((H(x, y), P(y, x)))),
            Implication(B(x), Conjunction((A(x), Q(y)))),
        )
    )
    cpl_program = CPLogicProgram()
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    cpl_program.walk(code)
    qpred = H(x, y)
    result = solve_succ_query(qpred, cpl_program)
    expected = testing.make_prov_set(
        [
            (0.1 * 0.1, "a", "b"),
            (0.1 * 0.9, "a", "c"),
            (0.4 * 0.1, "b", "b"),
            (0.4 * 0.9, "b", "c"),
            (0.5 * 0.1, "c", "b"),
            (0.5 * 0.9, "c", "c"),
        ],
        ("_p_", "x", "y"),
    )
    assert testing.eq_prov_relations(result, expected)
    qpred = B(z)
    exp, result = testing.inspect_resolution(qpred, cpl_program,)
    expected = testing.make_prov_set([(0.5 * 0.1 * 0.5, "c")], ("_p_", "z"),)
    assert testing.eq_prov_relations(result, expected)


@pytest.mark.skip("not implemented yet")
def test_repeated_antecedent_predicate_symbol():
    """
    We consider the simple program

        P(a) : 0.4  <-  T
        P(b) : 0.7  <-  T
           Q(x, y)  <-  P(x), P(y)

    Possible outcomes are

        { }                     with prob   (1 - 0.4) * (1 - 0.7)
        { P(a), Q(a, a) }       with prob   0.4 * (1 - 0.7)
        { P(b), Q(b, b) }       with prob   0.7 * (1 - 0.4)
        { P(a), P(b),
          Q(a, a), Q(a, b),     with prob   0.4 * 0.7
          Q(b, b), Q(b, a) }

    We expected the following provenance set to result from the
    succ query prob[Q(x, y)]?

        _p_                         | x | y
        ----------------------------|---|---
        0.4 * (1 - 0.7) + 0.4 * 0.7 | a | a
        0.7 * (1 - 0.4) + 0.4 * 0.7 | b | b
        0.4 * 0.7                   | a | b
        0.4 * 0.7                   | b | a

    """
    pfact_sets = {
        P: {(0.4, "a"), (0.7, "b")},
    }
    code = Union((Implication(Q(x, y), Conjunction((P(x), P(y)))),))
    cpl_program = CPLogicProgram()
    for pred_symb, pfact_set in pfact_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(pred_symb, pfact_set)
    cpl_program.walk(code)
    qpred = Q(x, y)
    result = solve_succ_query(qpred, cpl_program)
    expected = testing.make_prov_set(
        [
            (0.4 * (1 - 0.7) + 0.4 * 0.7, "a", "a"),
            (0.7 * (1 - 0.4) + 0.4 * 0.7, "b", "b"),
            (0.4 * 0.7, "a", "b"),
            (0.4 * 0.7, "b", "a"),
        ],
        ("_p_", "x", "y"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_fake_neurosynth():
    TermInStudy = Symbol("TermInStudy")
    ActivationReported = Symbol("ActivationReported")
    SelectedStudy = Symbol("SelectedStudy")
    TermAssociation = Symbol("TermAssociation")
    Activation = Symbol("Activation")
    s = Symbol("s")
    t = Symbol("t")
    v = Symbol("v")
    code = Union(
        (
            Implication(
                TermAssociation(t),
                Conjunction([TermInStudy(t, s), SelectedStudy(s)]),
            ),
            Implication(
                Activation(v),
                Conjunction([ActivationReported(v, s), SelectedStudy(s)]),
            ),
        )
    )
    pfact_sets = {
        TermInStudy: {
            (0.001, "memory", "1"),
            (0.002, "memory", "2"),
            (0.015, "visual", "2"),
            (0.004, "memory", "3"),
            (0.005, "visual", "4"),
            (0.0001, "memory", "4"),
            (0.01, "visual", "5"),
        },
        ActivationReported: {
            (1.0, "v1", "1"),
            (1.0, "v2", "1"),
            (1.0, "v3", "2"),
            (1.0, "v1", "3"),
            (1.0, "v1", "4"),
        },
    }
    pchoice_as_sets = {
        SelectedStudy: {
            (0.2, "1"),
            (0.2, "2"),
            (0.2, "3"),
            (0.2, "4"),
            (0.2, "5"),
        }
    }
    cpl_program = CPLogicProgram()
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    for pred_symb, pfact_set in pfact_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(pred_symb, pfact_set)
    cpl_program.walk(code)
    qpred = TermAssociation(t)
    result = solve_succ_query(qpred, cpl_program)
    expected = testing.make_prov_set(
        [
            (
                sum(t[0] for t in pfact_sets[TermInStudy] if t[1] == term) / 5,
                term,
            )
            for term in ("memory", "visual")
        ],
        ("_p_", "t"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_disjunctive_program():
    pfact_sets = {
        P: {(0.8, "a"), (0.4, "b")},
        Q: {(0.1, "a"), (0.2, "c"),},
    }
    code = Union((Implication(Z(x), P(x)), Implication(Z(x), Q(x)),))
    cpl_program = CPLogicProgram()
    for pred_symb, pfact_set in pfact_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(pred_symb, pfact_set)
    cpl_program.walk(code)
    qpred = Z(x)
    result = solve_succ_query(qpred, cpl_program)
    expected = testing.make_prov_set(
        [(1 - (1 - 0.8) * (1 - 0.1), "a"), (0.4, "b"), (0.2, "c"),],
        columns=("_p_", "x"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_simple_marg_query():
    code = Union(())
    cpl_program = CPLogicProgram()
    cpl_program.walk(code)
    cpl_program.add_probabilistic_facts_from_tuples(
        P, {(0.7, "a"), (0.8, "b")}
    )
    cpl_program.add_probabilistic_facts_from_tuples(Q, {(0.8, "b")})
    result = solve_marg_query(P(x), Q(b), cpl_program)
    expected = testing.make_prov_set([(0.7, "a"), (0.8, "b")], ("_p_", "x"))
    assert testing.eq_prov_relations(result, expected)


def test_marg_query_intensional():
    pchoice_as_sets = {P: {(0.2, "a", "b"), (0.4, "b", "c"), (0.4, "b", "b")}}
    pfact_sets = {Z: {(0.5, "b"), (0.5, "c")}}
    code = Union((Implication(Q(x), Conjunction((P(x, y), Z(y)))),))
    cpl_program = CPLogicProgram()
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    for pred_symb, pfact_set in pfact_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(pred_symb, pfact_set)
    cpl_program.walk(code)
    result = solve_marg_query(Q(x), Z(b), cpl_program)
    expected = testing.make_prov_set([(0.2, "a"), (0.6, "b")], ("_p_", "x"))
    assert testing.eq_prov_relations(result, expected)


def test_marg_query_shared_variables_between_query_and_evidence_preds():
    pchoice_as_sets = {P: {(0.2, "a"), (0.5, "b"), (0.3, "c")}}
    code = Union((Implication(Q(x), P(x)),))
    cpl_program = CPLogicProgram()
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    cpl_program.walk(code)
    result = solve_marg_query(Q(x), P(x), cpl_program)
    expected = testing.make_prov_set(
        [(1.0, "a"), (1.0, "b"), (1.0, "c")], ("_p_", "x")
    )
    assert testing.eq_prov_relations(result, expected)


def test_marg_query_multiple_evidence_predicates():
    pchoice_as_sets = {
        P: {(0.2, "a"), (0.8, "b")},
        Q: {(0.5, "a"), (0.5, "b")},
    }
    code = Union((Implication(Z(x), Conjunction((P(x), Q(x)))),))
    cpl_program = CPLogicProgram()
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    cpl_program.walk(code)
    query_pred = Z(x)
    evidence = Conjunction((P(a), Q(b)))
    result = solve_marg_query(query_pred, evidence, cpl_program)
    expected = testing.make_prov_set([], ("_p_", "x"))
    assert testing.eq_prov_relations(result, expected)


def test_solve_succ_all():
    cpl = CPLogicProgram()
    cpl.add_extensional_predicate_from_tuples(P, {("a", "b"), ("b", "c")})
    cpl.add_probabilistic_facts_from_tuples(Q, {(0.2, "a"), (0.9, "b")})
    cpl.add_probabilistic_choice_from_tuples(H, {(0.5, "a"), (0.5, "c")})
    cpl.walk(
        Implication(
            Z(x, ProbabilisticQuery(PROB, (x,))), Conjunction((H(x), Q(x)))
        )
    )
    result = problog_solve_succ_all(cpl)
    assert set(result.keys()) == {Z}
    assert len(result[Z].value) == 1
    res = next(iter(result[Z].value))
    assert res[0] == "a"
    assert np.isclose(res[1], 0.1)
