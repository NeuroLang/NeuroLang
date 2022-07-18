import operator
from typing import AbstractSet

import numpy as np
import pytest

from ...config import config
from ...datalog import Fact
from ...exceptions import NonLiftableException, UnsupportedSolverError
from ...expressions import Constant, Symbol
from ...logic import Conjunction, Implication, Negation, Union
from ...relational_algebra import (
    NamedRelationalAlgebraFrozenSet,
    str2columnstr_constant
)
from ...relational_algebra_provenance import ProvenanceAlgebraSet
from .. import (
    dalvi_suciu_lift,
    small_dichotomy_theorem_based_solver,
    weighted_model_counting
)
from ..cplogic import testing
from ..cplogic.program import CPLogicProgram
from ..exceptions import NotHierarchicalQueryException

try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import suppress as nullcontext

EQ = Constant(operator.eq)
GT = Constant(operator.gt)
NE = Constant(operator.ne)

ans = Symbol("ans")
P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
Z = Symbol("Z")
H = Symbol("H")
A = Symbol("A")
B = Symbol("B")
C = Symbol("C")
w = Symbol("w")
x = Symbol("x")
y = Symbol("y")
z = Symbol("z")
p = Symbol("p")

a = Constant("a")
b = Constant("b")
c = Constant("c")


@pytest.fixture(
    params=((
        weighted_model_counting,
        small_dichotomy_theorem_based_solver,
        dalvi_suciu_lift,
    )),
    ids=["SDD-WMC", "small-dichotomy", "dalvi-suciu"],
)
def solver(request):
    return request.param


def test_deterministic(solver):
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
    code = Union(
        (
            Fact(Q(a)),
            Fact(Q(b)),
            Implication(P(x), Q(x)),
        )
    )
    cpl_program = CPLogicProgram()
    cpl_program.walk(code)
    query = Implication(ans(x), P(x))
    result = solver.solve_succ_query(query, cpl_program)
    expected = testing.make_prov_set([(1.0, "a"), (1.0, "b")], ("_p_", "x"))
    assert testing.eq_prov_relations(result, expected)


def test_deterministic_conjunction_varying_arity(solver):
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
    query = Implication(ans(x, y), Z(x, y))
    result = solver.solve_succ_query(query, cpl_program)
    expected = testing.make_prov_set([(1.0, "a", "b")], ("_p_", "x", "y"))
    assert testing.eq_prov_relations(result, expected)


def test_deterministic_conjunction_varying_arity_empty(solver):
    code = Union(
        (
            Fact(P(a)),
            Fact(P(b)),
            Implication(
                Z(x, y),
                Conjunction((Q(x, y), P(x), P(y)))
            ),
        )
    )
    cpl_program = CPLogicProgram()
    cpl_program.walk(code)
    query = Implication(ans(x, y), Z(x, y))
    result = solver.solve_succ_query(query, cpl_program)
    assert result.relation.value.is_empty()


def test_simple_bernoulli(solver):
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
    query = Implication(ans(x), P(x))
    result = solver.solve_succ_query(query, cpl_program)
    expected = testing.make_prov_set([(0.7, "a"), (0.8, "b")], ("_p_", "x"))
    assert testing.eq_prov_relations(result, expected)


def test_bernoulli_conjunction(solver):
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
    query = Implication(ans(x), Z(x))
    result = solver.solve_succ_query(query, cpl_program)

    expected = testing.make_prov_set([(0.9 * 0.9 * 0.5, "b")], ("_p_", "x"))
    assert testing.eq_prov_relations(result, expected)


def test_probfact_existential():
    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_facts_from_tuples(
        P, {(0.7, 1, 2), (0.9, 1, 3), (0.88, 2, 4)}
    )
    query = Implication(ans(x), P(x, y))
    result = dalvi_suciu_lift.solve_succ_query(query, cpl_program)
    expected = testing.make_prov_set(
        [
            ((1 - (1 - 0.7) * (1 - 0.9)), 1),
            (0.88, 2),
        ],
        ("_p_", "x"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_multi_level_conjunction(solver):
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
    query = Implication(ans(x, y), H(x, y))
    result = solver.solve_succ_query(query, cpl_program)
    expected = testing.make_prov_set(
        [
            (0.2 * 0.9 * 0.1, "a", "a"),
            (0.2 * 0.9 * 0.5, "a", "b"),
        ],
        ("_p_", "x", "y"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_intertwined_conjunctions_and_probfacts(solver):
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
    query = Implication(ans(y), Z(y))
    result = solver.solve_succ_query(query, cpl)
    expected = testing.make_prov_set([(0.8 * 0.5, "a")], ("_p_", "y"))
    assert testing.eq_prov_relations(result, expected)


def test_simple_probchoice(solver):
    pchoice_as_sets = {P: {(0.2, "a"), (0.8, "b")}}
    cpl_program = CPLogicProgram()
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    query = Implication(ans(x), P(x))
    result = solver.solve_succ_query(query, cpl_program)
    expected = testing.make_prov_set(
        [
            (0.2, "a"),
            (0.8, "b"),
        ],
        ("_p_", "x"),
    )
    assert testing.eq_prov_relations(result, expected)


@pytest.mark.parametrize("solver", [
    pytest.param(weighted_model_counting, marks=pytest.mark.xfail(
        reason="WMC issue to be resolved"
    )),
    small_dichotomy_theorem_based_solver,
    dalvi_suciu_lift,
])
def test_mutual_exclusivity(solver):
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
    query = Implication(ans(x, y), Z(x, y))
    result = solver.solve_succ_query(query, cpl_program)
    expected = testing.make_prov_set([], ("_p_", "x", "y"))
    assert testing.eq_prov_relations(result, expected)


@pytest.mark.parametrize("solver", [
    pytest.param(weighted_model_counting, marks=pytest.mark.xfail(
        reason="WMC issue to be resolved"
    )),
    small_dichotomy_theorem_based_solver,
    dalvi_suciu_lift,
])
def test_fact_negation_choice(solver):
    pchoice_as_sets = {P: {(0.2, "a"), (0.8, "b")}}
    pfact_sets = {Q: {(0.5, "a", "b")}}
    code = Union((
        Implication(
            Z(x, y),
            Conjunction((Negation(P(y)), Q(x, y)))
        ),
    ))
    cpl_program = CPLogicProgram()
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    for pred_symb, pfact_set in pfact_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(pred_symb, pfact_set)
    cpl_program.walk(code)
    query = Implication(ans(x, y), Z(x, y))

    if solver is small_dichotomy_theorem_based_solver:
        context = pytest.raises(NotHierarchicalQueryException)
    else:
        context = nullcontext()

    with context:
        result = solver.solve_succ_query(query, cpl_program)
        expected = testing.make_prov_set([(0.1, "a", "b")], ("_p_", "x", "y"))
        assert testing.eq_prov_relations(result, expected)


@pytest.mark.parametrize("solver", [
    pytest.param(weighted_model_counting, marks=pytest.mark.xfail(
        reason="WMC issue to be resolved"
    )),
    small_dichotomy_theorem_based_solver,
    dalvi_suciu_lift,
    pytest.param(dalvi_suciu_lift, marks=pytest.mark.xfail(
        reason="Connected component issue addressed by PR #660"
    )),
])
def test_negation_fact_choice(solver):
    pchoice_as_sets = {P: {(0.2, "a"), (0.8, "b")}}
    pfact_sets = {Q: {(0.4, "a", "a")}}
    code = Union((
        Implication(
            Z(x),
            Conjunction((P(x), Negation(Q(x, x))))
        ),
    ))
    cpl_program = CPLogicProgram()
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    for pred_symb, pfact_set in pfact_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(pred_symb, pfact_set)
    cpl_program.walk(code)
    query = Implication(ans(x), Z(x))

    if solver is small_dichotomy_theorem_based_solver:
        context = pytest.raises(NotHierarchicalQueryException)
    else:
        context = nullcontext()

    with context:
        result = solver.solve_succ_query(query, cpl_program)
        expected = testing.make_prov_set([(.12, "a"), (.8, "b")], ("_p_", "x"))
        assert testing.eq_prov_relations(result, expected)


def test_multiple_probchoices_mutual_exclusivity(solver):
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
    query = Implication(ans(x, y), Z(x, y))
    result = solver.solve_succ_query(query, cpl_program)
    expected = testing.make_prov_set(
        [(0.2 * 0.1, "a", "b"), (0.8 * 0.1, "b", "b")], ("_p_", "x", "y")
    )
    assert testing.eq_prov_relations(result, expected)


def test_large_probabilistic_choice(solver):
    if (
        config["RAS"].get("backend", "pandas") == "dask"
        or solver is weighted_model_counting
    ):
        n = int(1000)
    else:
        n = int(10000)
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
    query = Implication(ans(x, y), Z(x, y))
    result = solver.solve_succ_query(query, cpl_program)
    expected = testing.make_prov_set(
        [
            (0.5 * probs[0], 0, 0),
            (0.5 * probs[0], 0, 1),
        ],
        ("_p_", "x", "y"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_simple_probchoice_existential(solver):
    """
    We define the following program

        P(a, a) : 0.2 v P(a, b) : 0.7 v P(c, c) : 0.1 <- T
        Q(x) <- ∃y, P(x, y)

    We expect the following to hold

        - Pr[Q(a)] = 0.9
        - Pr[Q(c)] = 0.1

    """
    pchoice_as_sets = {P: {(0.2, "a", "a"), (0.7, "a", "b"), (0.1, "c", "c")}}
    code = Union((Implication(Q(x), P(x, y)),))
    cpl_program = CPLogicProgram()
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    cpl_program.walk(code)
    query = Implication(ans(x), Q(x))
    result = solver.solve_succ_query(query, cpl_program)
    expected = testing.make_prov_set([(0.9, "a"), (0.1, "c")], ("_p_", "x"))
    assert testing.eq_prov_relations(result, expected)


def test_existential_in_conjunction(solver):
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
    query = Implication(ans(x), Q(x))
    result = solver.solve_succ_query(query, cpl_program)
    expected = testing.make_prov_set([(0.1, "a"), (0.2, "b")], ("_p_", "x"))
    assert testing.eq_prov_relations(result, expected)


def test_existential_alternative_variables(solver):
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
    query = Implication(ans(z), H(z))
    result = solver.solve_succ_query(query, cpl_program)
    expected = testing.make_prov_set([(0.2 * 0.8, "b")], ("prob", "z"))
    assert testing.eq_prov_relations(result, expected)


def test_multilevel_existential(solver):
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
            Implication(C(x), H(x, y)),
        )
    )
    cpl_program = CPLogicProgram()
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    cpl_program.walk(code)
    query = Implication(ans(x, y), H(x, y))
    result = solver.solve_succ_query(query, cpl_program)
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

    query = Implication(ans(z), C(z))
    result = solver.solve_succ_query(query, cpl_program)
    expected = testing.make_prov_set(
        [
            (0.1, "a"),
            (0.4, "b"),
            (0.5, "c"),
        ],
        ("_p_", "z"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_multilevel_existential_2(solver):
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
            Implication(C(x), H(x, y)),
        )
    )
    cpl_program = CPLogicProgram()
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    cpl_program.walk(code)

    query = Implication(ans(z), B(z))

    if solver is small_dichotomy_theorem_based_solver:
        context = pytest.raises(NotHierarchicalQueryException)
    else:
        context = nullcontext()

    with context:
        result = solver.solve_succ_query(query, cpl_program)
        expected = testing.make_prov_set(
            [(0.5 * 0.1 * 0.5, "c")],
            ("_p_", "z"),
        )
        assert testing.eq_prov_relations(result, expected)


def test_repeated_antecedent_predicate_symbol(solver):
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
    query = Implication(ans(x, y), Q(x, y))

    if solver is small_dichotomy_theorem_based_solver:
        context = pytest.raises(NotHierarchicalQueryException)
    elif solver is dalvi_suciu_lift:
        context = pytest.raises(NonLiftableException)
    else:
        context = nullcontext()

    with context:
        result = solver.solve_succ_query(query, cpl_program)
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


def test_tiny_cbma_example(solver):
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
    query = Implication(ans(t), TermAssociation(t))
    result = solver.solve_succ_query(query, cpl_program)
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


def test_conjunct_pfact_equantified_pchoice(solver):
    pfact_sets = {P: {(0.8, "a", "s1"), (0.5, "a", "s2"), (0.1, "b", "s2")}}
    pchoice_as_sets = {Z: {(0.6, "s1"), (0.4, "s2")}}
    code = Union((Implication(Q(x), Conjunction((P(x, y), Z(y)))),))
    cpl_program = CPLogicProgram()
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    for pred_symb, pfact_set in pfact_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(pred_symb, pfact_set)
    cpl_program.walk(code)
    query = Implication(ans(x), Q(x))
    result = solver.solve_succ_query(query, cpl_program)
    expected = testing.make_prov_set(
        [
            (0.6 * 0.8 + 0.4 * 0.5, "a"),
            (0.1 * 0.4, "b"),
        ],
        ("_p_", "x"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_shatterable_query(solver):
    pfact_sets = {P: {(0.8, "a", "1"), (0.5, "a", "2"), (0.1, "b", "2")}}
    code = Union((Implication(Q(x), Conjunction((P(a, x), P(b, x)))),))
    cpl_program = CPLogicProgram()
    for pred_symb, pfact_set in pfact_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(pred_symb, pfact_set)
    cpl_program.walk(code)
    query = Implication(ans(x), Q(x))
    result = solver.solve_succ_query(query, cpl_program)
    expected = testing.make_prov_set(
        [
            (0.5 * 0.1, "2"),
        ],
        ("_p_", "x"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_shatterable_query_2(solver):
    pfact_sets = {
        P: {(0.8, "a", "c", "1"), (0.5, "a", "c", "2"), (0.1, "b", "b", "2")}
    }
    code = Union((Implication(Q(x), Conjunction((P(a, c, x), P(b, b, x)))),))
    cpl_program = CPLogicProgram()
    for pred_symb, pfact_set in pfact_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(pred_symb, pfact_set)
    cpl_program.walk(code)
    query = Implication(ans(x), Q(x))
    result = solver.solve_succ_query(query, cpl_program)
    expected = testing.make_prov_set(
        [
            (0.5 * 0.1, "2"),
        ],
        ("_p_", "x"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_program_with_variable_equality(solver):
    pfact_sets = {
        Q: {(0.2, "a"), (0.3, "b"), (0.4, "c")},
    }
    pchoice_as_sets = {
        P: {(0.5, "a", "b"), (0.5, "b", "c")},
    }
    code = Union(
        (Implication(Z(x, y), Conjunction((P(y, x), Q(y), EQ(y, a)))),)
    )
    cpl_program = CPLogicProgram()
    for pred_symb, pfact_set in pfact_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(
            pred_symb, pfact_set
        )
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    cpl_program.walk(code)
    query = Implication(ans(x, y), Z(x, y))
    result = solver.solve_succ_query(query, cpl_program)
    expected = testing.make_prov_set(
        [
            (0.5 * 0.2, "b", "a"),
        ],
        ("_p_", "x", "y"),
    )
    assert testing.eq_prov_relations(result, expected)


@pytest.mark.parametrize("solver", [
    pytest.param(weighted_model_counting, marks=pytest.mark.xfail(
        reason="WMC issue to be resolved"
    )),
    small_dichotomy_theorem_based_solver,
    dalvi_suciu_lift,
])
def test_program_with_segregation_constant_case(solver):
    pfact_sets = {
        P: {(0.5, "a", "b"), (0.5, "b", "c"), (0.2, "b", "d")},
    }
    code = Union((
        Implication(Z(x, y), Conjunction((P(x, y), Negation(A(x, y))))),
        Implication(A(x, y), Conjunction((P(w, y), NE(w, x)))),
    ))
    cpl_program = CPLogicProgram()
    for pred_symb, pfact_set in pfact_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(
            pred_symb, pfact_set
        )
    cpl_program.walk(code)
    query = Implication(ans(), Z(a, b))

    if solver is small_dichotomy_theorem_based_solver:
        context = pytest.raises(UnsupportedSolverError)
    else:
        context = nullcontext()

    with context:
        result = solver.solve_succ_query(query, cpl_program)
        expected = testing.make_prov_set(
            [
                (0.5,),
            ],
            ("_p_",),
        )
        assert testing.eq_prov_relations(result, expected)


@pytest.mark.parametrize("solver", [
    pytest.param(weighted_model_counting, marks=pytest.mark.xfail(
        reason="WMC issue to be resolved"
    )),
    small_dichotomy_theorem_based_solver,
    dalvi_suciu_lift,
])
def test_program_with_segregation_first_constant(solver):
    pfact_sets = {
        P: {(0.5, "a", "b"), (0.5, "b", "c"), (0.2, "b", "d")},
    }
    code = Union((
        Implication(Z(x, y), Conjunction((P(x, y), Negation(A(x, y))))),
        Implication(A(x, y), Conjunction((P(w, y), NE(w, x)))),
    ))
    cpl_program = CPLogicProgram()
    for pred_symb, pfact_set in pfact_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(
            pred_symb, pfact_set
        )
    cpl_program.walk(code)
    query = Implication(ans(y), Z(a, y))

    if solver is small_dichotomy_theorem_based_solver:
        context = pytest.raises(UnsupportedSolverError)
    else:
        context = nullcontext()

    with context:
        result = solver.solve_succ_query(query, cpl_program)
        expected = testing.make_prov_set(
            [
                (0.5, "b"),
            ],
            ("_p_", "y"),
        )
        assert testing.eq_prov_relations(result, expected)


@pytest.mark.parametrize("solver", [
    pytest.param(weighted_model_counting, marks=pytest.mark.xfail(
        reason="WMC issue to be resolved"
    )),
    small_dichotomy_theorem_based_solver,
    dalvi_suciu_lift,
])
def test_program_with_segregation_second_constant(solver):
    pfact_sets = {
        P: {(0.5, "a", "b"), (0.5, "b", "c"), (0.2, "b", "d")},
    }
    code = Union((
        Implication(Z(x, y), Conjunction((P(x, y), Negation(A(x, y))))),
        Implication(A(x, y), Conjunction((P(w, y), NE(w, x)))),
    ))
    cpl_program = CPLogicProgram()
    for pred_symb, pfact_set in pfact_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(
            pred_symb, pfact_set
        )
    cpl_program.walk(code)
    query = Implication(ans(x), Z(x, b))

    if (
        solver is small_dichotomy_theorem_based_solver or
        solver is dalvi_suciu_lift
    ):
        context = pytest.raises(UnsupportedSolverError)
    else:
        context = nullcontext()

    with context:
        result = solver.solve_succ_query(query, cpl_program)
        expected = testing.make_prov_set(
            [
                (0.5, "a"),
            ],
            ("_p_", "x"),
        )
        assert testing.eq_prov_relations(result, expected)


@pytest.mark.parametrize("solver", [
    pytest.param(weighted_model_counting, marks=pytest.mark.xfail(
        reason="WMC issue to be resolved"
    )),
    small_dichotomy_theorem_based_solver,
    dalvi_suciu_lift,
])
def test_program_with_segregation(solver):
    pfact_sets = {
        P: {(0.5, "a", "b"), (0.5, "b", "c"), (0.2, "b", "d")},
    }
    code = Union((
        Implication(Z(x, y), Conjunction((P(x, y), Negation(A(x, y))))),
        Implication(A(x, y), Conjunction((P(w, y), NE(w, x)))),
    ))
    cpl_program = CPLogicProgram()
    for pred_symb, pfact_set in pfact_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(
            pred_symb, pfact_set
        )
    cpl_program.walk(code)
    query = Implication(ans(x, y), Z(x, y))

    if (
        solver is small_dichotomy_theorem_based_solver or
        solver is dalvi_suciu_lift
    ):
        context = pytest.raises(UnsupportedSolverError)
    else:
        context = nullcontext()

    with context:
        result = solver.solve_succ_query(query, cpl_program)
        expected = testing.make_prov_set(
            [
                (0.5, "a", "b"),
            ],
            ("_p_", "x", "y"),
        )
        assert testing.eq_prov_relations(result, expected)


@pytest.mark.parametrize("solver", [
    pytest.param(weighted_model_counting, marks=pytest.mark.xfail(
        reason="WMC issue to be resolved"
    )),
    small_dichotomy_theorem_based_solver,
    dalvi_suciu_lift,
])
def test_program_with_segregation_first_existential(solver):
    pfact_sets = {
        P: {(0.5, "a", "b"), (0.5, "b", "c"), (0.2, "b", "d")},
    }
    code = Union((
        Implication(Z(x, y), Conjunction((P(x, y), Negation(A(x, y))))),
        Implication(A(x, y), Conjunction((P(w, y), NE(w, x)))),
    ))
    cpl_program = CPLogicProgram()
    for pred_symb, pfact_set in pfact_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(
            pred_symb, pfact_set
        )
    cpl_program.walk(code)
    query = Implication(ans(y), Z(x, y))

    if (
        solver is small_dichotomy_theorem_based_solver or
        solver is dalvi_suciu_lift
    ):
        context = pytest.raises(UnsupportedSolverError)
    else:
        context = nullcontext()

    with context:
        result = solver.solve_succ_query(query, cpl_program)
        expected = testing.make_prov_set(
            [
                (0.5, "b"),
            ],
            ("_p_", "y"),
        )
        assert testing.eq_prov_relations(result, expected)


@pytest.mark.parametrize("solver", [
    pytest.param(weighted_model_counting, marks=pytest.mark.xfail(
        reason="WMC issue to be resolved"
    )),
    small_dichotomy_theorem_based_solver,
    dalvi_suciu_lift,
])
def test_program_with_segregation_second_existential(solver):
    pfact_sets = {
        P: {(0.5, "a", "b"), (0.5, "b", "c"), (0.2, "b", "d")},
    }
    code = Union((
        Implication(Z(x, y), Conjunction((P(x, y), Negation(A(x, y))))),
        Implication(A(x, y), Conjunction((P(w, y), NE(w, x)))),
    ))
    cpl_program = CPLogicProgram()
    for pred_symb, pfact_set in pfact_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(
            pred_symb, pfact_set
        )
    cpl_program.walk(code)
    query = Implication(ans(x), Z(x, y))

    if (
        solver is small_dichotomy_theorem_based_solver or
        solver is dalvi_suciu_lift
    ):
        context = pytest.raises(UnsupportedSolverError)
    else:
        context = nullcontext()

    with context:
        result = solver.solve_succ_query(query, cpl_program)
        expected = testing.make_prov_set(
            [
                (0.5, "a"),
            ],
            ("_p_", "x"),
        )
        assert testing.eq_prov_relations(result, expected)


@pytest.mark.parametrize("solver", [
    pytest.param(weighted_model_counting, marks=pytest.mark.xfail(
        reason="WMC issue to be resolved"
    )),
    small_dichotomy_theorem_based_solver,
    dalvi_suciu_lift,
])
def test_program_with_negative_existential(solver):
    pfact_sets = {
        P: {(0.5, "a", "b"), (0.5, "b", "c"), (0.2, "b", "d")},
        Q: {(1., "b", "c")}
    }
    code = Union((
        Implication(Z(x, y), Conjunction((P(x, y), Negation(A(y))))),
        Implication(A(y), Q(w, y)),
    ))
    cpl_program = CPLogicProgram()
    for pred_symb, pfact_set in pfact_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(
            pred_symb, pfact_set
        )
    cpl_program.walk(code)
    query = Implication(ans(x, y), Z(x, y))

    if solver is small_dichotomy_theorem_based_solver:
        context = pytest.raises(UnsupportedSolverError)
    else:
        context = nullcontext()

    with context:
        result = solver.solve_succ_query(query, cpl_program)
        expected = testing.make_prov_set(
            [
                (0.5, "a", "b"),
                (0.2, "b", "d")
            ],
            ("_p_", "x", "y"),
        )
        assert testing.eq_prov_relations(result, expected)


@pytest.mark.parametrize("solver", [
    weighted_model_counting,
    pytest.param(small_dichotomy_theorem_based_solver, marks=pytest.mark.xfail(
        reason="Existential variable in probfact leads to unsupported noisy-or"
    )),
    dalvi_suciu_lift,
])
def test_repeated_variable_probabilistic_rule(solver):
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(
        Q, [(0.2, 7, 7, 2), (0.5, 7, 8, 4)]
    )
    cpl.walk(Implication(H(x, x), Q(x, x, y)))
    query = Implication(ans(x, y), H(x, y))
    result = solver.solve_succ_query(query, cpl)
    expected = testing.make_prov_set([(0.2, 7, 7)], ("_p_", "x", "y"))
    assert testing.eq_prov_relations(result, expected)


def test_repeated_variable_with_constant_in_head(solver):
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(
        Q,
        [(0.2, 7, 8), (0.6, 8, 9), (0.9, 8, 8)],
    )
    cpl.add_probabilistic_choice_from_tuples(
        P,
        [(0.4, 8), (0.6, 9)],
    )
    cpl.walk(
        Implication(R(Constant[int](8), x), Conjunction((Q(x, x), P(x))))
    )
    query = Implication(ans(x, y), R(x, y))
    result = solver.solve_succ_query(query, cpl)
    expected = testing.make_prov_set(
        [(0.4 * 0.9, 8, 8)],
        ("_p_", "x", "y"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_empty_result_program(solver):
    rule = Implication(R(Constant(2), Constant(3)), Conjunction((Q(x),)))
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(
        Q,
        [(0.2, 7)],
    )
    cpl.walk(rule)
    query = Implication(ans(x), R(x, x))
    result = solver.solve_succ_query(query, cpl)
    expected = testing.make_prov_set([], ("_p_", "x"))
    assert testing.eq_prov_relations(result, expected)


@pytest.mark.parametrize("solver", [
    pytest.param(weighted_model_counting, marks=pytest.mark.xfail(
        reason="WMC issue to be resolved"
    )),
    small_dichotomy_theorem_based_solver,
    dalvi_suciu_lift,
])
def test_simple_negation(solver):
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(
        Q,
        [(0.2, 'a'), (0.1, 'b')]
    )
    cpl.add_probabilistic_facts_from_tuples(
        R,
        [(0.1, 'a'), (0.3, 'c')]
    )
    rule = Implication(P(x), Conjunction((Q(x), Negation(R(x)))))
    cpl.walk(rule)
    query = Implication(ans(x), P(x))

    if solver is small_dichotomy_theorem_based_solver:
        context = pytest.raises(NotHierarchicalQueryException)
    else:
        context = nullcontext()

    with context:
        result = solver.solve_succ_query(query, cpl)
        expected = testing.make_prov_set(
            [(.18, 'a'), (.10, 'b')], ("_p_", "x")
        )
        assert testing.eq_prov_relations(result, expected)


@pytest.mark.parametrize("solver", [
    pytest.param(weighted_model_counting, marks=pytest.mark.xfail(
        reason="WMC issue to be resolved"
    )),
    small_dichotomy_theorem_based_solver,
    dalvi_suciu_lift,
])
def test_program_with_probchoice_selfjoin(solver):
    cpl = CPLogicProgram()
    cpl.add_probabilistic_choice_from_tuples(
        P,
        [(0.2, "a"), (0.8, "b")],
    )
    cpl.add_probabilistic_facts_from_tuples(
        Q,
        [(0.6, "a"), (0.8, "b")],
    )
    rule = Implication(R(x, y), Conjunction((Q(x), P(x), P(y))))
    cpl.walk(rule)
    query = Implication(ans(x, y), R(x, y))
    result = solver.solve_succ_query(query, cpl)
    expected = testing.make_prov_set(
        [(0.2 * 0.6, "a", "a"), (0.8 * 0.8, "b", "b")], ("_p_", "x", "y")
    )
    assert testing.eq_prov_relations(result, expected)


@pytest.mark.parametrize("solver", [
    pytest.param(weighted_model_counting, marks=pytest.mark.xfail(
        reason="WMC issue to be resolved"
    )),
    small_dichotomy_theorem_based_solver,
    dalvi_suciu_lift,
])
def test_probchoice_selfjoin_multiple_variables(solver):
    cpl = CPLogicProgram()
    cpl.add_probabilistic_choice_from_tuples(
        P,
        [(0.2, "a", "b"), (0.8, "b", "c")],
    )
    cpl.add_probabilistic_facts_from_tuples(
        Q,
        [(0.6, "a"), (0.8, "b")],
    )
    rule = Implication(R(x, y, z, w), Conjunction((Q(x), P(x, y), P(z, w))))
    cpl.walk(rule)
    query = Implication(ans(x, y, z, w), R(x, y, z, w))
    result = solver.solve_succ_query(query, cpl)
    expected = testing.make_prov_set(
        [(0.2 * 0.6, "a", "b", "a", "b"), (0.8 * 0.8, "b", "c", "b", "c")],
        ("_p_", "x", "y", "z", "w"),
    )
    assert testing.eq_prov_relations(result, expected)


@pytest.mark.parametrize("solver", [
    pytest.param(weighted_model_counting, marks=pytest.mark.xfail(
        reason="WMC issue to be resolved"
    )),
    small_dichotomy_theorem_based_solver,
    dalvi_suciu_lift,
])
def test_probchoice_selfjoin_multiple_variables_shared_var(solver):
    cpl = CPLogicProgram()
    cpl.add_probabilistic_choice_from_tuples(
        P,
        [(0.2, "a", "b"), (0.8, "b", "b")],
    )
    cpl.add_probabilistic_facts_from_tuples(
        Q,
        [(0.6, "a", "b"), (0.8, "b", "a"), (0.2, "b", "b")],
    )
    rule = Implication(R(x, y, z), Conjunction((Q(x, y), P(x, y), P(z, x))))
    cpl.walk(rule)
    query = Implication(ans(x, y, z), R(x, y, z))
    result = solver.solve_succ_query(query, cpl)
    expected = testing.make_prov_set(
        [(0.2 * 0.8, "b", "b", "b")],
        ("_p_", "x", "y", "z"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_simple_boolean_query(solver):
    pchoice_as_sets = {Z: {(0.6, "s1"), (0.4, "s2")}}
    cpl_program = CPLogicProgram()
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    query = Implication(ans(), Z(x))
    result = solver.solve_succ_query(query, cpl_program)
    expected = testing.make_prov_set([(1.0,)], ("_p_",))
    assert testing.eq_prov_relations(result, expected)


def test_boolean_query_equality(solver):
    pfacts_as_sets = {Z: {(0.6, "s1"), (0.4, "s2")}}
    cpl_program = CPLogicProgram()
    for pred_symb, pfacts_as_set in pfacts_as_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(
            pred_symb, pfacts_as_set
        )
    query = Implication(ans(), Z(Constant[str]("s1")))
    result = solver.solve_succ_query(query, cpl_program)
    expected = testing.make_prov_set([(.6,)], ("_p_",))
    assert testing.eq_prov_relations(result, expected)


def test_dalvi_suciu_fails_unate():
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(
        Q,
        [(0.2, 'a'), (0.1, 'b')]
    )
    cpl.add_probabilistic_facts_from_tuples(
        R,
        [(0.1, 'a'), (0.3, 'c')]
    )
    rule = Implication(P(x), Conjunction((R(y), Q(x), Negation(R(x)))))
    cpl.walk(rule)
    query = Implication(ans(x), P(x))
    with pytest.raises(NonLiftableException):
        dalvi_suciu_lift.solve_succ_query(query, cpl)


def test_dalvi_suciu_fails_single_quantifier():
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(
        Q,
        [(0.2, 'a'), (0.1, 'b')]
    )
    cpl.add_probabilistic_facts_from_tuples(
        R,
        [(0.1, 'a'), (0.3, 'c')]
    )
    cpl.add_probabilistic_facts_from_tuples(
        P,
        [(0.1, 'a'), (0.3, 'c')]
    )
    rule = Implication(A(x), Conjunction((Q(y), R(x))))
    cpl.walk(rule)
    query = Implication(ans(), Conjunction((P(x), Negation(A(y)))))
    with pytest.raises(NonLiftableException):
        dalvi_suciu_lift.solve_succ_query(query, cpl)


def test_dalvi_suciu_fails_not_ranked():
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(
        Q,
        [(0.2, 'a', 'b'), (0.1, 'b', 'c')]
    )
    rule = Implication(P(x), Conjunction((Q(x, y), Q(y, x))))
    cpl.walk(rule)
    query = Implication(ans(x), P(x))
    with pytest.raises(NonLiftableException):
        dalvi_suciu_lift.solve_succ_query(query, cpl)


@pytest.mark.parametrize("solver", [
    pytest.param(weighted_model_counting, marks=pytest.mark.xfail(
        reason="WMC issue to be resolved"
    )),
    small_dichotomy_theorem_based_solver,
    dalvi_suciu_lift,
])
def test_nested_negation(solver):
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(
        Q,
        [
            (0.2, 'a', 1, 2),
            (0.8, 'b', 1, 5),
            (0.1, 'b', 2, 2),
        ]
    )
    cpl.add_probabilistic_facts_from_tuples(
        R,
        [(0.1, 'a'), (0.3, 'c')]
    )
    program = Union((
        Implication(Z(x, y), Q(x, y, z)),
    ))
    query = Implication(ans(x), Conjunction((
            R(x),
            Negation(Z(x, Constant(1)))
        )))
    cpl.walk(program)

    if solver is small_dichotomy_theorem_based_solver:
        context = pytest.raises(NotHierarchicalQueryException)
    else:
        context = nullcontext()

    with context:
        res = solver.solve_succ_query(query, cpl)

        expected = ProvenanceAlgebraSet(
            Constant[AbstractSet](NamedRelationalAlgebraFrozenSet(
                ("_p_", "x"),
                [
                    (0.08, "a"),
                    (0.30, "c"),
                ],
            )),
            str2columnstr_constant("_p_"),
        )

        assert testing.eq_prov_relations(res, expected)


@pytest.mark.parametrize("solver", [
    pytest.param(weighted_model_counting, marks=pytest.mark.xfail(
        reason="WMC issue to be resolved"
    )),
    small_dichotomy_theorem_based_solver,
    dalvi_suciu_lift,
])
def test_nested_negated_existential(solver):
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(
        Q,
        [
            (0.2, 'a', 1, 2),
            (0.8, 'b', 1, 5),
            (0.1, 'b', 2, 2),
        ]
    )
    cpl.add_probabilistic_facts_from_tuples(
        R,
        [(0.1, 'a'), (0.3, 'c')]
    )
    program = Union((
        Implication(Z(x), Q(x, y, z)),
    ))
    query = Implication(ans(x), Conjunction((
            R(x),
            Negation(Z(x))
        )))
    cpl.walk(program)

    if solver is small_dichotomy_theorem_based_solver:
        context = pytest.raises(NotHierarchicalQueryException)
    else:
        context = nullcontext()

    with context:
        res = solver.solve_succ_query(query, cpl)

        expected = ProvenanceAlgebraSet(
            Constant[AbstractSet](NamedRelationalAlgebraFrozenSet(
                ("_p_", "x"),
                [
                    (0.08, "a"),
                    (0.30, "c"),
                ],
            )),
            str2columnstr_constant("_p_"),
        )

        assert testing.eq_prov_relations(res, expected)


def test_disjunctive_query_with_probchoice():
    pchoice_as_sets = {
        Q: {(0.6, 1), (0.4, 2)},
        A: {(0.7, 1), (0.2, 2), (0.1, 3)},
    }
    pfact_as_sets = {
        P: {(0.9, 3), (0.8, 2)},
        R: {(0.4, 1), (0.1, 2)},
    }
    cpl_program = CPLogicProgram()
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    for pred_symb, pfact_as_set in pfact_as_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(
            pred_symb, pfact_as_set
        )
    code = Union((
        Implication(Z(x), Conjunction((P(x), Q(x), A(y)))),
        Implication(Z(x), Conjunction((R(x), Q(x)))),
    ))
    cpl_program.walk(code)
    query = Implication(ans(x), Z(x))
    result = dalvi_suciu_lift.solve_succ_query(query, cpl_program)
    expected = testing.make_prov_set(
        [
            (0.6 * 0.4, 1),
            (0.4 * (1 - (1 - 0.8) * (1 - 0.1)), 2),
        ],
        ("_p_", "x"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_small_dichotomy_fails_on_disjunctive_query():
    solver = small_dichotomy_theorem_based_solver
    pchoice_as_sets = {
        Q: {(0.6, 1), (0.4, 2)},
        A: {(0.7, 1), (0.2, 2), (0.1, 3)},
    }
    pfact_as_sets = {
        P: {(0.9, 3), (0.8, 2)},
        R: {(0.4, 1), (0.1, 2)},
    }
    cpl_program = CPLogicProgram()
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    for pred_symb, pfact_as_set in pfact_as_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(
            pred_symb, pfact_as_set
        )
    code = Union((
        Implication(Z(x), Conjunction((P(x), Q(x), A(y)))),
        Implication(Z(x), Conjunction((R(x), Q(x)))),
    ))
    cpl_program.walk(code)
    query = Implication(ans(x), Z(x))
    with pytest.raises(UnsupportedSolverError):
        solver.solve_succ_query(query, cpl_program)


def test_small_dichotomy_fails_on_noisy_or_projection():
    solver = small_dichotomy_theorem_based_solver
    pfact_as_sets = {
        P: {(0.9, 3, 2), (0.8, 3, 1)},
    }
    cpl_program = CPLogicProgram()
    for pred_symb, pfact_as_set in pfact_as_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(
            pred_symb, pfact_as_set
        )
    query = Implication(ans(x), P(x, y))
    with pytest.raises(UnsupportedSolverError):
        solver.solve_succ_query(query, cpl_program)


def test_small_dichotomy_fails_on_noisy_or_projection_with_pchoice_in_query():
    solver = small_dichotomy_theorem_based_solver
    pfact_as_sets = {
        P: {(0.9, 3, 2), (0.8, 3, 1)},
    }
    pchoice_as_sets = {
        Q: {(0.6, 1), (0.4, 2)},
    }
    cpl_program = CPLogicProgram()
    for pred_symb, pfact_as_set in pfact_as_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(
            pred_symb, pfact_as_set
        )
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    query = Implication(ans(x), Conjunction((Q(x), P(x, y))))
    with pytest.raises(UnsupportedSolverError):
        solver.solve_succ_query(query, cpl_program)


def test_dalvi_suciu_fails_builtin():
    solver = dalvi_suciu_lift
    pfact_as_sets = {
        P: {(0.9, 3, 2), (0.8, 3, 1)},
    }
    pchoice_as_sets = {
        Q: {(0.6, 1), (0.4, 2)},
    }
    cpl_program = CPLogicProgram()
    for pred_symb, pfact_as_set in pfact_as_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(
            pred_symb, pfact_as_set
        )
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    query = Implication(ans(x), Conjunction((Q(x), P(x, y), GT(x, y))))
    with pytest.raises(UnsupportedSolverError):
        solver.solve_succ_query(query, cpl_program)


def test_deterministic_simplification():
    pchoice_as_sets = {
        Q: [(0.6, 'b'), (0.4, 'c')],
    }
    deterministic_as_sets = {
        P: [('a', 'b'), ('b', 'c')]
    }
    cpl_program = CPLogicProgram()
    for k, v in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(k, v)
    for k, v in deterministic_as_sets.items():
        cpl_program.add_extensional_predicate_from_tuples(k, v)

    query = Implication(
        ans(x, y),
        Conjunction((P(x, y), P(Constant('a'), y), Q(y)))
    )

    res = dalvi_suciu_lift.solve_succ_query(
        query, cpl_program
    )

    assert testing.eq_prov_relations(
        res, testing.make_prov_set({(0.6, 'a', 'b')}, ['_p_', 'x', 'y'])
    )
