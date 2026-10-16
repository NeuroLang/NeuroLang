"""
Unification of the existential variables of the disjuncts of a union of
conjunctive queries, used to find separator variables.

∃x C1 ∨ ∃y C2 ≡ ∃z (C1[z/x] ∨ C2[z/y]) only for renamings of variables
by variables that are injective within each disjunct: identifying two
variables of a disjunct, or a variable with a constant, changes the
query.
"""
import itertools
import random

import pytest

from ...exceptions import UnsupportedSolverError
from ...expressions import Constant, Symbol
from ...logic import (
    Conjunction,
    Disjunction,
    ExistentialPredicate,
    Implication,
    Union
)
from ...logic.expression_processing import (
    extract_logic_atoms,
    extract_logic_free_variables
)
from ...logic.transformations import (
    MakeExistentialsImplicit,
    convert_to_pnf_with_dnf_matrix
)
from .. import dalvi_suciu_lift
from ..cplogic import testing
from ..cplogic.program import CPLogicProgram
from ..transforms import unify_existential_variables

ans = Symbol("ans")
Gt = Symbol("Gt")
Lt = Symbol("Lt")
R = Symbol("R")
S = Symbol("S")
Z = Symbol("Z")
t = Symbol("t")
u = Symbol("u")
v = Symbol("v")
w = Symbol("w")
x = Symbol("x")
y = Symbol("y")


def _closed(formula):
    for variable in sorted(
        extract_logic_free_variables(formula), key=lambda s: s.name
    ):
        formula = ExistentialPredicate(variable, formula)
    return formula


def _disjuncts(query):
    matrix = MakeExistentialsImplicit().walk(
        convert_to_pnf_with_dnf_matrix(query)
    )
    if isinstance(matrix, Disjunction):
        return matrix.formulas
    return (matrix,)


def test_variables_of_a_disjunct_are_not_identified():
    """S constrains the first variable of Lt in the first disjunct and its
    second in the other: unifying both S and Lt atoms must not identify u
    and v."""
    query = Disjunction((
        _closed(Conjunction((Lt(u, v), Gt(u, v), S(u)))),
        _closed(Conjunction((Lt(w, t), Gt(w, t), S(t)))),
    ))
    for disjunct in _disjuncts(unify_existential_variables(query)):
        atoms = [a for a in extract_logic_atoms(disjunct) if a.functor == Lt]
        assert all(len(set(atom.args)) == 2 for atom in atoms), disjunct


def test_variables_are_not_replaced_by_constants():
    a = Constant("a")
    query = Disjunction((_closed(R(x, y)), R(a, a)))
    unified = unify_existential_variables(query)
    assert any(
        len(extract_logic_free_variables(atom)) == 2
        for atom in extract_logic_atoms(unified)
    ), unified


TUPLES = (("a", "b"), ("b", "a"), ("a", "a"), ("b", "c"), ("c", "b"))


def _solve_lifted(rules, r_facts, s_facts):
    cpl_program = CPLogicProgram()
    for symbol, facts in ((R, r_facts), (S, s_facts)):
        cpl_program.add_probabilistic_facts_from_tuples(
            symbol, [(p,) + tuple_ for tuple_, p in facts.items()]
        )
    cpl_program.walk(Union(tuple(
        Implication(Z(), Conjunction(atoms)) for atoms in rules
    )))
    return dalvi_suciu_lift.solve_succ_query(
        Implication(ans(), Z()), cpl_program
    )


def _enumerated(rules, r_facts, s_facts):
    """Probability that a conjunction of `rules` has a valuation in a
    world, by enumerating the worlds."""
    facts = [("R", f, p) for f, p in r_facts.items()]
    facts += [("S", f, p) for f, p in s_facts.items()]
    values = sorted({v for f, _ in r_facts.items() for v in f} |
                    {v for f, _ in s_facts.items() for v in f})
    total = 0.
    for present in itertools.product((False, True), repeat=len(facts)):
        world = {"R": set(), "S": set()}
        probability = 1.
        for (name, f, p), is_present in zip(facts, present):
            probability *= p if is_present else 1 - p
            if is_present:
                world[name].add(f)
        if any(_holds(atoms, world, values) for atoms in rules):
            total += probability
    return total


def _holds(atoms, world, values):
    variables = sorted(
        {a for atom in atoms for a in atom.args}, key=lambda s: s.name
    )
    for assignment in itertools.product(values, repeat=len(variables)):
        valuation = dict(zip(variables, assignment))
        if all(
            tuple(valuation[a] for a in atom.args) in
            world[atom.functor.name]
            for atom in atoms
        ):
            return True
    return False


def test_union_with_shared_relations_is_lifted_correctly():
    """The union of the ranked forms of ∃x, y R(x, y) ∧ R(y, x) ∧ S(x)."""
    lt, gt, eq = Symbol("Rlt"), Symbol("Rgt"), Symbol("Req")
    r_facts = {
        ("a", "b"): 0.5, ("b", "a"): 0.4, ("a", "a"): 0.3,
        ("b", "c"): 0.6, ("c", "b"): 0.2,
    }
    s_facts = {("a",): 0.7, ("b",): 0.1}
    relations = {
        lt: {f: p for f, p in r_facts.items() if f[0] < f[1]},
        gt: {f[::-1]: p for f, p in r_facts.items() if f[0] > f[1]},
        eq: {f[:1]: p for f, p in r_facts.items() if f[0] == f[1]},
        S: s_facts,
    }
    cpl_program = CPLogicProgram()
    for symbol, facts in relations.items():
        cpl_program.add_probabilistic_facts_from_tuples(
            symbol, [(p,) + f for f, p in facts.items()]
        )
    cpl_program.walk(Union((
        Implication(Z(), Conjunction((lt(x, y), gt(x, y), S(x)))),
        Implication(Z(), Conjunction((eq(x), S(x)))),
        Implication(Z(), Conjunction((lt(y, x), gt(y, x), S(x)))),
    )))
    expected = _enumerated(
        [[R(x, y), R(y, x), S(x)]], r_facts, s_facts
    )
    result = dalvi_suciu_lift.solve_succ_query(
        Implication(ans(), Z()), cpl_program
    )
    assert testing.eq_prov_relations(
        result, testing.make_prov_set([(expected,)], ("_p_",))
    )


ATOM_POOL = (
    R(x, y), R(y, x), R(x, x), R(y, w), S(x), S(y), S(w),
)


@pytest.mark.parametrize("seed", range(400))
def test_random_unions_lifted_are_correct(seed):
    """Unions of two or three conjunctions of atoms of R and S, with
    self-joins, on random tuple-independent relations: every lifted
    probability is the enumerated one."""
    rng = random.Random(seed)
    rules = [
        rng.sample(ATOM_POOL, rng.randint(2, 3))
        for _ in range(rng.randint(2, 3))
    ]
    r_facts = {
        f: round(rng.uniform(0.1, 0.9), 2)
        for f in rng.sample(TUPLES, rng.randint(2, 4))
    }
    s_facts = {
        (value,): round(rng.uniform(0.1, 0.9), 2)
        for value in rng.sample(("a", "b", "c"), 2)
    }
    try:
        result = _solve_lifted(rules, r_facts, s_facts)
    except UnsupportedSolverError:
        return
    expected = _enumerated(rules, r_facts, s_facts)
    probabilities = [row[0] for row in result.relation.value]
    assert sum(probabilities) == pytest.approx(expected), rules
