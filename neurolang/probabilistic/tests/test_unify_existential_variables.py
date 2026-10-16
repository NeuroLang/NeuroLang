"""
Unification of the existential variables of the disjuncts of a union of
conjunctive queries, used to find separator variables.

∃x C1 ∨ ∃y C2 ≡ ∃z (C1[z/x] ∨ C2[z/y]) only for renamings of variables
by variables that are injective within each disjunct: identifying two
variables of a disjunct, or a variable with a constant, changes the
query.
"""
import pytest

from ...exceptions import NonLiftableException
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


def test_union_with_shared_relations_is_not_lifted_wrongly():
    """The union of the ranked forms of ∃x, y R(x, y) ∧ R(y, x) ∧ S(x):
    its first and last disjuncts differ by the variable of S. Unifying
    their variables identified the two variables of the first disjunct,
    and the lifted probability was 0.2258 instead of 0.321584."""
    lt, gt, eq = Symbol("Rlt"), Symbol("Rgt"), Symbol("Req")
    relations = {
        lt: [(0.5, "a", "b"), (0.6, "b", "c")],
        gt: [(0.4, "a", "b"), (0.2, "b", "c")],
        eq: [(0.3, "a")],
        S: [(0.7, "a"), (0.1, "b")],
    }
    cpl_program = CPLogicProgram()
    for symbol, facts in relations.items():
        cpl_program.add_probabilistic_facts_from_tuples(symbol, facts)
    cpl_program.walk(Union((
        Implication(Z(), Conjunction((lt(x, y), gt(x, y), S(x)))),
        Implication(Z(), Conjunction((eq(x), S(x)))),
        Implication(Z(), Conjunction((lt(y, x), gt(y, x), S(x)))),
    )))
    with pytest.raises(NonLiftableException):
        dalvi_suciu_lift.solve_succ_query(
            Implication(ans(), Z()), cpl_program
        )
