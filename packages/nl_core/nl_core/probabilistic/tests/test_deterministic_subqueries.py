import operator
from typing import Callable

from ...datalog import Fact
from nl_expr.expressions import Constant, Symbol
from nl_expr.logic import Conjunction, ExistentialPredicate, Negation, Union
from ...relational_algebra import Materialise, RelationalAlgebraSolver
from ..cplogic.program import CPLogicProgram
from ..deterministic_subqueries import replace_deterministic_negations

EQ = Constant(operator.eq)
NE = Constant(operator.ne)
ADD = Constant(operator.add)
STARTSWITH = Constant[Callable[[str, str], bool]](
    lambda prefix, string: string.startswith(prefix)
)

Mentions = Symbol("Mentions")
Selected = Symbol("Selected")
Term = Symbol("Term")
s = Symbol("s")
t = Symbol("t")
t1 = Symbol("t1")


def _program():
    cpl = CPLogicProgram()
    cpl.add_probabilistic_choice_from_tuples(
        Selected, [(0.5, 0), (0.5, 1)]
    )
    cpl.walk(Union(
        tuple(
            Fact(Mentions(Constant(study), Constant(term)))
            for study, term in ((0, "pain"), (1, "pain"), (1, "joy"))
        ) + tuple(Fact(Term(Constant(term))) for term in ("pain", "joy"))
    ))
    return cpl


def test_deterministic_negation_replaced_by_unevaluated_relation():
    """The relation replacing the negation is a relational algebra
    expression marked to be materialised, evaluated with the rest of the
    plan, not a set.
    """
    cpl = _program()
    formula = Conjunction((
        Selected(s), Mentions(s, t),
        Negation(ExistentialPredicate(
            t1, Conjunction((Mentions(s, t1), Term(t1), NE(t1, t)))
        )),
    ))

    new_formula, relations = replace_deterministic_negations(formula, cpl)

    assert len(relations) == 1
    symbol, relation = next(iter(relations.items()))
    assert new_formula == Conjunction((
        Selected(s), Mentions(s, t), symbol(s, t)
    ))
    assert isinstance(relation, Materialise)
    result = RelationalAlgebraSolver().walk(relation).value
    assert set(result) == {(0, "pain")}


def test_probabilistic_negation_is_not_replaced():
    cpl = _program()
    formula = Conjunction((
        Mentions(s, t),
        Negation(ExistentialPredicate(
            t1, Conjunction((Selected(s), Mentions(s, t1), NE(t1, t)))
        )),
    ))

    new_formula, relations = replace_deterministic_negations(formula, cpl)

    assert new_formula == formula
    assert relations == {}


def test_builtin_on_deterministic_atom_replaced_by_relation():
    """R(x̄) ∧ B(x̄) ≡ D(x̄): the builtin, which the lifted solvers do not
    plan, filters the deterministic relation instead."""
    cpl = _program()
    formula = Conjunction((
        Selected(s), Mentions(s, t), STARTSWITH(Constant("j"), t)
    ))

    new_formula, relations = replace_deterministic_negations(formula, cpl)

    assert len(relations) == 1
    symbol, relation = next(iter(relations.items()))
    assert new_formula == Conjunction((Selected(s), symbol(s, t)))
    result = RelationalAlgebraSolver().walk(relation).value
    assert set(result) == {(1, "joy")}


def test_builtin_on_probabilistic_atom_or_comparison_is_not_replaced():
    """A builtin of a probabilistic atom's variables, or a comparison,
    which the lifting rules plan, is left unchanged."""
    cpl = _program()
    formula = Conjunction((
        Selected(s), STARTSWITH(Constant("1"), s),
        Mentions(Constant(1), t), NE(t, Constant("joy")),
    ))

    new_formula, relations = replace_deterministic_negations(formula, cpl)

    assert new_formula == formula
    assert relations == {}


def test_function_of_a_probabilistic_atom_replaced_by_relation():
    """R(s) ∧ u = s + 1 ≡ R(s) ∧ D(s, u), D holding the values of the
    function on the tuples of R: a deterministic relation which is a
    function of s, for the separator variables."""
    cpl = _program()
    u = Symbol("u")
    formula = Conjunction((Selected(s), EQ(u, ADD(s, Constant(1)))))

    new_formula, relations = replace_deterministic_negations(formula, cpl)

    assert len(relations) == 1
    symbol, relation = next(iter(relations.items()))
    assert new_formula == Conjunction((Selected(s), symbol(s, u)))
    result = RelationalAlgebraSolver().walk(relation).value
    assert set(result) == {(0, 1), (1, 2)}


def test_function_of_a_probabilistic_atom_named_in_the_program():
    """The function of u = f(s) is a symbol of the program, as the
    functions the user adds to an engine are: the relation computes it."""
    cpl = _program()
    f = Symbol("f")
    cpl.symbol_table[f] = Constant[Callable[[int], int]](lambda x: x + 10)
    u = Symbol("u")
    formula = Conjunction((Selected(s), EQ(u, f(s))))

    new_formula, relations = replace_deterministic_negations(formula, cpl)

    symbol, relation = next(iter(relations.items()))
    assert new_formula == Conjunction((Selected(s), symbol(s, u)))
    result = RelationalAlgebraSolver().walk(relation).value
    assert set(result) == {(0, 10), (1, 11)}
