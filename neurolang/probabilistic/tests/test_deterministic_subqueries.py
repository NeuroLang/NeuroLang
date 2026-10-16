import operator

from ...datalog import Fact
from ...expressions import Constant, Symbol
from ...logic import Conjunction, ExistentialPredicate, Negation, Union
from ...relational_algebra import (
    RelationalAlgebraOperation,
    RelationalAlgebraSolver
)
from ..cplogic.program import CPLogicProgram
from ..deterministic_subqueries import replace_deterministic_negations

NE = Constant(operator.ne)

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
    expression, evaluated with the rest of the plan, not a set.
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
    assert isinstance(relation, RelationalAlgebraOperation)
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
