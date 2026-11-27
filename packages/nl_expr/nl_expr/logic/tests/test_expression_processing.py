import operator

from ...expressions import Constant, Symbol
from .. import Conjunction, Disjunction, ExistentialPredicate, Negation
from ..expression_processing import (
    ExtractConjunctAtoms,
    ExtractExistentialPrefix,
    ExtractRelationalAtoms,
    extract_logic_free_variables
)

P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
X = Symbol("X")
Y = Symbol("Y")
Z = Symbol("Z")
NE = Constant(operator.ne)


def test_free_variables_of_a_conjunction_applied_to_logic_operands():
    """``a(x) & <formula>`` of the query builder holds the formula as an
    argument of ``and_``; its quantified variables are not free."""
    conjunction = operator.and_
    assert set(extract_logic_free_variables(
        Constant(conjunction)(P(X), Conjunction((Q(X, Y),)))
    )) == {X, Y}
    assert set(extract_logic_free_variables(
        Constant(conjunction)(P(X), ExistentialPredicate(Y, Q(X, Y)))
    )) == {X}


def test_extract_conjunct_atoms():
    formula = Conjunction((
        P(X), NE(X, Y), Negation(Q(X)), ExistentialPredicate(Z, R(X, Z)),
        Disjunction((Q(Y), R(X, Y))), R(X, Y)
    ))
    assert list(ExtractConjunctAtoms().walk(formula)) == [P(X), R(X, Y)]
    assert list(ExtractConjunctAtoms().walk(P(X))) == [P(X)]


def test_extract_relational_atoms():
    formula = Conjunction((
        P(X), NE(X, Y), Negation(Q(X)), ExistentialPredicate(Z, R(X, Z))
    ))
    assert list(ExtractRelationalAtoms().walk(formula)) == [
        P(X), Q(X), R(X, Z)
    ]


def test_extract_existential_prefix():
    body = Conjunction((P(X), Q(Y), ExistentialPredicate(Z, R(X, Z))))
    formula = ExistentialPredicate(X, ExistentialPredicate(Y, body))
    variables, extracted = ExtractExistentialPrefix().walk(formula)
    assert list(variables) == [X, Y]
    assert extracted == body
    variables, extracted = ExtractExistentialPrefix().walk(body)
    assert len(variables) == 0 and extracted == body
