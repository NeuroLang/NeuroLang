import operator

from ...expressions import Constant, Symbol
from .. import Conjunction, Disjunction, ExistentialPredicate, Negation
from ..expression_processing import (
    ExtractConjunctAtoms,
    ExtractExistentialPrefix
)

P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
X = Symbol("X")
Y = Symbol("Y")
Z = Symbol("Z")
NE = Constant(operator.ne)


def test_extract_conjunct_atoms():
    formula = Conjunction((
        P(X), NE(X, Y), Negation(Q(X)), ExistentialPredicate(Z, R(X, Z)),
        Disjunction((Q(Y), R(X, Y))), R(X, Y)
    ))
    assert list(ExtractConjunctAtoms().walk(formula)) == [P(X), R(X, Y)]
    assert list(ExtractConjunctAtoms().walk(P(X))) == [P(X)]


def test_extract_existential_prefix():
    body = Conjunction((P(X), Q(Y), ExistentialPredicate(Z, R(X, Z))))
    formula = ExistentialPredicate(X, ExistentialPredicate(Y, body))
    variables, extracted = ExtractExistentialPrefix().walk(formula)
    assert list(variables) == [X, Y]
    assert extracted == body
    variables, extracted = ExtractExistentialPrefix().walk(body)
    assert len(variables) == 0 and extracted == body
