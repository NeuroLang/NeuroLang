import typing

from ..expressions import Symbol, Constant
from ..logic import Implication, Disjunction, Conjunction, Negation, UniversalPredicate, ExistentialPredicate
from ..logic.horn_clauses import EliminateImplications, MoveNegationsToAtoms



def test_remove_implication():
    A = Symbol('A')
    B = Symbol('B')
    exp = Implication(B, A)
    res = EliminateImplications().walk(exp)
    
    assert isinstance(res, Disjunction)
    assert res == Disjunction((B, Negation(A)))


def test_remove_nested_implication():
    A = Symbol('A')
    B = Symbol('B')
    exp = Implication(
            Implication(Negation(A), Negation(B)),
            Implication(B, A)
          )
    res = EliminateImplications().walk(exp)
    
    assert isinstance(res, Disjunction)
    assert res == Disjunction((
            Disjunction((Negation(A), Negation(Negation(B)))),
            Negation(Disjunction((B, Negation(A))))
        ))


def test_remove_double_negation():
    A = Symbol('A')
    exp = Negation(Negation(A))
    res = MoveNegationsToAtoms().walk(exp)
    assert res == A


def test_negated_universal():
    X = Symbol('X')
    P = Symbol('P')
    exp = Negation(UniversalPredicate(X, P(X)))
    res = MoveNegationsToAtoms().walk(exp)
    assert res == ExistentialPredicate(X, Negation(P(X)))


def test_negated_existensial():
    X = Symbol('X')
    P = Symbol('P')
    exp = Negation(ExistentialPredicate(X, P(X)))
    res = MoveNegationsToAtoms().walk(exp)
    assert res == UniversalPredicate(X, Negation(P(X)))


def test_negated_disjunction():
    A = Symbol('A')
    B = Symbol('B')
    exp = Negation(Disjunction((A, B)))
    res = MoveNegationsToAtoms().walk(exp)
    assert res == Conjunction((Negation(A), Negation(B)))


def test_negated_conjunction():
    A = Symbol('A')
    B = Symbol('B')
    exp = Negation(Conjunction((A, B)))
    res = MoveNegationsToAtoms().walk(exp)
    assert res == Disjunction((Negation(A), Negation(B)))


def test_long_negated_chain():
    A = Symbol('A')
    X = Symbol('X')
    Y = Symbol('Y')
    R = Symbol('R')
    exp = Negation(
            Disjunction((
                A,
                UniversalPredicate(
                    X,
                    ExistentialPredicate(
                        Y,
                        Negation(R(X, Y))
                    )
                )
            ))
        )
    res = MoveNegationsToAtoms().walk(exp)
    assert res == Conjunction((
            Negation(A),
            ExistentialPredicate(
                X,
                UniversalPredicate(
                    Y,
                    R(X, Y)
                )
            )
        ))
