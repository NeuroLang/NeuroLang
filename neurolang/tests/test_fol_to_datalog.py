import typing
from unittest.mock import patch

from ..expressions import Symbol, Constant
from ..logic import (
    Implication,
    Disjunction,
    Conjunction,
    Negation,
    UniversalPredicate,
    ExistentialPredicate,
)
from ..logic.horn_clauses import (
    EliminateImplications,
    MoveNegationsToAtoms,
    Skolemize,
    Symbol,
)


def test_remove_implication():
    A = Symbol("A")
    B = Symbol("B")
    exp = Implication(B, A)
    res = EliminateImplications().walk(exp)

    assert isinstance(res, Disjunction)
    assert res == Disjunction((B, Negation(A)))


def test_remove_nested_implication():
    A = Symbol("A")
    B = Symbol("B")
    exp = Implication(Implication(Negation(A), Negation(B)), Implication(B, A))
    res = EliminateImplications().walk(exp)

    assert isinstance(res, Disjunction)
    assert res == Disjunction(
        (
            Disjunction((Negation(A), Negation(Negation(B)))),
            Negation(Disjunction((B, Negation(A)))),
        )
    )


def test_remove_nested_implication():
    A = Symbol("A")
    B = Symbol("B")
    exp = Disjunction((A, Implication(B, A)))
    res = EliminateImplications().walk(exp)
    assert res == Disjunction((A, Disjunction((B, Negation(A)))))


def test_remove_double_negation():
    A = Symbol("A")
    exp = Negation(Negation(A))
    res = MoveNegationsToAtoms().walk(exp)
    assert res == A


def test_negated_universal():
    X = Symbol("X")
    P = Symbol("P")
    exp = Negation(UniversalPredicate(X, P(X)))
    res = MoveNegationsToAtoms().walk(exp)
    assert res == ExistentialPredicate(X, Negation(P(X)))


def test_negated_existensial():
    X = Symbol("X")
    P = Symbol("P")
    exp = Negation(ExistentialPredicate(X, P(X)))
    res = MoveNegationsToAtoms().walk(exp)
    assert res == UniversalPredicate(X, Negation(P(X)))


def test_negated_disjunction():
    A = Symbol("A")
    B = Symbol("B")
    exp = Negation(Disjunction((A, B)))
    res = MoveNegationsToAtoms().walk(exp)
    assert res == Conjunction((Negation(A), Negation(B)))


def test_negated_conjunction():
    A = Symbol("A")
    B = Symbol("B")
    exp = Negation(Conjunction((A, B)))
    res = MoveNegationsToAtoms().walk(exp)
    assert res == Disjunction((Negation(A), Negation(B)))


def test_long_negated_chain():
    A = Symbol("A")
    X = Symbol("X")
    Y = Symbol("Y")
    R = Symbol("R")
    exp = Negation(
        Disjunction(
            (
                A,
                UniversalPredicate(
                    X, ExistentialPredicate(Y, Negation(R(X, Y)))
                ),
            )
        )
    )
    res = MoveNegationsToAtoms().walk(exp)
    assert res == Conjunction(
        (Negation(A), ExistentialPredicate(X, UniversalPredicate(Y, R(X, Y))))
    )


def test_skolemize_without_existentials():
    A = Symbol("A")
    B = Symbol("B")
    exp = Negation(Conjunction((A, B)))
    res = Skolemize().walk(exp)
    assert res == exp


def test_skolemize_simple_expression():
    P = Symbol("P")
    X = Symbol("X")
    exp = ExistentialPredicate(X, P(X))
    sk = Skolemize()
    res = sk.walk(exp)
    c0 = sk.used_symbols[0]
    assert res == P(c0)


def test_skolemize_universally_quantified():
    X = Symbol("X")
    Y = Symbol("Y")
    exp = UniversalPredicate(Y, ExistentialPredicate(X, Disjunction((Y, X))))
    sk = Skolemize()
    res = sk.walk(exp)
    c0 = sk.used_symbols[0]
    assert res == UniversalPredicate(Y, Disjunction((Y, c0(Y))))


def test_skolemize_nested_universals():
    X = Symbol("X")
    Y = Symbol("Y")
    Z = Symbol("Z")
    exp = UniversalPredicate(
        Z,
        UniversalPredicate(Y, ExistentialPredicate(X, Disjunction((Z, Y, X)))),
    )
    sk = Skolemize()
    res = sk.walk(exp)
    c0 = sk.used_symbols[0]
    assert res == UniversalPredicate(
        Z, UniversalPredicate(Y, Disjunction((Z, Y, c0(Z, Y)))),
    )


def test_skolemize_multiple_existentials():
    X = Symbol("X")
    Y = Symbol("Y")
    Z = Symbol("Z")
    exp = ExistentialPredicate(
        X,
        UniversalPredicate(Y, ExistentialPredicate(Z, Conjunction((X, Y, Z)))),
    )
    sk = Skolemize()
    res = sk.walk(exp)
    c0, c1 = sk.used_symbols
    assert res == UniversalPredicate(Y, Conjunction((c0, Y, c1(Y))))


def test_skolemize_repeated_symbols():
    X = Symbol("X")
    Y = Symbol("Y")
    P = Symbol("P")
    exp = UniversalPredicate(
        X, Disjunction((P(X), ExistentialPredicate(X, P(X))))
    )
    sk = Skolemize()
    res = sk.walk(exp)
    c0 = sk.used_symbols[0]
    assert res == UniversalPredicate(X, Disjunction((P(X), P(c0(X)))))
