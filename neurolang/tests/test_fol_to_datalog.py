import typing
from unittest.mock import patch

from ..expressions import Symbol, Constant, FunctionApplication
from ..logic.unification import most_general_unifier
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
    RemoveUniversalPredicates,
    Symbol,
    DistributeDisjunctions,
    DesambiguateQuantifiedVariables,
    CollapseDisjunctions,
    CollapseConjunctions,
    convert_to_pnf_with_cnf_matrix,
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


def test_skolemize_nested_existentials():
    X = Symbol("X")
    Y = Symbol("Y")
    P = Symbol("P")
    R = Symbol("R")
    exp = ExistentialPredicate(
        X, Disjunction((P(X), ExistentialPredicate(X, P(X)), R(X)))
    )
    sk = Skolemize()
    res = sk.walk(exp)
    c0, c1 = sk.used_symbols
    assert res == Disjunction((P(c0), P(c1), R(c0)))


def test_skolemize_nested_universal_with_repeated_symbols():
    X = Symbol("X")
    Y = Symbol("Y")
    P = Symbol("P")
    R = Symbol("R")
    exp = UniversalPredicate(
        X,
        Disjunction(
            (P(X), UniversalPredicate(X, ExistentialPredicate(Y, R(Y, X))))
        ),
    )
    sk = Skolemize()

    res = sk.walk(exp)
    c0 = sk.used_symbols[0]

    assert isinstance(res, UniversalPredicate)
    assert res.head == X
    assert isinstance(res.body, Disjunction)
    left, right = res.body.formulas
    assert left == P(X)
    assert isinstance(right, UniversalPredicate)
    assert right.body == R(c0(X, right.head), right.head)


def test_remove_universal_predicate():
    X = Symbol("X")
    P = Symbol("P")
    exp = UniversalPredicate(X, P(X))
    res = RemoveUniversalPredicates().walk(exp)
    assert isinstance(res, FunctionApplication)
    assert res.functor == P
    assert isinstance(res.args[0], Symbol)


def test_remove_nested_universal_predicates():
    Y = Symbol("Y")
    X = Symbol("X")
    P = Symbol("P")
    R = Symbol("R")
    exp = UniversalPredicate(
        X, Disjunction((P(X), UniversalPredicate(Y, R(X, Y))))
    )
    res = RemoveUniversalPredicates().walk(exp)

    c0 = res.formulas[0].args[0]
    c1 = res.formulas[1].args[1]

    assert c0 != c1
    assert res == Disjunction((P(c0), R(c0, c1)))


def test_remove_repeated_symbols_in_nested_universal_predicates():
    X = Symbol("X")
    P = Symbol("P")
    exp = UniversalPredicate(
        X, Disjunction((P(X), UniversalPredicate(X, P(X))))
    )
    res = DesambiguateQuantifiedVariables().walk(exp)

    c0 = res.body.formulas[0].args[0]
    c1 = res.body.formulas[1].body.args[0]

    assert c0 != c1


def test_remove_sibling_universal_predicates():
    Y = Symbol("Y")
    X = Symbol("X")
    P = Symbol("P")
    R = Symbol("R")
    exp = Disjunction(
        (
            UniversalPredicate(X, P(X)),
            UniversalPredicate(X, Conjunction((P(X), R(X, Y)))),
            P(Y),
        )
    )
    res = DesambiguateQuantifiedVariables().walk(exp)
    res = RemoveUniversalPredicates().walk(res)

    c0 = res.formulas[0].args[0]
    c1 = res.formulas[1].formulas[0].args[0]

    assert c0 != c1
    assert res == Disjunction((P(c0), Conjunction((P(c1), R(c1, Y))), P(Y)))


def test_distribute_disjunction():
    P = Symbol("P")
    Q = Symbol("Q")
    R = Symbol("R")
    exp = Disjunction((P, Conjunction((Q, R))))
    res = DistributeDisjunctions().walk(exp)
    assert res == Conjunction((Disjunction((P, Q)), Disjunction((P, R)),))


def test_distribute_disjunction():
    P = Symbol("P")
    Q = Symbol("Q")
    R = Symbol("R")
    exp = Disjunction((P, Conjunction((Q, R))))
    res = DistributeDisjunctions().walk(exp)
    assert res == Conjunction((Disjunction((P, Q)), Disjunction((P, R)),))


def test_distribute_non_binary_disjunction():
    P = Symbol("P")
    Q = Symbol("Q")
    R = Symbol("R")
    S = Symbol("S")
    exp = Disjunction((P, Conjunction((Q, R)), S))
    res = DistributeDisjunctions().walk(exp)
    assert res == Conjunction(
        (
            Disjunction((P, Disjunction((Q, S)))),
            Disjunction((P, Disjunction((R, S)))),
        )
    )


def test_collapse_disjunctions():
    P = Symbol("P")
    Q = Symbol("Q")
    R = Symbol("R")
    S = Symbol("S")
    exp = Conjunction(
        (
            Disjunction((P, Disjunction((Q, S)))),
            Disjunction((P, Disjunction((R, S)))),
        )
    )
    res = CollapseDisjunctions().walk(exp)
    assert res == Conjunction(
        (Disjunction((P, Q, S)), Disjunction((P, R, S)),)
    )


def test_collapse_conjunctions():
    P = Symbol("P")
    Q = Symbol("Q")
    R = Symbol("R")
    S = Symbol("S")
    exp = Conjunction((Conjunction((P, Conjunction((Q, R)))), S))
    res = CollapseConjunctions().walk(exp)
    assert res == Conjunction((P, Q, R, S))


def test_transform_to_cnf():
    A = Symbol("A")
    B = Symbol("B")
    C = Symbol("C")
    D = Symbol("D")
    P = Symbol("P")
    Q = Symbol("Q")
    exp = Disjunction((P, Conjunction((A, B, Disjunction((C, D)))), Q))
    res = convert_to_pnf_with_cnf_matrix(exp)
    assert res == Conjunction(
        (
            Disjunction((P, A, Q)),
            Disjunction((P, B, Q)),
            Disjunction((P, C, D, Q)),
        )
    )


def test_transform_to_cnf_idempotence():
    P = Symbol("P")
    Q = Symbol("Q")
    R = Symbol("R")

    exp = Disjunction((P, Q))
    res = convert_to_pnf_with_cnf_matrix(exp)
    assert res == exp

    exp = Conjunction((P, Q))
    res = convert_to_pnf_with_cnf_matrix(exp)
    assert res == exp

    exp = Conjunction((Disjunction((P, Q)), Disjunction((P, R))))
    res = convert_to_pnf_with_cnf_matrix(exp)
    assert res == exp


def test_transform_to_cnf_1():
    Y = Symbol("Y")
    X = Symbol("X")
    P = Symbol("P")
    R = Symbol("R")
    exp = UniversalPredicate(
        Y,
        Implication(P(Y), UniversalPredicate(X, Disjunction((R(Y, X), P(X))))),
    )
    res = convert_to_pnf_with_cnf_matrix(exp)
    Y, X = res.head, res.body.head
    assert res == UniversalPredicate(
        Y,
        ExistentialPredicate(
            X,
            Conjunction(
                (
                    Disjunction((P(Y), Negation(R(Y, X)),)),
                    Disjunction((P(Y), Negation(P(X)),)),
                )
            ),
        ),
    )


def test_transform_to_cnf_2():
    socrates = Symbol("socrates")
    mortal = Symbol("mortal")
    man = Symbol("man")
    X = Symbol("X")
    exp = Conjunction(
        (UniversalPredicate(X, Implication(mortal(X), man(X))), man(socrates))
    )
    res = convert_to_pnf_with_cnf_matrix(exp)
    X = res.head
    assert res == UniversalPredicate(
        X,
        Conjunction(
            (Disjunction((mortal(X), Negation(man(X)))), man(socrates))
        ),
    )


def test_transform_to_cnf_3():
    father = Symbol("father")
    sister = Symbol("sister")
    Alice = Symbol("Alice")
    Bob = Symbol("Bob")
    Carol = Symbol("Carol")
    X = Symbol("X")
    Y = Symbol("Y")
    Z = Symbol("Z")

    exp = Conjunction(
        (
            UniversalPredicate(
                X,
                UniversalPredicate(
                    Y,
                    ExistentialPredicate(
                        Z,
                        Implication(
                            sister(X, Y),
                            Conjunction((father(X, Z), father(Y, Z))),
                        ),
                    ),
                ),
            ),
            father(Alice, Bob),
            father(Carol, Bob),
        )
    )

    res = convert_to_pnf_with_cnf_matrix(exp)

    X, Y, Z = res.head, res.body.head, res.body.body.head
    assert res == UniversalPredicate(
        X,
        UniversalPredicate(
            Y,
            ExistentialPredicate(
                Z,
                Conjunction(
                    (
                        Disjunction(
                            (
                                sister(X, Y),
                                Negation(father(X, Z)),
                                Negation(father(Y, Z)),
                            )
                        ),
                        father(Alice, Bob),
                        father(Carol, Bob),
                    )
                ),
            ),
        ),
    )
