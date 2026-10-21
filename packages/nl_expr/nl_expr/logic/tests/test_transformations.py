from ...expressions import Symbol
from .. import (
    Conjunction,
    Disjunction,
    ExistentialPredicate,
    Negation,
    UniversalPredicate
)
from ..transformations import (
    AbsorbAssertedConjuncts,
    AbsorbAssertedConjunctsInNegations,
    PushExistentialsDown,
    PushUniversalsDown,
    standardise_apart
)

P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
X = Symbol("X")
Y = Symbol("Y")
Z = Symbol("Z")


PED = PushExistentialsDown()
PUD = PushUniversalsDown()
AACN = AbsorbAssertedConjunctsInNegations()
AAC = AbsorbAssertedConjuncts()


def test_push_existentials_down():
    exp = ExistentialPredicate(
        X,
        Disjunction((P(X), Q(X)))
    )

    assert PED.walk(exp) == Disjunction((
        ExistentialPredicate(X, P(X)),
        ExistentialPredicate(X, Q(X))
    ))

    exp = ExistentialPredicate(
        X, Conjunction((
            P(X), Q(X)
        ))
    )

    assert PED.walk(exp) == exp

    exp = ExistentialPredicate(
        X, Conjunction((
            P(Y), Q(X)
        ))
    )

    assert PED.walk(exp) == Conjunction((
        ExistentialPredicate(X, Q(X)),
        P(Y)
    ))

    exp = ExistentialPredicate(
        X, Conjunction((
            P(Y), Q(X)
        ))
    )

    assert PED.walk(exp) == Conjunction((
        ExistentialPredicate(X, Q(X)),
        P(Y)
    ))

    exp = ExistentialPredicate(
        X, Conjunction((
            P(Y, Z), Negation(Q(X, Z))
        ))
    )

    assert PED.walk(exp) == exp

    exp = ExistentialPredicate(
        X,
        ExistentialPredicate(
            Y,
            Conjunction((
                P(X), Q(Y)
            ))
        )
    )

    assert PED.walk(exp) == Conjunction((
        ExistentialPredicate(X, P(X)),
        ExistentialPredicate(Y, Q(Y))
    ))

    exp = ExistentialPredicate(
        X,
        UniversalPredicate(
            Y,
            Conjunction((P(X), Q(Y)))
        )
    )

    assert PED.walk(exp) == exp


def test_push_universals_down():
    exp = UniversalPredicate(
        X,
        Conjunction((P(X), Q(X)))
    )

    assert PUD.walk(exp) == Conjunction((
        UniversalPredicate(X, P(X)),
        UniversalPredicate(X, Q(X))
    ))

    exp = UniversalPredicate(
        X, Disjunction((
            P(X), Q(X)
        ))
    )

    assert PUD.walk(exp) == exp

    exp = UniversalPredicate(
        X, Disjunction((
            P(Y), Q(X)
        ))
    )

    assert PUD.walk(exp) == Disjunction((
        UniversalPredicate(X, Q(X)),
        P(Y)
    ))

    exp = UniversalPredicate(
        X, Disjunction((
            P(Y), Q(X)
        ))
    )

    assert PUD.walk(exp) == Disjunction((
        UniversalPredicate(X, Q(X)),
        P(Y)
    ))

    exp = UniversalPredicate(
        X, Conjunction((
            P(Y, Z), Negation(Q(X, Z))
        ))
    )

    assert PUD.walk(exp) == Conjunction((
        P(Y, Z),
        UniversalPredicate(X, Negation(Q(X, Z)))
    ))

    exp = UniversalPredicate(
        X,
        UniversalPredicate(
            Y,
            Conjunction((
                P(X), Q(Y)
            ))
        )
    )

    assert PUD.walk(exp) == Conjunction((
        UniversalPredicate(X, P(X)),
        UniversalPredicate(Y, Q(Y))
    ))

    exp = UniversalPredicate(
        X,
        ExistentialPredicate(
            Y,
            Conjunction((P(X), Q(Y)))
        )
    )

    assert PUD.walk(exp) == exp


def test_absorb_asserted_conjuncts_in_negation():
    exp = Conjunction((P(X), Negation(Conjunction((P(X), Q(X))))))

    assert AACN.walk(exp) == Conjunction((P(X), Negation(Q(X))))


def test_absorb_asserted_conjuncts_in_negated_existential():
    exp = Conjunction((
        P(X), Q(X, Z),
        Negation(ExistentialPredicate(
            Y, Conjunction((P(X), Q(X, Z), R(X, Y)))
        ))
    ))

    assert AACN.walk(exp) == Conjunction((
        P(X), Q(X, Z),
        Negation(ExistentialPredicate(Y, R(X, Y)))
    ))


def test_absorb_asserted_conjuncts_keeps_bound_variables():
    exp = Conjunction((
        P(Y),
        Negation(ExistentialPredicate(Y, Conjunction((P(Y), Q(X, Y)))))
    ))

    assert AACN.walk(exp) == exp


def test_absorb_asserted_conjuncts_keeps_contradictions():
    exp = Conjunction((P(X), Negation(P(X))))

    assert AACN.walk(exp) == exp

    exp = Conjunction((P(X), Negation(Conjunction((P(X), P(X))))))

    assert AACN.walk(exp) == exp


def test_absorb_asserted_conjuncts_within_quantifiers():
    exp = ExistentialPredicate(X, Conjunction((
        P(X), Negation(Conjunction((P(X), Q(X))))
    )))

    assert AACN.walk(exp) == ExistentialPredicate(
        X, Conjunction((P(X), Negation(Q(X))))
    )


def test_absorb_disjuncts_contradicting_asserted_conjuncts():
    """A ∧ (B ∨ (D ∧ ¬A)) ≡ A ∧ B, the remaining disjunct being
    collapsed into the conjunction."""
    exp = Conjunction((
        P(X), Disjunction((Q(X), Conjunction((R(X), Negation(P(X))))))
    ))

    assert AAC.walk(exp) == Conjunction((P(X), Q(X)))

    exp = Conjunction((
        P(X), Disjunction((
            Conjunction((Q(X), R(X))), Negation(P(X)),
            Conjunction((R(X), Negation(P(X))))
        ))
    ))

    assert AAC.walk(exp) == Conjunction((P(X), Q(X), R(X)))


def test_absorb_self_contradicting_disjuncts():
    exp = Disjunction((Q(X), Conjunction((P(X), R(X), Negation(P(X))))))

    assert AAC.walk(exp) == Q(X)


def test_absorb_contradicting_disjuncts_through_existentials():
    exp = Conjunction((
        P(X), Disjunction((
            Q(X),
            ExistentialPredicate(Y, Conjunction((R(Y), Negation(P(X)))))
        ))
    ))

    assert AAC.walk(exp) == Conjunction((P(X), Q(X)))


def test_absorb_contradicting_disjuncts_keeps_bound_variables():
    exp = Conjunction((
        P(Y), Disjunction((
            Q(X),
            ExistentialPredicate(Y, Conjunction((R(Y), Negation(P(Y)))))
        ))
    ))

    assert AAC.walk(exp) == exp


def test_absorb_contradicting_disjuncts_keeps_contradictions():
    """If every disjunct is a contradiction, the formula is left
    unchanged."""
    exp = Conjunction((
        P(X), Disjunction((
            Negation(P(X)), Conjunction((Q(X), Negation(P(X))))
        ))
    ))

    assert AAC.walk(exp) == exp

    exp = Disjunction((
        Conjunction((P(X), Negation(P(X)))),
        Conjunction((Q(X), Negation(Q(X)))),
    ))

    assert AAC.walk(exp) == exp


def test_absorb_asserted_conjuncts_still_absorbs_in_negations():
    exp = Conjunction((P(X), Negation(Conjunction((P(X), Q(X))))))

    assert AAC.walk(exp) == Conjunction((P(X), Negation(Q(X))))


def test_standardise_apart():
    """Each quantified variable is renamed to a fresh one, the free
    variables are kept."""
    exp = Conjunction((
        P(X),
        ExistentialPredicate(
            X, Conjunction((Q(X), Negation(ExistentialPredicate(X, R(X)))))
        ),
        UniversalPredicate(Y, P(Y)),
    ))
    res = standardise_apart(exp)
    free, outer, universal = res.formulas
    assert free == P(X)
    inner = outer.body.formulas[1].formula
    heads = {outer.head, inner.head, universal.head}
    assert len(heads) == 3 and heads.isdisjoint({X, Y})
    assert outer.body.formulas[0] == Q(outer.head)
    assert inner.body == R(inner.head)
    assert universal.body == P(universal.head)
