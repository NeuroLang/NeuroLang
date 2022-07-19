from ...expressions import Symbol
from .. import (
    Conjunction,
    Disjunction,
    ExistentialPredicate,
    Negation,
    UniversalPredicate
)
from ..transformations import (
    PushExistentialsDown,
    PushUniversalsDown
)

P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
X = Symbol("X")
Y = Symbol("Y")
Z = Symbol("Z")


PED = PushExistentialsDown()
PUD = PushUniversalsDown()


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
