import pytest

from ...datalog import Fact
from ...exceptions import ForbiddenExpressionError
from ...expressions import Constant, Symbol
from ...logic import Implication, Union
from ...logic.unification import apply_substitution, most_general_unifier
from ..ppdl import PPDL, PPDLDeltaTerm, get_dterm

C_ = Constant
S_ = Symbol

x = S_("x")
y = S_("y")
z = S_("z")
P = S_("P")
Q = S_("Q")
R = S_("R")
W = S_("W")
K = S_("K")
Z = S_("Z")
p = S_("p")
q = S_("q")
a = C_("a")
b = C_("b")

bernoulli = S_("bernoulli")


def test_get_dterm():
    datom = P(x, y, PPDLDeltaTerm(S_("Hi"), tuple()))
    assert get_dterm(datom) == PPDLDeltaTerm(S_("Hi"), tuple())
    datom = P(x, y, PPDLDeltaTerm(S_("Hi"), (C_(2),)))
    assert get_dterm(datom) == PPDLDeltaTerm(S_("Hi"), (C_(2),))


def test_apply_substitution_to_delta_term():
    dterm = PPDLDeltaTerm(bernoulli, (p,))
    new_dterm = apply_substitution(dterm, {p: q})
    assert new_dterm == PPDLDeltaTerm(bernoulli, (q,))

    substitution = {S_("random_symbol"): S_("another_random_symbol")}
    new_dterm = apply_substitution(dterm, substitution)
    assert new_dterm == dterm


def test_apply_substitution_to_delta_atom():
    datom = P(x, PPDLDeltaTerm(bernoulli, (p,)))
    new_datom = apply_substitution(datom, {p: q})
    assert new_datom == P(x, PPDLDeltaTerm(bernoulli, (q,)))


def test_unification_of_delta_atom():
    a = P(x, PPDLDeltaTerm(bernoulli, (p,)))
    b = P(y, PPDLDeltaTerm(bernoulli, (q,)))
    mgu = most_general_unifier(a, b)
    assert mgu is not None
    unifier, _ = mgu
    assert unifier == {x: y, p: q}


def test_ppdl_program():
    tau_1 = Implication(P(x, PPDLDeltaTerm(bernoulli, (C_(0.5),))), Q(x))
    program = Union((tau_1,))
    ppdl = PPDL()
    ppdl.walk(program)
    edb = ppdl.extensional_database()
    idb = ppdl.intensional_database()
    assert tau_1 in idb[P].formulas

    with pytest.raises(ForbiddenExpressionError):
        tau_2 = Implication(
            P(
                x,
                PPDLDeltaTerm(bernoulli, tuple()),
                PPDLDeltaTerm(S_("Flap"), tuple()),
            ),
            Q(x),
        )
        ppdl = PPDL()
        ppdl.walk(Union((tau_2, Fact(Q(C_(2))))))
        edb = ppdl.extensional_database()
        idb = ppdl.intensional_database()
        assert tau_2 in idb[P].formulas
        assert Fact(Q(C_(2))) in edb[Q]


def test_burglar():
    City = S_("City")
    House = S_("House")
    Business = S_("Business")
    Unit = S_("Unit")
    Earthquake = S_("Earthquake")
    Burglary = S_("Burglary")
    Trig = S_("Trig")
    Alarm = S_("Alarm")
    x, h, b, c, r = S_("x"), S_("h"), S_("b"), S_("c"), S_("r")
    program = Union(
        (
            Implication(Unit(h, c), House(h, c)),
            Implication(Unit(b, c), Business(b, c)),
            Implication(
                Earthquake(c, PPDLDeltaTerm(bernoulli, (C_(0.01),))),
                City(c, r),
            ),
            Implication(
                Burglary(x, c, PPDLDeltaTerm(bernoulli, (r,))),
                Unit(x, c) & City(c, r),
            ),
            Implication(
                Trig(x, PPDLDeltaTerm(bernoulli, (C_(0.6),))),
                Unit(x, c) & Earthquake(c, C_(1)),
            ),
            Implication(
                Trig(x, PPDLDeltaTerm(bernoulli, (C_(0.9),))),
                Burglary(x, c, C_(1)),
            ),
            Implication(Alarm(x), Trig(x, C_(1))),
        )
    )


def test_pcs_example():
    Uniform = S_("uniform")
    Gender = S_("Gender")
    Subject = S_("Subject")
    pGender = S_("pGender")
    pHasLPC = S_("pHasLPC")
    pHasRPC = S_("pHasRPC")
    HasLPC = S_("HasLPC")
    HasRPC = S_("HasRPC")
    x = S_("x")
    p = S_("p")
    program = Union(
        (
            Implication(
                Gender(x, PPDLDeltaTerm(bernoulli, (p,))),
                Subject(x) & pGender(p),
            ),
            Implication(
                HasLPC(x, PPDLDeltaTerm(bernoulli, (p,))),
                Subject(x) & pHasLPC(p),
            ),
            Implication(
                HasRPC(x, PPDLDeltaTerm(bernoulli, (p,))),
                Subject(x) & pHasRPC(p),
            ),
            Implication(
                pGender(PPDLDeltaTerm(Uniform, (C_(0), C_(1)))), C_("True")
            ),
            Implication(
                pHasLPC(PPDLDeltaTerm(Uniform, (C_(0), C_(1)))), C_("True")
            ),
            Implication(
                pHasRPC(PPDLDeltaTerm(Uniform, (C_(0), C_(1)))), C_("True")
            ),
        )
    )
