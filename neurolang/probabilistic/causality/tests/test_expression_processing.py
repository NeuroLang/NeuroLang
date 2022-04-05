import pytest

from ....expressions import Constant, Symbol
from ....logic import Conjunction
from ...exceptions import MalformedCausalOperatorError
from ...expressions import Condition
from ..expressions import CausalIntervention, CausalInterventionWalker

def test_symbols_not_allowed():
    P = Symbol("P")
    Q = Symbol("Q")

    x = Symbol("x")
    with pytest.raises(MalformedCausalOperatorError, match="The atoms intervened by the*"):
        Condition(
            P(x),
            Conjunction((
                CausalIntervention((
                    Q(x),
                )),
            ))
        )

def test_do_instantiation():
    P = Symbol("P")
    Q = Symbol("Q")

    x = Symbol("x")
    a = Constant("a")

    imp = Condition(
        P(x),
        Conjunction((
            CausalIntervention((
                Q(a),
            )),
        ))
    )
    ciw = CausalInterventionWalker()

    ci = ciw.walk(imp)
    assert ci == CausalIntervention((Q(a),))


def test_do_instantiation_more_atoms():
    P = Symbol("P")
    Q = Symbol("Q")
    R = Symbol("R")

    x = Symbol("x")
    a = Constant("a")
    b = Constant("b")

    imp = Condition(
        P(x),
        Conjunction((
            CausalIntervention((
                Q(a),R(b)
            )),
        ))
    )
    ciw = CausalInterventionWalker()

    ciw.walk(imp)
    assert CausalIntervention((Q(a),R(b)))

def test_multiple_instantiation_exception():
    P = Symbol("P")
    Q = Symbol("Q")
    R = Symbol("R")

    x = Symbol("x")
    a = Constant("a")

    imp = Condition(
        P(x),
        Conjunction((
            CausalIntervention((
                Q(a),
            )),
            CausalIntervention((
                R(a),
            )),
        ))
    )
    ciw = CausalInterventionWalker()
    with pytest.raises(MalformedCausalOperatorError, match="The use of more than one DO operator*"):
        ciw.walk(imp)

