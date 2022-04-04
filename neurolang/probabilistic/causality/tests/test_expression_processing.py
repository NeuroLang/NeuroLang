import pytest

from ....logic import Conjunction
from ...exceptions import MalformedCausalOperatorError
from ...expressions import Condition, Symbol
from ..expressions import CausalIntervention, CausalInterventionWalker


def test_do_instantiation():
    P = Symbol("P")
    Q = Symbol("Q")

    x = Symbol("x")
    y = Symbol("y")

    imp = Condition(
        P(x),
        Conjunction((
            CausalIntervention((
                Q(y),
            )),
        ))
    )
    ciw = CausalInterventionWalker()

    ciw.walk(imp)
    assert True

def test_do_instantiation_more_atoms():
    P = Symbol("P")
    Q = Symbol("Q")
    R = Symbol("R")

    x = Symbol("x")
    y = Symbol("y")

    imp = Condition(
        P(x),
        Conjunction((
            CausalIntervention((
                Q(y),R(x)
            )),
        ))
    )
    ciw = CausalInterventionWalker()

    ciw.walk(imp)
    assert True

def test_do_multiple_instantiation_exception():
    P = Symbol("P")
    Q = Symbol("Q")
    R = Symbol("R")

    x = Symbol("x")
    y = Symbol("y")

    imp = Condition(
        P(x),
        Conjunction((
            CausalIntervention((
                Q(y),
            )),
            CausalIntervention((
                R(y),
            )),
        ))
    )
    ciw = CausalInterventionWalker()
    with pytest.raises(MalformedCausalOperatorError, match="The use of more than one DO operator*"):
        ciw.walk(imp)

