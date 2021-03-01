import pytest
from unittest.mock import MagicMock, ANY
from ... import expression_walker, expressions
from .. import DatalogProgram, Fact, Implication
from ..chase import ChaseNamedStratified, ChaseNR, ChaseSN
from ..chase.general import NoValidChaseClassForStratumException
from ..expressions import TranslateToLogic
from ..instance import MapInstance

C_ = expressions.Constant
S_ = expressions.Symbol
Imp_ = Implication
F_ = Fact
Eb_ = expressions.ExpressionBlock


class Datalog(
    TranslateToLogic,
    DatalogProgram,
    expression_walker.ExpressionBasicEvaluator,
):
    pass


x = S_("X")
y = S_("Y")
z = S_("Z")
anc = S_("anc")
par = S_("par")
q = S_("q")
a = C_("a")
b = C_("b")
c = C_("c")
d = C_("d")


@pytest.fixture
def datalog():
    edb = Eb_(
        [
            F_(par(a, b)),
            F_(par(b, c)),
            F_(par(c, d)),
        ]
    )

    code = Eb_(
        [
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
            Imp_(q(x), anc(a, x)),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    return dl


def test_stratification_resolution_works(datalog):
    solution = ChaseNamedStratified(datalog).build_chase_solution()
    assert solution[q].value == {C_((e,)) for e in (b, c, d)}


def test_stratified_chases(datalog):
    # Recursive stratum can be solved by just ChaseSN
    classes = (ChaseSN,)
    solution = ChaseNamedStratified(datalog, chase_classes=classes).build_chase_solution()
    assert solution[q].value == {C_((e,)) for e in (b, c, d)}

    # Recursive stratum cannot be solved by just ChaseNR
    classes = (ChaseNR,)
    with pytest.raises(
        NoValidChaseClassForStratumException,
    ):
        solution = ChaseNamedStratified(datalog, chase_classes=classes).build_chase_solution()

    # Check that each stratum is solved by the first possible class
    strata = list(datalog.intensional_database().values())
    spied_chase_sn = MagicMock(wraps=ChaseSN(datalog, strata[0]))
    spied_chase_nr = MagicMock(wraps=ChaseNR(datalog, strata[1]))
    classes = (
        MagicMock(return_value=spied_chase_nr),
        MagicMock(return_value=spied_chase_sn),
    )

    solution = ChaseNamedStratified(datalog, chase_classes=classes).build_chase_solution()
    assert solution[q].value == {C_((e,)) for e in (b, c, d)}
    spied_chase_sn.execute_chase.assert_called_with(
        list(strata[0].formulas),
        MapInstance(datalog.extensional_database()),
        ANY,
    )
    spied_chase_nr.execute_chase.assert_called_once()


def test_unstratifyable_code_resolution():
    Q = S_("Q")  # noqa: N806
    R = S_("R")  # noqa: N806
    S = S_("S")  # noqa: N806
    T = S_("T")  # noqa: N806
    x = S_("x")
    y = S_("y")
    code = Eb_(
        [
            F_(Q(C_(1), C_(2))),
            F_(Q(C_(2), C_(3))),
            Imp_(R(x, y), Q(x, y)),
            Imp_(R(x, y), Q(y, x)),
            Imp_(S(x), R(x, y) & T(y)),
            Imp_(T(x), R(x, y) & S(x)),
        ]
    )
    dl = Datalog()
    dl.walk(code)
    solution = ChaseNamedStratified(dl, chase_classes=(ChaseSN,)).build_chase_solution()
    assert solution[R].value == {(1, 2), (2, 1), (3, 2), (2, 3)}
