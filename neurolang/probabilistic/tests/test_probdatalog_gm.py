from typing import Mapping

from ..probdatalog_gm import (
    TranslateGroundedProbDatalogToGraphicalModel,
    always_true_cpd_factory,
)
from ..probdatalog import Grounding
from ...relational_algebra import NameColumns
from ...utils.relational_algebra_set import RelationalAlgebraFrozenSet
from ...expressions import Symbol, Constant, ExpressionBlock

P = Symbol("P")
x = Symbol("x")
y = Symbol("y")
a = Constant[str]("a")
b = Constant[str]("b")
c = Constant[str]("c")
d = Constant[str]("d")


def test_extensional_grounding():
    grounding = Grounding(
        P(x, y),
        NameColumns(RelationalAlgebraFrozenSet({(a, b), (c, d)}), (x, y)),
    )
    translator = TranslateGroundedProbDatalogToGraphicalModel()
    translator.walk(grounding)
    assert not translator.edges
    assert translator.cpd_factories == {P: always_true_cpd_factory}
    translator = TranslateGroundedProbDatalogToGraphicalModel()
    gm = translator.walk(ExpressionBlock([grounding]))
    assert not gm.edges.value
    assert gm.cpd_factories == Constant[Mapping]({P: always_true_cpd_factory})
    assert gm.groundings == Constant[Mapping]({P: grounding})
