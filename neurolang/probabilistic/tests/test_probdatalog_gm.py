from typing import Mapping

from ..probdatalog_gm import TranslateGroundedProbDatalogToGraphicalModel
from ..probdatalog import Grounding
from ...relational_algebra import NameColumns
from ...utils.relational_algebra_set import RelationalAlgebraFrozenSet
from ...expressions import Symbol, Constant, ExpressionBlock
from ..expressions import VectorisedTableDistribution

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
    assert translator.cpds == {
        P: VectorisedTableDistribution(
            Constant[Mapping](
                {
                    Constant[int](0): Constant[float](0.0),
                    Constant[int](1): Constant[float](1.0),
                }
            ),
            grounding,
        )
    }
    translator = TranslateGroundedProbDatalogToGraphicalModel()
    gm = translator.walk(ExpressionBlock([grounding]))
    assert not gm.edges.value
    assert gm.cpds == Constant[Mapping](
        {
            P: VectorisedTableDistribution(
                Constant[Mapping](
                    {
                        Constant[int](0): Constant[float](0.0),
                        Constant[int](1): Constant[float](1.0),
                    }
                ),
                grounding,
            )
        }
    )
    assert gm.groundings == Constant[Mapping]({P: grounding})
