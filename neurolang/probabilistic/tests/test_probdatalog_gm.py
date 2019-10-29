from typing import Mapping, AbstractSet

from ..probdatalog_gm import TranslateGroundedProbDatalogToGraphicalModel
from ..probdatalog import Grounding
from ...relational_algebra import NameColumns
from ...utils.relational_algebra_set import NamedRelationalAlgebraFrozenSet
from ...expressions import Symbol, Constant, ExpressionBlock
from ...datalog.expressions import Implication, Conjunction
from ..expressions import VectorisedTableDistribution, ProbabilisticPredicate

P = Symbol("P")
Q = Symbol("Q")
T = Symbol("T")
x = Symbol("x")
y = Symbol("y")
a = Constant[str]("a")
b = Constant[str]("b")
c = Constant[str]("c")
d = Constant[str]("d")


def test_extensional_grounding():
    grounding = Grounding(
        P(x, y),
        Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(
                columns=("x", "y"), iterable={(a, b), (c, d)}
            )
        ),
    )
    translator = TranslateGroundedProbDatalogToGraphicalModel()
    translator.walk(grounding)
    assert not translator.edges
    assert translator.cpds == {
        P: VectorisedTableDistribution(
            Constant[Mapping](
                {
                    Constant[bool](False): Constant[float](0.0),
                    Constant[bool](True): Constant[float](1.0),
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
                        Constant[bool](False): Constant[float](0.0),
                        Constant[bool](True): Constant[float](1.0),
                    }
                ),
                grounding,
            )
        }
    )
    assert gm.groundings == Constant[Mapping]({P: grounding})


def test_probabilistic_grounding():
    probfact = Implication(
        ProbabilisticPredicate(Constant[float](0.3), P(x)),
        Constant[bool](True),
    )
    grounding = Grounding(
        probfact,
        Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(iterable=[1, 2, 3], columns=["x"])
        ),
    )
    gm = TranslateGroundedProbDatalogToGraphicalModel().walk(
        ExpressionBlock([grounding])
    )
    assert not gm.edges.value
    assert gm.cpds == Constant[Mapping](
        {
            P: VectorisedTableDistribution(
                Constant[Mapping](
                    {
                        Constant[bool](False): Constant[float](1.0)
                        - Constant[float](0.3),
                        Constant[bool](True): Constant[float](0.3),
                    }
                ),
                grounding,
            )
        }
    )
    assert gm.groundings == Constant[Mapping]({P: grounding})


def test_intensional_grounding():
    extensional_grounding = Grounding(
        T(x),
        Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(iterable=[1, 2, 3], columns=["x"])
        ),
    )
    probabilistic_grounding = Grounding(
        Implication(
            ProbabilisticPredicate(Constant[float](0.3), P(x)),
            Constant[bool](True),
        ),
        Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(iterable=[1, 2, 3], columns=["x"])
        ),
    )
    intensional_grounding = Grounding(
        Implication(Q(x), Conjunction([P(x), T(x)])),
        Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(iterable=[1, 2, 3], columns=["x"])
        ),
    )
    groundings = ExpressionBlock(
        [extensional_grounding, probabilistic_grounding, intensional_grounding]
    )
    gm = TranslateGroundedProbDatalogToGraphicalModel().walk(groundings)
    assert gm.edges == Constant[Mapping]({Q: {P, T}})
