from typing import Mapping, AbstractSet

from ..probdatalog_gm import TranslateGroundedProbDatalogToGraphicalModel
from ..probdatalog import Grounding
from ...relational_algebra import NameColumns
from ...utils.relational_algebra_set import NamedRelationalAlgebraFrozenSet
from ...expressions import (
    Symbol,
    Constant,
    ExpressionBlock,
    ExistentialPredicate,
)
from ...datalog.expressions import Implication, Conjunction
from ..expressions import (
    VectorisedTableDistribution,
    ProbabilisticPredicate,
    ParameterVectorPointer,
    SubtractVectors,
)

P = Symbol("P")
Q = Symbol("Q")
T = Symbol("T")
x = Symbol("x")
y = Symbol("y")
p = Symbol("p")
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


def test_existential_probfact_grounding():
    x_algebra_set = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(iterable=[1, 2, 3, 4], columns=["x"])
    )
    existential_probfact = Implication(
        ExistentialPredicate(p, ProbabilisticPredicate(p, P(x))),
        Constant[bool](True),
    )
    eprobfact_grounding = Grounding(existential_probfact, x_algebra_set)
    extensional_grounding = Grounding(T(x), x_algebra_set)
    intensional_grounding = Grounding(
        Implication(Q(x), Conjunction([P(x), T(x)])), x_algebra_set
    )
    gm = TranslateGroundedProbDatalogToGraphicalModel().walk(
        ExpressionBlock(
            [extensional_grounding, eprobfact_grounding, intensional_grounding]
        )
    )
    assert P in gm.cpds.value
    assert len(gm.cpds.value[P].parameters.value) == 1
    params_vect_symb = next(iter(gm.cpds.value[P].parameters.value))
    assert gm.cpds.value[P].table.value[Constant[bool](True)] == (
        ParameterVectorPointer(params_vect_symb)
    )
    assert gm.cpds.value[P].table.value[
        Constant[bool](False)
    ] == SubtractVectors(
        Constant[float](1.0), ParameterVectorPointer(params_vect_symb)
    )
