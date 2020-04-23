from typing import AbstractSet

from ....datalog import Fact
from ....expressions import Constant, ExpressionBlock, Symbol
from ....logic import Implication
from ....relational_algebra import NamedRelationalAlgebraFrozenSet
from .. import testing
from ..cplogic_to_gm import (
    AndCPDFactory,
    BernoulliCPDFactory,
    CPLogicGroundingToGraphicalModelTranslator,
)
from ..grounding import ground_cplogic_program

P = Symbol("P")
Q = Symbol("Q")
x = Symbol("x")
a = Constant("a")
b = Constant("b")


def test_empty_program():
    code = ExpressionBlock(tuple())
    gm = testing.build_gm(code)
    assert len(gm.edges.value) == 0
    assert len(gm.cpd_factories.value) == 0


def test_simple_deterministic_program():
    code = ExpressionBlock((Fact(Q(a)), Fact(Q(b)), Implication(P(x), Q(x))))
    gm = testing.build_gm(code)
    assert set(gm.cpd_factories.value) == {Q, P}
    assert isinstance(gm.cpd_factories.value[Q], BernoulliCPDFactory)
    assert isinstance(gm.cpd_factories.value[P], AndCPDFactory)


def test_program_with_probfacts():
    code = ExpressionBlock((Implication(Q(x), P(x)),))
    probfacts_sets = {P: {(1.0, "a"), (0.5, "b"), (0.3, "c")}}
    gm = testing.build_gm(code, probfacts_sets=probfacts_sets)
    assert set(gm.edges.value) == {Q}
    assert gm.edges.value[Q] == {P}
    assert isinstance(gm.cpd_factories.value[Q], AndCPDFactory)
    assert isinstance(gm.cpd_factories.value[P], BernoulliCPDFactory)
    assert (
        testing.get_named_relation_tuples(gm.cpd_factories.value[P].relation)
        == probfacts_sets[P]
    )
