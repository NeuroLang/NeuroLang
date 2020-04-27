from ..datalog import DatalogProgram
from ..datalog.aggregation import (AggregationApplication,
                                   DatalogWithAggregationMixin)
from ..datalog.constraints_representation import (
    DatalogConstraintsProgramMixin, RightImplication)
from ..datalog.expressions import Fact
from ..datalog.negation import DatalogProgramNegationMixin
from ..expression_walker import ExpressionWalker
from ..expressions import Constant, Symbol
from ..logic import Conjunction, Implication, Negation, Union
from ..probabilistic.cplogic.program import CPLogicMixin
from ..probabilistic.expressions import ProbabilisticPredicate
from ..type_system import Unknown

A = Symbol('A')
B = Symbol('B')
C = Symbol('C')
D = Symbol('D')
E = Symbol('E')
F = Symbol('F')


def test_all_composed():
    class Datalog(
        CPLogicMixin,
        DatalogWithAggregationMixin,
        DatalogProgramNegationMixin,
        DatalogConstraintsProgramMixin,
        DatalogProgram,
        ExpressionWalker
    ):
        def function_length(self, x: Unknown) -> int:
            return len(x)

    x = Symbol('x')
    program = Union((
        Fact(A(Constant(1))),
        Fact(A(Constant(2))),
        Fact(A(Constant(3))),
        Fact(C(Constant(3))),
        Fact(ProbabilisticPredicate(Constant(.5), B(Constant(2)))),
        Implication(D(AggregationApplication(Symbol('length'), (x,))), A(x)),
        Implication(E(x), Conjunction((A(x), Negation(D(x))))),
        Implication(F(x), A(x)),
        RightImplication(C(x), F(x))
    ))

    dl = Datalog()
    dl.walk(program)

    assert {A, C} == set(dl.extensional_database())
    assert {D, E, F} == set(dl.intensional_database())
    assert {B} == set(dl.probabilistic_facts())
    assert {F} == set(dl.constraints())
