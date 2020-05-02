from ...expression_walker import ExpressionBasicEvaluator
from ...expressions import Constant, Symbol
from ...logic import Union
from ..constraints_representation import (
    DatalogConstraintsProgram,
    RightImplication,
)
from ..expressions import Implication


class Datalog(DatalogConstraintsProgram, ExpressionBasicEvaluator):
    pass


P = Symbol("P")
Q = Symbol("Q")
x = Symbol("x")
y = Symbol("y")
z = Symbol("z")


def test_protected_word():
    dl = Datalog()
    assert "__constraints__" in dl.protected_keywords


def test_one_constraints():
    cons = RightImplication(P(y, z), Q(y, z))

    dl = Datalog()
    dl.walk(cons)

    assert dl.constraints() == Union((cons,))


def test_cero_constraints():
    imp = Implication(P(x), Q(x))

    dl = Datalog()
    dl.walk(imp)

    assert dl.constraints() == Union(())
    assert P in dl.symbol_table


def test_all_constraints():
    cons1 = RightImplication(P(y, x), Q(y, x))
    cons2 = RightImplication(P(z), Q(z))
    cons3 = RightImplication(Q(y, z), P(y, z))

    dl1 = Datalog()
    dl1.walk(cons1)
    dl1.walk(cons2)
    dl1.walk(cons3)

    dl2 = Datalog()
    dl2.walk(Union((cons1, cons2, cons3)))

    assert dl1.constraints().formulas == dl2.constraints().formulas
    assert len(dl1.constraints().formulas) == 3


def test_non_constraints():
    cons1 = Implication(P(y, x), Q(y, x))
    cons2 = Implication(Q(y, z), P(y, z))

    dl1 = Datalog()
    dl1.walk(cons1)
    dl1.walk(cons2)

    dl2 = Datalog()
    dl2.walk(Union((cons1, cons2)))

    assert dl1.constraints().formulas == dl2.constraints().formulas
    assert len(dl1.constraints().formulas) == 0
    assert P in dl1.symbol_table and P in dl2.symbol_table
    assert Q in dl1.symbol_table and Q in dl2.symbol_table


def test_mix_constraints():
    cons1 = Implication(P(y, x), Q(y, x))
    cons2 = RightImplication(P(z), Q(z))
    cons3 = RightImplication(Q(y, z), P(y, z))

    dl1 = Datalog()
    dl1.walk(cons1)
    dl1.walk(cons2)
    dl1.walk(cons3)

    dl2 = Datalog()
    dl2.walk(Union((cons1, cons2, cons3)))

    assert dl1.constraints().formulas == dl2.constraints().formulas
    assert len(dl1.constraints().formulas) == 2
    assert P in dl2.symbol_table
    assert P in dl1.symbol_table
