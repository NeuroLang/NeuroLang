from unittest.mock import Mock

from ....datalog import Conjunction, Implication, Union
from ....expression_walker import ExpressionWalker, ResolveSymbolMixin
from ....expressions import Constant, Symbol
from .. import intermediate_sugar as sugar


class TranslateColumnsToAtoms(sugar.TranslateColumnsToAtoms, ExpressionWalker):
    pass


def test_columns_to_atoms():
    A = Symbol("A")
    B = Symbol("B")
    C = Symbol("C")
    x = Symbol("x")
    one = Constant(1)
    test_rule = Implication(A(x), C(sugar.Column(B, one), x))

    tcta = TranslateColumnsToAtoms({B: Constant(Mock(arity=3))})
    out_rule = tcta.walk(test_rule)
    antecedent_formulas = out_rule.antecedent.formulas
    if antecedent_formulas[0].functor == "C":
        c = antecedent_formulas[0]
        b = antecedent_formulas[1]
    else:
        c = antecedent_formulas[1]
        b = antecedent_formulas[0]

    assert isinstance(out_rule.antecedent, Conjunction)
    assert c.functor == C
    assert c.args[1] == x
    assert all(arg.is_fresh for arg in b.args)
    assert c.args[0] == b.args[1]

    test_rule = Implication(A(x), Conjunction((B(x, x, x), C(x))))
    out_rule = tcta.walk(test_rule)

    assert test_rule == out_rule
