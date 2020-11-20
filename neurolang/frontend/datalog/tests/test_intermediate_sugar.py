import operator
from typing import AbstractSet, Tuple
from unittest.mock import Mock

import pytest

from ....datalog import Fact
from ....datalog.expression_processing import extract_logic_atoms
from ....exceptions import ForbiddenExpressionError
from ....expression_walker import ExpressionWalker, IdentityWalker
from ....expressions import Constant, Symbol
from ....logic import Conjunction, Implication
from ....probabilistic.expressions import (
    PROB,
    Condition,
    ProbabilisticPredicate,
    ProbabilisticQuery,
)
from .. import intermediate_sugar as sugar


class SymbolTableMixin:
    def __init__(self, symbol_table=None):
        if symbol_table is None:
            symbol_table = dict()
        self.symbol_table = symbol_table


class TranslateColumnsToAtoms(
    sugar.TranslateColumnsToAtoms, SymbolTableMixin, ExpressionWalker
):
    pass


class TranslateSelectByFirstColumn(
    sugar.TranslateSelectByFirstColumn, SymbolTableMixin, ExpressionWalker
):
    pass


def test_columns_to_atoms_rules():
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

    test_rule = Implication(
        A(sugar.Column(B, one)), C(sugar.Column(B, one), x)
    )

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
    assert out_rule.consequent.functor == A
    assert out_rule.consequent.args == (b.args[1],)

    test_rule = Implication(A(x), Conjunction((B(x, x, x), C(x))))
    out_rule = tcta.walk(test_rule)

    assert test_rule == out_rule


def test_columns_to_atoms_facts():
    A = Symbol("A")
    B = Symbol("B")
    one = Constant(1)

    test_rule = Fact(A(sugar.Column(B, one)))

    tcta = TranslateColumnsToAtoms({B: Constant(Mock(arity=3))})
    out_rule = tcta.walk(test_rule)
    atoms = extract_logic_atoms(out_rule.antecedent)
    assert len(atoms) == 1
    b = atoms.pop()
    assert b.functor == B
    assert all(arg.is_fresh for arg in b.args)
    assert out_rule.consequent.functor == A
    assert out_rule.consequent.args[0] == b.args[1]


def test_select_by_first_implication():
    A = Symbol("A")
    B = Symbol("B")
    C = Symbol("C")
    c = Constant("c")
    x = Symbol("x")

    test_rule = Implication(A(x), C(sugar.SelectByFirstColumn(B, c), x))

    tr = TranslateSelectByFirstColumn().walk(test_rule)
    fs = next(s for s in tr._symbols if s.is_fresh)
    assert tr == Implication(A(x), Conjunction((C(fs, x), B(c, fs))))


def test_select_by_first_implication_builtin():
    A = Symbol("A")
    B = Symbol("B")
    C = Symbol("C")
    c = Constant("c")
    eq = Constant(lambda x, y: x == y)
    x = Symbol("x")
    y = Symbol("y")

    test_rule = Implication(
        A(x),
        Conjunction(
            (
                C(sugar.SelectByFirstColumn(B, c), x),
                eq(sugar.SelectByFirstColumn(B, c), y),
            )
        ),
    )

    tr = TranslateSelectByFirstColumn().walk(test_rule)
    fs = next(s for s in tr._symbols if s.is_fresh)
    res = Implication(A(x), Conjunction((C(fs, x), eq(fs, y), B(c, fs))))
    assert tr == res


def test_select_by_first_implication_builtin_head():
    A = Symbol("A")
    B = Symbol("B")
    C = Symbol[AbstractSet[Tuple[int, int]]]("C")
    c = Constant("c")
    eq = Constant(lambda x, y: x == y)
    x = Symbol("x")
    y = Symbol("y")

    test_rule = Implication(
        sugar.SelectByFirstColumn(A, c), Conjunction((C(x), eq(y), B(x)))
    )

    tr = TranslateSelectByFirstColumn({C: C, B: B}).walk(test_rule)
    fresh_symbols = [s for s in tr._symbols if s.is_fresh]
    assert len(fresh_symbols) == 1

    fs = fresh_symbols[0]
    res = Implication(A(c, fs), Conjunction((C(fs, x), eq(fs, y), B(x))))
    assert tr == res


class _TestTranslator(sugar.TranslateProbabilisticQueryMixin, ExpressionWalker):
    pass


def test_wlq_floordiv_translation():
    P = Symbol("P")
    Q = Symbol("Q")
    R = Symbol("R")
    x = Symbol("x")
    y = Symbol("y")
    wlq = Implication(
        Q(x, y, PROB(x, y)), Constant(operator.floordiv)(P(x), R(x, y))
    )
    translator = _TestTranslator()
    result = translator.walk(wlq)
    assert result == Implication(
        Q(x, y, ProbabilisticQuery(PROB, (x, y))), Condition(P(x), R(x, y))
    )


def test_wlq_marg_bad_syntax():
    P = Symbol("P")
    Q = Symbol("Q")
    Z = Symbol("Z")
    x = Symbol("x")
    y = Symbol("y")
    bad_wlq = Implication(Q(x, y), Constant(operator.floordiv)(P(x), Z(x, y)))
    translator = _TestTranslator()
    with pytest.raises(ForbiddenExpressionError):
        translator.walk(bad_wlq)


class TestTranslateQueryBasedProbabilisticFact(
    sugar.TranslateQueryBasedProbabilisticFactMixin,
    IdentityWalker,
):
    pass


def test_translation_sugar_syntax():
    P = Symbol("P")
    Q = Symbol("Q")
    x = Symbol("x")
    p = Symbol("p")
    pfact = Implication(
        Constant(operator.matmul)(P, (p / Constant(2)))(x), Q(x, p)
    )
    translator = TestTranslateQueryBasedProbabilisticFact()
    result = translator.walk(pfact)
    expected = Implication(
        ProbabilisticPredicate(p / Constant(2), P(x)), Q(x, p)
    )
    assert result == expected
