from typing import AbstractSet, Tuple
from itertools import product

import pytest

from .. import solver_datalog_intentional_db as sdb
from .. import solver
from .. import expressions
from ..expressions import NeuroLangTypeException

C_ = expressions.Constant
S_ = expressions.Symbol
F_ = expressions.FunctionApplication


class IntentionalTestSolver(sdb.IntentionalDatabaseSolver, solver.GenericSolver):
    pass


def test_simple_set():
    ts = IntensionalTestSolver()

    s1 = C_[AbstractSet[int]](frozenset(C_(i) for i in range(5)))

    ts.symbol_table[S_[s1.type]('R')] = s1
    fa = F_(s1, (C_(2), ))
    res = ts.walk(fa)

    assert isinstance(res, expressions.Constant[bool]) and res.value

    fa = F_(s1, (C_(20), ))
    res = ts.walk(fa)

    assert isinstance(res, expressions.Constant[bool]) and not res.value


def test_tuple_set():
    ts = IntensionalTestSolver()

    s1 = C_[AbstractSet[Tuple[int, str]]](
        frozenset(
            C_[Tuple[int, str]]((C_(i), C_[str](chr(ord('a') + 1))))
            for i in range(5)
        )
    )

    ts.symbol_table[S_[s1.type]('R')] = s1
    f = F_(s1, (C_(2), C_('b')))

    res = ts.walk(f)
    assert isinstance(res, expressions.Constant[bool]) and res.value

    with pytest.raises(NeuroLangTypeException):
        f = F_(s1, (C_(2), ))
        ts.walk(f)


def test_simple_set_in_function():
    ts = IntensionalTestSolver()

    s1 = C_[AbstractSet[int]](frozenset(C_(i) for i in range(5)))

    fa = F_(S_('isin'), (C_(2), s1))
    res = ts.walk(fa)

    assert isinstance(res, expressions.Constan[bool]) and res.value

    fa = F_(S_('isin'), (C_(20), s1))
    res = ts.walk(fa)

    assert isinstance(res, expressions.Constant[bool]) and not res.value


def test_join():
    ts = IntensionalTestSolver()

    set1 = C_(frozenset((i, i * 2) for i in range(2)))

    set2 = C_(frozenset((i, i * 2) for i in range(4)))

    set3 = frozenset(
        e1 + e2
        for e1, e2 in product(set1.value, set2.value)
        if e1[1] == e2[0]
    )  # yapf: disable

    fa = F_(S_('join'), (set1, C_(1), set2, C_(0)))
    res = ts.walk(fa)

    assert isinstance(res, expressions.Constant)
    assert res.value == set3


def test_join_notuple():
    ts = IntensionalTestSolver()

    set1 = C_(frozenset(i for i in range(2)))

    set2 = C_(frozenset((i, i * 2) for i in range(4)))

    set3 = frozenset(
        (e1, ) + e2
        for e1, e2 in product(set1.value, set2.value)
        if e1 == e2[1]
    )  # yapf: disable

    fa = F_(S_('join'), (set1, C_(0), set2, C_(1)))
    res = ts.walk(fa)

    assert isinstance(res, expressions.Constant)
    assert res.value == set3
