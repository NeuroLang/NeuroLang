import pytest

from typing import AbstractSet

from .. import solver_datalog_naive as sdb
from .. import solver_datalog_extensional_db
from .. import expression_walker
from ..expressions import (
    Symbol, Constant, Statement,
    FunctionApplication, Lambda, ExpressionBlock,
    ExistentialPredicate, UniversalPredicate,
    Query,
    is_subtype, NeuroLangException
)

S_ = Symbol
C_ = Constant
ST_ = Statement
F_ = FunctionApplication
L_ = Lambda
B_ = ExpressionBlock
EP_ = ExistentialPredicate
UP_ = UniversalPredicate
Q_ = Query
CNone = C_(None)


class Datalog(
    solver_datalog_extensional_db.ExtensionalDatabaseSolver,
    sdb.NaiveDatalog,
    expression_walker.ExpressionBasicEvaluator
):
    pass


def test_facts_constants():
    dl = Datalog()

    f1 = ST_(S_('Q')(C_(1), C_(2)), CNone)

    dl.walk(f1)

    assert 'Q' in dl.symbol_table
    assert isinstance(dl.symbol_table['Q'], ExpressionBlock)
    fact_set = dl.symbol_table['Q'].expressions[0]
    assert isinstance(fact_set, Constant)
    assert is_subtype(fact_set.type, AbstractSet)
    assert {C_((C_(1), C_(2)))} == fact_set.value

    f2 = ST_(S_('Q')(C_(3), C_(4)), CNone)
    dl.walk(f2)
    assert (
        {C_((C_(1), C_(2))), C_((C_(3), C_(4)))} ==
        fact_set.value
    )

    f = S_('Q')(C_(1), C_(2))
    g = S_('Q')(C_(18), C_(23))

    assert dl.walk(f).value is True
    assert dl.walk(g).value is False


def test_atoms_variables():
    dl = Datalog()

    eq = S_('equals')
    x = S_('x')
    y = S_('y')
    Q = S_('Q')

    f1 = ST_(Q(x,), eq(x, x))

    dl.walk(f1)

    assert 'Q' in dl.symbol_table
    isinstance(dl.symbol_table['Q'], ExpressionBlock)
    fact = dl.symbol_table['Q'].expressions[-1]
    assert isinstance(fact, Lambda)
    assert len(fact.args) == 1
    assert fact.function_expression == eq(x, x)

    f2 = ST_(Q(x, y), eq(x, y))

    dl.walk(f2)

    assert 'Q' in dl.symbol_table
    isinstance(dl.symbol_table['Q'], ExpressionBlock)
    fact = dl.symbol_table['Q'].expressions[-1]
    assert isinstance(fact, Lambda)
    assert len(fact.args) == 2
    assert fact.function_expression == eq(x, y)

    with pytest.raises(NeuroLangException):
        dl.walk(ST_(Q(x), ...))

    f = Q(C_(10))
    g = Q(C_(1), C_(5))
    h = Q(C_(1), C_(1))

    assert dl.walk(f).value is True
    assert dl.walk(g).value is False
    assert dl.walk(h).value is True


def test_facts_intensional():
    dl = Datalog()

    Q = S_('Q')
    R = S_('R')
    T = S_('T')
    U = S_('U')
    x = S_('x')
    y = S_('y')
    z = S_('z')

    extensional = ExpressionBlock((
        ST_(Q(C_(1), C_(1)), CNone),
        ST_(Q(C_(1), C_(2)), CNone),
        ST_(Q(C_(1), C_(4)), CNone),
        ST_(Q(C_(2), C_(4)), CNone),
    ))

    intensional = ExpressionBlock((
        ST_(R(x, y, z), Q(x, y) & Q(y, z)),
        ST_(T(x, z), EP_(y, Q(x, y) & Q(y, z))),
        ST_(U(x), UP_(y, Q(x, y))),
    ))

    dl.walk(extensional)
    dl.walk(intensional)

    res = dl.walk(R(C_(1), C_(2), C_(4)))
    assert res.value is True

    res = dl.walk(R(C_(1), C_(2), C_(5)))
    assert res.value is False

    res = dl.walk(T(C_(1), C_(4)))
    assert res.value is True

    res = dl.walk(R(C_(1), C_(5)))
    assert res.value is False

    res = dl.walk(U(C_(1)))
    assert res.value is True

    res = dl.walk(U(C_(2)))
    assert res.value is False

    with pytest.raises(NeuroLangException):
        res = dl.walk(ST_(Q(x, y), Q(x)))


def test_query():
    dl = Datalog()

    Q = S_('Q')
    R = S_('R')
    T = S_('T')
    U = S_('U')
    x = S_('x')
    y = S_('y')
    z = S_('z')

    extensional = ExpressionBlock((
        ST_(Q(C_(1), C_(1)), CNone),
        ST_(Q(C_(1), C_(2)), CNone),
        ST_(Q(C_(1), C_(4)), CNone),
        ST_(Q(C_(2), C_(4)), CNone),
    ))

    intensional = ExpressionBlock((
        ST_(R(x, y, z), Q(x, y) & Q(y, z)),
        ST_(T(x, z), Q(x, y) & Q(y, z)),
        ST_(U(x), UP_(y, Q(x, y))),
    ))

    dl.walk(extensional)
    dl.walk(intensional)

    query = Q_(x, U(x))
    res = dl.walk(query)

    assert res.value == {C_(1)}

    query = Q_((x, y), T(x, y))
    res = dl.walk(query)

    assert res.value == {
        C_((C_(1), C_(1))),
        C_((C_(1), C_(2))),
        C_((C_(1), C_(4))),
    }


def test_conjunctive_expression():
    Q = S_('Q')
    R = S_('R')
    x = S_('x')
    y = S_('y')

    assert sdb.is_conjunctive_expression(
        ST_(R(x), Q())
    )

    assert sdb.is_conjunctive_expression(
        ST_(R(x), Q(x))
    )

    assert sdb.is_conjunctive_expression(
        ST_(R(x), Q(x) & R(y, C_(1)))
    )

    assert not sdb.is_conjunctive_expression(
        ST_(Q(x, y), R(x) | R(y))
    )

    assert not sdb.is_conjunctive_expression(
        ST_(Q(x, y), R(x) & R(y) | R(x))
    )

    assert not sdb.is_conjunctive_expression(
        ST_(Q(x, y), ~R(x))
    )

    assert not sdb.is_conjunctive_expression(
        ST_(Q(x, y), R(Q(x)))
    )


def test_not_conjunctive():

    dl = Datalog()

    Q = S_('Q')
    R = S_('R')
    x = S_('x')
    y = S_('y')

    with pytest.raises(NeuroLangException):
        dl.walk(ST_(Q(x, y), R(x) | R(y)))

    with pytest.raises(NeuroLangException):
        dl.walk(ST_(Q(x, y), R(x) & R(y) | R(x)))

    with pytest.raises(NeuroLangException):
        dl.walk(ST_(Q(x, y), ~R(x)))

    with pytest.raises(NeuroLangException):
        dl.walk(ST_(Q(x, y), R(Q(x))))


def test_extract_free_variables():
    Q = S_('Q')
    R = S_('R')
    x = S_('x')
    y = S_('y')

    emptyset = set()
    assert sdb.extract_datalog_free_variables(Q) == emptyset
    assert sdb.extract_datalog_free_variables(Q(x, y)) == {x, y}
    assert sdb.extract_datalog_free_variables(Q(x, C_(1))) == {x}
    assert sdb.extract_datalog_free_variables(Q(x) & R(y)) == {x, y}
    assert sdb.extract_datalog_free_variables(EP_(x, Q(x, y))) == {y}
    assert sdb.extract_datalog_free_variables(UP_(x, Q(x, y))) == {y}
    assert sdb.extract_datalog_free_variables(ST_(R(x), Q(x, y))) == {y}
    assert sdb.extract_datalog_free_variables(ST_(R(x), Q(y) & Q(x))) == {y}

    with pytest.raises(NeuroLangException):
        assert sdb.extract_datalog_free_variables(Q(x) | R(y))

    with pytest.raises(NeuroLangException):
        assert sdb.extract_datalog_free_variables(Q(R(y)))

    with pytest.raises(NeuroLangException):
        assert sdb.extract_datalog_free_variables(~(R(y)))
