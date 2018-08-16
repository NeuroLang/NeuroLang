from typing import AbstractSet

from .. import solver_datalog_naive
from .. import solver_datalog_extensional_db
from .. import expression_walker
from ..expressions import (
    Symbol, Constant, Statement,
    FunctionApplication, Lambda, ExpressionBlock,
    ExistentialPredicate, UniversalPredicate,
    Query,
    is_subtype
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


class Datalog(
    solver_datalog_extensional_db.ExtensionalDatabaseSolver,
    solver_datalog_naive.NaiveDatalog,
    expression_walker.ExpressionBasicEvaluator
):
    pass


def test_facts_constants():
    dl = Datalog()

    f1 = ST_(S_('Q')(C_(1), C_(2)), None)

    dl.walk(f1)

    assert 'Q' in dl.symbol_table
    isinstance(dl.symbol_table['Q'], ExpressionBlock)
    fact_set = dl.symbol_table['Q'].expressions[0]
    assert isinstance(fact_set, Constant)
    assert is_subtype(fact_set.type, AbstractSet)
    assert {C_((C_(1), C_(2)))} == fact_set.value

    f2 = ST_(S_('Q')(C_(3), C_(4)), None)
    dl.walk(f2)
    assert (
        {C_((C_(1), C_(2))), C_((C_(3), C_(4)))} ==
        fact_set.value
    )

    f = S_('Q')(C_(1), C_(2))
    g = S_('Q')(C_(18), C_(23))

    assert dl.walk(f).value is True
    assert dl.walk(g).value is False


def test_facts_variables():
    dl = Datalog()

    f1 = ST_(S_('Q')(S_('x'),), None)

    dl.walk(f1)

    assert 'Q' in dl.symbol_table
    isinstance(dl.symbol_table['Q'], ExpressionBlock)
    fact = dl.symbol_table['Q'].expressions[-1]
    assert isinstance(fact, Lambda)
    assert len(fact.args) == 1
    assert fact.function_expression.value is True

    f2 = ST_(S_('Q')(S_('x'), S_('y')), None)

    dl.walk(f2)

    assert 'Q' in dl.symbol_table
    isinstance(dl.symbol_table['Q'], ExpressionBlock)
    fact = dl.symbol_table['Q'].expressions[-1]
    assert isinstance(fact, Lambda)
    assert len(fact.args) == 2
    assert fact.function_expression.value is True

    f3 = ST_(S_('Q')(S_('x'), S_('y'), S_('x')), None)

    dl.walk(f3)

    assert 'Q' in dl.symbol_table
    isinstance(dl.symbol_table['Q'], ExpressionBlock)
    fact = dl.symbol_table['Q'].expressions[-1]
    assert isinstance(fact, Lambda)
    assert len(fact.args) == 3
    assert fact.function_expression.functor == S_('equals')
    assert fact.function_expression.args == (S_('a0'), S_('a2'))

    f = S_('Q')(C_(10))
    g = S_('Q')(C_(1), C_(5))
    h = S_('Q')(C_(18), C_(23), C_(18))
    i = S_('Q')(C_(18), C_(23), C_(19))

    assert dl.walk(f).value is True
    assert dl.walk(g).value is True
    assert dl.walk(h).value is True
    assert dl.walk(i).value is False


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
        ST_(Q(C_(1), C_(1)), None),
        ST_(Q(C_(1), C_(2)), None),
        ST_(Q(C_(1), C_(4)), None),
        ST_(Q(C_(2), C_(4)), None),
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
        ST_(Q(C_(1), C_(1)), None),
        ST_(Q(C_(1), C_(2)), None),
        ST_(Q(C_(1), C_(4)), None),
        ST_(Q(C_(2), C_(4)), None),
    ))

    intensional = ExpressionBlock((
        ST_(R(x, y, z), Q(x, y) & Q(y, z)),
        ST_(T(x, z), EP_(y, Q(x, y) & Q(y, z))),
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
