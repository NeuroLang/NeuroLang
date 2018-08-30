from typing import Callable

from .. import solver_datalog_seminaive as sds

from .. import solver_datalog_naive as sdb
from .. import solver_datalog_extensional_db
from .. import expression_walker
from ..expressions import (
    Symbol, Constant, Statement,
    FunctionApplication, Lambda, ExpressionBlock,
    ExistentialPredicate, UniversalPredicate,
    Query,
)

S_ = Symbol
C_ = Constant
St_ = Statement
F_ = FunctionApplication
L_ = Lambda
B_ = ExpressionBlock
EP_ = ExistentialPredicate
UP_ = UniversalPredicate
Q_ = Query
T_ = sdb.Fact


class Datalog(
    solver_datalog_extensional_db.ExtensionalDatabaseSolver,
    sdb.DatalogBasic,
    sds.DatalogSeminaiveEvaluator,
    expression_walker.ExpressionBasicEvaluator
):
    pass


def test_extensional():

    Q = S_('Q')
    x = S_('x')
    y = S_('y')

    extensional = B_(tuple(
        T_(Q(C_(i), C_(2 * i)))
        for i in range(5)
    ))

    dl = Datalog()
    dl.walk(extensional)

    res = dl.walk(Q(x, y))
    assert res == dl.extensional_database()[Q].value

    res = dl.walk(Q(C_(0), y))
    assert res == {C_((C_(0), C_(0)))}

    res = dl.walk(Q(y, C_(2)))
    assert res == {C_((C_(1), C_(2)))}

    res = dl.walk(Q(C_(2), C_(4)))
    assert res == {C_((C_(2), C_(4)))}

    res = dl.walk(Q(y, C_(1)))
    assert res == set()


def test_intensional_single_case():

    Q = S_('Q')
    R = S_('R')
    S = S_('S')
    T = S_('T')
    x = S_('x')
    y = S_('y')

    extensional = B_(tuple(
        T_(Q(C_(i), C_(2 * i)))
        for i in range(50)
    ))

    intensional_1 = B_((
        St_(R(x), Q(x, C_(2))),
    ))
    dl = Datalog()
    dl.walk(extensional)
    dl.walk(intensional_1)

    res = dl.walk(R(x))
    assert res == {C_((C_(1),))}

    intensional_2 = B_((
        St_(S(x), Q(x, C_(2))),
        St_(S(x), Q(x, x)),
        St_(T(x), Q(x, y) & Q(y, x)),
    ))
    dl.walk(intensional_2)

    res = dl.walk(Query(x, S(x)))
    assert res == {
        C_((C_(1),)),
        C_((C_(0),)),
    }

    res = dl.walk(Query(x, T(x)))
    assert res == {
       C_((C_(0),)),
    }


def test_intensional_recursive():
    Q = S_('Q')
    R = S_('R')

    x = S_('x')
    y = S_('y')
    z = S_('z')

    extensional = B_(tuple(
        T_(Q(C_(i), C_(2 * i)))
        for i in range(5)
    ))

    intensional = B_((
        St_(R(x, y), Q(x, y)),
        St_(R(x, y), Q(x, z) & R(z, y)),
    ))

    dl = Datalog()
    dl.walk(extensional)
    dl.walk(intensional)

    q = Query((x, y), R(x, y))
    res = dl.walk(q)

    assert res == {
        (0, 0),
        (1, 2),
        (2, 4),
        (3, 6),
        (4, 8),
        (1, 4),
        (2, 8),
        (1, 8)
    }

    S = S_('S')
    program_2 = B_((
        St_(S(x, y), R(x, y)),
        St_(S(x, y), R(x, z) & S(z, y)),
    ))

    dl.walk(program_2)

    q = Query((x, y), S(x, y))
    res = dl.walk(q)

    assert res == {
        (0, 0),
        (1, 2),
        (2, 4),
        (3, 6),
        (4, 8),
        (1, 4),
        (2, 8),
        (1, 8),
    }


def test_intensional_defined():
    Q = S_('Q')
    R = S_('R')
    T = S_[Callable[[int], bool]]('T')

    x = S_('x')
    y = S_('y')

    t = C_[Callable[[int], bool]](lambda x: x % 2 == 0)

    extensional = B_(tuple(
        T_(Q(C_(i), C_(2 * i)))
        for i in range(500)
    ))

    intensional = B_((
        St_(R(x, y), T(y) & Q(x, y) & T(x)),
    ))

    dl = Datalog()
    dl.walk(extensional)
    dl.walk(intensional)
    dl.symbol_table[T] = t

    q = Query((x, y), R(x, y))
    res = dl.walk(q)
    res == {
        (0, 0),
        (2, 4),
        (4, 8),
    }
