import pytest

from typing import Callable

from .. import solver_datalog_seminaive as sds

from .. import solver_datalog_naive as sdb
from .. import solver_datalog_extensional_db
from .. import expression_walker
from ..expressions import (
    Symbol, Constant,
    FunctionApplication, Lambda, ExpressionBlock,
    ExistentialPredicate, UniversalPredicate,
    Query,
)

S_ = Symbol
C_ = Constant
Imp_ = sdb.Implication
F_ = FunctionApplication
L_ = Lambda
B_ = ExpressionBlock
EP_ = ExistentialPredicate
UP_ = UniversalPredicate
Q_ = Query
T_ = sdb.Fact


class Datalog(
    sds.DatalogSeminaiveEvaluator,
    solver_datalog_extensional_db.ExtensionalDatabaseSolver,
    expression_walker.ExpressionBasicEvaluator
):
    pass


N = 5


@pytest.fixture
def extensional_single_double(N=N):
    Q = S_('Q')
    extensional = B_(tuple(
      T_(Q(C_(i), C_(2 * i)))
      for i in range(N)
    ))

    dl = Datalog()
    dl.walk(extensional)

    return dl, Q


def test_extensional(extensional_single_double):
    dl, Q = extensional_single_double
    x = S_('x')
    y = S_('y')

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


def test_intensional_single_case(extensional_single_double):

    dl, Q = extensional_single_double

    R = S_('R')
    S = S_('S')
    T = S_('T')
    U = S_('U')
    x = S_('x')
    y = S_('y')

    intensional_1 = B_((
        Imp_(R(x), Q(x, C_(2))),
        Imp_(S(x), Q(x, x)),
    ))
    dl.walk(intensional_1)

    res = dl.walk(R(x))
    assert res == {C_((C_(1),))}

    res = dl.walk(S(x))
    assert res == {C_((C_(0),))}

    res = dl.walk(Query(x, R(x)))
    assert res == {C_((C_(1),))}

    res = dl.walk(Query(x, S(x)))
    assert res == {C_((C_(0),))}

    intensional_3 = B_((
        Imp_(T(x), Q(x, C_(2))),
        Imp_(T(x), Q(x, x)),
        Imp_(U(x), Q(x, y) & Q(y, x)),
    ))
    dl.walk(intensional_3)

    res = dl.walk(T(x))
    assert res == {
        C_((C_(1),)),
        C_((C_(0),)),
    }

    res = dl.walk(Query(x, T(x)))
    assert res == {
        C_((C_(1),)),
        C_((C_(0),)),
    }

    res = dl.walk(Query(x, U(x)))
    assert res == {
       C_((C_(0),)),
    }


def test_intensional_recursive(extensional_single_double):
    dl, Q = extensional_single_double

    R = S_('R')

    x = S_('x')
    y = S_('y')
    z = S_('z')

    intensional = B_((
        Imp_(R(x, y), Q(x, y)),
        Imp_(R(x, y), Q(x, z) & R(z, y)),
    ))

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
        Imp_(S(x, y), R(x, y)),
        Imp_(S(x, y), R(x, z) & S(z, y)),
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


def test_intensional_defined(extensional_single_double):
    dl, Q = extensional_single_double

    R = S_('R')
    T = S_[Callable[[int], bool]]('T')

    x = S_('x')
    y = S_('y')

    t = C_[Callable[[int], bool]](lambda x: x % 2 == 0)

    intensional = B_((
        Imp_(R(x, y), T(y) & Q(x, y) & T(x)),
    ))

    dl.walk(intensional)
    dl.symbol_table[T] = t

    q = Query((x, y), R(x, y))
    res = dl.walk(q)
    assert res == {
        (i, 2 * i)
        for i in range(0, N, 2)
    }
