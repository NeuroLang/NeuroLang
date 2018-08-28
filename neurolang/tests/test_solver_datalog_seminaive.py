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

    dse = sds.DatalogSeminaiveEvaluator(dl)

    res = dse.walk(Q(x, y))
    assert res == dl.extensional_database()[Q].value

    res = dse.walk(Q(C_(0), y))
    assert res == {C_((C_(0), C_(0)))}

    res = dse.walk(Q(y, C_(2)))
    assert res == {C_((C_(1), C_(2)))}

    res = dse.walk(Q(C_(2), C_(4)))
    assert res == {C_((C_(2), C_(4)))}

    res = dse.walk(Q(y, C_(1)))
    assert res == set()


def test_intensional_single_case():

    Q = S_('Q')
    R = S_('r')
    S = S_('S')
    T = S_('T')
    x = S_('x')
    y = S_('y')

    extensional = B_(tuple(
        T_(Q(C_(i), C_(2 * i)))
        for i in range(500)
    ))

    intensional_1 = B_((
        St_(R(x), Q(x, C_(2))),
    ))
    dl = Datalog()
    dl.walk(extensional)
    dl.walk(intensional_1)

    dse = sds.DatalogSeminaiveEvaluator(dl)

    # res = dse.walk(R(x))
    # assert res == {C_((C_(1),))}

    intensional_2 = B_((
        St_(S(x), Q(x, C_(2))),
        St_(S(x), Q(x, x)),
        St_(T(x), Q(x, y) & Q(y, x)),
    ))
    dl.walk(intensional_2)

    dse = sds.DatalogSeminaiveEvaluator(dl)

    # res = dse.walk(S(x))
    # assert res == {
    #     C_((C_(1),)),
    #     C_((C_(0),)),
    # }

    res = dse.walk(T(x))
    assert res == {
       C_((C_(0),)),
    }
