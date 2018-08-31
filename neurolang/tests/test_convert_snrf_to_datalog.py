from operator import and_

from .. import convet_snrf_to_datalog as csd
from .. import expressions

C_ = expressions.Constant
S_ = expressions.Symbol
F_ = expressions.FunctionApplication
E_ = expressions.ExistentialPredicate


def test_simple_conjunction():
    P = S_('P')
    Q = S_('Q')

    x = S_('x')
    y = S_('y')

    exp = P(x, y) & Q(x)

    cs2d = csd.ConvertSNRFToDatalogWalker()
    lhs, datalog = cs2d.walk(exp)

    assert set(lhs.args) == {x, y}
    assert isinstance(lhs.functor, expressions.Symbol)
    assert lhs.functor not in exp._symbols
    assert len(datalog.expressions) == 1
    assert datalog.expressions[0].lhs == lhs
    assert datalog.expressions[0].rhs == exp


def test_simple_disjunction():
    P = S_('P')
    Q = S_('Q')

    x = S_('x')
    y = S_('y')

    exp = P(x, y) | Q(y, x)

    cs2d = csd.ConvertSNRFToDatalogWalker()
    lhs, datalog = cs2d.walk(exp)

    assert set(lhs.args) == {x, y}
    assert isinstance(lhs.functor, expressions.Symbol)
    assert lhs.functor not in exp._symbols
    assert len(datalog.expressions) == 2
    assert datalog.expressions[0].lhs == lhs
    assert datalog.expressions[1].lhs == lhs
    assert {e.rhs for e in datalog.expressions} == {P(x, y), Q(y, x)}


def test_simple_existential():
    P = S_('P')

    x = S_('x')
    y = S_('y')

    exp = E_(x, P(x, y))

    cs2d = csd.ConvertSNRFToDatalogWalker()
    lhs, datalog = cs2d.walk(exp)

    assert set(lhs.args) == {y}
    assert isinstance(lhs.functor, expressions.Symbol)
    assert lhs.functor not in exp._symbols
    assert len(datalog.expressions) == 1
    assert datalog.expressions[0].lhs == lhs
    assert datalog.expressions[0].rhs == exp.body


def test_conjunction_disjunction():
    P = S_('P')
    Q = S_('Q')

    x = S_('x')
    y = S_('y')

    exp = P(x, y) & (Q(x) | P(x))

    cs2d = csd.ConvertSNRFToDatalogWalker()
    lhs, datalog = cs2d.walk(exp)

    assert set(lhs.args) == {x, y}
    assert isinstance(lhs.functor, expressions.Symbol)
    assert lhs.functor not in exp._symbols
    assert len(datalog.expressions) == 3

    rhs_set = {
        e.rhs
        for e in datalog.expressions
    }

    assert rhs_set.issuperset({Q(x), P(x)})

    disjunctions = []
    for e in datalog.expressions:
        if (
            isinstance(e.rhs, expressions.FunctionApplication) and
            isinstance(e.rhs.functor, expressions.Constant) and
            e.rhs.functor.value == and_
        ):
            conjunction = e
        else:
            disjunctions.append(e)

    assert conjunction.lhs == lhs
    assert disjunctions[0].lhs == disjunctions[1].lhs
    assert disjunctions[0].lhs in conjunction.rhs.args
    assert P(x, y) in conjunction.rhs.args
