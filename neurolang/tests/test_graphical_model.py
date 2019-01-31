from ..expressions import Symbol, Constant, ExpressionBlock
from ..solver_datalog_naive import Fact, Implication
from ..graphical_model import produce, infer, GraphicalModelSolver

C_ = Constant
S_ = Symbol

P = S_('P')
Q = S_('Q')
R = S_('R')
x = S_('x')
y = S_('y')
a = C_(2)
b = C_(3)


def test_produce():
    fact = Fact(P(a))
    rule = Implication(Q(x), P(x))
    result = produce(rule, [fact.fact])
    assert result is not None
    assert result == Q(a)


def test_infer():
    facts = {P(a), P(b)}
    rule1 = Implication(Q(x), P(x))
    assert infer(rule1, facts) == {Q(a), Q(b)}

    rule2 = Implication(Q(x, y), P(x) & P(y))
    assert infer(rule2, facts) == {Q(a, b), Q(b, a)}


def test_graphical_model_conversion_simple():
    program = ExpressionBlock((
        Fact(P(a)),
        Fact(P(b)),
        Implication(Q(x, y),
                    P(x) & P(y)),
    ))
    gm = GraphicalModelSolver()
    gm.walk(program)
    for predicate in ['P', 'Q']:
        assert predicate in gm.random_variables
        assert predicate in gm.cpds
    assert gm.parents['Q'] == {'Q_1'}
    assert gm.parents['Q_1'] == {'P'}
    assert gm.parents['P'] == set()
    assert gm.cpds['P'](gm.parents['P']) == {P(a), P(b)}
    assert gm.cpds['Q'](gm.parents['Q']) == {Q(a, b), Q(b, a)}

    program = ExpressionBlock((
        Fact(P(a, b)),
        Fact(P(b, a)),
        Implication(Q(x), P(x, x)),
        Implication(R(x),
                    P(x, y) & P(y, x)),
    ))
    gm = GraphicalModelSolver()
    gm.walk(program)
    assert gm.parents['Q'] == {'Q_1'}
    assert gm.parents['Q_1'] == {'P'}
    assert gm.parents['R'] == {'R_1'}
    assert gm.parents['R_1'] == {'P'}
    assert gm.cpds['Q'](gm.parents['Q']) == set()
    assert gm.cpds['R'](gm.parents['R']) == {R(b), R(a)}
    assert gm.sample('Q') == set()
    assert gm.sample('R') == {R(b), R(a)}
