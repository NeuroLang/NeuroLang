from ..expressions import Symbol, Constant, ExpressionBlock
from ..solver_datalog_naive import Fact, Implication
from ..graphical_model import produce, infer, GraphicalModelSolver

C_ = Constant
S_ = Symbol

P = S_('P')
Q = S_('Q')
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
