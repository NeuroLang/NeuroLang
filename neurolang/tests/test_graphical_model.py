from ..expressions import Symbol, Constant, ExpressionBlock
from ..expression_pattern_matching import add_match
from ..solver_datalog_naive import Fact, Implication, DatalogBasic
from ..graphical_model import (
    produce, infer, get_datom_vars, GraphicalModelSolver
)
from ..generative_datalog import (
    DeltaTerm, DeltaAtom, GenerativeDatalogSugarRemover
)

C_ = Constant
S_ = Symbol

P = S_('P')
Q = S_('Q')
R = S_('R')
Z = S_('Z')
x = S_('x')
y = S_('y')
z = S_('z')
a = C_(2)
b = C_(3)
p = S_('p')
p_a = C_(0.2)
p_b = C_(0.7)
bernoulli = C_('bernoulli')


class GenerativeDatalogSugarRemoverTest(
    GenerativeDatalogSugarRemover, DatalogBasic
):
    @add_match(Implication(DeltaAtom, ...))
    def ignore_gdatalog_rule(self, rule):
        return rule


def sugar_remove(program):
    return GenerativeDatalogSugarRemoverTest().walk(program)


def test_produce():
    fact = Fact(P(a))
    rule = Implication(Q(x), P(x))
    result = produce(rule, [fact.fact])
    assert result is not None
    assert result == Q(a)


def test_delta_produce():
    fact_a = Fact(P(a, p_a))
    fact_b = Fact(P(b, p_b))
    rule = Implication(Q(x, DeltaTerm(bernoulli, p)), P(x, p))
    assert produce(rule, [fact_a.fact]) == Q(a, DeltaTerm(bernoulli, p_a))
    assert produce(rule, [fact_b.fact]) == Q(b, DeltaTerm(bernoulli, p_b))


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
        assert predicate in gm.samplers
    assert gm.parents['Q'] == {'Q_1'}
    assert gm.parents['Q_1'] == {'P'}
    assert gm.parents['P'] == set()
    assert gm.samplers['P'](gm.parents['P']) == {P(a), P(b)}
    assert gm.samplers['Q'](gm.parents['Q']) == {Q(a, b), Q(b, a)}

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
    assert gm.samplers['Q'](gm.parents['Q']) == set()
    assert gm.samplers['R'](gm.parents['R']) == {R(b), R(a)}
    assert gm.sample('Q') == set()
    assert gm.sample('R') == {R(b), R(a)}


def test_get_datom_vars():
    datom = DeltaAtom(Q, (x, DeltaTerm(C_('hello'), C_(0.5), x, y), y))
    assert get_datom_vars(datom) == {'x', 'y'}


def test_delta_term():
    program = ExpressionBlock((
        Fact(P(a)),
        Implication(Q(x, DeltaTerm(bernoulli, x)), P(x)),
    ))
    program = sugar_remove(program)
    gm = GraphicalModelSolver()
    gm.walk(program)
    assert 'Q' in gm.random_variables
    assert 'Q_1' in gm.random_variables
    assert gm.sample('Q') in [{Q(a, C_(0))}, {Q(a, C_(1))}]


def test_2levels_model():
    program = ExpressionBlock((
        Fact(P(a)),
        Implication(Q(x, DeltaTerm(C_('bernoulli'), C_(0.5))), P(x)),
        Implication(R(x, DeltaTerm(C_('bernoulli'), C_(0.5))), Q(x, y)),
    ))
    program = sugar_remove(program)
    gm = GraphicalModelSolver()
    gm.walk(program)
    assert gm.sample('R') in [{R(a, C_(0))}, {R(a, C_(1))}]

    program = sugar_remove(
        ExpressionBlock((
            Fact(P(a)),
            Fact(P(b)),
            Implication(Q(x, DeltaTerm(C_('bernoulli'), C_(0.5))), P(x)),
            Implication(Z(C_(0.1)), Q(x, C_(0))),
            Implication(Z(C_(0.9)), Q(x, C_(1))),
            Implication(R(x, DeltaTerm(C_('bernoulli'), z)),
                        Q(x, y) & Z(z)),
        ))
    )
    gm = GraphicalModelSolver()
    gm.walk(program)
    assert gm.random_variables == {
        'P', 'Q_1', 'Q', 'Z_1', 'Z_2', 'Z', 'R_1', 'R'
    }
    gm.sample('R')


def test_direct_connection():
    '''Test for when X and Y are directly connected via an edge X -> Y'''
