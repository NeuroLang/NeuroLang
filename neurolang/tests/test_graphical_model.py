import numpy as np

from ..expressions import Symbol, Constant, ExpressionBlock
from ..expression_pattern_matching import add_match
from ..solver_datalog_naive import Fact, Implication, DatalogBasic
from ..probabilistic.graphical_model import (
    produce, GraphicalModel, GDatalogToGraphicalModelTranslator,
    substitute_dterm, gdatalog2gm, delta_infer1,
    solve_conditional_probability_query, solve_map_query
)
from ..probabilistic.ppdl import DeltaTerm, is_gdatalog_rule
from ..probabilistic.distributions import TableDistribution
from ..datalog.instance import SetInstance

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
c = C_(4)
p = S_('p')
p_a = C_(0.2)
p_b = C_(0.7)
bernoulli = S_('bernoulli')

program_1 = ExpressionBlock((
    Fact(P(a)),
    Fact(P(b)),
    Implication(Q(x, y),
                P(x) & P(y)),
))
program_2 = ExpressionBlock((
    Fact(P(a, b)),
    Fact(P(b, a)),
    Implication(Q(x), P(x, x)),
    Implication(R(x),
                P(x, y) & P(y, x)),
))
program_3 = ExpressionBlock((
    Fact(P(a)),
    Fact(P(b)),
    Implication(Q(x, DeltaTerm(bernoulli, (C_(0.7), ))), P(x)),
    Implication(Z(x, C_(0.2)), Q(x, C_(1))),
    Implication(Z(x, C_(0.8)), Q(x, C_(0))),
    Implication(R(x, DeltaTerm(bernoulli, (y, ))), Z(x, y)),
))
program_4 = ExpressionBlock((
    Fact(P(a)),
    Implication(Q(x, DeltaTerm(bernoulli, (C_(0.2), ))), P(x)),
    Implication(R(x, y, DeltaTerm(bernoulli, (C_(0.9), ))), Q(x, y)),
))


def test_produce():
    fact = Fact(P(a))
    rule = Implication(Q(x), P(x))
    result = produce(rule, [fact])
    assert result is not None
    assert result == Fact(Q(a))


def test_substitute_dterm():
    fa = Q(x, DeltaTerm(bernoulli, (p, )))
    assert substitute_dterm(fa, a) == Q(x, a)


def test_delta_produce():
    fact_a = Fact(P(a, p_a))
    fact_b = Fact(P(b, p_b))
    rule = Implication(Q(x, DeltaTerm(bernoulli, (p, ))), P(x, p))
    assert produce(rule, [fact_a]) == Fact(Q(a, DeltaTerm(bernoulli, (p_a, ))))
    assert produce(rule, [fact_b]) == Fact(Q(b, DeltaTerm(bernoulli, (p_b, ))))


def test_delta_infer1():
    fact_a = Fact(P(a, p_a))
    fact_b = Fact(P(b, p_b))
    rule = Implication(Q(x, DeltaTerm(bernoulli, (p, ))), P(x, p))
    result = delta_infer1(rule, SetInstance(frozenset({fact_a, fact_b})))
    expected_dist = TableDistribution({
        SetInstance(frozenset({Fact(Q(a, C_(0))),
                               Fact(Q(b, C_(0)))})):
        (1 - p_a.value) * (1 - p_b.value),
        SetInstance(frozenset({Fact(Q(a, C_(1))),
                               Fact(Q(b, C_(0)))})):
        p_a.value * (1 - p_b.value),
        SetInstance(frozenset({Fact(Q(a, C_(0))),
                               Fact(Q(b, C_(1)))})):
        (1 - p_a.value) * p_b.value,
        SetInstance(frozenset({Fact(Q(a, C_(1))),
                               Fact(Q(b, C_(1)))})):
        p_a.value * p_b.value,
    })
    assert expected_dist == result


def test_get_dependency_sorted_random_variables():
    gm = gdatalog2gm(program_1)
    assert gm.parents['Q_1'] == {'P'}
    assert gm.parents['Q'] == {'Q_1'}
    assert gm.get_dependency_sorted_random_variables() == ['P', 'Q_1', 'Q']


def test_gdatalog_translation_to_gm():
    gm = gdatalog2gm(program_1)
    assert set(gm.rv_to_cpd_functor.keys()) == {'P', 'Q', 'Q_1'}
    assert gm.parents['P'] == set()
    assert gm.parents['Q_1'] == {'P'}
    assert gm.parents['Q'] == {'Q_1'}

    gm = gdatalog2gm(program_2)
    assert gm.parents['Q_1'] == {'P'}
    assert gm.parents['Q'] == {'Q_1'}
    assert gm.parents['R_1'] == {'P'}
    assert gm.parents['R'] == {'R_1'}


def test_delta_term():
    program = ExpressionBlock((
        Fact(P(a)),
        Implication(Q(x, DeltaTerm(bernoulli, (x, ))), P(x)),
    ))
    gm = gdatalog2gm(program)
    assert 'Q_1' in gm.rv_to_cpd_functor


def test_2levels_model():
    gm = gdatalog2gm(program_4)

    program = ExpressionBlock((
        Fact(P(a)),
        Fact(P(b)),
        Implication(Q(x, DeltaTerm(bernoulli, (C_(0.5), ))), P(x)),
        Implication(Z(C_(0.1)), Q(x, C_(0))),
        Implication(Z(C_(0.9)), Q(x, C_(1))),
        Implication(R(x, DeltaTerm(bernoulli, (z, ))),
                    Q(x, y) & Z(z)),
    ))
    gm = gdatalog2gm(program)
    assert set(gm.rv_to_cpd_functor.keys()
               ) == {'P', 'Q_1', 'Q', 'Z_1', 'Z_2', 'Z', 'R_1', 'R'}


def test_gm_solver_program_1():
    gm = gdatalog2gm(program_1)
    outcomes = solve_conditional_probability_query(gm, SetInstance())
    assert len(outcomes.value.table) == 1
    outcome = list(outcomes.value.table.items())[0]
    assert outcome[1] == 1.0


def test_gm_solver_program_4():
    gm = gdatalog2gm(program_4)
    outcomes = solve_conditional_probability_query(gm, SetInstance())
    expected_dist = TableDistribution({
        SetInstance(
            frozenset({
                Fact(P(a)),
                Fact(Q(a, C_(1))),
                Fact(R(a, C_(1), C_(1)))
            })
        ):
        0.2 * 0.9,
        SetInstance(
            frozenset({
                Fact(P(a)),
                Fact(Q(a, C_(1))),
                Fact(R(a, C_(1), C_(0)))
            })
        ):
        0.2 * 0.1,
        SetInstance(
            frozenset({
                Fact(P(a)),
                Fact(Q(a, C_(0))),
                Fact(R(a, C_(0), C_(0)))
            })
        ):
        0.8 * 0.1,
        SetInstance(
            frozenset({
                Fact(P(a)),
                Fact(Q(a, C_(0))),
                Fact(R(a, C_(0), C_(1)))
            })
        ):
        0.8 * 0.9,
    })
    assert expected_dist == outcomes.value


def test_conditional_probability_query_resolution():
    gm = gdatalog2gm(program_4)
    evidence = SetInstance(frozenset({Fact(Q(a, C_(0)))}))
    outcomes = solve_conditional_probability_query(gm, evidence)
    expected_dist = TableDistribution({
        SetInstance(
            frozenset({
                Fact(P(a)),
                Fact(Q(a, C_(0))),
                Fact(R(a, C_(0), C_(0)))
            })
        ):
        0.1,
        SetInstance(
            frozenset({
                Fact(P(a)),
                Fact(Q(a, C_(0))),
                Fact(R(a, C_(0), C_(1)))
            })
        ):
        0.9,
    })
    assert expected_dist == outcomes.value

    evidence = SetInstance(
        frozenset({
            Fact(Q(a, C_(0))),
            Fact(R(a, C_(0), C_(1))),
        })
    )
    outcomes = solve_conditional_probability_query(gm, evidence)
    expected_dist = TableDistribution({
        SetInstance(
            frozenset({
                Fact(P(a)),
                Fact(Q(a, C_(0))),
                Fact(R(a, C_(0), C_(1)))
            })
        ):
        1.0,
    })
    assert expected_dist == outcomes.value


def test_conditional_probability_query_resolution_multiple_rules_same_pred():
    program = ExpressionBlock((
        Fact(P(a)),
        Implication(Q(x, DeltaTerm(bernoulli, (C_(0.5), ))), P(x)),
        Implication(Q(x, DeltaTerm(bernoulli, (C_(0.5), ))), P(x)),
    ))
    evidence = SetInstance({Fact(Q(a, C_(0)))})
    gm = gdatalog2gm(program)
    outcomes = solve_conditional_probability_query(gm, evidence)
    expected_dist = TableDistribution({
        SetInstance(frozenset({Fact(P(a)), Fact(Q(a, C_(0)))})):
        1.0 / 3.0,
        SetInstance(
            frozenset({Fact(P(a)),
                       Fact(Q(a, C_(0))),
                       Fact(Q(a, C_(1)))})
        ):
        2.0 / 3.0,
    })
    assert expected_dist == outcomes.value


def test_map_query_resolution_simple():
    program = ExpressionBlock((
        Fact(P(a)),
        Implication(Q(x, DeltaTerm(bernoulli, (C_(0.9), ))), P(x)),
    ))
    gm = gdatalog2gm(program)
    query_atoms = frozenset({Q(x, y)})
    evidence = SetInstance()
    solution = solve_map_query(gm, query_atoms, evidence)
    assert solution == SetInstance({
        Q: frozenset({(a, C_(1))}),
    })
