from ..expressions import Symbol, Constant, ExpressionBlock
from ..expression_pattern_matching import add_match
from ..solver_datalog_naive import Fact, Implication, DatalogBasic
from ..graphical_model import (
    produce,
    infer,
    GDatalogToGraphicalModelTranslator,
    substitute_dterm,
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
symb_p_a = S_('p_a')
symb_p_b = S_('p_b')
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


def test_substitute_dterm():
    fa = DeltaAtom(Q, (x, DeltaTerm(bernoulli, p)))
    assert substitute_dterm(fa, a) == Q(x, a)


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


# def test_delta_infer1():
    # fact_a = Fact(P(a, symb_p_a))
    # fact_b = Fact(P(b, symb_p_b))
    # rule = Implication(DeltaAtom(Q, (x, DeltaTerm(bernoulli, p))), P(x, p))
    # result = delta_infer1(rule, {fact_a.fact, fact_b.fact})
    # result_as_dict = dict(result)
    # dist = {
        # frozenset({Q(a, C_(0)), Q(b, C_(0))}):
        # Multiplication(
            # Subtraction(C_(1), symb_p_a), Subtraction(C_(1), symb_p_b)
        # ),
        # frozenset({Q(a, C_(1)), Q(b, C_(0))}):
        # Multiplication(symb_p_a, Subtraction(C_(1), symb_p_b)),
        # frozenset({Q(a, C_(0)), Q(b, C_(1))}):
        # Multiplication(Subtraction(C_(1), symb_p_a), symb_p_b),
        # frozenset({Q(a, C_(1)), Q(b, C_(1))}):
        # Multiplication(symb_p_a, symb_p_b),
    # }
    # for outcome, prob in dist.items():
        # assert outcome in result_as_dict
        # assert arithmetic_eq(prob, result_as_dict[outcome])


def test_gdatalog_translation_to_gm():
    program = ExpressionBlock((
        Fact(P(a)),
        Fact(P(b)),
        Implication(Q(x, y),
                    P(x) & P(y)),
    ))
    gm = GDatalogToGraphicalModelTranslator().walk(program)
    assert 'P' in gm.random_variables
    assert 'Q_1' in gm.random_variables
    assert gm.parents['Q_1'] == {'P'}
    assert gm.parents['P'] == set()

    program = ExpressionBlock((
        Fact(P(a, b)),
        Fact(P(b, a)),
        Implication(Q(x), P(x, x)),
        Implication(R(x),
                    P(x, y) & P(y, x)),
    ))
    gm = GDatalogToGraphicalModelTranslator().walk(program)
    assert gm.parents['Q_1'] == {'P'}
    assert gm.parents['R_1'] == {'P'}


def test_delta_term():
    program = ExpressionBlock((
        Fact(P(a)),
        Implication(Q(x, DeltaTerm(bernoulli, x)), P(x)),
    ))
    program = sugar_remove(program)
    gm = GDatalogToGraphicalModelTranslator().walk(program)
    assert 'Q_1' in gm.random_variables


def test_2levels_model():
    program = ExpressionBlock((
        Fact(P(a)),
        Implication(Q(x, DeltaTerm(bernoulli, C_(0.5))), P(x)),
        Implication(R(x, DeltaTerm(bernoulli, C_(0.5))), Q(x, y)),
    ))
    program = sugar_remove(program)
    gm = GDatalogToGraphicalModelTranslator().walk(program)

    program = sugar_remove(
        ExpressionBlock((
            Fact(P(a)),
            Fact(P(b)),
            Implication(Q(x, DeltaTerm(bernoulli, C_(0.5))), P(x)),
            Implication(Z(C_(0.1)), Q(x, C_(0))),
            Implication(Z(C_(0.9)), Q(x, C_(1))),
            Implication(R(x, DeltaTerm(bernoulli, z)),
                        Q(x, y) & Z(z)),
        ))
    )
    gm = GDatalogToGraphicalModelTranslator().walk(program)
    assert set(gm.random_variables.keys()) == {'P', 'Q_1', 'Z_1', 'Z_2', 'R_1'}
