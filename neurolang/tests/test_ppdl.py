import pytest

from .. import solver_datalog_extensional_db
from .. import expression_walker
from ..expression_walker import ExpressionBasicEvaluator
from ..exceptions import NeuroLangException
from ..expressions import ExpressionBlock, Constant, Symbol, Query
from ..existential_datalog import Implication
from ..probabilistic.ppdl import (
    add_to_expression_block, GenerativeDatalog,
    SolverNonRecursiveGenerativeDatalog, TranslateGDatalogToEDatalog,
    DeltaTerm, get_dterm
)
from ..solver_datalog_naive import Fact

C_ = Constant
S_ = Symbol

x = S_('x')
y = S_('y')
z = S_('z')
P = S_('P')
Q = S_('Q')
W = S_('W')
K = S_('K')
Z = S_('Z')
a = C_('a')
b = C_('b')


class GenerativeDatalogTest(GenerativeDatalog, ExpressionBasicEvaluator):
    pass


class GenerativeDatalogTestSolver(
    solver_datalog_extensional_db.ExtensionalDatabaseSolver,
    SolverNonRecursiveGenerativeDatalog, ExpressionBasicEvaluator
):
    pass


class TranslateGDatalogToEDatalogTestSolver(
    TranslateGDatalogToEDatalog,
):
    pass


def test_get_dterm():
    datom = P(x, y, DeltaTerm('Hi'))
    assert get_dterm(datom) == DeltaTerm('Hi')
    datom = P(x, y, DeltaTerm('Hi', C_(2)))
    assert get_dterm(datom) == DeltaTerm('Hi', C_(2))


def test_generative_datalog():
    tau_1 = Implication(P(x, DeltaTerm('Flip', C_(0.5))), Q(x))
    program = ExpressionBlock((tau_1, ))
    gdatalog = GenerativeDatalogTest()
    gdatalog.walk(program)
    assert tau_1 in gdatalog.symbol_table[P].expressions

    with pytest.raises(NeuroLangException):
        tau_2 = Implication(P(x, DeltaTerm('Flip'), DeltaTerm('Flap')), Q(x))
        gdatalog = GenerativeDatalogTest()
        gdatalog.walk(ExpressionBlock((tau_2, )))


def test_translation_of_gdatalog_program_to_edatalog_program():
    solver = TranslateGDatalogToEDatalogTestSolver()
    res = solver.walk(Implication(P(x, DeltaTerm('Flip', C_(0.5))), Q(x)))
    assert isinstance(res, ExpressionBlock)
    assert len(res.expressions) == 2


def test_non_generative_rule_preserved_when_block_translated():
    block = ExpressionBlock((
        Implication(P(x, DeltaTerm('Flip', C_(0.5))), Q(x)),
        Fact(Q(a)),
        Fact(Q(b)),
        Implication(Z(x, y, z),
                    W(x, y) & K(y, z)),
    ))
    translator = TranslateGDatalogToEDatalogTestSolver()
    translated_block = translator.walk(block)
    assert Fact(Q(a)) in translated_block.expressions
    assert (
        Implication(P(x, DeltaTerm('Flip', C_(0.5))),
                    Q(x)) not in translated_block.expressions
    )
    assert (
        Implication(Z(x, y, z),
                    W(x, y) & K(y, z)) in translated_block.expressions
    )


def test_burglar():
    City = S_('City')
    House = S_('House')
    Business = S_('Business')
    Unit = S_('Unit')
    Earthquake = S_('Earthquake')
    Burglary = S_('Burglary')
    Trig = S_('Trig')
    Alarm = S_('Alarm')
    Flip = C_('Flip')
    x, h, b, c, r = S_('x'), S_('h'), S_('b'), S_('c'), S_('r')
    program = ExpressionBlock((
        Implication(Unit(h, c), House(h, c)),
        Implication(Unit(b, c), Business(b, c)),
        Implication(Earthquake(c, DeltaTerm(Flip, C_(0.01))), City(c, r)),
        Implication(
            Burglary(x, c, DeltaTerm(Flip, r)),
            Unit(x, c) & City(c, r)
        ),
        Implication(
            Trig(x, DeltaTerm(Flip, C_(0.6))),
            Unit(x, c) & Earthquake(c, C_(1))
        ),
        Implication(Trig(x, DeltaTerm(Flip, C_(0.9))), Burglary(x, c, C_(1))),
        Implication(Alarm(x), Trig(x, C_(1)))
    ))

    translator = TranslateGDatalogToEDatalogTestSolver()
    translated = translator.walk(program)
    assert not any(
        isinstance(e, ExpressionBlock) for e in translated.expressions
    )
    solver = GenerativeDatalogTestSolver()
    solver.walk(translated)


def test_pcs_example():
    Bernoulli = C_('bernoulli')
    Uniform = C_('uniform')
    Gender = S_('Gender')
    Subject = S_('Subject')
    pGender = S_('pGender')
    pHasLPC = S_('pHasLPC')
    pHasRPC = S_('pHasRPC')
    HasLPC = S_('HasLPC')
    HasRPC = S_('HasRPC')
    x = S_('x')
    p = S_('p')
    program = ExpressionBlock((
        Implication(
            Gender(x, DeltaTerm(Bernoulli, p)),
            Subject(x) & pGender(p)
        ),
        Implication(
            HasLPC(x, DeltaTerm(Bernoulli, p)),
            Subject(x) & pHasLPC(p)
        ),
        Implication(
            HasRPC(x, DeltaTerm(Bernoulli, p)),
            Subject(x) & pHasRPC(p)
        ),
        Implication(pGender(DeltaTerm(Uniform, C_(0), C_(1))), C_('True')),
        Implication(pHasLPC(DeltaTerm(Uniform, C_(0), C_(1))), C_('True')),
        Implication(pHasRPC(DeltaTerm(Uniform, C_(0), C_(1))), C_('True')),
    ))
    translator = TranslateGDatalogToEDatalogTestSolver()
    translated = translator.walk(program)
    assert not any(
        isinstance(e, ExpressionBlock) for e in translated.expressions
    )
    solver = GenerativeDatalogTestSolver()
    solver.walk(translated)


def test_add_to_expression_block():
    x = S_('x')
    y = S_('y')
    z = S_('z')
    P = S_('P')
    Q = S_('Q')
    block1 = ExpressionBlock((P(x), ))
    block2 = ExpressionBlock((
        Q(z),
        Q(y),
        Q(x),
    ))
    block3 = ExpressionBlock((
        P(z),
        P(y),
    ))
    assert P(y) in add_to_expression_block(block1, P(y)).expressions
    for expression in block2.expressions:
        new_block = add_to_expression_block(block1, block2)
        assert expression in new_block.expressions
        new_block = add_to_expression_block(block1, [block2])
        assert expression in new_block.expressions
    for expression in (
        block1.expressions + block2.expressions + block3.expressions
    ):
        new_block = add_to_expression_block(block1, [block2, block3])
        assert expression in new_block.expressions
