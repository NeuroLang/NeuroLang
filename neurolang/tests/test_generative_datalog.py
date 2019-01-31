import pytest

from .. import solver_datalog_extensional_db
from .. import expression_walker
from ..exceptions import NeuroLangException
from ..expressions import ExpressionBlock, Constant, Symbol, Query
from ..existential_datalog import Implication
from ..generative_datalog import (
    GenerativeDatalogSugarRemover, SolverNonRecursiveGenerativeDatalog,
    TranslateGDatalogToEDatalog, DeltaTerm, DeltaAtom
)
from ..solver_datalog_naive import Fact

C_ = Constant
S_ = Symbol


class GenerativeDatalogTestSolver(
    solver_datalog_extensional_db.ExtensionalDatabaseSolver,
    SolverNonRecursiveGenerativeDatalog,
    expression_walker.ExpressionBasicEvaluator
):
    pass


class TranslateGDatalogToEDatalogTestSolver(
    GenerativeDatalogSugarRemover,
    TranslateGDatalogToEDatalog,
):
    pass


def test_delta_atom_without_delta_term():
    with pytest.raises(NeuroLangException):
        DeltaAtom('TestAtom', (S_('x'), ))


def test_delta_atom_delta_term():
    delta_atom = DeltaAtom('TestAtom', (S_('x'), S_('y'), DeltaTerm('Hi')))
    assert delta_atom.delta_term == DeltaTerm('Hi')
    delta_atom = DeltaAtom(
        'TestAtom', (S_('x'), S_('y'), DeltaTerm('Hi', C_(2)))
    )
    assert delta_atom.delta_term == DeltaTerm('Hi', C_(2))
    assert delta_atom.delta_term != DeltaTerm('Hi')


def test_translation_of_gdatalog_program_to_edatalog_program():
    solver = TranslateGDatalogToEDatalogTestSolver()
    P = S_('P')
    Q = S_('Q')
    x = S_('x')
    res = solver.walk(Implication(P(x, DeltaTerm('Flip', C_(0.5))), Q(x)))
    assert isinstance(res, ExpressionBlock)
    assert len(res.expressions) == 2


def test_non_generative_rule_preserved_when_block_translated():
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
    Bernoulli = C_('Bernoulli')
    Uniform = C_('Uniform')
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
