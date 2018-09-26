import pytest

from .. import solver_datalog_extensional_db
from .. import expression_walker
from ..exceptions import NeuroLangException
from ..expressions import ExpressionBlock, Constant, Symbol, Query, Statement
from ..existential_datalog import Implication
from ..generative_datalog import (
    GenerativeDatalog, SolverNonRecursiveGenerativeDatalog,
    TranslateGDatalogToEDatalog, DeltaTerm, DeltaAtom
)

C_ = Constant
S_ = Symbol
St_ = Statement


class GenerativeDatalogTestSolver(
    solver_datalog_extensional_db.ExtensionalDatabaseSolver,
    SolverNonRecursiveGenerativeDatalog,
    expression_walker.ExpressionBasicEvaluator
):
    pass


class TranslateGDatalogToEDatalogTestSolver(
    solver_datalog_extensional_db.ExtensionalDatabaseSolver,
    TranslateGDatalogToEDatalog,
    GenerativeDatalog,
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
    solver.walk(Implication(P(x, DeltaTerm('Flip', C_(0.5))), Q(x)))
    assert 'P' in solver.intensional_database()

    # block = ExpressionBlock((
        # Implication(P(x, DeltaTerm('Flip', C_(0.5))), Q(x)),
    # ))


def test_burglar():
    solver = GenerativeDatalogTestSolver()
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

    extensional = ExpressionBlock(())

    intensional = ExpressionBlock((
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
