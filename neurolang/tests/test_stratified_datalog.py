import pytest

from operator import and_, or_, invert

from .. import expressions, exceptions
from .. import solver_datalog_naive as sdb
from .. import solver_datalog_extensional_db
from .. import expression_walker as ew
from ..stratified_datalog import (
    StratifiedDatalog,
    ConsequentSymbols,
    NeuroLangDataLogNonStratifiable
)
from ..solver_datalog_naive import (
    Implication,
    Fact,
)

C_ = expressions.Constant
S_ = expressions.Symbol
F_ = expressions.FunctionApplication
L_ = expressions.Lambda
E_ = expressions.ExistentialPredicate
U_ = expressions.UniversalPredicate
Eb_ = expressions.ExpressionBlock

class Datalog(
    sdb.DatalogBasic,
    solver_datalog_extensional_db.ExtensionalDatabaseSolver,
    ew.ExpressionBasicEvaluator
):
    def function_gt(self, x: int, y: int)->bool:
        return x > y


def test_consequent_symbols():
    w = S_('w')
    x = S_('x')
    y = S_('y')
    z = S_('z')

    imp0 = Implication(y(), invert(x()))
    imp1 = Implication(invert(x()), and_(y(), invert(z())))
    imp2 = Implication(invert(x()), or_(invert(y()), invert(z())))
    imp3 = Implication(x(), z())
    imp4 = Fact(w())
    imp5 = Implication(w(), and_(C_(False), z()))
    imp6 = Implication(w(), w)

    program = Eb_((
        imp0,
        imp1,
        imp2,
        imp3,
        imp4,
        imp5,
        imp6,
    ))

    cSymbols = ConsequentSymbols()
    con = cSymbols.walk(program)

    assert con[0] == [invert(x())]
    assert con[1] == [y(), invert(z())]
    assert con[2] == [invert(y()), invert(z())]
    assert con[3] == [z()]
    assert con[4] == []
    assert con[5] == [z()]
    assert con[6] == [w]


def test_empty_program():
    program = Eb_(())

    sDatalog = StratifiedDatalog()
    stratified = sDatalog.stratify(program)

    assert stratified.expressions == ()
    assert stratified == program


def test_negated_consequent():
    x = S_('x')
    y = S_('y')
    z = S_('z')

    imp0 = Implication(y(), invert(x()))
    imp1 = Implication(invert(x()), and_(y(), invert(z())))
    imp2 = Implication(x(), z())

    program = Eb_((imp0, imp1, imp2))

    sDatalog = StratifiedDatalog()
    with pytest.raises(
        exceptions.NeuroLangException,
        match=r"Symbol in the consequent can not be .*"
    ):
        sDatalog.stratify(program)


def test_valid_stratification():
    x = S_('x')
    y = S_('y')
    z = S_('z')

    imp0 = Implication(y(), invert(x()))
    imp1 = Implication(x(), and_(y(), invert(z())))
    imp2 = Implication(x(), z())

    program = Eb_((imp0, imp1, imp2))

    sDatalog = StratifiedDatalog()
    stratified = sDatalog.stratify(program)

    assert stratified.expressions[0] == imp1
    assert stratified.expressions[1] == imp2
    assert stratified.expressions[2] == imp0

    w = S_('w')

    imp0 = Implication(y(), invert(x()))
    imp1 = Implication(w(), and_(y(), invert(z())))
    imp2 = Implication(x(), z())

    program = Eb_((imp0, imp1, imp2))

    sDatalog = StratifiedDatalog()
    stratified = sDatalog.stratify(program)

    assert stratified.expressions[0] == imp1
    assert stratified.expressions[1] == imp2
    assert stratified.expressions[2] == imp0

    imp0 = Implication(y(), invert(x()))
    imp1 = Implication(w(), and_(y(), invert(z())))
    imp2 = Implication(x(), z())
    imp3 = Implication(z(), invert(y()))

    program = Eb_((imp0, imp1, imp2, imp3))

    sDatalog = StratifiedDatalog()
    stratified = sDatalog.stratify(program)

    assert stratified.expressions[0] == imp2
    assert stratified.expressions[1] == imp0
    assert stratified.expressions[2] == imp3
    assert stratified.expressions[3] == imp1


def test_impossible_stratification():
    x = S_('x')
    y = S_('y')
    z = S_('z')

    imp0 = Implication(y(), invert(x()))
    imp1 = Implication(x(), and_(invert(y()), invert(z())))
    imp2 = Implication(x(), z())

    program = Eb_((imp0, imp1, imp2))

    sDatalog = StratifiedDatalog()
    with pytest.raises(
        NeuroLangDataLogNonStratifiable,
        match=r"The program cannot be stratifiable.*"
    ):
        sDatalog.stratify(program)


def test_positive_cicle():
    x = S_('x')
    y = S_('y')
    z = S_('z')

    imp0 = Implication(y(), x())
    imp1 = Implication(x(), and_(y(), invert(z())))

    program = Eb_((imp0, imp1))

    sDatalog = StratifiedDatalog()
    stratified = sDatalog.stratify(program)

    assert stratified.expressions[0] == imp0
    assert stratified.expressions[1] == imp1


def test_different_consts():
    x = S_('x')
    y = S_('y')
    z = S_('z')
    a = C_('a')
    b = C_('b')

    imp0 = Implication(y(), invert(x(a)))
    imp1 = Implication(x(a), and_(y(), invert(z(b))))
    imp2 = Implication(x(b), z(a))

    program = Eb_((imp0, imp1, imp2))

    sDatalog = StratifiedDatalog()
    stratified = sDatalog.stratify(program)

    assert stratified.expressions[0] == imp1
    assert stratified.expressions[1] == imp0
    assert stratified.expressions[2] == imp2

    w = S_('w')

    imp0 = Implication(y(b), invert(x(a)))
    imp1 = Implication(w(), and_(invert(y(b)), invert(z())))
    imp2 = Implication(x(b), z())

    program = Eb_((imp0, imp1, imp2))

    sDatalog = StratifiedDatalog()
    stratified = sDatalog.stratify(program)

    assert stratified.expressions[0] == imp0
    assert stratified.expressions[1] == imp1
    assert stratified.expressions[2] == imp2

    imp0 = Implication(y(a), invert(x(a)))
    imp1 = Implication(w(), and_(y(b), invert(z(b))))
    imp2 = Implication(x(b), z())
    imp3 = Implication(z(b), invert(y(a)))

    program = Eb_((imp0, imp1, imp2, imp3))

    sDatalog = StratifiedDatalog()
    stratified = sDatalog.stratify(program)

    assert stratified.expressions[0] == imp0
    assert stratified.expressions[1] == imp2
    assert stratified.expressions[2] == imp3
    assert stratified.expressions[3] == imp1
