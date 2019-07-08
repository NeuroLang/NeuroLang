import pytest

from .. import expressions

from operator import and_, or_, invert

from ..stratified_datalog import StratifiedDatalog

from ..existential_datalog import Implication

C_ = expressions.Constant
S_ = expressions.Symbol
F_ = expressions.FunctionApplication
L_ = expressions.Lambda
E_ = expressions.ExistentialPredicate
U_ = expressions.UniversalPredicate
Eb_ = expressions.ExpressionBlock


def test_negated_consequent():
    x = S_('x')
    y = S_('y')
    z = S_('z')

    imp0 = Implication(y(), invert(x()))
    imp1 = Implication(invert(x()), and_(y(), invert(z())))
    imp2 = Implication(x(), z())

    program = Eb_((
        imp0, imp1, imp2
    ))

    sDatalog = StratifiedDatalog()
    with pytest.raises(expressions.NeuroLangException, 
        match=r"Symbol in the consequent can not be .*"):
            sDatalog.solve(program)


def test_valid_stratification():
    x = S_('x')
    y = S_('y')
    z = S_('z')

    imp0 = Implication(y(), invert(x()))
    imp1 = Implication(x(), and_(y(), invert(z())))
    imp2 = Implication(x(), z())

    program = Eb_((
        imp0, imp1, imp2
    ))

    sDatalog = StratifiedDatalog()
    stratified = sDatalog.solve(program)

    assert stratified.expressions[0] == imp1
    assert stratified.expressions[1] == imp2
    assert stratified.expressions[2] == imp0


def test_imposible_stratification():
    x = S_('x')
    y = S_('y')
    z = S_('z')

    imp0 = Implication(y(), invert(x()))
    imp1 = Implication(x(), and_(invert(y()), invert(z())))
    imp2 = Implication(x(), z())

    program = Eb_((
        imp0, imp1, imp2
    ))

    sDatalog = StratifiedDatalog()
    with pytest.raises(expressions.NeuroLangException, 
        match=r"The program cannot be stratifiable"):
            sDatalog.solve(program)


def test_positive_cicle():
    x = S_('x')
    y = S_('y')
    z = S_('z')

    imp0 = Implication(y(), x())
    imp1 = Implication(x(), and_(y(), invert(z())))

    program = Eb_((
        imp0, imp1
    ))

    sDatalog = StratifiedDatalog()
    stratified = sDatalog.solve(program)

    assert stratified.expressions[0] == imp0
    assert stratified.expressions[1] == imp1
    
