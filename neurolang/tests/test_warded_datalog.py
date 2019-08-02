import pytest

from .. import expressions
from .. import solver_datalog_naive as sdb
from .. import solver_datalog_extensional_db
from .. import datalog_chase as dc
from .. import expression_walker as ew
from ..solver_datalog_naive import (
    Implication, Fact
)
from ..warded_datalog import WardedDatalog, NeuroLangDataLogNonWarded

class Datalog(
    sdb.DatalogBasic,
    solver_datalog_extensional_db.ExtensionalDatabaseSolver,
    ew.ExpressionBasicEvaluator
):
    def function_gt(self, x: int, y: int)->bool:
        return x > y

C_ = expressions.Constant
S_ = expressions.Symbol
Fa_ = expressions.FunctionApplication
Ep_ = expressions.ExistentialPredicate
Eb_ = expressions.ExpressionBlock


def test_warded_walker_1():
    x = S_('x')
    y = S_('y')
    z = S_('z')
    P = S_('P')
    Q = S_('Q')
    R = S_('R')

    datalog_program = Eb_((
        Fact(Q(x, z)),
        Implication(R(y, z), P(x, y)),
        Implication(R(x, z), Q(x,z) & P(x)),
    ))

    wd = WardedDatalog()
    warded = wd.walk(datalog_program)

    assert warded == True

def test_warded_walker_2():
    P = S_('P')
    Q = S_('Q')
    T = S_('T')
    x = S_('x')
    y = S_('y')
    z = S_('z')

    datalog_program = Eb_((
        Implication(Q(z, x), P(x)),
        Implication(T(x), Q(x, y) & P(y)),
    ))

    wd = WardedDatalog()
    warded = wd.walk(datalog_program)

    assert warded == True


def test_variables_outside_ward():
    P = S_('P')
    Q = S_('Q')
    T = S_('T')
    x = S_('x')
    y = S_('y')
    z = S_('z')

    datalog_program = Eb_((
        Implication(Q(z, x), P(x)),
        Implication(T(x), Q(x, y) & P(x)),
    ))

    wd = WardedDatalog()
    with pytest.raises(
        NeuroLangDataLogNonWarded,
        match=r"The program is not warded: there are dangerous variables outside the ward.*"
    ):
        warded = wd.walk(datalog_program)



def test_more_one_atom():
    P = S_('P')
    Q = S_('Q')
    R = S_('R')
    S = S_('S')
    T = S_('T')
    x = S_('x')
    y = S_('y')
    z = S_('z')

    datalog_program = Eb_((
        Implication(Q(z, x), P(x)),
        Implication(R(y, x), S(x)),
        Implication(T(x), Q(x, y) & P(y) & R(x, z) & S(z)),
    ))

    wd = WardedDatalog()
    with pytest.raises(
        NeuroLangDataLogNonWarded,
        match=r"The program is not warded: there are dangerous variables that appear in more than one atom of the body.*"
    ):
        warded = wd.walk(datalog_program)


def test_warded_chase():
    hsbc = C_('HSBC')
    hsb = C_('HSB')
    iba = C_('IBA')

    company = S_('Company')
    controls = S_('Controls')
    owns = S_('Owns')
    stock = S_('Stock')
    PSC = S_('PSC')
    strongLink = S_('StrongLink')

    x = S_('x')
    p = S_('p')
    s = S_('s')
    y = S_('y')

    datalog_program = Eb_((
        Fact(company(hsbc)),
        Fact(company(hsb)),
        Fact(company(iba)),
        Fact(controls(hsbc, hsb)),
        Fact(controls(hsb, iba)),

        Implication(owns(p, s, x), company(x)),
        Implication(stock(x, s), owns(p, s, x)),
        Implication(PSC(x, p), owns(p, s, x)),
        Implication(owns(p, s, y), PSC(x, p) & controls(x, y)),
        Implication(strongLink(x, y), PSC(x, p) & PSC(y, p)),
        Implication(owns(p, s, x), strongLink(x, y)),
        Implication(owns(p, s, y), strongLink(x, y)),
        Implication(company(x), stock(x, s)),
    ))

    wd = WardedDatalog()
    wd.walk(datalog_program)

    #dl = Datalog()
    #dl.walk(datalog_program)

    #solution_instance = dc.build_chase_solution(dl)

    #print(solution_instance)