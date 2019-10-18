import operator

from pytest import raises

from ... import expression_walker, expressions
from .. import DatalogProgram, Fact, Implication, magic_sets
from ..chase import Chase
from ..expressions import TranslateToLogic

C_ = expressions.Constant
S_ = expressions.Symbol
Imp_ = Implication
F_ = Fact
Eb_ = expressions.ExpressionBlock


class Datalog(
    TranslateToLogic,
    DatalogProgram,
    expression_walker.ExpressionBasicEvaluator
):
    pass


def test_adorned_expression():
    x = S_('x')
    a = C_('a')

    ax = magic_sets.AdornedExpression(x, 'b', 1)
    ax_ = magic_sets.AdornedExpression(x, 'b', 1)
    ax__ = magic_sets.AdornedExpression(a, 'b', 1)

    assert ax.expression == x
    assert ax.adornment == 'b'
    assert ax.number == 1
    assert ax == ax_
    assert ax != ax__

    ax = magic_sets.AdornedExpression(a, 'b', 1)
    ax_ = magic_sets.AdornedExpression(a, 'b', 1)
    ax__ = magic_sets.AdornedExpression(a, 'b', 2)

    assert ax.expression == a
    assert ax.adornment == 'b'
    assert ax.number == 1
    assert ax == ax_
    assert ax != ax__

    with raises(NotImplementedError):
        magic_sets.AdornedExpression(a, 'b', 1).name


def test_resolution_works():
    x = S_('X')
    y = S_('Y')
    z = S_('Z')
    anc = S_('anc')
    par = S_('par')
    q = S_('q')
    a = C_('a')
    b = C_('b')
    c = C_('c')
    d = C_('d')

    edb = Eb_([
        F_(par(a, b)),
        F_(par(b, c)),
        F_(par(c, d)),
    ])

    code = Eb_([
        Imp_(q(x), anc(a, x)),
        Imp_(anc(x, y), par(x, y)),
        Imp_(anc(x, y), anc(x, z) & par(z, y)),
    ])

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_sets.magic_rewrite(q(x), dl)

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)

    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {C_((e,)) for e in (b, c, d)}


def test_resolution_works_query_constant():
    x = S_('X')
    y = S_('Y')
    z = S_('Z')
    anc = S_('anc')
    par = S_('par')
    q = S_('q')
    a = C_('a')
    b = C_('b')
    c = C_('c')
    d = C_('d')

    edb = Eb_([
        F_(par(a, b)),
        F_(par(b, c)),
        F_(par(c, d)),
    ])

    code = Eb_([
        Imp_(q(x, a), anc(a, x)),
        Imp_(anc(x, y), par(x, y)),
        Imp_(anc(x, y), anc(x, z) & par(z, y)),
    ])

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_sets.magic_rewrite(q(x), dl)

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)

    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {C_((e, a)) for e in (b, c, d)}


def test_resolution_works_builtin():
    x = S_('X')
    y = S_('Y')
    z = S_('Z')
    anc = S_('anc')
    par = S_('par')
    anc2 = S_('anc2')
    q = S_('q')
    a = C_('a')
    b = C_('b')
    c = C_('c')
    d = C_('d')
    eq = C_(operator.eq)

    edb = Eb_([
        F_(par(a, b)),
        F_(par(b, c)),
        F_(par(c, d)),
    ])

    code = Eb_([
        Imp_(q(x), anc2(a, x)),
        Imp_(anc(x, y), par(x, y)),
        Imp_(anc(x, y), anc(x, z) & par(z, y)),
        Imp_(anc2(x, y), anc(x, z) & eq(z, y))
    ])

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_sets.magic_rewrite(q(x), dl)

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)

    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {C_((e,)) for e in (b, c, d)}
