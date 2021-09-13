import operator

from pytest import raises

from ... import expression_walker, expressions
from .. import DatalogProgram, Fact, Implication, magic_sets
from ..aggregation import (
    AGG_COUNT,
    BuiltinAggregationMixin,
    DatalogWithAggregationMixin,
    TranslateToLogicWithAggregation,
)
from ..chase import Chase
from ..negation import DatalogProgramNegationMixin

C_ = expressions.Constant
S_ = expressions.Symbol
Imp_ = Implication
F_ = Fact
Eb_ = expressions.ExpressionBlock


class Datalog(
    TranslateToLogicWithAggregation,
    BuiltinAggregationMixin,
    DatalogWithAggregationMixin,
    DatalogProgramNegationMixin,
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


def test_resolution_works_2():
    """
    Reverse Same Generation from Abiteboul et al. 1995.
    Foundations of databases. p.312

    l0:  l  m   n   o   p
    l1:  e  f   g   h   i   j   k
    l2:  a  b   c   d
    """
    x = S_("x")
    y = S_("y")
    x1 = S_("x1")
    y1 = S_("y1")
    rsg = S_("rsg")
    up = S_("up")
    down = S_("down")
    flat = S_("flat")
    q = S_("q")

    edb = Eb_(
        [
            F_(up(C_("a"), C_("e"))),
            F_(up(C_("a"), C_("f"))),
            F_(up(C_("f"), C_("m"))),
            F_(up(C_("g"), C_("n"))),
            F_(up(C_("h"), C_("n"))),
            F_(up(C_("i"), C_("o"))),
            F_(up(C_("j"), C_("o"))),
            F_(flat(C_("g"), C_("f"))),
            F_(flat(C_("m"), C_("n"))),
            F_(flat(C_("m"), C_("o"))),
            F_(flat(C_("p"), C_("m"))),
            F_(down(C_("l"), C_("f"))),
            F_(down(C_("m"), C_("f"))),
            F_(down(C_("g"), C_("b"))),
            F_(down(C_("h"), C_("c"))),
            F_(down(C_("i"), C_("d"))),
            F_(down(C_("p"), C_("k"))),
        ]
    )

    code = Eb_(
        [
            Imp_(q(y), rsg(C_("a"), y)),
            Imp_(rsg(x, y), flat(x, y)),
            Imp_(rsg(x, y), up(x, x1) & rsg(y1, x1) & down(y1, y)),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_sets.magic_rewrite(q(x), dl)

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)

    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {(C_("b"),), (C_("c"),), (C_("d"),)}


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


def test_resolution_works_aggregation():
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

    edb = Eb_([
        F_(par(a, b)),
        F_(par(b, c)),
        F_(par(c, d)),
    ])

    code = Eb_([
        Imp_(q(x), anc2(a, x)),
        Imp_(anc(x, y), par(x, y)),
        Imp_(anc(x, y), anc(x, z) & par(z, y)),
        Imp_(anc2(x, AGG_COUNT(y)), anc(x, y))
    ])

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_sets.magic_rewrite(q(x), dl)

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)

    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {(3,)}
