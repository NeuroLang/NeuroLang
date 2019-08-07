from .. import datalog_chase
from .. import datalog_magic_sets
from .. import expression_walker
from .. import solver_datalog_naive

C_ = solver_datalog_naive.Constant
S_ = solver_datalog_naive.Symbol
Imp_ = solver_datalog_naive.Implication
F_ = solver_datalog_naive.Fact
Eb_ = solver_datalog_naive.ExpressionBlock


class Datalog(
    solver_datalog_naive.DatalogBasic,
    expression_walker.ExpressionBasicEvaluator
):
    pass


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
    goal, mr = datalog_magic_sets.magic_rewrite(q(x), dl)

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)

    solution = datalog_chase.build_chase_solution(dl)
    assert solution[goal].value == {C_((e,)) for e in (b, c, d)}
