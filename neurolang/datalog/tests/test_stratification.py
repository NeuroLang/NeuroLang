from ... import expression_walker, expressions
from .. import DatalogProgram, Fact, Implication
from ..chase import Chase, ChaseN, ChaseSN
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

def test_resolution_works():
    x = S_('X')
    y = S_('Y')
    z = S_('Z')
    anc = S_('anc')
    par = S_('par')
    q = S_('q')
    M = S_('M')
    N = S_('N')
    O = S_('O')
    a = C_('a')
    b = C_('b')
    c = C_('c')
    d = C_('d')

    edb = Eb_([
        F_(par(a, b)),
        F_(par(b, c)),
        F_(par(c, d)),
        F_(M(a)),
    ])

    code = Eb_([
        Imp_(N(a), M(a)),
        Imp_(O(x), N(x)),
        Imp_(anc(x, y), par(x, y)),
        Imp_(q(x), anc(a, x)),
        Imp_(anc(x, y), anc(x, z) & par(z, y) & O(a)),
    ])

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)

    solution = Chase(dl).build_chase_solution()
    assert solution[q].value == {C_((e,)) for e in (b, c, d)}