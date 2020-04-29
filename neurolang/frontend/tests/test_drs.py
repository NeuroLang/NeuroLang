from ...expressions import Symbol, Constant
from ..chart_parser import Grammar, add_rule, ChartParser, _lu


S = Symbol("S")
NP = Symbol("NP")
PN = Symbol("PN")
VP = Symbol("VP")
V = Symbol("V")

a = Symbol("a")
b = Symbol("b")
w = Symbol("w")

plural = Constant("plural")
singular = Constant("singular")


class DRSGrammar(Grammar):
    @add_rule(NP(a), VP(a), root=True)
    def s(self, np, vp):
        return S(np.args[0])  # this could look better if unified too

    @add_rule(V(a), NP(b))
    def vp(self, v, np):
        return VP(v.args[0])

    @add_rule(NP(a), Constant("and"), NP(b))
    def np_and(self, first, _, second):
        return NP(plural)

    @add_rule(PN(a))
    def np_proper(self, pn):
        return NP(pn.args[0])

    @add_rule(w)
    def verb_singular(self, token):
        if token.value in ["owns"]:
            return V(singular)

    @add_rule(w)
    def verb_plural(self, token):
        if token.value in ["own"]:
            return V(plural)

    @add_rule(w)
    def proper_name(self, token):
        if token.value in ["Jones", "Smith", "Ulysses"]:
            return PN(singular)


def test_mgu():
    u = _lu.unify(a, Constant("hehe"))
    assert u is not None
    u = _lu.unify(NP(a), NP(Constant("hoho")))
    assert u is not None


def test_parser():
    g = DRSGrammar()
    cp = ChartParser(g)
    assert cp.recognize("Jones owns Ulysses")
    assert not cp.recognize("Jones own Ulysses")
    assert cp.recognize("Jones and Smith own Ulysses")
