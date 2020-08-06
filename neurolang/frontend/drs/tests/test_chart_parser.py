from ....expressions import Symbol, Constant, FunctionApplication as Fa
from ....expression_walker import PatternWalker, add_match
from ..chart_parser import Grammar, add_rule, ChartParser, _lu, DictLexicon


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


class TestGrammar(Grammar):
    @add_rule(NP(a), VP(a), root=True)
    def s(self, np, vp):
        return S(a)

    @add_rule(V(a), NP(b))
    def vp(self, v, np):
        return VP(a)

    @add_rule(NP(a), Constant("and"), NP(b))
    def np_and(self, first, _, second):
        return NP(plural)

    @add_rule(PN(a))
    def np_proper(self, pn):
        return NP(a)


test_lexicon = DictLexicon({
    "owns": (V(singular),),
    "own": (V(plural),),
    "Jones": (PN(singular),),
    "Smith": (PN(singular),),
    "Ulysses": (PN(singular),),
})


def test_mgu():
    u = _lu.unify(a, Constant("hehe"))
    assert u is not None
    u = _lu.unify(NP(a), NP(Constant("hoho")))
    assert u is not None


def test_recognize():
    g = TestGrammar(test_lexicon)
    cp = ChartParser(g)
    assert cp.recognize("Jones owns Ulysses")
    assert not cp.recognize("Jones own Ulysses")
    assert cp.recognize("Jones and Smith own Ulysses")


def test_parse():
    g = TestGrammar(test_lexicon)
    cp = ChartParser(g)
    tree = S(singular)(
        NP(singular)(PN(singular)(Constant("Jones"))),
        VP(singular)(
            V(singular)(Constant("owns")),
            NP(singular)(PN(singular)(Constant("Ulysses"))),
        ),
    )
    assert tree == cp.parse("Jones owns Ulysses")[0]


class TestGrammarWalker(PatternWalker):
    @add_match(Fa(Fa(S, ...), ...))
    def s(self, exp):
        (np, vp) = exp.args
        return self.walk(np) + " " + self.walk(vp)

    @add_match(Fa(Fa(NP, ...), ...))
    def np_proper(self, exp):
        (pn,) = exp.args
        return self.walk(pn)

    @add_match(Fa(Fa(VP, ...), ...))
    def vp(self, exp):
        (v, np) = exp.args
        return self.walk(v) + " " + self.walk(np)

    @add_match(Fa(Fa(PN, ...), ...))
    def pn(self, exp):
        (w,) = exp.args
        return w.value

    @add_match(Fa(Fa(V, ...), ...))
    def v(self, exp):
        (w,) = exp.args
        return w.value


def test_walk_parsed():
    g = TestGrammar(test_lexicon)
    cp = ChartParser(g)
    sentence = "Jones owns Ulysses"
    tree = cp.parse(sentence)[0]
    r = TestGrammarWalker().walk(tree)
    assert sentence == r


class TestGrammarWalker2(TestGrammarWalker):
    @add_match(Fa(V(singular), ...))
    def v_sing(self, exp):
        return "SV"

    @add_match(
        Fa(Fa(NP, ...), (..., "and", ...,),)
    )
    def np_conj(self, exp):
        (pn1, c, pn2) = exp.args
        return self.walk(pn1) + " and " + self.walk(pn2)

    @add_match(Fa(V(plural), ...))
    def v_plur(self, exp):
        return "PV"

    @add_match(Fa(Fa(PN, ...), ...))
    def pn_(self, exp):
        (num,) = exp.functor.args
        return "SN" if num == singular else "PN"


def test_walk_parsed_2():
    g = TestGrammar(test_lexicon)
    cp = ChartParser(g)
    tree = cp.parse("Jones owns Ulysses")[0]
    r = TestGrammarWalker2().walk(tree)
    assert "SN SV SN" == r

    tree = cp.parse("Jones and Smith own Ulysses")[0]
    r = TestGrammarWalker2().walk(tree)
    assert "SN and SN PV SN"
