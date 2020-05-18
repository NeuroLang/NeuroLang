from ...expressions import Symbol, Constant
from .chart_parser import Grammar, add_rule


S = Symbol("S")
NP = Symbol("NP")
PN = Symbol("PN")
VP = Symbol("VP")
V = Symbol("V")
DET = Symbol("DET")
N = Symbol("N")
PRO = Symbol("PRO")

n = Symbol("n")
m = Symbol("m")
g = Symbol("g")
h = Symbol("h")
w = Symbol("w")


class num:
    plural = Constant("plural")
    singular = Constant("singular")


class gen:
    female = Constant("female")
    male = Constant("male")
    thing = Constant("thing")


class EnglishGrammar(Grammar):
    def __init__(self, lexicon):
        super().__init__()
        self.lexicon = lexicon

    @add_rule(NP(n, g), VP(n), root=True)
    def s(self, np, vp):
        return S(np.args[0])  # this could look better if unified too

    @add_rule(V(n), NP(m, g))
    def vp(self, v, np):
        return VP(v.args[0])

    @add_rule(NP(n, g), Constant("and"), NP(m, h))
    def np_and(self, first, _, second):
        return NP(num.plural, Symbol.fresh())

    @add_rule(PN(n, g))
    def np_proper(self, pn):
        return NP(pn.args[0], pn.args[1])

    @add_rule(DET(n), N(n, g))
    def np_indefinite(self, det, n):
        return NP(n.args[0], n.args[1])

    @add_rule(PRO(n, g))
    def np_pronoun(self, pro):
        return NP(pro.args[0], pro.args[1])

    @add_rule(w)
    def verb_singular(self, token):
        if token.value in ["owns", "has", "likes"]:
            return V(num.singular)

    @add_rule(w)
    def verb_plural(self, token):
        if token.value in ["own", "have", "like"]:
            return V(num.plural)

    @add_rule(w)
    def proper_name_male(self, token):
        if token.value in ["Jones", "Smith"]:
            return PN(num.singular, gen.male)

    @add_rule(w)
    def proper_name_thing(self, token):
        if token.value in ["Jones", "Smith", "Ulysses"]:
            return PN(num.singular, gen.thing)

    @add_rule(w)
    def determinant(self, token):
        if token.value in ["a", "an", "every", "the"]:
            return DET(num.singular)

    @add_rule(w)
    def noun_female(self, token):
        if token.value in ["woman", "stockbroker"]:
            return N(num.singular, gen.female)

    @add_rule(w)
    def noun_male(self, token):
        if token.value in ["man", "stockbroker"]:
            return N(num.singular, gen.male)

    @add_rule(w)
    def noun_thing(self, token):
        if token.value in ["book", "donkey", "horse", "region", "ending"]:
            return N(num.singular, gen.thing)

    @add_rule(w)
    def prop(self, token):
        props = {
            "he": gen.male,
            "she": gen.female,
            "it": gen.thing,
        }
        if token.value in props:
            return PRO(num.singular, props[token.value])


class BaseLexicon:
    pass
