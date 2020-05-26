from ...expressions import Symbol, Constant
from .chart_parser import Grammar, DictLexicon, add_rule


S = Symbol("S")
NP = Symbol("NP")
PN = Symbol("PN")
VP = Symbol("VP")
V = Symbol("V")
DET = Symbol("DET")
N = Symbol("N")
PRO = Symbol("PRO")
VAR = Symbol("VAR")

c = Symbol("c")
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


class case:
    nom = Constant("nom")
    notnom = Constant("notnom")


class EnglishGrammar(Grammar):
    def __init__(self, lexicon):
        super().__init__(lexicon)

    @add_rule(NP(n, g, case.nom), VP(n), root=True)
    def s(self, np, vp):
        return S(n)

    @add_rule(V(n), NP(m, g, case.notnom))
    def vp(self, v, np):
        return VP(n)

    @add_rule(NP(n, g, c), Constant("and"), NP(m, h, c))
    def np_and(self, first, _, second):
        return NP(num.plural, Symbol.fresh(), c)

    @add_rule(NP(n, g, c), VAR())
    def np_apposition(self, np, var):
        return NP(n, g, c)

    @add_rule(PN(n, g))
    def np_proper(self, pn):
        return NP(n, g, Symbol.fresh())

    @add_rule(VAR())
    def np_var(self, var):
        return NP(Symbol.fresh(), Symbol.fresh(), Symbol.fresh())

    @add_rule(DET(n), N(n, g))
    def np_indefinite(self, det, noun):
        return NP(n, g, Symbol.fresh())

    @add_rule(PRO(n, g, c))
    def np_pronoun(self, pro):
        return NP(n, g, c)


class EnglishBaseLexicon(DictLexicon):
    def __init__(self):
        super().__init__({
            "he": (PRO(num.singular, gen.male, case.nom),),
            "him": (PRO(num.singular, gen.male, case.notnom),),
            "she": (PRO(num.singular, gen.female, case.nom),),
            "her": (PRO(num.singular, gen.female, case.notnom),),
            "it": (
                PRO(num.singular, gen.thing, case.nom),
                PRO(num.singular, gen.thing, case.notnom),
            ),
            "owns": (V(num.singular),),
            "has": (V(num.singular),),
            "likes": (V(num.singular),),
            "intersects": (V(num.singular),),
            "own": (V(num.plural),),
            "have": (V(num.plural),),
            "like": (V(num.plural),),
            "Jones": (PN(num.singular, gen.male),),
            "Smith": (PN(num.singular, gen.male),),
            "Ulysses": (PN(num.singular, gen.thing),),
            "a": (DET(num.singular),),
            "an": (DET(num.singular),),
            "every": (DET(num.singular),),
            "the": (DET(num.singular),),
            "woman": (N(num.singular, gen.female),),
            "stockbroker": (
                N(num.singular, gen.female),
                N(num.singular, gen.male)
            ),
            "man": (N(num.singular, gen.male),),
            "book": (N(num.singular, gen.thing),),
            "donkey": (N(num.singular, gen.thing),),
            "horse": (N(num.singular, gen.thing),),
            "region": (N(num.singular, gen.thing),),
            "ending": (N(num.singular, gen.thing),),
        })

    def get_meanings(self, word):
        m = super().get_meanings(word)
        if word.isupper():
            m += (VAR(),)
        return m
