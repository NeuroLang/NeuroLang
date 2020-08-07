from ...expressions import Symbol, Constant
from .chart_parser import (
    Grammar,
    DictLexicon,
    Rule,
    RootRule,
    Quote,
    CODE_QUOTE,
    STRING_QUOTE,
)


S = Symbol("S")
NP = Symbol("NP")
PN = Symbol("PN")
VP = Symbol("VP")
V = Symbol("V")
DET = Symbol("DET")
N = Symbol("N")
PRO = Symbol("PRO")
VAR = Symbol("VAR")
LIT = Symbol("LIT")

c = Symbol("c")
n = Symbol("n")
m = Symbol("m")
g = Symbol("g")
h = Symbol("h")
w = Symbol("w")
v = Symbol("v")


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


EnglishGrammar = Grammar(
    (
        RootRule(S(n), (NP(n, g, case.nom), VP(n))),
        RootRule(S(n), (S(n), Constant("if"), S(m))),
        RootRule(S(n), (Constant("if"), S(n), Constant("then"), S(m))),
        Rule(VP(n), (V(n), NP(m, g, case.notnom))),
        Rule(
            NP(num.plural, Symbol.fresh(), c),
            (NP(n, g, c), Constant("and"), NP(m, h, c)),
        ),
        Rule(NP(n, g, c), (NP(n, g, c), VAR())),
        Rule(NP(n, g, Symbol.fresh()), (PN(n, g, v),)),
        Rule(NP(Symbol.fresh(), Symbol.fresh(), Symbol.fresh()), (VAR(),)),
        Rule(NP(n, g, Symbol.fresh()), (DET(n), N(n, g))),
        Rule(NP(n, g, c), (PRO(n, g, c),)),
        Rule(S(n), (Quote(Constant(CODE_QUOTE), v),)),
        Rule(LIT(v), (Quote(Constant(STRING_QUOTE), v),)),
        Rule(NP(Symbol.fresh(), Symbol.fresh(), Symbol.fresh()), (LIT(v),)),
    )
)


class EnglishBaseLexicon(DictLexicon):
    def __init__(self):
        super().__init__(FIXED_VOCABULARY)

    def get_meanings(self, token):
        m = super().get_meanings(token)
        if isinstance(token, Constant) and token.value.isupper():
            m += (VAR(),)
        return m


FIXED_VOCABULARY = {
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
    "references": (V(num.singular),),
    "own": (V(num.plural),),
    "have": (V(num.plural),),
    "like": (V(num.plural),),
    "Jones": (PN(num.singular, gen.male, Constant("Jones")),),
    "Smith": (PN(num.singular, gen.male, Constant("Smith")),),
    "Ulysses": (PN(num.singular, gen.thing, Constant("Ulysses")),),
    "Odyssey": (PN(num.singular, gen.thing, Constant("Odyssey")),),
    "a": (DET(num.singular),),
    "an": (DET(num.singular),),
    "every": (DET(num.singular),),
    "the": (DET(num.singular),),
    "woman": (N(num.singular, gen.female),),
    "stockbroker": (N(num.singular, gen.female), N(num.singular, gen.male),),
    "man": (N(num.singular, gen.male),),
    "book": (N(num.singular, gen.thing),),
    "donkey": (N(num.singular, gen.thing),),
    "horse": (N(num.singular, gen.thing),),
    "region": (N(num.singular, gen.thing),),
    "ending": (N(num.singular, gen.thing),),
}
