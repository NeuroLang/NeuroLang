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
SL = Symbol("SL")
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
_x = Symbol("_x")
_y = Symbol("_y")
_z = Symbol("_z")


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
            NP(num.plural, _x, c), (NP(n, g, c), Constant("and"), NP(m, h, c)),
        ),
        Rule(NP(n, g, c), (NP(n, g, c), VAR())),
        Rule(NP(n, g, _x), (PN(n, g, v),)),
        Rule(NP(_x, _y, _z), (VAR(),)),
        Rule(NP(n, g, _x), (DET(n), N(n, g))),
        Rule(NP(n, g, c), (PRO(n, g, c),)),
        Rule(SL(), (S(n),)),
        Rule(SL(), (SL(), Constant(","), S(_y),)),
        RootRule(S(_x), (SL(), Constant(","), Constant("and"), S(_y),)),
        RootRule(S(_x), (S(_y), Constant("and"), S(_z),)),
        Rule(S(n), (Quote(Constant(CODE_QUOTE), v),)),
        Rule(LIT(v), (Quote(Constant(STRING_QUOTE), v),)),
        Rule(NP(_x, _y, _z), (LIT(v),)),
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
    "provides": (V(num.singular),),
    "affects": (V(num.singular),),
    "reaches": (V(num.singular),),
    "affect": (V(num.plural),),
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
    "that": (DET(num.singular),),
    "woman": (N(num.singular, gen.female),),
    "stockbroker": (N(num.singular, gen.female), N(num.singular, gen.male),),
    "man": (N(num.singular, gen.male),),
    "book": (N(num.singular, gen.thing),),
    "donkey": (N(num.singular, gen.thing),),
    "horse": (N(num.singular, gen.thing),),
    "region": (N(num.singular, gen.thing),),
    "ending": (N(num.singular, gen.thing),),
    "function": (N(num.singular, gen.thing),),
}
