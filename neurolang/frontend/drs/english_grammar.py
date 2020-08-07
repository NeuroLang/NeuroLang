from ...expressions import Symbol, Constant, FunctionApplication as Fa
from ...expression_walker import PatternWalker, add_match
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
UNK = Symbol("UNK")

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


class stype:
    if_ = Constant("if_")
    notif = Constant("notif")


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
        RootRule(S(n, stype.notif), (NP(n, g, case.nom), VP(n))),
        RootRule(S(n, stype.if_), (S(n, _x), Constant("if"), S(m, _y))),
        RootRule(
            S(n, stype.if_),
            (Constant("if"), S(n, _x), Constant("then"), S(m, _y)),
        ),
        Rule(VP(n), (V(n), NP(m, g, case.notnom))),
        Rule(V(n), (UNK(),)),
        Rule(
            NP(num.plural, _x, c), (NP(n, g, c), Constant("and"), NP(m, h, c)),
        ),
        Rule(NP(n, g, c), (NP(n, g, c), VAR())),
        Rule(NP(n, g, _x), (PN(n, g, v),)),
        Rule(NP(_x, _y, _z), (VAR(),)),
        Rule(NP(n, g, _x), (DET(n), N(n, g))),
        Rule(NP(n, g, c), (PRO(n, g, c),)),
        Rule(SL(), (S(n, stype.notif),)),
        Rule(SL(), (SL(), Constant(","), S(_y, stype.notif),)),
        RootRule(
            S(_x, stype.notif),
            (SL(), Constant(","), Constant("and"), S(_y, stype.notif),),
        ),
        RootRule(
            S(_x, stype.notif),
            (S(_y, stype.notif), Constant("and"), S(_z, stype.notif),),
        ),
        RootRule(S(n, stype.notif), (Quote(Constant(CODE_QUOTE), v),)),
        Rule(LIT(v), (Quote(Constant(STRING_QUOTE), v),)),
        Rule(NP(_x, _y, _z), (LIT(v),)),
    )
)


class UnknownWordsInSentence(PatternWalker):
    @add_match(Constant)
    def constant(self, _):
        return set()

    @add_match(Quote)
    def quote(self, _):
        return set()

    @add_match(Fa(Fa(V, ...), (Fa(Fa(UNK, ...), ...),)))
    def unknown_verb(self, v):
        (unk,) = v.args
        verb = unk.args[0].value
        return {(V, verb)}

    @add_match(Fa)
    def node(self, fa):
        uwords = set()
        for a in fa.args:
            uwords |= self.walk(a)
        return uwords


class EnglishBaseLexicon(DictLexicon):
    def __init__(self):
        super().__init__(FIXED_VOCABULARY)

    def get_meanings(self, token):
        m = super().get_meanings(token)
        if isinstance(token, Constant) and token.value.isupper():
            m += (VAR(),)
        return m


class DatalogLexicon(EnglishBaseLexicon):
    def __init__(self, nl):
        super().__init__()
        self.nl = nl
        self.initialize()

    def initialize(self):
        with self.nl.environment as e:
            e.verb[e.w, e.n] = e.singular_verb[e.w] & (e.n == num.singular)

    def get_meanings(self, token):
        m = super().get_meanings(token)

        if isinstance(token, Constant):
            n = self.nl.new_symbol(name="n")
            verb = self.nl.new_symbol(name="verb")
            sol = self.nl.query((n,), verb(token, n))

            for (n,) in sol:
                m += (V(Constant(n)),)

        return m


class UnknownWordLexicon(DatalogLexicon):
    def get_meanings(self, token):
        m = super().get_meanings(token)
        if len(m) == 0:
            m += (UNK(),)
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
    "contains": (V(num.singular),),
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
