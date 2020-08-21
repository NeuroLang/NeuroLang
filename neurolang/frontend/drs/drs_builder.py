from ...expressions import (
    Expression,
    Symbol,
    FunctionApplication as Fa,
    Constant as C,
)
from ...expression_walker import (
    add_match,
    ExpressionWalker,
    ReplaceSymbolWalker,
)
from .chart_parser import Quote, CODE_QUOTE
from .english_grammar import (
    S,
    V,
    NP,
    VP,
    PN,
    DET,
    N,
    VAR,
    SL,
    LIT,
)
from ...logic import (
    Negation,
    Implication,
    Conjunction,
    ExistentialPredicate,
    UniversalPredicate,
)
import re


def indent(s, tab="    "):
    return "".join(tab + l for l in s.splitlines(keepend=True))


class DRS(Expression):
    def __init__(self, referents, expressions):
        self.referents = referents
        self.expressions = expressions

    def __repr__(self):
        return (
            "DRS <"
            + ", ".join(map(repr, self.referents))
            + "> [\n"
            + "".join("    " + repr(e) + ",\n" for e in self.expressions)
            + "]"
        )


class DRSBuilderBase(ExpressionWalker):
    def __init__(self, grammar):
        self.grammar = grammar
        self.accessible_referents = set()

    @add_match(DRS)
    def walk_drs(self, drs):
        old = set(self.accessible_referents)
        self.accessible_referents |= drs.referents

        changed = False
        refs = set(drs.referents)
        exps = tuple()

        for e in drs.expressions:
            if changed:
                exps += (e,)
            elif isinstance(e, DRS):
                refs |= set(r for r in e.referents if r not in refs)
                exps += e.expressions
                changed = True
            else:
                new_e = self.walk(e)
                changed |= new_e is not e
                exps += (new_e,)

        self.accessible_referents = old

        if changed:
            return self.walk(DRS(refs, exps))
        else:
            return drs

    @add_match(Fa, lambda fa: any(isinstance(e, DRS) for e in fa.args))
    def float_drs(self, fa):
        args = ()
        drs = None
        for e in fa.args:
            if isinstance(e, DRS) and not drs:
                drs = e
                args += (drs.expressions[0],)
            else:
                args += (e,)
        exps = (Fa(fa.functor, args),) + tuple(drs.expressions[1:])
        return self.walk(DRS(drs.referents, exps))

    @add_match(Implication)
    def implication(self, impl):
        drs_ant = self.walk(impl.antecedent)
        old = set(self.accessible_referents)
        self.accessible_referents |= drs_ant.referents
        drs_con = self.walk(impl.consequent)
        self.accessible_referents = old
        if drs_ant is not impl.antecedent or drs_con is not impl.consequent:
            return Implication(drs_con, drs_ant)
        return impl


class DRSBuilder(DRSBuilderBase):
    @add_match(Fa(Fa(NP, ...), (Fa(Fa(PN, ...), ...),)))
    def proper_names(self, np):
        (pn,) = np.args
        (_, _, const) = pn.functor.args
        return self.walk(DRS(set(), (const,)))

    @add_match(
        Fa(Fa(S, ...), (..., Fa(Fa(VP, ...), (Fa(Fa(V, ...), ...), ...)),),)
    )
    def predicate(self, s):
        (subject, vp) = s.args
        (v, object_) = vp.args
        exp = Symbol(v.args[0].value)(subject, object_)
        return self.walk(DRS(set(), (exp,)))

    @add_match(
        Fa(Fa(NP, ...), (Fa(Fa(DET, ...), ...), Fa(Fa(N, ...), ...),)),
        lambda np: np.args[0].args[0].value in ["a", "an"],
    )
    def indefinite_noun_phrase(self, np):
        (det, n) = np.args
        x = Symbol.fresh()
        exp = Symbol(n.args[0].value)(x)
        return self.walk(DRS({x}, (x, exp)))

    @add_match(Fa(Fa(NP, ...), (Fa(Fa(VAR, ...), ...),)),)
    def var_noun_phrase(self, np):
        (var,) = np.args
        v = Symbol(var.args[0].value)
        refs = {v} - self.accessible_referents
        return self.walk(DRS(refs, (v,)))

    @add_match(
        Fa(Fa(NP, ...), (Fa(Fa(NP, ...), ...), Fa(Fa(VAR, ...), ...),)),
    )
    def var_apposition(self, np):
        (np, var) = np.args
        np_drs = self.walk(np)

        y = Symbol(var.args[0].value)
        x = np_drs.expressions[0]
        rsw = ReplaceSymbolWalker({x: y})

        exps = ()
        for e in np_drs.expressions:
            exps += (rsw.walk(e),)

        refs = set()
        for r in np_drs.referents:
            refs |= {rsw.walk(r)}

        refs -= self.accessible_referents
        return self.walk(DRS(refs, exps))

    @add_match(Fa(Fa(S, ...), (C("if"), ..., C("then"), ...),),)
    def conditional(self, s):
        (_, ant, _, cons) = s.args
        return self.walk(DRS(set(), (Implication(cons, ant),)))

    @add_match(Fa(Fa(S, ...), (..., C("if"), ...),),)
    def inverse_conditional(self, s):
        (cons, _, ant) = s.args
        return self.walk(DRS(set(), (Implication(cons, ant),)))

    @add_match(Fa(Fa(S, ...), (Fa(Quote, (C(CODE_QUOTE), ...)),),),)
    def quoted_predicate(self, s):
        exp = _parse_predicate(s.args[0].args[1].value)
        refs = set(a for a in exp.args if isinstance(a, Symbol))
        refs -= self.accessible_referents
        return self.walk(DRS(refs, (exp,)))

    @add_match(Fa(Fa(S, ...), (..., C("and"), ...)),)
    def simple_and(self, s):
        (a, _, b) = s.args
        return self.walk(DRS(set(), (self.walk(a), b,)))

    @add_match(Fa(Fa(S, ...), (..., C(","), C("and"), ...),),)
    def comma_and(self, s):
        (sl, _, _, s) = s.args
        sl = self.walk(sl)
        return self.walk(DRS(set(), sl + (s,)))

    @add_match(Fa(Fa(SL, ...), (...,),),)
    def single_sentence_list(self, sl):
        (s,) = sl.args
        return (s,)

    @add_match(Fa(Fa(SL, ...), (Fa(Fa(SL, ...), ...), C(","), ...),),)
    def sentence_list(self, sl):
        (sl, _, s) = sl.args
        sl = self.walk(sl)
        return sl + (s,)

    @add_match(Fa(Fa(NP, ...), (Fa(Fa(LIT, ...), ...),)),)
    def lit_noun_phrase(self, np):
        (lit,) = np.args
        (const,) = lit.functor.args
        return self.walk(DRS(set(), (const,)))

    @add_match(
        Fa(
            Fa(S, ...),
            (C("is"), C("not"), C("the"), C("case"), C("that"), ...),
        ),
    )
    def sentence_negation(self, s):
        (_, _, _, _, _, inner_s) = s.args
        drs = self.walk(inner_s)
        return self.walk(DRS(set(), (Negation(drs),)))


r = re.compile(r'^(\w+)\(((\w+|"\w+")(,\s?(\w+|"\w+"))*)\)$')


def _parse_predicate(string):
    # This could totally use the datalog parser
    m = r.match(string)
    if not m:
        raise Exception(f"Quoted predicate is not valid datalog: {string}")
    functor = Symbol(m.group(1))
    args = map(_parse_argument, map(str.strip, m.group(2).split(",")))
    return functor(*args)


def _parse_argument(s):
    if s[0] == '"':
        return C(s.strip('"'))
    return Symbol(s)


class DRS2FOL(ExpressionWalker):
    @add_match(DRS)
    def drs(self, drs):
        exp = Conjunction(tuple(map(self.walk, drs.expressions)))
        for r in sorted(drs.referents, key=lambda s: s.name):
            exp = ExistentialPredicate(r, exp)
        return self.walk(exp)

    @add_match(Conjunction((...,)))
    def unary_conjunction(self, conj):
        return self.walk(conj.formulas[0])

    @add_match(Implication(DRS, DRS))
    def implication(self, impl):
        drs_ant = impl.antecedent
        drs_con = impl.consequent
        ant = Conjunction(tuple(map(self.walk, drs_ant.expressions)))
        con = self.walk(drs_con)
        exp = Implication(con, ant)
        for r in sorted(drs_ant.referents, key=lambda s: s.name):
            exp = UniversalPredicate(r, exp)
        return self.walk(exp)
