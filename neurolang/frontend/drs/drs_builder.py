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
from .english_grammar import S, V, NP, VP, PN, DET, N, VAR, SL, LIT
from .exceptions import ParseDatalogPredicateException
from ...logic import (
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


class DRSBuilder(ExpressionWalker):
    def __init__(self, grammar):
        self.grammar = grammar

    @add_match(
        DRS, lambda drs: any(isinstance(e, DRS) for e in drs.expressions),
    )
    def join_drs(self, drs):
        refs = drs.referents
        exps = ()
        for e in drs.expressions:
            if isinstance(e, DRS):
                refs += tuple(r for r in e.referents if r not in refs)
                exps += e.expressions
            else:
                exps += (e,)
        return self.walk(DRS(refs, exps))

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

    @add_match(Fa(Fa(NP, ...), (Fa(Fa(PN, ...), ...),)))
    def proper_names(self, np):
        (pn,) = np.args
        (_, _, const) = pn.functor.args
        return self.walk(DRS((), (const,)))

    @add_match(
        Fa(Fa(S, ...), (..., Fa(Fa(VP, ...), (Fa(Fa(V, ...), ...), ...)),),)
    )
    def predicate(self, s):
        (subject, vp) = s.args
        (v, object_) = vp.args
        exp = Symbol(v.args[0].value)(subject, object_)
        return self.walk(DRS((), (exp,)))

    @add_match(
        Fa(Fa(NP, ...), (Fa(Fa(DET, ...), ...), Fa(Fa(N, ...), ...),)),
        lambda np: np.args[0].args[0].value in ["a", "an"],
    )
    def indefinite_noun_phrase(self, np):
        (det, n) = np.args
        x = Symbol.fresh()
        exp = Symbol(n.args[0].value)(x)
        return self.walk(DRS((x,), (x, exp)))

    @add_match(Fa(Fa(NP, ...), (Fa(Fa(VAR, ...), ...),)),)
    def var_noun_phrase(self, np):
        (var,) = np.args
        v = Symbol(var.args[0].value)
        return self.walk(DRS((v,), (v,)))

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

        refs = ()
        for r in np_drs.referents:
            refs += (rsw.walk(r),)

        return self.walk(DRS(refs, exps))

    @add_match(
        Fa(
            Fa(S, ...),
            (C("if"), Fa(Fa(S, ...), ...), C("then"), Fa(Fa(S, ...), ...),),
        ),
    )
    def conditional(self, s):
        (_, ant, _, cons) = s.args
        return self.walk(DRS((), (Implication(cons, ant),)))

    @add_match(Fa(Fa(S, ...), (Fa(Quote, (C(CODE_QUOTE), ...)),),),)
    def quoted_predicate(self, s):
        exp = _parse_predicate(s.args[0].args[1].value)
        return self.walk(DRS(exp.args, (exp,)))

    @add_match(
        Implication(DRS, DRS),
        lambda impl: (
            set(impl.antecedent.referents) & set(impl.consequent.referents)
        ),
    )
    def implication(self, impl):
        drs_ant = impl.antecedent
        drs_con = impl.consequent
        drs_con.referents = tuple(
            set(drs_con.referents) - set(drs_ant.referents)
        )
        return self.walk(Implication(drs_con, drs_ant))

    @add_match(
        Fa(Fa(S, ...), (Fa(Fa(S, ...), ...), C("and"), Fa(Fa(S, ...), ...),),),
    )
    def simple_and(self, s):
        (a, _, b) = s.args
        a = self.walk(a)
        b = self.walk(b)
        return self.walk(DRS((), (a, b,)))

    @add_match(
        Fa(
            Fa(S, ...),
            (Fa(Fa(SL, ...), ...), C(","), C("and"), Fa(Fa(S, ...), ...),),
        ),
    )
    def comma_and(self, s):
        (sl, _, _, s) = s.args
        sl = self.walk(sl)
        s = self.walk(s)
        return self.walk(DRS((), sl + (s,)))

    @add_match(
        Fa(
            Fa(SL, ...),
            (Fa(Fa(S, ...), ...),),
        ),
    )
    def single_sentence_list(self, sl):
        (s,) = sl.args
        return (self.walk(s),)

    @add_match(
        Fa(
            Fa(SL, ...),
            (Fa(Fa(SL, ...), ...), C(","), Fa(Fa(S, ...), ...)),
        ),
    )
    def sentence_list(self, sl):
        (sl, _, s) = sl.args
        sl = self.walk(sl)
        s = self.walk(s)
        return sl + (s,)

    @add_match(Fa(Fa(NP, ...), (Fa(Fa(LIT, ...), ...),)),)
    def lit_noun_phrase(self, np):
        (lit,) = np.args
        (const,) = lit.functor.args
        return self.walk(DRS((), (const,)))


r = re.compile(r"^(\w+)\((\w+(,\s\w+)*)\)$")


def _parse_predicate(string):
    # This could totally use the datalog parser
    m = r.match(string)
    if not m:
        raise ParseDatalogPredicateException(
            f"Quoted predicate is not valid datalog: {string}"
        )
    functor = Symbol(m.group(1))
    args = map(Symbol, map(str.strip, m.group(2).split(",")))
    return functor(*args)


class DRS2FOL(ExpressionWalker):
    @add_match(DRS)
    def drs(self, drs):
        exp = Conjunction(tuple(map(self.walk, drs.expressions)))
        for r in drs.referents:
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
        for r in drs_ant.referents:
            exp = UniversalPredicate(r, exp)
        return self.walk(exp)
