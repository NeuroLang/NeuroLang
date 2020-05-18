from ...expressions import Expression, Symbol, FunctionApplication as Fa
from ...expression_walker import add_match, ExpressionWalker
from .english_grammar import S, V, NP, VP, PN, DET, N


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
        self.trace = []

    @add_match(
        DRS, lambda drs: any(isinstance(e, DRS) for e in drs.expressions),
    )
    def join_drs(self, drs):
        refs = drs.referents
        exps = ()
        for e in drs.expressions:
            if isinstance(e, DRS):
                refs += e.referents
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

        x = Symbol.fresh()
        exp = Symbol(pn.args[0].value)(x)
        self.trace.append(exp)
        return self.walk(DRS((x,), (x, exp)))

    @add_match(
        Fa(
            Fa(S, ...),
            (Symbol, Fa(Fa(VP, ...), (Fa(Fa(V, ...), ...), Symbol,)),),
        )
    )
    def predicate(self, s):
        (subject, vp) = s.args
        (v, object_) = vp.args
        exp = Symbol(v.args[0].value)(subject, object_)
        self.trace.append(exp)
        return self.walk(DRS((), (exp,)))

    @add_match(
        Fa(Fa(NP, ...), (
            Fa(Fa(DET, ...), ...),
            Fa(Fa(N, ...), ...),
        )),
        lambda np: np.args[0].args[0].value in ['a', 'an']
    )
    def indefinite_noun_phrase(self, np):
        (det, n) = np.args
        x = Symbol.fresh()
        exp = Symbol(n.args[0].value)(x)
        self.trace.append(exp)
        return self.walk(DRS((x,), (x, exp)))
