"""
Pretty-printer for NeuroLang IR expressions as human-readable Datalog text.

Strips type annotations, uses ``:-`` notation for rules/queries, and
shortens fresh variables to short names (``s₀``, ``s₁``, …).
"""

import operator as _operator

from neurolang.expression_pattern_matching import add_match
from neurolang.expression_walker import PatternWalker
from neurolang.expressions import (
    Constant, Expression, ExpressionBlock,
    FunctionApplication, Lambda, Query, Symbol,
)
from neurolang.datalog.expressions import Implication
from neurolang.logic import (
    Conjunction, Disjunction, Negation,
    ExistentialPredicate, UniversalPredicate,
)
from neurolang.probabilistic.expressions import Condition

# Binary operators that should be printed infix.
# Maps operator function → display symbol.
_INFIX_OPS = {
    _operator.eq: "=",
    _operator.ne: "\u2260",
    _operator.lt: "<",
    _operator.le: "\u2264",
    _operator.gt: ">",
    _operator.ge: "\u2265",
    _operator.add: "+",
    _operator.sub: "\u2212",
    _operator.mul: "\u00d7",
    _operator.truediv: "\u00f7",
}


class DatalogPrettyPrinter(PatternWalker):
    """Pretty-print NeuroLang IR expressions as clean, human-readable text.

    Uses the PatternWalker pattern-matching framework to recursively format
    each expression type. Strips type annotations, uses ``:-`` notation for
    rules/queries, and renames fresh variables to short names (``s₀``, ``s₁``,
    …). Handles probabilistic ``Condition`` objects (``[A | B]``),
    ``ExpressionBlock``, ``Union``, and all standard FOL/Datalog constructs.

    Parameters
    ----------
    fresh_map : dict, optional
        Pre-populated mapping from original fresh names to short aliases.
    counter : list, optional
        Mutable one-element list ``[int]`` for generating fresh-variable
        aliases.

    Examples
    --------
    >>> printer = DatalogPrettyPrinter()
    >>> printer.walk(some_implication)
    'ans(X) :-\n    Pred(X)'
    """

    def __init__(self, fresh_map=None, counter=None):
        super().__init__()
        self._fresh_map = fresh_map if fresh_map is not None else {}
        self._counter = counter if counter is not None else [0]

    def _sym_name(self, sym: Symbol) -> str:
        if sym.is_fresh:
            if sym.name not in self._fresh_map:
                n = self._counter[0]
                if n < 10:
                    self._fresh_map[sym.name] = f"s{chr(0x2080 + n)}"
                else:
                    self._fresh_map[sym.name] = f"s{n}"
                self._counter[0] += 1
            return self._fresh_map[sym.name]
        return sym.name

    def _indent_body(self, body: str) -> str:
        return "\n".join("    " + line for line in body.split("\n"))

    @add_match(Symbol)
    def format_symbol(self, expr: Symbol) -> str:
        return self._sym_name(expr)

    @add_match(Constant)
    def format_constant(self, expr: Constant) -> str:
        v = expr.value
        if callable(v) and hasattr(v, "__qualname__"):
            return f"\u27e8{v.__qualname__}\u27e9"
        return repr(v)

    @add_match(FunctionApplication)
    def format_fa(self, expr: FunctionApplication) -> str:
        if (
            isinstance(expr.functor, Constant)
            and callable(expr.functor.value)
            and expr.functor.value in _INFIX_OPS
            and len(expr.args) == 2
        ):
            left = self.walk(expr.args[0])
            right = self.walk(expr.args[1])
            return f"{left} {_INFIX_OPS[expr.functor.value]} {right}"
        args = ", ".join(self.walk(a) for a in expr.args)
        return f"{self.walk(expr.functor)}({args})"

    @add_match(Lambda)
    def format_lambda(self, expr: Lambda) -> str:
        body_s = self.walk(expr.function_expression)
        args_s = ", ".join(
            self._sym_name(a) if isinstance(a, Symbol) else self.walk(a)
            for a in expr.args
        )
        return f"\u03bb({args_s}). {body_s}"

    @add_match(Conjunction)
    def format_conjunction(self, expr: Conjunction) -> str:
        return " \u2227 ".join(self.walk(f) for f in expr.formulas)

    @add_match(Disjunction)
    def format_disjunction(self, expr: Disjunction) -> str:
        return " \u2228 ".join(self.walk(f) for f in expr.formulas)

    @add_match(Negation)
    def format_negation(self, expr: Negation) -> str:
        return f"\u00ac({self.walk(expr.formula)})"

    @add_match(ExistentialPredicate)
    def format_exists(self, expr: ExistentialPredicate) -> str:
        return (
            f"\u2203{self._sym_name(expr.head)}."
            f" ({self.walk(expr.body)})"
        )

    @add_match(UniversalPredicate)
    def format_forall(self, expr: UniversalPredicate) -> str:
        return (
            f"\u2200{self._sym_name(expr.head)}."
            f" ({self.walk(expr.body)})"
        )

    @add_match(Condition)
    def format_condition(self, expr: Condition) -> str:
        cond_s = self.walk(expr.conditioned)
        conding_s = self.walk(expr.conditioning)
        return f"[{cond_s} | {conding_s}]"

    @add_match(ExpressionBlock)
    def format_block(self, expr: ExpressionBlock) -> str:
        return "\n\n".join(self.walk(e) for e in expr.expressions)

    @add_match(Implication)
    def format_implication(self, imp: Implication) -> str:
        cons_s = self.walk(imp.consequent)
        ante_s = self._format_body(imp.antecedent)
        return f"{cons_s} :-\n{self._indent_body(ante_s)}"

    @add_match(Query)
    def format_query(self, expr: Query) -> str:
        head_walked = self.walk(expr.head)
        if isinstance(head_walked, tuple):
            head_s = "({})".format(", ".join(head_walked))
        else:
            head_s = head_walked
        body_s = self._format_body(expr.body)
        return f"{head_s} :-\n{self._indent_body(body_s)}"

    @add_match(Expression)
    def format_default(self, expr: Expression) -> str:
        return repr(expr)

    # ── body formatting ─────────────────────────────────────

    def _format_body(self, e):
        """Format an Implication/Query body.

        Conjunctions are displayed as a comma-separated list (the standard
        Datalog convention).  Everything else is dispatched through the
        regular walker.
        """
        if isinstance(e, Conjunction):
            return ",\n".join(self.walk(f) for f in e.formulas)
        return self.walk(e)
