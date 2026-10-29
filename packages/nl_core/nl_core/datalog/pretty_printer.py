"""
Pretty-printer for NeuroLang IR expressions as human-readable Datalog text.

Strips type annotations, uses ``:-`` notation for rules/queries, and
shortens fresh variables to short names (``s₀``, ``s₁``, …).
"""

import operator as _operator
from collections import Counter

from nl_expr.expression_pattern_matching import add_match
from nl_expr.expression_walker import PatternWalker, ReplaceExpressionWalker
from nl_expr.expressions import (
    Constant,
    Expression,
    ExpressionBlock,
    FunctionApplication,
    Lambda,
    Query,
    Symbol,
)
from .expressions import Implication
from .magic_sets import AdornedSymbol
from nl_expr.logic import (
    Conjunction,
    Disjunction,
    Negation,
    ExistentialPredicate,
    UniversalPredicate,
)
from nl_expr.logic.expression_processing import extract_logic_atoms
from ..probabilistic.expressions import (
    PROB,
    Condition,
    ProbabilisticChoice,
    ProbabilisticFact,
    ProbabilisticQuery,
)

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
        if isinstance(sym, AdornedSymbol):
            return str(sym)
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
        if isinstance(v, (tuple, list)):
            return self._format_arg(v)
        return repr(v)

    @add_match(ProbabilisticQuery(PROB, ...), lambda e: _of_formula(e))
    def format_probabilistic_query(self, expr: ProbabilisticQuery) -> str:
        """Format the probability of a formula as ``PROB[A | B]``."""
        body = expr.args[0]
        if isinstance(body, Condition):
            return f"PROB{self.walk(body)}"
        return f"PROB[{self.walk(body)}]"

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
        args = ", ".join(self._format_arg(a) for a in expr.args)
        return f"{self.walk(expr.functor)}({args})"

    def _format_arg(self, arg):
        if isinstance(arg, tuple):
            items = ", ".join(self._format_arg(a) for a in arg)
            return f"({items})" if len(arg) != 1 else f"({items},)"
        if isinstance(arg, list):
            return "[" + ", ".join(self._format_arg(a) for a in arg) + "]"
        return self.walk(arg)

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

    @add_match(Negation(FunctionApplication(Symbol, ...)))
    def format_negated_atom(self, expr: Negation) -> str:
        """Format the negation of an atom without parentheses: ``¬S(y)``."""
        return f"\u00ac{self.walk(expr.formula)}"

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

    @add_match(ProbabilisticFact)
    def format_probabilistic_fact(self, expr: ProbabilisticFact) -> str:
        """Format a probabilistic fact as ``head :: probability``."""
        return f"{self.walk(expr.body)} :: {self._probability(expr)}"

    @add_match(ProbabilisticChoice)
    def format_probabilistic_choice(self, expr: ProbabilisticChoice) -> str:
        """Format a probabilistic choice as ``head ^ probability``."""
        return f"{self.walk(expr.body)} ^ {self._probability(expr)}"

    def _probability(self, expr) -> str:
        """Format the probability of *expr*, in parentheses if computed."""
        probability = self.walk(expr.probability)
        if isinstance(expr.probability, FunctionApplication):
            return f"({probability})"
        return probability

    @add_match(ExpressionBlock)
    def format_block(self, expr: ExpressionBlock) -> str:
        return "\n\n".join(self.walk(e) for e in expr.expressions)

    @add_match(Implication)
    def format_implication(self, imp: Implication) -> str:
        return self._rule(self.walk(imp.consequent), imp.antecedent)

    @add_match(Query)
    def format_query(self, expr: Query) -> str:
        head_walked = self.walk(expr.head)
        if isinstance(head_walked, tuple):
            head_s = "({})".format(", ".join(head_walked))
        else:
            head_s = head_walked
        return self._rule(head_s, expr.body)

    @add_match(Expression)
    def format_default(self, expr: Expression) -> str:
        return repr(expr)

    # ── body formatting ─────────────────────────────────────

    def _rule(self, head_s, body):
        """Format a rule or query of formatted head *head_s* and *body*."""
        body_s = self._format_body(body)
        return f"{head_s} :-\n{self._indent_body(body_s)}"

    def _format_body(self, e):
        """Format an Implication/Query body.

        Conjunctions are displayed as a comma-separated list (the standard
        Datalog convention).  Everything else is dispatched through the
        regular walker.
        """
        if isinstance(e, Conjunction):
            return ",\n".join(self.walk(f) for f in e.formulas)
        return self.walk(e)


def _is_prob(arg) -> bool:
    """Whether *arg* is the ``PROB(...)`` argument of a rule's head."""
    return isinstance(arg, FunctionApplication) and arg.functor == PROB


def _of_formula(probability) -> bool:
    """Whether *probability* is ``PROB[A]``, of a formula: not ``PROB(x)``."""
    args = probability.args
    return len(args) == 1 and not isinstance(args[0], Symbol)


def _body(formula):
    """Return the body of the rule or query *formula*."""
    return formula.body if isinstance(formula, Query) else formula.antecedent


def _is_helper(formula) -> bool:
    """Whether *formula* is a rule of a fresh predicate computing PROB."""
    return (
        isinstance(formula, Implication)
        and isinstance(formula.consequent, FunctionApplication)
        and getattr(formula.consequent.functor, "is_fresh", False)
        and any(_is_prob(arg) for arg in formula.consequent.args)
    )


def fold_probabilities(formulas) -> list:
    """
    Fold the helper rules of ``PROB[A // B] = p`` back into their use.

    The parser desugars ``PROB[A // B] = p`` (and ``MARG``) in a body into
    a rule ``f(x, PROB(x)) :- A // B`` of a fresh predicate ``f`` and the
    atom ``f(x, p)`` in the body. A helper used once is removed and its
    atom replaced by ``PROB[A // B] = p``, a :class:`ProbabilisticQuery`
    equal to ``p``, put first in the body.
    """
    helpers = {f.consequent.functor: f for f in formulas if _is_helper(f)}
    uses = Counter(
        atom.functor for f in formulas if not _is_helper(f)
        for atom in extract_logic_atoms(_body(f))
        if atom.functor in helpers
    )
    folds = {}
    for f in formulas:
        for atom in extract_logic_atoms(_body(f)):
            if uses[atom.functor] == 1 and not _is_helper(f):
                rule = helpers[atom.functor]
                n = next(
                    n for n, arg in enumerate(rule.consequent.args)
                    if _is_prob(arg)
                )
                folds[atom] = Constant(_operator.eq)(
                    ProbabilisticQuery(PROB, (rule.antecedent,)), atom.args[n]
                )
    replace = ReplaceExpressionWalker(folds)
    return [
        _fold(f, replace, set(folds.values())) for f in formulas
        if not (_is_helper(f) and uses[f.consequent.functor] == 1)
    ]


def _fold(formula, replace, probabilities):
    """Replace the atoms of *formula*'s body, the *probabilities* first."""
    body = _body(formula)
    if isinstance(body, Conjunction):
        body = Conjunction(tuple(sorted(
            (replace.walk(f) for f in body.formulas),
            key=lambda f: f not in probabilities,
        )))
    else:
        body = replace.walk(body)
    if isinstance(formula, Query):
        return Query(formula.head, body)
    return Implication(formula.consequent, body)


class LogicPrettyPrinter(DatalogPrettyPrinter):

    """
    Pretty-print NeuroLang IR in logic notation.

    Rules read ``head ← body``, their variables implicitly universally
    quantified, the conjuncts of the body aligned one per line under the
    first. A rule computing a probability, ``p(x, PROB) :- A // B``,
    reads ``p(x, P[A | B])``, and ``p(x, PROB) :- A`` reads ``p(x, P[A])``;
    a probability that does not fit on a line has ``| B`` on the next,
    under ``[``, and its conjuncts wrapped at :attr:`width` columns. A
    disjunction that does not fit, or of conjunctions, has one disjunct
    per line, ``∨`` under ``(``. :func:`fold_probabilities` shows the
    helper rules of ``PROB[A // B] = p`` as ``p = P[A | B]``.

    Examples
    --------
    >>> print(LogicPrettyPrinter().walk(rule))
    close(q, s) ← similar_study(q, s)
                ∧ selected_study(s)

    """

    width = 80
    # The column where the formula being formatted starts.
    _column = 0

    def _aligned(self, body, column: int) -> str:
        """Format *body*, one conjunct per line, ``∧`` at *column*."""
        formulas = body.formulas if isinstance(body, Conjunction) else (body,)
        lines = []
        for formula in formulas:
            self._column = column + 2
            lines.append(self.walk(formula))
        return ("\n" + " " * column + "\u2227 ").join(lines)

    def _wrapped(self, body, column: int) -> str:
        """Format *body* inline, wrapped at :attr:`width`, from *column*."""
        formulas = body.formulas if isinstance(body, Conjunction) else (body,)
        self._column = column
        lines = [self.walk(formulas[0])]
        for formula in formulas[1:]:
            self._column = column + len(lines[-1].split("\n")[-1]) + 3
            text = self.walk(formula)
            if self._column + len(text) > self.width or "\n" in text:
                self._column = column + 2
                lines.append(f"\u2227 {self.walk(formula)}")
            else:
                lines[-1] += f" \u2227 {text}"
        return ("\n" + " " * column).join(lines)

    def _p(self, body, column: int) -> str:
        """Format ``P[body]``, its ``[`` at *column*."""
        if not isinstance(body, Condition):
            return f"P[{self._wrapped(body, column + 1)}]"
        conditioned, conditioning = body.conditioned, body.conditioning
        inline = f"P[{self.walk(conditioned)} | {self.walk(conditioning)}]"
        if column + len(inline) <= self.width and "\n" not in inline:
            return inline
        return (
            f"P[{self._wrapped(body.conditioned, column + 1)}\n"
            f"{' ' * column}| {self._wrapped(body.conditioning, column + 2)}]"
        )

    def _rule(self, head_s, body):
        """Format ``head ← body``, the conjuncts aligned under the first."""
        return f"{head_s} \u2190 {self._aligned(body, len(head_s) + 1)}"

    @add_match(Disjunction)
    def format_disjunction(self, expr: Disjunction) -> str:
        """
        Format a disjunction in parentheses, not to read ``∧`` in it.

        A disjunction too long for a line, or of conjunctions, has one
        disjunct per line, ``∨`` under ``(``.
        """
        column = self._column
        disjuncts = [self.walk(f) for f in expr.formulas]
        inline = f"({' ∨ '.join(disjuncts)})"
        if (
            column + len(inline) <= self.width and "\n" not in inline
            and not any(isinstance(f, Conjunction) for f in expr.formulas)
        ):
            return inline
        disjuncts = [self._wrapped(f, column + 2) for f in expr.formulas]
        separator = "\n" + " " * column + "\u2228 "
        return f"( {separator.join(disjuncts)} )"

    @add_match(
        FunctionApplication(
            Constant(_operator.eq), (ProbabilisticQuery, Symbol)
        ),
        lambda e: _of_formula(e.args[0]),
    )
    def format_probability_equality(self, expr: FunctionApplication) -> str:
        """Format ``PROB[A // B] = p`` as ``p = P[A | B]``."""
        value = self.walk(expr.args[1])
        column = self._column + len(value) + 4
        return f"{value} = {self._p(expr.args[0].args[0], column)}"

    @add_match(
        Implication(FunctionApplication, ...),
        lambda imp: any(_is_prob(arg) for arg in imp.consequent.args),
    )
    def format_probability_rule(self, imp: Implication) -> str:
        """Format ``p(x, PROB) :- A // B`` as ``p(x, P[A | B])``."""
        head = imp.consequent
        args = [self._format_arg(arg) for arg in head.args]
        n = next(n for n, arg in enumerate(head.args) if _is_prob(arg))
        prefix = f"{self.walk(head.functor)}("
        prefix += "".join(f"{arg}, " for arg in args[:n])
        body_s = self._p(imp.antecedent, len(prefix) + 1)
        rest = "".join(f", {arg}" for arg in args[n + 1:])
        return f"{prefix}{body_s}{rest})"
