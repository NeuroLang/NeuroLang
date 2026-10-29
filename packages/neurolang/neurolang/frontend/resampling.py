"""
Confidence intervals by resampling the outcomes of a probabilistic choice.

A resampling of a choice ``C`` (for instance ``selected_study``, the uniform
choice of a study) is a relation ``boot(b, c, w)``: ``n`` replicates ``b`` of
the choice, each one reweighting the outcomes ``c`` of the choice as a
bootstrap of them does.  ``w`` is the ratio of the probability of ``c`` in the
replicate to its probability in ``C``, so that for any query ``q``

    P_b(q) = Σ_c P_b(C = c) P(q | C = c) = Σ_c w_bc P(q ∧ C = c).

``P(q ∧ C = c)`` is ``q`` with the outcome of the choice as one more answer
variable, which the probabilistic solvers compute as any other query, and the
sum over ``c`` is an aggregation.  Hence ``RESAMPLE[boot](pred(x)) = b``
needs no other solver: it rewrites the rules ``pred`` depends on into

* the rules below a ``PROB``, which depend on ``C``, with the outcome ``c``
  of ``C`` as their first argument (``q@boot:c``);
* each ``PROB`` over them as ``P(q ∧ C = c)``, followed by its sum weighted
  by the replicate's ``w`` (``q@boot``);
* the rules over those probabilities (arithmetic, ratios, aggregations) with
  the replicate ``b`` as their first argument, shared by all the atoms of a
  body, so that the measures of a replicate are computed on the same
  reweighting.

A conditional ``PROB[q // e]`` is already the ratio of two ``PROB`` rules,
both resampled.  The point estimates are those of the original rules, which
are left as they are.
"""
from operator import eq, mul

import numpy as np
import pandas as pd

from nl_core.datalog.expressions import AggregationApplication
from nl_core.probabilistic.expressions import PROB, ProbabilisticQuery
from nl_expr import expressions as ir
from nl_expr.exceptions import ForbiddenExpressionError
from nl_expr.expression_walker import ReplaceExpressionWalker
from nl_expr.logic import Conjunction, Implication, Negation

RESAMPLE_MARKER = ir.Symbol("__RESAMPLE__")

RESAMPLING_SCHEMES = ("bayesian", "multinomial", "poisson")


def resampling_ratios(probabilities, n, scheme="bayesian", seed=0):
    """
    The ``n × N`` ratios ``P_b(C = c) / P(C = c)`` of ``n`` replicates of a
    choice with the probabilities of its ``N`` outcomes.

    The outcomes are the resampled units, each with a count ``m_bc`` in the
    replicate: an Exp(1) weight for the Bayesian bootstrap (Rubin, 1981),
    a Multinomial(N, 1/N) count for the classical one, a Poisson(1) count
    for the Poisson bootstrap.  The replicate's probability of ``c`` is
    then ``P(C = c) m_bc``, normalised.
    """
    if scheme not in RESAMPLING_SCHEMES:
        raise ValueError(
            f"Unknown resampling scheme {scheme!r}, "
            f"use one of {', '.join(RESAMPLING_SCHEMES)}"
        )
    probabilities = np.asarray(probabilities, dtype=float)
    size = len(probabilities)
    rng = np.random.default_rng(seed)
    if scheme == "bayesian":
        counts = rng.exponential(size=(n, size))
    elif scheme == "multinomial":
        counts = rng.multinomial(size, np.full(size, 1 / size), size=n)
    else:
        counts = rng.poisson(size=(n, size))
    return counts / (counts @ probabilities)[:, None]


def resampling_relation(choice_rows, n, scheme="bayesian", seed=0):
    """
    The rows ``(b, c..., w)`` of a resampling of the choice whose rows are
    ``(p, c...)``, ordered by outcome so that a seed always gives the same
    replicates.
    """
    choice = pd.DataFrame(choice_rows).sort_values(
        by=list(range(1, len(choice_rows.columns)))
    )
    ratios = resampling_ratios(choice[0].to_numpy(), n, scheme, seed)
    outcomes = choice.iloc[:, 1:].to_numpy()
    size = len(choice)
    columns = {"b": np.repeat(np.arange(n), size)}
    for i in range(outcomes.shape[1]):
        columns[f"c{i}"] = np.tile(outcomes[:, i], n)
    columns["w"] = ratios.ravel()
    return pd.DataFrame(columns)


def is_resample_marker(formula):
    return (
        isinstance(formula, ir.FunctionApplication)
        and formula.functor == RESAMPLE_MARKER
    )


def uses_resampling(formula):
    """Whether the body of the rule or query *formula* has a RESAMPLE."""
    body = (
        formula.body if isinstance(formula, ir.Query)
        else getattr(formula, "antecedent", None)
    )
    return body is not None and any(
        map(is_resample_marker, _body_formulas(body))
    )


def _body_formulas(body):
    if isinstance(body, Conjunction):
        return body.formulas
    return (body,)


def _atom_of(formula):
    """The atom of a literal, or None for a builtin or another formula."""
    if isinstance(formula, Negation):
        formula = formula.formula
    if isinstance(formula, ir.FunctionApplication) and isinstance(
        formula.functor, ir.Symbol
    ):
        return formula
    return None


def _is_prob(arg):
    return isinstance(arg, ProbabilisticQuery) or (
        isinstance(arg, ir.FunctionApplication) and arg.functor == PROB
    )


def _name(functor):
    adornment = getattr(functor, "adornment", None)
    return functor.name + (f"^{adornment}" if adornment else "")


class ResampledProgram:
    """
    The rewriting, for a resampling ``boot`` of a choice, of the rules of a
    program into rules computing their values on every replicate.

    Predicates are of one of four kinds:

    * ``det``, which don't depend on the choice through a ``PROB`` and are
      left as they are;
    * ``choice``, the choice and the rules below a ``PROB`` that depend on it,
      rewritten with the outcome of the choice as first argument;
    * ``prob``, a ``PROB`` over ``choice`` rules, rewritten as its sum over
      the outcomes weighted by each replicate;
    * ``replicate``, the rules over ``prob`` predicates, rewritten with the
      replicate as first argument.
    """

    def __init__(self, idb, name, choice, weights, arity, versions):
        self.idb = idb
        self.name = name
        self.choice = choice
        self.weights = weights
        self.arity = arity
        # The rewritings of earlier RESAMPLEs: for each predicate, its
        # rules and the rewritten predicates they used, and its rewriting.
        self.versions = versions
        self.kinds = {}
        self.symbols = {}
        self.rules = []

    def predicate(self, functor):
        """The replicate version of *functor*, whose first argument is the
        replicate, adding the rules that define it."""
        if self.kind(functor) not in ("prob", "replicate"):
            raise ForbiddenExpressionError(
                f"{functor.name} does not depend on a probability over "
                f"{self.choice.name}: there is nothing to resample"
            )
        return self._rewrite(functor)

    def kind(self, functor):
        if functor == self.choice:
            return "choice"
        if functor == self.weights:
            raise ForbiddenExpressionError(
                f"RESAMPLE over a predicate that uses {self.weights.name} "
                "itself: nested resamplings are not supported"
            )
        if functor not in self.idb:
            return "det"
        if functor not in self.kinds:
            # Recursive rules are taken as deterministic while their kind is
            # computed: a recursion through the choice is not supported.
            self.kinds[functor] = "det"
            self.kinds[functor] = self._rules_kind(functor)
        return self.kinds[functor]

    def _rules_kind(self, functor):
        kinds = {self._rule_kind(rule) for rule in self.idb[functor].formulas}
        kinds.discard("det")
        if len(kinds) > 1:
            raise ForbiddenExpressionError(
                f"The rules of {functor.name} mix probabilities over "
                f"{self.choice.name} with the choice itself"
            )
        return kinds.pop() if kinds else "det"

    def _rule_kind(self, rule):
        body = {
            self.kind(atom.functor)
            for atom in map(_atom_of, _body_formulas(rule.antecedent))
            if atom is not None
        }
        if any(_is_prob(arg) for arg in rule.consequent.args):
            return "prob" if "choice" in body else "det"
        if "choice" in body and body & {"prob", "replicate"}:
            raise ForbiddenExpressionError(
                f"{rule.consequent.functor.name} uses the probabilities over "
                f"{self.choice.name} inside a probabilistic rule"
            )
        if "choice" in body:
            return "choice"
        if body & {"prob", "replicate"}:
            return "replicate"
        return "det"

    def _rewrite(self, functor):
        if functor in self.symbols:
            return self.symbols[functor]
        rules = self.idb[functor]
        uses = tuple(
            self._rewrite(atom.functor)
            for rule in rules.formulas
            for atom in map(_atom_of, _body_formulas(rule.antecedent))
            if atom is not None and atom.functor != functor
            and self.kind(atom.functor) != "det"
            and atom.functor != self.choice
        )
        earlier = self.versions.get(functor)
        if (
            earlier is not None and earlier[0] is rules
            and earlier[1] == uses and earlier[2] in self.idb
        ):
            self.symbols[functor] = earlier[2]
            return earlier[2]
        symbol = self._new_symbol(functor)
        self.symbols[functor] = symbol
        self.versions[functor] = (rules, uses, symbol)
        kind = self.kind(functor)
        for rule in rules.formulas:
            if kind == "choice":
                self.rules.append(self._choice_rule(rule, symbol))
            elif kind == "prob":
                self.rules.extend(self._prob_rules(rule, symbol))
            else:
                self.rules.append(self._replicate_rule(rule, symbol))
        return symbol

    def _new_symbol(self, functor):
        """``f@boot`` (``f@boot:c`` below a PROB), numbered if the rules it
        rewrites changed since an earlier RESAMPLE."""
        suffix = ":c" if self.kind(functor) == "choice" else ""
        base = f"{_name(functor)}@{self.name.name}"
        symbol = ir.Symbol(base + suffix)
        version = 1
        while symbol in self.idb:
            version += 1
            symbol = ir.Symbol(f"{base}#{version}{suffix}")
        return symbol

    def _choice_body(self, body):
        """
        The literals of *body* with the outcome of the choice, the arguments
        of its first atom, as first argument of the ``choice`` atoms.
        """
        formulas = list(_body_formulas(body))
        choices = [
            f for f in formulas
            if isinstance(f, ir.FunctionApplication)
            and f.functor == self.choice
        ]
        if choices:
            outcome = choices[0].args
        else:
            outcome = tuple(ir.Symbol.fresh() for _ in range(self.arity))
            formulas.append(self.choice(*outcome))
        # In every world the choice holds for one outcome: its atoms are
        # the same outcome.
        replacements = {}
        for atom in choices[1:]:
            for arg, out in zip(atom.args, outcome):
                if arg != out:
                    if not isinstance(arg, ir.Symbol):
                        raise ForbiddenExpressionError(
                            f"{self.choice.name} with two constants in a rule "
                            "is not supported by RESAMPLE"
                        )
                    replacements[arg] = out
        walker = ReplaceExpressionWalker(replacements)
        formulas = [walker.walk(f) for f in formulas]
        rewritten = []
        for formula in formulas:
            atom = _atom_of(formula)
            if atom is None or self.kind(atom.functor) != "choice" or (
                atom.functor == self.choice
            ):
                rewritten.append(formula)
                continue
            new_atom = self._rewrite(atom.functor)(*outcome, *atom.args)
            rewritten.append(
                Negation(new_atom) if isinstance(formula, Negation)
                else new_atom
            )
        return outcome, walker, Conjunction(tuple(rewritten))

    def _choice_rule(self, rule, symbol):
        outcome, walker, body = self._choice_body(rule.antecedent)
        head = walker.walk(rule.consequent)
        return Implication(symbol(*outcome, *head.args), body)

    def _prob_rules(self, rule, symbol):
        """
        ``f(x, PROB(x)) :- body`` as ``f:c(c, x, PROB(c, x)) :- body:c``,
        the probability ``P(body ∧ C = c)``, then its sum weighted by each
        replicate, ``f(b, x, sum(w p))``.
        """
        outcome, walker, body = self._choice_body(rule.antecedent)
        head = walker.walk(rule.consequent)
        per_outcome = ir.Symbol(f"{symbol.name}:c")
        weighted = ir.Symbol(f"{symbol.name}:w")
        args = tuple(
            ProbabilisticQuery(PROB, (*outcome, *arg.args))
            if _is_prob(arg) else arg
            for arg in head.args
        )
        rules = [Implication(per_outcome(*outcome, *args), body)]
        b, w, p, v = (ir.Symbol.fresh() for _ in range(4))
        c = tuple(ir.Symbol.fresh() for _ in outcome)
        values = tuple(p if _is_prob(arg) else arg for arg in head.args)
        others = tuple(arg for arg in head.args if not _is_prob(arg))
        rules.append(Implication(
            weighted(b, *c, *others, v),
            Conjunction((
                self.weights(b, *c, w),
                per_outcome(*c, *values),
                ir.Constant(eq)(v, ir.Constant(mul)(w, p)),
            )),
        ))
        sums = tuple(
            AggregationApplication(ir.Symbol("sum"), (v,))
            if _is_prob(arg) else arg
            for arg in head.args
        )
        rules.append(Implication(
            symbol(b, *sums), weighted(b, *c, *others, v)
        ))
        return rules

    def _replicate_rule(self, rule, symbol):
        b = ir.Symbol.fresh()
        formulas = []
        bound = False
        for formula in _body_formulas(rule.antecedent):
            atom = _atom_of(formula)
            if atom is None or self.kind(atom.functor) not in (
                "prob", "replicate"
            ):
                formulas.append(formula)
                continue
            new_atom = self._rewrite(atom.functor)(b, *atom.args)
            if isinstance(formula, Negation):
                formulas.append(Negation(new_atom))
            else:
                bound = True
                formulas.append(new_atom)
        if not bound:
            others = (ir.Symbol.fresh() for _ in range(self.arity + 1))
            formulas.append(self.weights(b, *others))
        return Implication(
            symbol(b, *rule.consequent.args), Conjunction(tuple(formulas))
        )


def rewrite_resample_markers(formula, resampled_predicate):
    """
    *formula* with each ``RESAMPLE[boot](pred(x)) = b`` of its body replaced
    by ``pred@boot(b, x)``, where ``resampled_predicate(boot, pred)`` gives
    ``pred@boot``.
    """
    if not uses_resampling(formula):
        return formula
    if isinstance(formula, ir.Query):
        head, body = formula.head, formula.body
    else:
        head, body = formula.consequent, formula.antecedent
    formulas = []
    for f in _body_formulas(body):
        if is_resample_marker(f):
            name, atom, replicate = f.args
            f = resampled_predicate(name, atom.functor)(replicate, *atom.args)
        formulas.append(f)
    return formula.apply(head, Conjunction(tuple(formulas)))
