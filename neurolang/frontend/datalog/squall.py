"""
Logic simplifier for SQUALL parsed expressions.

Collapses nested conjunctions, removes trivial operations,
and normalizes quantifier expressions produced by the SQUALL parser.
"""
from collections import defaultdict
from functools import lru_cache

from ...datalog.expressions import AggregationApplication as _AggApp
from ...expression_walker import ExpressionWalker, PatternWalker, add_match
from ...expressions import Constant, Expression, FunctionApplication, Query, Symbol
from ...logic import (
    Conjunction,
    ExistentialPredicate,
    Implication,
    Negation,
    UniversalPredicate,
)
from ...logic.transformations import (
    CollapseConjunctionsMixin,
    RemoveTrivialOperationsMixin,
)
from ...exceptions import NeuroLangException, PredicateArityError




class InvertedFunctionApplication(FunctionApplication):

    """Intermediate IR node for transitive verbs prefixed with ``~``.

    Carries the same functor and args as a FunctionApplication built with
    the *surface* argument order (subject first, object(s) after), but
    signals that the order must be reversed before the rule enters the engine.

    Examples
    --------
    SQUALL: ``every study ~reports a voxel``
    Transformer emits: ``InvertedFunctionApplication(reports, (study, voxel))``
    Mixin resolves to:  ``reports(voxel, study)``

    Resolved by ResolveInvertedFunctionApplicationMixin.

    """


class ResolveInvertedFunctionApplicationMixin(PatternWalker):

    """Rewrites ``InvertedFunctionApplication(f, (a, b, …))`` to ``f(…, b, a)`` at walk time.

    Must appear before ``LogicSimplifier`` and ``ExpressionBasicEvaluator``
    in any walker/solver MRO that processes SQUALL output containing ``~``
    verbs, to prevent ``LogicSimplifier.walk_function_application`` from
    silently demoting the node to a plain ``FunctionApplication``.
    """

    @add_match(InvertedFunctionApplication)
    def resolve_inverted(self, expr):
        return expr.functor(*reversed(expr.args))


class LogicSimplifier(
    CollapseConjunctionsMixin,
    RemoveTrivialOperationsMixin,
    ExpressionWalker,
):
    """Simplifies logical expressions produced by the SQUALL transformer.

    Combines CollapseConjunctionsMixin (flattens nested Conjunctions)
    and RemoveTrivialOperationsMixin (removes True/False tautologies)
    into a single walker.
    """

    @staticmethod
    def _flatten_nested_quantifier(expression, cls, walk_body):
        """Shared logic for flattening nested quantifier predicates.

        Walks the body; if unchanged, returns the original expression to
        prevent infinite re-wrapping of structurally stable chains.
        """
        inner = walk_body
        if inner is expression.body:
            return expression
        return cls(expression.head, inner)

    @add_match(
        ExistentialPredicate,
        lambda e: isinstance(e.body, ExistentialPredicate),
    )
    def flatten_nested_existentials(self, expression):
        return self._flatten_nested_quantifier(
            expression, ExistentialPredicate, self.walk(expression.body)
        )

    @add_match(
        UniversalPredicate,
        lambda e: isinstance(e.body, UniversalPredicate),
    )
    def flatten_nested_universals(self, expression):
        return self._flatten_nested_quantifier(
            expression, UniversalPredicate, self.walk(expression.body)
        )

    @add_match(Implication)
    def walk_implication(self, expression):
        consequent = self.walk(expression.consequent)
        antecedent = self.walk(expression.antecedent)
        if (
            consequent is not expression.consequent
            or antecedent is not expression.antecedent
        ):
            expression = Implication(consequent, antecedent)
            return self.walk(expression)
        return expression

    @add_match(Negation)
    def walk_negation(self, expression):
        formula = self.walk(expression.formula)
        if formula != expression.formula:
            return Negation(formula)
        return expression

    @add_match(ExistentialPredicate)
    def walk_existential(self, expression):
        body = self.walk(expression.body)
        if body != expression.body:
            return ExistentialPredicate(expression.head, body)
        return expression

    @add_match(UniversalPredicate)
    def walk_universal(self, expression):
        body = self.walk(expression.body)
        if body != expression.body:
            return UniversalPredicate(expression.head, body)
        return expression

    @add_match(_AggApp)
    def walk_aggregation_application(self, expression):
        new_args = tuple(self.walk(a) for a in expression.args)
        if new_args != expression.args:
            return _AggApp(expression.functor, new_args)
        return expression

    @add_match(
        Conjunction,
        lambda e: any(f == Constant(True) for f in e.formulas),
    )
    def remove_true_from_conjunction(self, expression):
        """Remove Constant(True) conjuncts, e.g. from 'has a noun' clauses."""
        remaining = tuple(
            f for f in expression.formulas if f != Constant(True)
        )
        if not remaining:
            return Constant(True)
        if len(remaining) == 1:
            return self.walk(remaining[0])
        return self.walk(Conjunction(remaining))

    @add_match(Query)
    def walk_query(self, expression):
        body = self.walk(expression.body)
        if body != expression.body:
            return Query(expression.head, body)
        return expression

    @add_match(InvertedFunctionApplication)
    def walk_inverted_function_application(self, expression):
        new_args = tuple(self.walk(a) for a in expression.args)
        if new_args != expression.args:
            return InvertedFunctionApplication(expression.functor, new_args)
        return expression

    @add_match(FunctionApplication)
    def walk_function_application(self, expression):
        new_args = tuple(self.walk(a) for a in expression.args)
        if new_args != expression.args:
            return expression.functor(*new_args)
        return expression


@lru_cache(maxsize=4096)
def predicate_lemma(name):
    """
    Return the lemma used to match the predicate *name* to a verb form.

    Only the last underscore-separated word is lemmatised, as it carries
    the inflection (``peak_reported`` → ``peak_report``). The verb lemma
    also folds plural nouns (``studies`` → ``study``).
    """
    from lemminflect import getLemma

    prefix, separator, word = name.rpartition("_")
    lemmas = getLemma(word, upos="VERB", lemmatize_oov=True)
    return prefix + separator + (lemmas[0] if lemmas else word)


def atom_arity(args):
    """
    Return the number of columns the arguments *args* of an atom fill.

    A SQUALL tuple argument such as ``(?i; ?j; ?k)`` fills one column per
    element.
    """
    return sum(len(arg) if isinstance(arg, tuple) else 1 for arg in args)


def _unknown_arity(name):
    """Default arity lookup: every arity is unknown."""
    return None


class ResolvePredicateLemmas(ExpressionWalker):
    """
    Rename undefined predicates to the defined one sharing their lemma.

    SQUALL uses the typed word as the predicate name, so ``does not
    mention`` refers to ``mention`` even if only ``mentions`` is defined.
    A rule-body or query predicate whose name is not in *defined_names* is
    renamed to the defined predicate with the same `predicate_lemma` and
    the atom's `atom_arity`, only if there is exactly one. *arity* maps a
    defined name to its arity, or to None when unknown, which accepts any
    atom. Several candidates leave the name unchanged and are recorded in
    ``ambiguous``. Rule heads, variables and constants are never renamed,
    nor are SQUALL's dimension-type predicates (``probability``,
    ``value``).
    """

    def __init__(self, defined_names, arity=_unknown_arity):
        self.defined_names = frozenset(defined_names)
        self.arity = arity
        self.ambiguous = {}
        self._names_by_lemma = None

    @add_match(Implication)
    def resolve_rule_body(self, rule):
        antecedent = self.walk(rule.antecedent)
        if antecedent is rule.antecedent:
            return rule
        return rule.apply(rule.consequent, antecedent)

    @add_match(Query)
    def resolve_query_body(self, query):
        body = self.walk(query.body)
        if body is query.body:
            return query
        return query.apply(query.head, body)

    @add_match(FunctionApplication(Symbol, ...))
    def resolve_predicate(self, atom):
        functor = self._resolve(atom.functor, atom_arity(atom.args))
        if functor is atom.functor:
            return atom
        return atom.apply(functor, atom.args)

    def _resolve(self, functor, arity):
        if (
            functor.is_fresh
            or functor.name in self.defined_names
            or functor.name in _DIMENSION_TYPE_PREDICATE_NAMES
        ):
            return functor
        candidates = self._candidates(predicate_lemma(functor.name), arity)
        if len(candidates) == 1:
            return Symbol(candidates[0])
        if candidates:
            self.ambiguous[functor.name] = candidates
        return functor

    def _candidates(self, lemma, arity):
        if self._names_by_lemma is None:
            self._names_by_lemma = defaultdict(list)
            for name in sorted(self.defined_names):
                self._names_by_lemma[predicate_lemma(name)].append(name)
        return tuple(
            name for name in self._names_by_lemma.get(lemma, ())
            if self.arity(name) in (None, arity)
        )


_DIMENSION_TYPE_PREDICATE_NAMES = frozenset({
    "probability",
    "value",
})


class StripDimensionTypePredicatesMixin(PatternWalker):
    """Replaces dimension-type atoms (probability/1, value/1) with Constant(True).

    Probability and Value are type annotations in SQUALL — they introduce a
    variable into scope without representing a database predicate. This mixin
    replaces their atoms with Constant(True) so they are simplified away by
    RemoveTrivialOperationsMixin or ``LogicSimplifier.remove_true_from_conjunction``.

    Override ``dimension_type_predicate_names`` on a subclass for custom names::

        class MyMixin(StripDimensionTypePredicatesMixin):
            dimension_type_predicate_names = {"probability", "value", "intensity"}

    Use ``make_dimension_type_stripper()`` to create one from engine YAML config.
    """

    dimension_type_predicate_names = _DIMENSION_TYPE_PREDICATE_NAMES

    @add_match(FunctionApplication(Symbol, (Expression,)))
    def replace_dimension_type_atom(self, fa):
        if fa.functor.name not in self.dimension_type_predicate_names:
            return fa
        return Constant(True)


def make_dimension_type_stripper(predicate_names=None):
    """Create a StripDimensionTypePredicatesMixin subclass with custom names.

    Parameters
    ----------
    predicate_names : iterable of str, optional
        Predicate names to treat as dimension-type annotations.
        Defaults to ``{"probability", "value"}``.

    Returns
    -------
    type
        A ``StripDimensionTypePredicatesMixin`` subclass that replaces
        only *predicate_names* atoms with ``Constant(True)``.

    Examples
    --------
    Used from engine YAML config::

        from neurolang.frontend.datalog.squall import (
            make_dimension_type_stripper,
        )

        Mixin = make_dimension_type_stripper(
            ["probability", "value", "intensity"]
        )
        # Then insert Mixin into the frontend solver MRO
    """
    if predicate_names is not None:
        predicate_names = frozenset(predicate_names)
    else:
        predicate_names = _DIMENSION_TYPE_PREDICATE_NAMES

    class _Stripper(StripDimensionTypePredicatesMixin):
        dimension_type_predicate_names = predicate_names

    _Stripper.__qualname__ = "StripDimensionTypePredicatesMixin"
    return _Stripper


def predicate_arity(program, symbol):
    """
    Return the number of arguments *symbol* takes in *program*, or None.

    None stands for an unknown arity: a symbol that is not a relation of
    *program*, such as a function or a predicate not defined yet. The
    probability column of stored probabilistic facts and choices is not an
    argument of their atoms, so it is not counted.
    """
    try:
        arity = len(program.predicate_terms(symbol))
    except (NeuroLangException, AttributeError):
        return None
    probabilistic = (
        getattr(program, "pfact_pred_symbs", set())
        | getattr(program, "pchoice_pred_symbs", set())
    )
    if symbol in probabilistic and isinstance(
        program.symbol_table[symbol], Constant
    ):
        arity -= 1
    return arity or None


class CheckPredicateArities(ExpressionWalker):
    """
    Raise `PredicateArityError` for an atom whose arity is not its
    predicate's.

    The atoms of a query body or of a rule body are checked against the
    relations of *program*; rule heads and predicates of unknown arity
    (see `predicate_arity`) are not checked.
    """

    def __init__(self, program):
        self.program = program

    @add_match(Implication)
    def check_rule_body(self, rule):
        self.walk(rule.antecedent)
        return rule

    @add_match(FunctionApplication(Symbol, ...))
    def check_atom(self, atom):
        arity = predicate_arity(self.program, atom.functor)
        if arity is not None and arity != len(atom.args):
            raise PredicateArityError(
                atom.functor.name, arity, len(atom.args)
            )
        return atom
