"""
Logic simplifier for SQUALL parsed expressions.

Collapses nested conjunctions, removes trivial operations,
and normalizes quantifier expressions produced by the SQUALL parser.
"""
from ...datalog.expressions import AggregationApplication as _AggApp
from ...expression_walker import ExpressionWalker, PatternWalker, add_match
from ...expressions import Constant, FunctionApplication, Query, Symbol
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

    @add_match(
        ExistentialPredicate,
        lambda e: isinstance(e.body, ExistentialPredicate),
    )
    def flatten_nested_existentials(self, expression):
        inner = self.walk(expression.body)
        # If walking the body produced no change, return the original expression.
        # This prevents infinite re-walking when nested ExistentialPredicate
        # chains (e.g. EP(x, EP(y, EP(z, Conj(...))))) are structurally stable
        # but the handler below always creates new wrapper objects.
        if inner is expression.body:
            return expression
        expression = ExistentialPredicate(expression.head, inner)
        return expression

    @add_match(
        UniversalPredicate,
        lambda e: isinstance(e.body, UniversalPredicate),
    )
    def flatten_nested_universals(self, expression):
        inner = self.walk(expression.body)
        if inner is expression.body:
            return expression
        expression = UniversalPredicate(expression.head, inner)
        return expression

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


_DIMENSION_TYPE_PREDICATE_NAMES = frozenset({
    "probability",
    "value",
})


class StripDimensionTypePredicatesMixin(PatternWalker):
    """Strips dimension-type atoms (probability/1, value/1) from rule bodies.

    Probability and Value are type annotations in SQUALL — they introduce a
    variable into scope without representing a database predicate. The SQUALL
    parser generates ``probability(v)`` / ``value(v)`` atoms from quantifiers
    like ``for every Probability``. This mixin removes those atoms from rule
    bodies so they never reach the Datalog engine.

    Must be placed in the frontend solver MRO after any expression-simplifying
    walkers but before the Datalog program solver (e.g.,
    ``TranslateToLogicWithAggregation``).
    """

    @add_match(
        Conjunction,
        lambda conjunction: any(
            isinstance(f, FunctionApplication)
            and isinstance(f.functor, Symbol)
            and f.functor.name in _DIMENSION_TYPE_PREDICATE_NAMES
            and len(f.args) == 1
            for f in conjunction.formulas
        ),
    )
    def strip_from_conjunction(self, conjunction):
        filtered = tuple(
            f for f in conjunction.formulas
            if not self._is_dimension_type_atom(f)
        )
        if len(filtered) == 0:
            return Constant(True)
        if len(filtered) == 1:
            return self.walk(filtered[0])
        if len(filtered) == len(conjunction.formulas):
            return conjunction
        return Conjunction(filtered)

    @staticmethod
    def _is_dimension_type_atom(expr):
        return (
            isinstance(expr, FunctionApplication)
            and isinstance(expr.functor, Symbol)
            and expr.functor.name in _DIMENSION_TYPE_PREDICATE_NAMES
            and len(expr.args) == 1
        )
