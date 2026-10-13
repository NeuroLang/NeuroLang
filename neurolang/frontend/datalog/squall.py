"""
Logic simplifier for SQUALL parsed expressions.

Collapses nested conjunctions, removes trivial operations,
and normalizes quantifier expressions produced by the SQUALL parser.
"""
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
