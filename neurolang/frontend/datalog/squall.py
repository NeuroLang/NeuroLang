"""
Logic simplifier for SQUALL parsed expressions.

Collapses nested conjunctions, removes trivial operations,
and normalizes quantifier expressions produced by the SQUALL parser.
"""
from ...datalog.aggregation import AggregationApplication
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
from ...probabilistic.expressions import ProbabilisticFact



class InvertedFunctionApplication(FunctionApplication):
    """
    Intermediate IR node emitted by the SQUALL transformer for transitive
    verbs prefixed with '~'.  Carries the same functor and args as a
    FunctionApplication built with the *surface* argument order
    (subject first, object(s) after), but signals that the order must be
    reversed before the rule enters the engine.

    Example
    -------
    SQUALL: ``every study ~reports a voxel``
    Transformer emits: ``InvertedFunctionApplication(reports, (study, voxel))``
    Mixin resolves to:  ``reports(voxel, study)``

    Resolved by ResolveInvertedFunctionApplicationMixin.
    """


class ResolveInvertedFunctionApplicationMixin(PatternWalker):
    """
    Rewrites ``InvertedFunctionApplication(f, (a, b, …))`` to
    ``f(…, b, a)`` (fully reversed argument tuple) at walk time.

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
    """
    Simplifies logical expressions produced by the SQUALL transformer.

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
        if isinstance(inner, ExistentialPredicate):
            if isinstance(inner.body, Conjunction):
                return ExistentialPredicate(
                    expression.head,
                    ExistentialPredicate(
                        inner.head,
                        self.walk(inner.body),
                    ),
                )
        return ExistentialPredicate(expression.head, inner)

    @add_match(
        UniversalPredicate,
        lambda e: isinstance(e.body, UniversalPredicate),
    )
    def flatten_nested_universals(self, expression):
        inner = self.walk(expression.body)
        return UniversalPredicate(expression.head, inner)

    @add_match(Implication)
    def walk_implication(self, expression):
        consequent = self.walk(expression.consequent)
        antecedent = self.walk(expression.antecedent)
        if (
            consequent != expression.consequent
            or antecedent != expression.antecedent
        ):
            return Implication(consequent, antecedent)
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
