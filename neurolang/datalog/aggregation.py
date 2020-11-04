"""
Support for aggregations according to [1]_. For instance
the encoding of ``P(x, count<y>):-Q(x, y)`` in intermediate
representation ``Implication(P(x, AggregationApplication(count, y), Q(x, y))``
produces the set ``P(x, z)`` where ``z`` is the number of tuples ``(x, ...)``
in the set ``Q``.

.. [1] T. J. Green, S. S. Huang, B. T. Loo, W. Zhou,
   Datalog and Recursive Query Processing.
   FNT in Databases. 5, 105–195 (2012).
"""

from warnings import warn

from ..exceptions import ForbiddenUnstratifiedAggregation, NeuroLangException
from ..expression_walker import (
    FunctionApplicationToPythonLambda,
    PatternWalker,
    ReplaceSymbolsByConstants,
    add_match
)
from ..expressions import Constant, Expression, FunctionApplication, Symbol
from ..type_system import get_generic_type
from ..utils.relational_algebra_set import RelationalAlgebraStringExpression
from . import (
    Implication,
    Union,
    chase,
    is_conjunctive_expression_with_nested_predicates
)
from .basic_representation import UnionOfConjunctiveQueries
from .expression_processing import extract_logic_predicates, stratify
from .expressions import TranslateToLogic
from .instance import MapInstance

FA2L = FunctionApplicationToPythonLambda()


class AggregationApplication(FunctionApplication):
    def __repr__(self):
        r = u'\u03BB{{<{}>: {}}}'.format(self.functor, self.__type_repr__)
        if self.args is ...:
            r += '(...)'
        elif self.args is not None:
            r += (
                '(' +
                ', '.join(repr(arg) for arg in self.args)
                + ')'
                )

        return r


def is_aggregation_rule(rule):
    return is_aggregation_predicate(rule.consequent)


def is_aggregation_predicate(predicate):
    return any(
        isinstance(arg, AggregationApplication) for arg in predicate.args
    )


class TranslateToLogicWithAggregation(TranslateToLogic):
    @add_match(
        Implication(FunctionApplication(Symbol, ...), Expression),
        lambda rule: any(
            isinstance(arg, FunctionApplication)
            and get_generic_type(type(arg)) is FunctionApplication
            for arg in rule.consequent.args
        )
    )
    def transform_function_application_consequent_to_aggregation(self, rule):
        consequent_arguments = tuple()
        for arg in rule.consequent.args:
            if (
                isinstance(arg, FunctionApplication)
                and get_generic_type(type(arg)) is FunctionApplication
            ):
                arg = AggregationApplication(*arg.unapply())
            consequent_arguments += (arg,)

        return self.walk(
            Implication(
                rule.consequent.functor(*consequent_arguments),
                rule.antecedent
            )
        )


class DatalogWithAggregationMixin(PatternWalker):
    @add_match(
        Implication(FunctionApplication(Symbol, ...), Expression),
        is_aggregation_rule
    )
    def statement_intensional_aggregation(self, expression):
        consequent = expression.consequent
        antecedent = expression.antecedent

        self._validate_aggregation_implication_syntax(consequent, antecedent)

        if consequent.functor in self.symbol_table:
            eb = self._new_intensional_internal_representation(consequent)
        else:
            eb = tuple()

        eb = eb + (expression, )
        symbol = consequent.functor.cast(UnionOfConjunctiveQueries)
        self.symbol_table[symbol] = Union(eb)

        return expression

    def _validate_aggregation_implication_syntax(self, consequent, antecedent):
        if consequent.functor in self.protected_keywords:
            raise NeuroLangException(
                f'symbol {self.constant_set_name} is protected'
            )

        seen_aggregations = 0
        for arg in consequent.args:
            if isinstance(arg, AggregationApplication):
                seen_aggregations += 1
                aggregation_functor = arg.functor
            elif not isinstance(arg, (Constant, Symbol)):
                raise NeuroLangException(
                    f'The consequent {consequent} can only be '
                    'constants, symbols'
                )

            if seen_aggregations > 1:
                raise NeuroLangException(
                    f'Only one aggregation allowed in {consequent}'
                )

        consequent_symbols = (
            consequent._symbols - consequent.functor._symbols -
            aggregation_functor._symbols
        )

        if not consequent_symbols.issubset(antecedent._symbols):
            raise NeuroLangException(
                "All variables on the consequent need to be on the antecedent"
            )

        if not is_conjunctive_expression_with_nested_predicates(antecedent):
            raise NeuroLangException(
                f'Expression {antecedent} is not conjunctive'
            )


class Chase(chase.Chase):
    def check_constraints(self, instance_update):
        code = Union(tuple(self.rules))
        stratified_code, stratifiable = stratify(code, self.datalog_program)
        self.stratified_code = stratified_code

        for stratum in self.stratified_code:
            seen_in_stratum = set()
            aggregate_rules = []
            for rule in stratum:
                seen_in_stratum.add(rule.consequent.functor)
                if is_aggregation_rule(rule):
                    aggregate_rules.append(rule)

            self._stratum_is_aggregation_viable(
                seen_in_stratum, aggregate_rules
            )

        return super().check_constraints(instance_update)

    def build_chase_solution(self):
        instance_update = MapInstance(
            self.datalog_program.extensional_database()
        )
        self.check_constraints(instance_update)
        instance = MapInstance()
        for stratum in self.stratified_code:
            instance = self.execute_chase(stratum, instance_update, instance)
            instance_update = instance
            instance = MapInstance()
        return instance_update

    def _stratum_is_aggregation_viable(self, seen_in_stratum, aggregate_rules):
        for rule in aggregate_rules:
            if any(
                predicate.functor in seen_in_stratum
                for predicate
                in extract_logic_predicates(rule.antecedent)
            ):
                raise ForbiddenUnstratifiedAggregation(
                    f"Unstratifiable aggregation {rule.consequent.functor}"
                )

    def compute_result_set(
        self, rule, substitutions, instance, restriction_instance=None
    ):
        if not is_aggregation_rule(rule):
            return super().compute_result_set(
                rule,
                substitutions,
                instance,
                restriction_instance=restriction_instance
            )
        if restriction_instance is None:
            restriction_instance = dict()

        group_vars, output_args = self._get_groups_and_aggregations(rule)
        new_tuples = (
            substitutions
            .aggregate(group_vars, output_args)
            .projection_to_unnamed(*(oa[0] for oa in output_args))
        )
        new_tuples = self.datalog_program.new_set(new_tuples)
        return self.compute_instance_update(
            rule, new_tuples, instance, restriction_instance
        )

    def _get_groups_and_aggregations(self, rule):
        output_args = []
        group_vars = []
        for arg in rule.consequent.args:
            if isinstance(arg, Symbol):
                group_vars.append(arg.name)
                aggregation_args = arg.name
                fun = RelationalAlgebraStringExpression('first')
            elif isinstance(arg, AggregationApplication):
                aggregation_args, fun = self._obtain_aggregations(arg)
            output_args.append(
                (
                    Symbol.fresh().name,
                    aggregation_args,
                    fun
                )
            )
        return group_vars, output_args

    def _obtain_aggregations(self, arg):
        if (
            len(arg.args) == 1 and
            isinstance(arg.args[0], Symbol)
        ):
            fun = self.datalog_program.walk(arg.functor).value
            aggregation_args = arg.args[0].name
        else:
            aggregation_args = None
            fa = ReplaceSymbolsByConstants(
                self.datalog_program.symbol_table
            ).walk(FunctionApplication(*arg.unapply()))
            fun_, arg_names = FA2L.walk(fa)
            fun_str = (
                "lambda t: fun_(" +
                ", ".join(f't.{arg_}' for arg_ in arg_names) +
                ")"
            )
            gs = globals()
            ls = locals()
            gs['fun_'] = fun_
            fun = eval(fun_str, gs, ls)
        return aggregation_args, fun

    def eliminate_already_computed(self, consequent, instance, substitutions):
        if is_aggregation_predicate(consequent):
            return substitutions

        return super().eliminate_already_computed(
            consequent, instance, substitutions
        )
