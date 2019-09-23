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

from typing import AbstractSet, Tuple
from uuid import uuid1
from warnings import warn

from ..exceptions import NeuroLangException
from ..expression_walker import PatternWalker, add_match
from ..expressions import Constant, Expression, FunctionApplication, Symbol
from ..unification import apply_substitution_arguments
from ..utils import OrderedSet
from . import (Disjunction, Implication, chase, extract_datalog_free_variables,
               is_conjunctive_expression_with_nested_predicates)


class AggregationApplication(FunctionApplication):
    pass


def is_aggregation_rule(rule):
    return is_aggregation_predicate(rule.consequent)


def is_aggregation_predicate(predicate):
    return any(
        isinstance(arg, AggregationApplication) for arg in predicate.args
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

        self.symbol_table[consequent.functor] = Disjunction(eb)

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


def extract_aggregation_atom_free_variables(atom):
    free_variables = OrderedSet()
    aggregation_fresh_variable = Symbol(str(uuid1()))
    for arg in atom.args:
        free_variables_arg = extract_datalog_free_variables(arg)
        if isinstance(arg, AggregationApplication):
            free_variables_aggregation = free_variables_arg
            free_variables.add(aggregation_fresh_variable)
            aggregation_application = arg
        else:
            free_variables |= free_variables_arg

    return (
        free_variables, free_variables_aggregation, aggregation_fresh_variable,
        aggregation_application
    )


class Chase(chase.Chase):
    def check_constraints(self, instance_update):
        warn(
            "No check performed. Should implement check for stratified"
            " aggregation"
        )
        return super().check_constraints(instance_update)

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

        args = extract_datalog_free_variables(rule.consequent)
        new_tuples = self.datalog_program.new_set(
            Constant[Tuple](
                apply_substitution_arguments(args, substitution)
            )
            for substitution in substitutions
        )

        fvs, substitutions = self.compute_aggregation_substitutions(
            rule, new_tuples, args
        )

        new_tuples = self.datalog_program.new_set(
            Constant[Tuple](
                apply_substitution_arguments(fvs, substitution)
            )
            for substitution in substitutions
        )

        return self.compute_instance_update(
            rule, new_tuples, instance, restriction_instance
        )

    def compute_aggregation_substitutions(self, rule, new_tuples, args):
        fvs, fvs_aggregation, agg_fresh_var, agg_application = \
            extract_aggregation_atom_free_variables(rule.consequent)

        group_ind_vars = tuple(
            zip(
                *((i, v) for i, v in enumerate(fvs) if v is not agg_fresh_var)
            )
        )

        if len(group_ind_vars) > 0:
            group_indices, group_vars = group_ind_vars
            grouped_iterator = new_tuples.groupby(group_indices)
        else:
            group_vars = tuple()
            grouped_iterator = [(None, new_tuples)]

        substitutions = []
        for g_id, group in grouped_iterator:
            substitution = self.compute_group_substitution(
                group, args, fvs_aggregation, agg_application, agg_fresh_var
            )

            if len(group_vars) == 1:
                substitution[group_vars[0]] = Constant(g_id)
            elif len(group_vars) > 1:
                substitution.update({
                    v: Constant(val)
                    for v, val in zip(group_vars, g_id)
                })

            substitutions.append(substitution)
        return fvs, substitutions

    def compute_group_substitution(
        self, group, args, fvs_aggregation, agg_application, agg_fresh_var
    ):
        agg_substitution = tuple(
            Constant[AbstractSet](
                frozenset(
                    v.value[0] for v in group.projection(args.index(v))
                ),
                auto_infer_type=False,
                verify_type=False
            ) for v in fvs_aggregation
        )
        if any(len(rs.value) == 0 for rs in agg_substitution):
            return {}
        else:
            fa_ = agg_application.functor(*agg_substitution)
            substitution = {agg_fresh_var: self.datalog_program.walk(fa_)}
            return substitution

    def eliminate_already_computed(self, consequent, instance, substitutions):
        if is_aggregation_predicate(consequent):
            return substitutions

        return super().eliminate_already_computed(
            consequent, instance, substitutions
        )
