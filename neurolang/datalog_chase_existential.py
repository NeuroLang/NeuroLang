from itertools import chain
from typing import AbstractSet

from .datalog_chase import DatalogChase
from .solver_datalog_naive import (
    NullConstant,
    Any,
    Unknown,
    extract_datalog_predicates,
    extract_datalog_free_variables,
)
from .unification import (
    apply_substitution, apply_substitution_arguments, compose_substitutions,
    most_general_unifier_arguments
)
from .expressions import Constant, ExistentialPredicate
from .solver_datalog_naive import Implication
from .expression_walker import expression_iterator


class DatalogExistentialChaseRestricted(DatalogChase):
    '''
    Based on definitions 2.5 and 2.7 of Cali et. al. [1]_.


    .. [1] Calì, A., G. Gottlob, and M. Kifer. “Taming the Infinite Chase:
    Query Answering under Expressive Relational Constraints.”
    Journal of Artificial Intelligence Research 48 (October 22, 2013):
    115–74. https://doi.org/10.1613/jair.3873.
    '''

    def __init__(self, datalog_program, **kwargs):
        super().__init__(datalog_program, **kwargs)
        self.fresh_nulls = dict({})
        self.fired_triggers = set()

    def chase_step(self, instance, rule, restriction_instance=None):
        if restriction_instance is None:
            restriction_instance = dict()

        substitutions = [{}]

        free_variables = self.get_free_variables(rule)
        if len(free_variables) > 0:
            substitutions = self.replace_fresh_null(free_variables)

        rule_predicates = self.extract_rule_predicates(
            rule, instance, restriction_instance=restriction_instance
        )

        if all(len(predicate_list) == 0 for predicate_list in rule_predicates):
            return {}

        restricted_predicates, nonrestricted_predicates, builtin_predicates =\
            rule_predicates

        rule_predicates_iterator = chain(
            restricted_predicates, nonrestricted_predicates
        )

        substitutions = self.obtain_substitutions(
            rule_predicates_iterator, substitutions
        )

        substitutions = self.evaluate_builtins(
            builtin_predicates, substitutions
        )

        return self.compute_result_set(
            rule, substitutions, instance, restriction_instance
        )

    def get_free_variables(self, rule):
        free_variable_consequent = extract_datalog_free_variables(
            rule.consequent
        )
        free_variable_antecedent = extract_datalog_free_variables(
            rule.antecedent
        )
        free_variables = free_variable_consequent._set.difference(
            free_variable_antecedent._set
        )

        return free_variables

    def compute_result_set(
        self, rule, substitutions, instance, restriction_instance=None
    ):
        if restriction_instance is None:
            restriction_instance = dict()

        args = rule.consequent.args
        functor = rule.consequent.functor

        new_tuples = set()
        for substitution in substitutions:
            new_args = apply_substitution_arguments(args, substitution)
            trigger = (rule, self.remove_nulls(new_args))
            if trigger not in self.fired_triggers:
                self.fired_triggers.add(trigger)
                new_tuples.add(Constant(new_args))

        if functor in instance:
            new_tuples = self.restricted_evaluation(
                new_tuples, instance[functor].value
            )
        elif functor in restriction_instance:
            new_tuples -= restriction_instance[functor].value

        if len(new_tuples) == 0:
            return {}
        else:
            set_type = next(iter(new_tuples)).type
            new_instance = {
                functor: Constant[AbstractSet[set_type]](new_tuples)
            }
            return new_instance

    def restricted_evaluation(self, new_tuples, instance_values):
        for value in instance_values:
            new_tuples = self.compare_and_remove_predicate(new_tuples, value)

        return new_tuples

    def compare_and_remove_predicate(self, tuples, value):
        temp_tuples = tuples.copy()
        for new_tuple in temp_tuples:
            if self.compare_predicates_values(new_tuple.value, value.value):
                tuples -= {new_tuple}
        return tuples

    def compare_predicates_values(self, new_values, instance_values):
        for k, v in enumerate(new_values):
            if v.value not in self.fresh_nulls and v == instance_values[k]:
                continue
            elif v.value in self.fresh_nulls and instance_values[
                k] in self.fresh_nulls:
                continue
            else:
                return False
        return True

    def new_fresh_null(self):
        name = f'NULL {len(self.fresh_nulls)}'
        fresh_null = NullConstant[Unknown](name, auto_infer_type=False)
        self.fresh_nulls[name] = fresh_null

        return fresh_null

    def replace_fresh_null(self, free_variables):
        substitutions = dict({})
        for var in free_variables:
            if var not in substitutions:
                new_fresh = self.new_fresh_null()
                substitutions[var] = new_fresh

        return [substitutions]

    def remove_nulls(self, new_args):
        without_nulls = tuple(
            filter(
                lambda arg_tuple: not isinstance(arg_tuple, NullConstant),
                new_args
            )
        )
        return without_nulls


class DatalogExistentialChaseOblivious(DatalogChase):
    '''
    Based on definitions 2.5 and 2.7 of Cali et. al. [1]_.
    .. [1] Calì, A., G. Gottlob, and M. Kifer. “Taming the Infinite Chase:
    Query Answering under Expressive Relational Constraints.”
    Journal of Artificial Intelligence Research 48 (October 22, 2013):
    115–74. https://doi.org/10.1613/jair.3873.
    '''

    def __init__(self, datalog_program, **kwargs):
        super().__init__(datalog_program, **kwargs)
        self.total_fresh_nulls = 0
        self.fired_triggers = set()

    def chase_step(self, instance, rule, restriction_instance=None):
        if restriction_instance is None:
            restriction_instance = dict()

        substitutions = [{}]

        free_variables = self.get_free_variables(rule)
        if len(free_variables) > 0:
            substitutions = self.replace_fresh_null(free_variables)

        rule_predicates = self.extract_rule_predicates(
            rule, instance, restriction_instance=restriction_instance
        )

        if all(len(predicate_list) == 0 for predicate_list in rule_predicates):
            return {}

        restricted_predicates, nonrestricted_predicates, builtin_predicates =\
            rule_predicates

        rule_predicates_iterator = chain(
            restricted_predicates, nonrestricted_predicates
        )

        substitutions = self.obtain_substitutions(
            rule_predicates_iterator, substitutions
        )

        substitutions = self.evaluate_builtins(
            builtin_predicates, substitutions
        )

        return self.compute_result_set(
            rule, substitutions, instance, restriction_instance
        )

    def get_free_variables(self, rule):
        free_variable_consequent = extract_datalog_free_variables(
            rule.consequent
        )
        free_variable_antecedent = extract_datalog_free_variables(
            rule.antecedent
        )
        free_variables = free_variable_consequent._set.difference(
            free_variable_antecedent._set
        )

        return free_variables

    def compute_result_set(
        self, rule, substitutions, instance, restriction_instance=None
    ):
        if restriction_instance is None:
            restriction_instance = dict()

        args = rule.consequent.args
        functor = rule.consequent.functor

        new_tuples = set()
        for substitution in substitutions:
            new_args = apply_substitution_arguments(args, substitution)
            trigger = (rule, self.remove_nulls(new_args))
            if trigger not in self.fired_triggers:
                self.fired_triggers.add(trigger)
                new_tuples.add(Constant(new_args))

        if functor in instance:
            new_tuples -= instance[functor].value
        elif functor in restriction_instance:
            new_tuples -= restriction_instance[functor].value

        if len(new_tuples) == 0:
            return {}
        else:
            set_type = next(iter(new_tuples)).type
            new_instance = {
                functor: Constant[AbstractSet[set_type]](new_tuples)
            }
            return new_instance

    def new_fresh_null(self):
        name = f'NULL {self.total_fresh_nulls}'
        fresh_null = NullConstant[Unknown](name, auto_infer_type=False)
        self.total_fresh_nulls += 1

        return fresh_null

    def replace_fresh_null(self, free_variables):
        substitutions = dict({})
        for var in free_variables:
            if var not in substitutions:
                new_fresh = self.new_fresh_null()
                substitutions[var] = new_fresh

        return [substitutions]

    def remove_nulls(self, new_args):
        without_nulls = tuple(
            filter(
                lambda arg_tuple: not isinstance(arg_tuple, NullConstant),
                new_args
            )
        )
        return without_nulls
