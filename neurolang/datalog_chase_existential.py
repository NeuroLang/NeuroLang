from itertools import chain
from typing import AbstractSet

from .datalog_chase import DatalogChase
from .solver_datalog_naive import NullConstant, Any, Unknown
from .unification import (
    apply_substitution, apply_substitution_arguments, compose_substitutions,
    most_general_unifier_arguments
)
from .expressions import Constant, ExistentialPredicate
from .solver_datalog_naive import Implication
from .expression_walker import expression_iterator


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
        self.fresh_nulls = dict({})

    def chase_step(self, instance, rule, restriction_instance=None):
        if restriction_instance is None:
            restriction_instance = dict()

        substitutions = [{}]
        if isinstance(rule, Implication
                      ) & isinstance(rule.consequent, ExistentialPredicate):
            substitutions = self.replace_fresh_null(rule)

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

    def compute_result_set(
        self, rule, substitutions, instance, restriction_instance=None
    ):
        if restriction_instance is None:
            restriction_instance = dict()

        if isinstance(rule.consequent, ExistentialPredicate):
            args = rule.consequent.body.args
            functor = rule.consequent.body.functor
        else:
            args = rule.consequent.args
            functor = rule.consequent.functor

        new_tuples = set(
            Constant(apply_substitution_arguments(args, substitution))
            for substitution in substitutions
        )

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
        total = len(self.fresh_nulls)
        name = f'NULL {total+1}'
        fresh_null = NullConstant[Unknown](name, auto_infer_type=False)
        self.fresh_nulls[name] = fresh_null

        return fresh_null

    def replace_fresh_null(self, rule):
        expression = expression_iterator(rule)
        substitutions = dict({})
        for exp in expression:
            if exp[0] == 'head' and exp[1] not in substitutions:
                new_fresh = self.new_fresh_null()
                substitutions[exp[1]] = new_fresh

        return [substitutions]
