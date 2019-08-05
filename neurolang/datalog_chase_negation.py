from itertools import chain
from operator import invert

from .expressions import Constant
from .exceptions import NeuroLangException
from . import solver_datalog_naive as sdb
from .unification import (
    apply_substitution_arguments, compose_substitutions,
    most_general_unifier_arguments
)
from .datalog_chase import DatalogChase


class DatalogChaseNegation(DatalogChase):
    def chase_step(self, instance, rule, restriction_instance=None):
        if restriction_instance is None:
            restriction_instance = set()

        rule_predicates = self.extract_rule_predicates(
            rule, instance, restriction_instance=restriction_instance
        )

        if all(len(predicate_list) == 0 for predicate_list in rule_predicates):
            return {}

        restricted_predicates, nonrestricted_predicates, negative_predicates, \
            builtin_predicates, negative_builtin_predicates = rule_predicates

        rule_predicates_iterator = chain(
            restricted_predicates, nonrestricted_predicates
        )

        substitutions = self.obtain_substitutions(rule_predicates_iterator)

        substitutions = self.obtain_negative_substitutions(
            negative_predicates, substitutions
        )

        substitutions = self.evaluate_builtins(
            builtin_predicates, substitutions
        )

        substitutions = self.evaluate_negative_builtins(
            negative_builtin_predicates, substitutions
        )

        return self.compute_result_set(
            rule, substitutions, instance, restriction_instance
        )

    @staticmethod
    def obtain_negative_substitutions(negative_predicates, substitutions):
        for predicate, representation in negative_predicates:
            new_substitutions = []
            for substitution in substitutions:
                new_substitutions += DatalogChaseNegation\
                    .unify_negative_substitution(
                        predicate, substitution, representation
                )
            substitutions = new_substitutions
        return substitutions

    @staticmethod
    def unify_negative_substitution(predicate, substitution, representation):
        new_substitutions = []
        subs_args = apply_substitution_arguments(predicate.args, substitution)

        for element in representation:
            mgu_substituted = most_general_unifier_arguments(
                subs_args, element.value
            )

            if mgu_substituted is not None:
                break
        else:
            new_substitution = {predicate: element.value}
            new_substitutions.append(
                compose_substitutions(substitution, new_substitution)
            )
        return new_substitutions

    def evaluate_negative_builtins(self, builtin_predicates, substitutions):
        for predicate, _ in builtin_predicates:
            functor = predicate.functor
            new_substitutions = []
            for substitution in substitutions:
                new_substitutions += self.unify_negative_builtin_substitution(
                    predicate, substitution, functor
                )
            substitutions = new_substitutions
        return substitutions

    def unify_negative_builtin_substitution(
        self, predicate, substitution, functor
    ):
        subs_args = apply_substitution_arguments(predicate.args, substitution)

        mgu_substituted = most_general_unifier_arguments(
            subs_args, predicate.args
        )

        if mgu_substituted is not None:
            predicate_res = self.datalog_program.walk(
                predicate.apply(functor, mgu_substituted[1])
            )

            if (
                isinstance(predicate_res, Constant[bool]) and
                not predicate_res.value
            ):
                return [
                    compose_substitutions(substitution, mgu_substituted[0])
                ]
        return []

    def extract_rule_predicates(
        self, rule, instance, restriction_instance=None
    ):
        if restriction_instance is None:
            restriction_instance = set()

        head_functor = rule.consequent.functor
        rule_predicates = sdb.extract_datalog_predicates(rule.antecedent)
        restricted_predicates = []
        nonrestricted_predicates = []
        negative_predicates = []
        negative_builtin_predicates = []
        builtin_predicates = []
        recursive_calls = 0
        for predicate in rule_predicates:
            functor = predicate.functor

            if functor == head_functor:
                recursive_calls += 1
                if recursive_calls > 1:
                    raise ValueError(
                        'Non-linear rule {rule}, solver non supported'
                    )

            if functor in restriction_instance:
                restricted_predicates.append(
                    (predicate, restriction_instance[functor].value)
                )
            elif functor in instance:
                nonrestricted_predicates.append(
                    (predicate, instance[functor].value)
                )
            elif functor in self.builtins:
                builtin_predicates.append((predicate, self.builtins[functor]))
            elif (
                functor == invert and
                predicate.args[0].functor in self.builtins
            ):
                negative_builtin_predicates.append((
                    predicate.args[0], self.builtins[predicate.args[0].functor]
                ))
            elif functor == invert:
                negative_predicates.append((
                    predicate.args[0],
                    instance[predicate.args[0].functor].value
                ))
            else:
                return ([], [], [], [])

        return (
            restricted_predicates,
            nonrestricted_predicates,
            negative_predicates,
            builtin_predicates,
            negative_builtin_predicates,
        )

    def check_constraints(self, instance_update):
        for symbol, args in self.datalog_program.negated_symbols.items():
            instance_values = [x for x in instance_update[symbol].value]
            if symbol in instance_update and next(
                iter(args.value)
            ) in instance_values:
                raise NeuroLangException(
                    f'There is a contradiction in your facts'
                )
