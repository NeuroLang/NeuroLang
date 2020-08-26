from itertools import chain, tee

from ...exceptions import NeuroLangException
from ...expressions import Constant
from ...logic.unification import (apply_substitution_arguments,
                                  compose_substitutions,
                                  most_general_unifier_arguments)
from ...utils import OrderedSet
from .. import Negation
from ..expression_processing import extract_logic_predicates
from . import (ChaseGeneral, ChaseMGUMixin,
               ChaseRelationalAlgebraPlusCeriMixin, ChaseSemiNaive)


class NegativeFactConstraints:
    def check_constraints(self, instance_update):
        super().check_constraints(instance_update)
        for symbol, args in self.datalog_program.negated_symbols.items():
            instance_values = [x for x in instance_update[symbol].value]
            if symbol in instance_update and next(
                iter(args.value)
            ) in instance_values:
                raise NeuroLangException(
                    f'There is a contradiction in your facts'
                )


class DatalogChaseNegationGeneral(
    ChaseGeneral, ChaseSemiNaive, NegativeFactConstraints
):
    def chase_step(self, instance, rule, restriction_instance=None):
        if restriction_instance is None:
            restriction_instance = set()

        rule_predicates = self.extract_rule_predicates(
            rule, instance, restriction_instance=restriction_instance
        )

        if all(len(predicate_list) == 0 for predicate_list in rule_predicates):
            return dict()

        restricted_predicates, nonrestricted_predicates, negative_predicates, \
            builtin_predicates, negative_builtin_predicates = rule_predicates

        builtin_predicates, builtin_predicates_ = tee(builtin_predicates)
        args_to_project = self.get_args_to_project(rule, builtin_predicates_)

        rule_predicates_iterator = chain(
            restricted_predicates, nonrestricted_predicates
        )

        substitutions = self.obtain_substitutions(
            args_to_project, rule_predicates_iterator
        )

        substitutions = self.obtain_negative_substitutions(
            args_to_project, negative_predicates, substitutions
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

    def get_args_to_project(self, rule, builtin_predicates_):
        args_to_project = self.extract_variable_arguments(rule.consequent)
        for predicate, _ in builtin_predicates_:
            args_to_project |= self.extract_variable_arguments(predicate)
        new_args_to_project = OrderedSet()
        for a in args_to_project:
            new_args_to_project.add(a)
        args_to_project = new_args_to_project
        return args_to_project

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

        rule_predicates = extract_logic_predicates(rule.antecedent)
        restricted_predicates = []
        nonrestricted_predicates = []
        negative_predicates = []
        negative_builtin_predicates = []
        builtin_predicates = []
        for predicate in rule_predicates:

            if isinstance(predicate, Negation):
                functor = predicate.formula.functor
                if functor in self.builtins:
                    negative_builtin_predicates.append((
                        predicate.formula,
                        self.builtins[functor]
                    ))
                elif isinstance(functor, Constant):
                    negative_builtin_predicates.append((
                        predicate.formula,
                        functor
                    ))
                else:
                    if functor in instance:
                        negative_predicates.append((
                            predicate.formula,
                            instance[functor].value
                        ))
                    if functor in restriction_instance:
                        negative_predicates.append((
                            predicate.formula,
                            restriction_instance[functor].value
                        ))
            else:
                functor = predicate.functor

                if functor in restriction_instance:
                    restricted_predicates.append(
                        (predicate, restriction_instance[functor].value)
                    )
                elif functor in instance:
                    nonrestricted_predicates.append(
                        (predicate, instance[functor].value)
                    )
                elif functor in self.builtins:
                    builtin_predicates.append(
                        (predicate, self.builtins[functor])
                    )
                elif isinstance(functor, Constant):
                    builtin_predicates.append((predicate, functor))
                else:
                    return ([], [], [], [], [])

        return (
            restricted_predicates,
            nonrestricted_predicates,
            negative_predicates,
            builtin_predicates,
            negative_builtin_predicates,
        )

    def check_non_linear(self, head_functor, functor, recursive_calls):
        if functor == head_functor:
            recursive_calls += 1
            if recursive_calls > 1:
                raise ValueError(
                    'Non-linear rule {rule}, solver non supported'
                )
        return recursive_calls


class DatalogChaseNegationRelationalAlgebraMixin(
    ChaseRelationalAlgebraPlusCeriMixin
):
    def obtain_negative_substitutions(
        self, args_to_project, rule_predicates_iterator, substitutions
    ):
        raise NotImplementedError()


class DatalogChaseNegationMGUMixin(ChaseMGUMixin):
    @staticmethod
    def obtain_negative_substitutions(
        args_to_project, negative_predicates, substitutions
    ):
        for predicate, representation in negative_predicates:
            new_substitutions = []
            for substitution in substitutions:
                new_substitutions += DatalogChaseNegationMGUMixin\
                    .unify_negative_substitution(
                        predicate, substitution, representation
                    )
            substitutions = new_substitutions
        return [{
            k: v
            for k, v in substitution.items()
            if k in args_to_project
        }
                for substitution in substitutions]

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


class DatalogChaseNegation(
    DatalogChaseNegationGeneral, DatalogChaseNegationMGUMixin
):
    pass
