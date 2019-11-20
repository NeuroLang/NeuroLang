from ...logic.unification import (apply_substitution_arguments,
                                  compose_substitutions,
                                  most_general_unifier_arguments)


class ChaseMGUMixin:
    @staticmethod
    def obtain_substitutions(args_to_project, rule_predicates_iterator):
        substitutions = [{}]
        for predicate, representation in rule_predicates_iterator:
            new_substitutions = []
            for substitution in substitutions:
                new_substitutions += ChaseMGUMixin.unify_substitution(
                    predicate, substitution, representation
                )
            substitutions = new_substitutions
        return [
            {
                k: v for k, v in substitution.items()
                if k in args_to_project
            }
            for substitution in substitutions
        ]

    @staticmethod
    def unify_substitution(predicate, substitution, representation):
        new_substitutions = []
        subs_args = apply_substitution_arguments(predicate.args, substitution)

        for element in representation:
            mgu_substituted = most_general_unifier_arguments(
                subs_args, element.value
            )

            if mgu_substituted is not None:
                new_substitution = mgu_substituted[0]
                new_substitutions.append(
                    compose_substitutions(substitution, new_substitution)
                )
        return new_substitutions
