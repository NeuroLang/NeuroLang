from collections import namedtuple
from itertools import chain
from typing import AbstractSet

from .expressions import Constant
from . import solver_datalog_naive as sdb
from .unification import (
    apply_substitution_arguments,
    compose_substitutions,
    most_general_unifier_arguments
)


def chase_step(datalog, instance, builtins, rule, restriction_instance=None):

    rule_predicates = extract_rule_predicates(
        rule, instance, builtins, restriction_instance=restriction_instance
    )

    if all(len(predicate_list) == 0 for predicate_list in rule_predicates):
        return {}

    restricted_predicates, nonrestricted_predicates, builtin_predicates =\
        rule_predicates

    rule_predicates_iterator = chain(
        restricted_predicates,
        nonrestricted_predicates
    )

    substitutions = [{}]
    for predicate, representation in rule_predicates_iterator:
        functor = predicate.functor
        new_substitutions = []
        for substitution in substitutions:
            subs_args = apply_substitution_arguments(
                predicate.args, substitution
            )

            for element in representation:
                mgu_substituted = most_general_unifier_arguments(
                    subs_args,
                    element.value
                )

                if mgu_substituted is not None:
                    new_substitution = mgu_substituted[0]
                    new_substitutions.append(
                        compose_substitutions(
                            substitution, new_substitution
                        )
                    )

        substitutions = new_substitutions

    for predicate, _ in builtin_predicates:
        functor = predicate.functor
        new_substitutions = []
        for substitution in substitutions:
            subs_args = apply_substitution_arguments(
                predicate.args, substitution
            )

            mgu_substituted = most_general_unifier_arguments(
                subs_args, predicate.args
            )

            if mgu_substituted is not None:
                predicate_res = datalog.walk(
                    predicate.apply(functor, mgu_substituted[1])
                )

                if (
                    isinstance(predicate_res, Constant[bool]) and
                    predicate_res.value
                ):
                    new_substitution = mgu_substituted[0]
                    new_substitutions.append(
                        compose_substitutions(
                            substitution, new_substitution
                        )
                    )
        substitutions = new_substitutions

    return compute_result_set(
        rule, substitutions, instance, restriction_instance
    )


def extract_rule_predicates(
    rule, instance, builtins, restriction_instance=None
):
    head_functor = rule.consequent.functor
    rule_predicates = sdb.extract_datalog_predicates(rule.antecedent)
    restricted_predicates = []
    nonrestricted_predicates = []
    builtin_predicates = []
    recursive_calls = 0
    for predicate in rule_predicates:
        functor = predicate.functor
        if restriction_instance is not None:
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
                continue

        if functor in instance:
            nonrestricted_predicates.append(
                (predicate, instance[functor].value)
            )
        elif functor in builtins:
            builtin_predicates.append((predicate, builtins[functor]))
        else:
            return ([], [], [])

    return (
        restricted_predicates,
        nonrestricted_predicates,
        builtin_predicates
    )


def compute_result_set(
    rule, substitutions, instance, restriction_instance=None
):
    new_tuples = set(
        Constant(
            apply_substitution_arguments(
                rule.consequent.args, substitution
            )
        )
        for substitution in substitutions
    )

    if rule.consequent.functor in instance:
        new_tuples -= instance[rule.consequent.functor].value
    if (
        restriction_instance is not None and
        rule.consequent.functor in restriction_instance
    ):
        new_tuples -= restriction_instance[rule.consequent.functor].value

    if len(new_tuples) == 0:
        return {}
    else:
        set_type = next(iter(new_tuples)).type
        new_instance = {
            rule.consequent.functor:
            Constant[AbstractSet[set_type]](new_tuples)
        }
        return new_instance


def merge_instances(*args):
    new_instance = args[0].copy()

    for next_instance in args:
        for k, v in next_instance.items():
            if k not in new_instance:
                new_instance[k] = v
            else:
                new_set = new_instance[k]
                new_set = Constant[new_set.type](
                    v.value | new_set.value
                )
                new_instance[k] = new_set

    return new_instance


ChaseNode = namedtuple('ChaseNode', 'instance children')


def build_chase_tree(datalog_program, chase_set=chase_step):
    builtins = datalog_program.builtins()
    root = ChaseNode(datalog_program.extensional_database(), dict())
    rules = []
    for expression_block in datalog_program.intensional_database().values():
        for rule in expression_block.expressions:
            rules.append(rule)

    nodes_to_process = [root]
    while len(nodes_to_process) > 0:
        node = nodes_to_process.pop()
        for rule in rules:
            instance_update = chase_step(
                datalog_program, node.instance, builtins, rule
            )
            if len(instance_update) > 0:
                new_instance = merge_instances(node.instance, instance_update)
                new_node = ChaseNode(new_instance, dict())
                nodes_to_process.append(new_node)
                node.children[rule] = new_node

    return root


def build_chase_solution(datalog_program, chase_step=chase_step):
    rules = []
    for expression_block in datalog_program.intensional_database().values():
        for rule in expression_block.expressions:
            rules.append(rule)

    instance = dict()
    builtins = datalog_program.builtins()
    instance_update = datalog_program.extensional_database()
    while len(instance_update) > 0:
        instance = merge_instances(instance, instance_update)
        instance_update = merge_instances(*(
            chase_step(
                datalog_program, instance, builtins, rule,
                restriction_instance=instance_update
            )
            for rule in rules
        ))

    return instance
