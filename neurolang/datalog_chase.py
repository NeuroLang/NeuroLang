from collections import namedtuple
from itertools import chain
from operator import eq
from typing import AbstractSet

from .expressions import Constant, Symbol, FunctionApplication
from . import solver_datalog_naive as sdb
from .unification import (
    apply_substitution,
    apply_substitution_arguments,
    compose_substitutions,
    most_general_unifier_arguments
)


def chase_step(datalog, instance, builtins, rule, restriction_instance=None):
    if restriction_instance is None:
        restriction_instance = dict()

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

    substitutions = obtain_substitutions(rule_predicates_iterator)

    substitutions = evaluate_builtins(
        builtin_predicates, substitutions, datalog
    )

    return compute_result_set(
        rule, substitutions, instance, restriction_instance
    )


def obtain_substitutions(rule_predicates_iterator):
    substitutions = [{}]
    for predicate, representation in rule_predicates_iterator:
        new_substitutions = []
        for substitution in substitutions:
            new_substitutions += unify_substitution(
                predicate, substitution, representation
            )
        substitutions = new_substitutions
    return substitutions


def unify_substitution(predicate, substitution, representation):
    new_substitutions = []
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
    return new_substitutions


def evaluate_builtins(builtin_predicates, substitutions, datalog):
    new_substitutions = []
    predicates = [p for p, _ in builtin_predicates]
    for substitution in substitutions:
        new_substitution = evaluate_builtins_predicates(
            predicates, substitution, datalog
        )
        if new_substitution is not None:
            new_substitutions.append(new_substitution)
    return new_substitutions


def evaluate_builtins_predicates(
    predicates_to_evaluate, substitution, datalog
):
    evaluate = True
    while evaluate:
        evaluate = False
        new_predicates_to_evaluate = []
        for predicate in predicates_to_evaluate:
            subs = unify_builtin_substitution(
                predicate, substitution, datalog
            )
            if subs is None:
                new_predicates_to_evaluate.append(predicate)
            else:
                substitution = compose_substitutions(
                    substitution, subs
                )
                evaluate = True
        predicates_to_evaluate = new_predicates_to_evaluate
    if len(predicates_to_evaluate) == 0:
        return substitution
    else:
        return None


def unify_builtin_substitution(predicate, substitution, datalog):
    substituted_predicate = apply_substitution(
        predicate, substitution
    )
    evaluated_predicate = datalog.walk(substituted_predicate)
    if (
        isinstance(evaluated_predicate, Constant[bool]) and
        evaluated_predicate.value
    ):
        return substitution
    elif is_equality_between_constant_and_symbol(evaluated_predicate):
        if isinstance(evaluated_predicate.args[0], Symbol):
            substitution = {
                evaluated_predicate.args[0]: evaluated_predicate.args[1]
            }
        else:
            substitution = {
                evaluated_predicate.args[1]: evaluated_predicate.args[0]
            }
        return substitution
    else:
        return None


def is_equality_between_constant_and_symbol(predicate):
    return (
        isinstance(predicate, FunctionApplication) and
        isinstance(predicate.functor, Constant) and
        predicate.functor.value is eq and
        any(isinstance(arg, Constant) for arg in predicate.args) and
        any(isinstance(arg, Symbol) for arg in predicate.args)
    )


def extract_rule_predicates(
    rule, instance, builtins, restriction_instance=None
):
    if restriction_instance is None:
        restriction_instance = dict()

    head_functor = rule.consequent.functor
    rule_predicates = sdb.extract_datalog_predicates(rule.antecedent)
    restricted_predicates = []
    nonrestricted_predicates = []
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
        elif functor in builtins:
            builtin_predicates.append((predicate, builtins[functor]))
        elif isinstance(functor, Constant):
            builtin_predicates.append((predicate, functor))
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
    if restriction_instance is None:
        restriction_instance = dict()
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
    elif rule.consequent.functor in restriction_instance:
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
            new_node = build_nodes_from_rules(
                datalog_program, node, builtins, rule
            )
            if new_node is not None:
                nodes_to_process.append(new_node)
    return root


def build_nodes_from_rules(datalog_program, node, builtins, rule):
    instance_update = chase_step(
        datalog_program, node.instance, builtins, rule
    )
    if len(instance_update) > 0:
        new_instance = merge_instances(node.instance, instance_update)
        new_node = ChaseNode(new_instance, dict())
        node.children[rule] = new_node
        return new_node
    else:
        return None


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
