from collections import namedtuple
from typing import AbstractSet

from . import unification
from .expressions import Constant
from . import solver_datalog_naive as sdb


def chase_step(instance, rule):
    rule_predicates = sdb.extract_datalog_predicates(rule.antecedent)

    substitutions = [{}]
    for i, predicate in enumerate(rule_predicates):
        functor = predicate.functor
        new_substitutions = []
        while len(substitutions) > 0:
            substitution = substitutions.pop()
            subs_pred = unification.apply_substitution(predicate, substitution)
            if functor not in instance:
                continue

            for element in instance[functor].value:
                mgu_substituted = unification.most_general_unifier(
                    subs_pred, functor(*element.value)
                )

                if mgu_substituted is not None:
                    new_substitutions.append(
                        unification.compose_substitutions(
                            substitution, mgu_substituted[0]
                        )
                    )
        substitutions = new_substitutions

    new_tuples = set(
        Constant(
            unification.apply_substitution(rule.consequent, substitution).args
        )
        for substitution in substitutions
    )

    if rule.consequent.functor in instance:
        new_tuples -= instance[rule.consequent.functor].value

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


def build_chase_tree(datalog_instance):
    root = ChaseNode(datalog_instance.extensional_database(), dict())
    rules = []
    for v in datalog_instance.intensional_database().values():
        for rule in v.expressions:
            rules.append(rule)

    nodes_to_process = [root]
    while len(nodes_to_process) > 0:
        node = nodes_to_process.pop()
        for rule in rules:
            DeltaI = chase_step(node.instance, rule)
            if len(DeltaI) > 0:
                new_instance = merge_instances(node.instance, DeltaI)
                new_node = ChaseNode(new_instance, dict())
                nodes_to_process.append(new_node)
                node.children[rule] = new_node

    return root


def build_chase_solution(datalog_instance):
    rules = []
    for v in datalog_instance.intensional_database().values():
        for rule in v.expressions:
            rules.append(rule)

    instance = dict()
    DeltaI = datalog_instance.extensional_database()
    while len(DeltaI) > 0:
        instance = merge_instances(instance, DeltaI)
        DeltaI = merge_instances(*(
            chase_step(instance, rule) for rule in rules
        ))

    return instance
