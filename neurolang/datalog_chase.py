from collections import namedtuple, OrderedDict
from itertools import chain, product, tee
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
        return dict()

    restricted_predicates, nonrestricted_predicates, builtin_predicates =\
        rule_predicates

    args_to_project = extract_variable_arguments(rule.consequent)
    builtin_predicates, builtin_predicates_ = tee(builtin_predicates)
    for predicate, _ in builtin_predicates_:
        args_to_project += extract_variable_arguments(predicate)
    new_args_to_project = tuple()
    for i, a in enumerate(args_to_project):
        if a not in args_to_project[:i]:
            new_args_to_project += (a,)
    args_to_project = new_args_to_project

    rule_predicates_iterator = chain(
        restricted_predicates,
        nonrestricted_predicates
    )

    substitutions = obtain_substitutions(
        args_to_project, rule_predicates_iterator
    )

    substitutions = evaluate_builtins(
        builtin_predicates, substitutions, datalog
    )

    return compute_result_set(
        datalog,
        rule, substitutions, instance, restriction_instance
    )


def extract_variable_arguments(predicate):
    stack = list(predicate.args[::-1])
    variables = tuple()
    while stack:
        arg = stack.pop()
        if isinstance(arg, Symbol):
            variables += (arg,)
        elif isinstance(arg, FunctionApplication):
            stack += arg.args[::-1]
    return variables


def obtain_substitutions_mgu(args_to_project, rule_predicates_iterator):
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


def obtain_substitutions_relational_algebra(
    args_to_project, rule_predicates_iterator
):
    new_representations, joins, vars = \
        filter_constants_obtain_joins(
            args_to_project, rule_predicates_iterator
        )

    new_representations, var_replacements = execute_joins(
        new_representations, joins
    )

    projections_per_relationship = OrderedDict()
    var_names = []
    for v, pos in vars.items():
        if v not in args_to_project:
            continue
        r, c = var_replacements.get(pos, pos)
        if r not in projections_per_relationship:
            projections_per_relationship[r] = tuple()
        projections_per_relationship[r] += (c,)
        var_names.append(v)

    projected_representations = [
        new_representations[r].projection(*cs)
        for r, cs in projections_per_relationship.items()
    ]

    substitutions = []
    for tuples in product(*projected_representations):
        subs = {
            var: value
            for var, value in zip(
                var_names,
                chain(*(t.value for t in tuples))
            )
        }
        substitutions.append(subs)

    return substitutions


def filter_constants_obtain_joins(args_to_project, rule_predicates_iterator):
    selections = {}
    joins = {}
    projections = {}
    vars = {}
    new_representations = []
    for p, pred_rep in enumerate(rule_predicates_iterator):
        predicate, representation = pred_rep
        for i, arg in enumerate(predicate.args):
            if isinstance(arg, Constant):
                if p not in selections:
                    selections[p] = dict()
                selections[p][i] = arg.value
            else:
                if arg in args_to_project and arg not in projections:
                    projections[arg] = (p, i)
                if arg not in vars:
                    vars[arg] = (p, i)
                    joins[(p, i)] = []
                else:
                    joins[vars[arg]].append((p, i))

        if p in selections:
            new_representation = representation.selection(selections[p])
        else:
            new_representation = representation
        new_representations.append(new_representation)
    return new_representations, joins, vars


def execute_joins(representations, joins):
    join_replacements = {}
    representations = OrderedDict(
        (i, r)
        for i, r in enumerate(representations)
    )
    for main_var, joins in joins.items():
        main_var = join_replacements.get(main_var, main_var)
        r1 = representations[main_var[0]]
        col1 = main_var[1]
        r1_a = r1.arity
        for join in joins:
            join = join_replacements.get(join, join)
            r2 = representations[join[0]]
            r12 = r1.natural_join(r2, ((col1, join[1]),))
            r1 = r12
            for i in range(r2.arity):
                join_replacements[(join[0], i)] = (main_var[0], r1_a + i)
            r1_a = r1.arity
            del representations[join[0]]
        representations[main_var[0]] = r1

    return representations, join_replacements


obtain_substitutions = obtain_substitutions_relational_algebra


def evaluate_builtins(builtin_predicates, substitutions, datalog):
    new_substitutions = []
    predicates = [datalog.walk(p) for p, _ in builtin_predicates]
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
    predicates_to_evaluate = predicates_to_evaluate.copy()
    unresolved_predicates = []
    while predicates_to_evaluate:
        predicate = predicates_to_evaluate.pop(0)

        subs = unify_builtin_substitution(
            predicate, substitution, datalog
        )
        if subs is None:
            unresolved_predicates.append(predicate)
        else:
            substitution = compose_substitutions(
                substitution, subs
            )
            predicates_to_evaluate += unresolved_predicates
            unresolved_predicates = []

    if len(unresolved_predicates) == 0:
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
        return unify_builtin_substitution_equality(evaluated_predicate)
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


def unify_builtin_substitution_equality(evaluated_predicate):
    if isinstance(evaluated_predicate.args[0], Symbol):
        substitution = {
            evaluated_predicate.args[0]: evaluated_predicate.args[1]
        }
    else:
        substitution = {
            evaluated_predicate.args[1]: evaluated_predicate.args[0]
        }
    return substitution


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
    datalog, rule, substitutions, instance, restriction_instance=None
):
    if restriction_instance is None:
        restriction_instance = dict()
    new_tuples = list(
        tuple(
            v.value for v in
            apply_substitution_arguments(
                rule.consequent.args, substitution
            )
        )
        for substitution in substitutions
    )
    new_tuples = datalog.new_set(new_tuples)

    if rule.consequent.functor in instance:
        new_tuples -= instance[rule.consequent.functor].value
    elif rule.consequent.functor in restriction_instance:
        new_tuples -= restriction_instance[rule.consequent.functor].value

    if len(new_tuples) == 0:
        return dict()
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
        node = nodes_to_process.pop(0)
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
