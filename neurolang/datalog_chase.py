from collections import namedtuple
from itertools import chain, tee
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
from .relational_algebra import (
    Column, Selection, Product, Projection, eq_,
    RelationAlgebraRewriteOptimiser, RelationAlgebraSolver
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


def translate_to_ra_plus(
    args_to_project,
    rule_predicates_iterator
):
    seen_vars = dict()
    selections = []
    column = -1
    new_representations = tuple()
    projections = tuple()
    projected_var_names = dict()
    rule_predicates_iterator = list(rule_predicates_iterator)
    for p, pred_rep in enumerate(rule_predicates_iterator):
        predicate, representation = pred_rep
        local_selections = []
        for i, arg in enumerate(predicate.args):
            column += 1
            c = Constant[Column](Column(column))
            if isinstance(arg, Constant):
                l_c = Constant[Column](Column(i))
                local_selections.append((l_c, arg))
            elif isinstance(arg, Symbol):
                if arg in seen_vars:
                    selections.append((seen_vars[arg], c))
                else:
                    if arg in args_to_project:
                        projected_var_names[arg] = len(projections)
                        projections += (c,)
                    seen_vars[arg] = c
        new_representation = sdb.Constant[AbstractSet](representation)
        for s1, s2 in local_selections:
            new_representation = Selection(new_representation, eq_(s1, s2))
        new_representations += (new_representation,)
    if len(new_representations) > 0:
        relation = Product(new_representations)
        for s1, s2 in selections:
            relation = Selection(relation, eq_(s1, s2))
        relation = Projection(relation, projections)
    else:
        relation = Constant[AbstractSet](set())
    return relation, projected_var_names


def obtain_substitutions_relational_algebra_plus(
    args_to_project, rule_predicates_iterator
):
    ra_code, projected_var_names = translate_to_ra_plus(
        args_to_project,
        rule_predicates_iterator
    )
    ra_code = RelationAlgebraRewriteOptimiser().walk(ra_code)

    if not isinstance(ra_code, Constant) or len(ra_code.value) > 0:
        result = RelationAlgebraSolver().walk(ra_code)
    else:
        return [{}]

    substitutions = []
    for tuple_ in result.value:
        subs = {
            var: tuple_.value[col]
            for var, col in projected_var_names.items()
        }
        substitutions.append(subs)

    return substitutions


obtain_substitutions = obtain_substitutions_relational_algebra_plus


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
