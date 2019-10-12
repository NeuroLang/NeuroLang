from collections import namedtuple
from itertools import chain, tee
from operator import eq
from typing import AbstractSet

from ...exceptions import NeuroLangException
from ...expressions import Constant, FunctionApplication, Symbol
from ...unification import (apply_substitution, apply_substitution_arguments,
                            compose_substitutions)
from ...utils import OrderedSet
from ..expression_processing import (extract_datalog_free_variables,
                                     extract_datalog_predicates,
                                     is_linear_rule)
from ..instance import MapInstance
from ...type_system import infer_type


ChaseNode = namedtuple('ChaseNode', 'instance children')


class NeuroLangNonLinearProgramException(NeuroLangException):
    pass


class ChaseGeneral():
    """Chase implementation using the naive resolution algorithm.

    """
    def __init__(self, datalog_program, rules=None):
        self.datalog_program = datalog_program
        self._set_rules(rules)

        self.builtins = datalog_program.builtins()
        self.idb_edb_symbols = set(
            chain(
                self.datalog_program.extensional_database(),
                self.datalog_program.intensional_database()
            )
        ) | set(
            rule.consequent.functor
            for rule in self.rules
        )

    def _set_rules(self, rules):
        self.rules = []
        if rules is None:
            for disjunction in \
                    self.datalog_program.intensional_database().values():
                self.rules += disjunction.formulas
        else:
            self.rules += rules.formulas

    def check_constraints(self, instance_update):
        pass

    def chase_step(self, instance, rule, restriction_instance=None):
        if restriction_instance is None:
            restriction_instance = MapInstance()

        rule_predicates = self.extract_rule_predicates(
            rule, instance, restriction_instance=restriction_instance
        )

        if all(len(predicate_list) == 0 for predicate_list in rule_predicates):
            return MapInstance()

        restricted_predicates, nonrestricted_predicates, builtin_predicates =\
            rule_predicates

        builtin_predicates, builtin_predicates_ = tee(builtin_predicates)
        args_to_project = self.get_args_to_project(rule, builtin_predicates_)

        rule_predicates_iterator = chain(
            restricted_predicates, nonrestricted_predicates
        )

        substitutions = self.obtain_substitutions(
            args_to_project, rule_predicates_iterator
        )

        substitutions = self.eliminate_already_computed(
            rule.consequent, instance, substitutions
        )

        substitutions = self.evaluate_builtins(
            builtin_predicates, substitutions
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

    @staticmethod
    def extract_variable_arguments(predicate):
        return extract_datalog_free_variables(predicate)

    def evaluate_builtins(self, builtin_predicates, substitutions):
        new_substitutions = []
        predicates = [p for p, _ in builtin_predicates]
        if len(predicates) == 0:
            return substitutions
        for substitution in substitutions:
            new_substitution = self.evaluate_builtins_predicates(
                predicates, substitution
            )
            if new_substitution is not None:
                new_substitutions.append(new_substitution)
        return new_substitutions

    def evaluate_builtins_predicates(
        self, predicates_to_evaluate, substitution
    ):
        predicates_to_evaluate = predicates_to_evaluate.copy()
        unresolved_predicates = []
        while predicates_to_evaluate:
            predicate = predicates_to_evaluate.pop(0)

            subs = self.unify_builtin_substitution(predicate, substitution)
            if subs is None:
                unresolved_predicates.append(predicate)
            else:
                substitution = compose_substitutions(substitution, subs)
                predicates_to_evaluate += unresolved_predicates
                unresolved_predicates = []

        if len(unresolved_predicates) == 0:
            return substitution
        else:
            return None

    def unify_builtin_substitution(self, predicate, substitution):
        substituted_predicate = apply_substitution(predicate, substitution)
        evaluated_predicate = self.datalog_program.walk(substituted_predicate)
        if (
            isinstance(evaluated_predicate, Constant[bool]) and
            evaluated_predicate.value
        ):
            return substitution
        elif self.is_equality_between_constant_and_symbol(evaluated_predicate):
            return self.unify_builtin_substitution_equality(
                evaluated_predicate
            )
        else:
            return None

    @staticmethod
    def is_equality_between_constant_and_symbol(predicate):
        return (
            isinstance(predicate, FunctionApplication) and
            isinstance(predicate.functor, Constant) and
            predicate.functor.value is eq and
            any(isinstance(arg, Constant) for arg in predicate.args) and
            any(isinstance(arg, Symbol) for arg in predicate.args)
        )

    @staticmethod
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
        self, rule, instance, restriction_instance=None
    ):
        if restriction_instance is None:
            restriction_instance = dict()

        rule_predicates = extract_datalog_predicates(rule.antecedent)
        restricted_predicates = []
        nonrestricted_predicates = []
        builtin_predicates = []
        for predicate in rule_predicates:
            functor = predicate.functor
            if functor in restriction_instance:
                restricted_predicates.append(
                    (predicate, restriction_instance[functor])
                )
            elif functor in instance:
                nonrestricted_predicates.append(
                    (predicate, instance[functor])
                )
            elif functor in self.builtins:
                builtin_predicates.append((predicate, self.builtins[functor]))
            elif isinstance(functor, Constant):
                builtin_predicates.append((predicate, functor))
            else:
                return ([], [], [])

        return (
            restricted_predicates, nonrestricted_predicates, builtin_predicates
        )

    def compute_result_set(
        self, rule, substitutions, instance, restriction_instance=None
    ):
        if restriction_instance is None:
            restriction_instance = MapInstance()

        tuples = [
            tuple(
                a.value for a in
                apply_substitution_arguments(
                    rule.consequent.args, substitution
                )
            )
            for substitution in substitutions
            if len(substitutions) > 0
        ]
        new_tuples = self.datalog_program.new_set(tuples)
        instance_update = MapInstance({rule.consequent.functor: new_tuples})
        instance_update -= instance
        instance_update -= restriction_instance

        return instance_update

    def compute_instance_update(
        self, rule, new_tuples, instance, restriction_instance
    ):
        instance_update = MapInstance({rule.consequent.functor: new_tuples})
        instance_update -= instance
        instance_update -= restriction_instance
        return instance_update

    def merge_instances(self, *args):
        new_instance = args[0].copy()

        for next_instance in args:
            for k, v in next_instance.items():
                if k not in new_instance:
                    new_instance[k] = v
                else:
                    new_set = new_instance[k]
                    new_set = Constant[new_set.type](v.value | new_set.value)
                    new_instance[k] = new_set

        return new_instance

    def build_chase_tree(self, chase_set=chase_step):
        root = ChaseNode(
            self.datalog_program.extensional_database(), dict()
        )
        rules = []
        for disjunction in self.datalog_program.intensional_database(
        ).values():
            for rule in disjunction.formulas:
                rules.append(rule)

        nodes_to_process = [root]
        while len(nodes_to_process) > 0:
            node = nodes_to_process.pop(0)
            for rule in rules:
                new_node = self.build_nodes_from_rules(node, rule)
                if new_node is not None:
                    nodes_to_process.append(new_node)
        return root

    def build_nodes_from_rules(self, node, rule):
        instance_update = self.chase_step(node.instance, rule)
        if len(instance_update) > 0:
            new_instance = self.merge_instances(node.instance, instance_update)
            new_node = ChaseNode(new_instance, dict())
            node.children[rule] = new_node
            return new_node
        else:
            return None

    def eliminate_already_computed(self, consequent, instance, substitutions):
        return substitutions


class ChaseNaive:
    """Chase implementation using the naive algorithm.
    """

    def build_chase_solution(self):
        instance = MapInstance()
        edb = {
            k: v.value
            for k, v in self.datalog_program.extensional_database().items()
        }
        instance_update = MapInstance(edb)
        self.check_constraints(instance_update)
        while len(instance_update) > 0:
            instance |= instance_update
            new_update = MapInstance()
            for rule in self.rules:
                upd = self.chase_step(
                    instance, rule, restriction_instance=instance_update
                )
                new_update |= upd
            instance_update = new_update

        constant_instance = dict()
        for k, v in instance.items():
            if len(v) > 0:
                el = next(iter(v))
                if not isinstance(el, Constant):
                    el_type = infer_type(el)
                else:
                    el_type = el.type
                set_type = AbstractSet[el_type]
            else:
                set_type = AbstractSet
            value_set = Constant[set_type](v, verify_type=False)
            k = k.cast(set_type)
            constant_instance[k] = value_set
        return constant_instance


class ChaseSemiNaive:
    """Chase implementation using the semi-naive algorithm.
    This algorithm will not work if there are non-linear rules.
       """
    def build_chase_solution(self):
        instance = dict()
        instance_update = self.datalog_program.extensional_database()
        self.check_constraints(instance_update)
        continue_chase = len(instance_update) > 0
        while continue_chase:
            instance = self.merge_instances(instance, instance_update)
            instance_update = dict()
            continue_chase = False
            for rule in self.rules:
                instance_update = self.per_rule_update(
                    rule, instance, instance_update
                )
                continue_chase |= len(instance_update) > 0

        return instance

    def per_rule_update(self, rule, instance, instance_update):
        new_instance_update = self.chase_step(
            instance, rule, restriction_instance=instance_update
        )
        if len(new_instance_update) > 0:
            instance_update = self.merge_instances(
                instance_update,
                new_instance_update
            )
        return instance_update

    def check_constraints(self, instance_update):
        for rule in self.rules:
            if not is_linear_rule(rule):
                raise NeuroLangNonLinearProgramException(
                    f"Rule {rule} is non-linear. "
                    "Use a different resolution algorithm"
                )
