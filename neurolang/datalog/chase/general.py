from collections import namedtuple
from itertools import chain, tee
import logging
from operator import contains, eq
from typing import Iterable, Tuple

from ...exceptions import NeuroLangException
from ...expressions import Constant, FunctionApplication, Symbol
from ...logic.unification import (apply_substitution,
                                  apply_substitution_arguments,
                                  compose_substitutions)
from ...type_system import (NeuroLangTypeException, Unknown, get_args,
                            is_leq_informative, unify_types)
from ...utils import OrderedSet, log_performance
from ..expression_processing import (extract_logic_free_variables,
                                     extract_logic_predicates, is_linear_rule,
                                     dependency_matrix, program_has_loops)
from ..instance import MapInstance


LOG = logging.getLogger(__name__)


ChaseNode = namedtuple('ChaseNode', 'instance children')


class NeuroLangNonLinearProgramException(NeuroLangException):
    pass


class NeuroLangProgramHasLoopsException(NeuroLangException):
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
            for union in \
                    self.datalog_program.intensional_database().values():
                self.rules += union.formulas
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
        return extract_logic_free_variables(predicate)

    def evaluate_builtins(self, builtin_predicates, substitutions):
        new_substitutions = []
        predicates = [p for p, _ in builtin_predicates]
        if len(predicates) == 0:
            return substitutions
        for substitution in substitutions:
            updated_substitutions = self.evaluate_builtins_predicates(
                predicates, substitution
            )
            new_substitutions += updated_substitutions
        return new_substitutions

    def evaluate_builtins_predicates(
        self, predicates_to_evaluate, substitution
    ):
        substitutions = [substitution]
        predicates_to_evaluate = predicates_to_evaluate.copy()
        unresolved_predicates = []
        while predicates_to_evaluate:
            predicate = predicates_to_evaluate.pop(0)

            new_substitutions = []
            for sub in substitutions:
                new_substitutions += self.unify_builtin_substitution(
                    predicate, sub
                )

            if len(new_substitutions) == 0:
                unresolved_predicates.append(predicate)
            else:
                substitutions = self.compose_substitutions_no_conflict(
                    substitutions, new_substitutions
                )
                predicates_to_evaluate += unresolved_predicates
                unresolved_predicates = []

        if len(unresolved_predicates) == 0:
            return substitutions
        else:
            return []

    @staticmethod
    def compose_substitutions_no_conflict(substitutions, new_substitutions):
        composed_substitutions = []
        for substitution in substitutions:
            subs_keys = set(
                k for k, v in substitution.items()
                if not isinstance(v, Symbol)
            )
            composed_substitutions += ChaseGeneral.compose_all_subtitutions(
                new_substitutions,
                subs_keys,
                substitution
            )
        substitutions = composed_substitutions
        return substitutions

    @staticmethod
    def compose_all_subtitutions(new_substitutions, subs_keys, substitution):
        composed_subtitutions_ = []
        for new_substitution in new_substitutions:
            overlap = subs_keys & set(new_substitution)
            if any(
                not isinstance(new_substitution[k], Symbol) and
                new_substitution[k] != substitution[k]
                for k in overlap
            ):
                continue
            subs = compose_substitutions(
                substitution, new_substitution
            )
            composed_subtitutions_.append(subs)
        return composed_subtitutions_

    @staticmethod
    def compose_substitutions_ignoring_conflicts(
        new_substitutions,
        subs_keys,
        substitution
    ):
        for new_substitution in new_substitutions:
            overlap = subs_keys & set(new_substitution)
            if any(
                not isinstance(new_substitution[k], Symbol) and
                new_substitution[k] != substitution[k]
                for k in overlap
            ):
                continue
            subs = compose_substitutions(
                substitution, new_substitution
            )
        return subs

    def unify_builtin_substitution(self, predicate, substitution):
        substituted_predicate = apply_substitution(predicate, substitution)
        evaluated_predicate = self.datalog_program.walk(substituted_predicate)
        if (
            isinstance(evaluated_predicate, Constant[bool]) and
            evaluated_predicate.value
        ):
            return [substitution]
        elif self.is_equality_between_constant_and_symbol(evaluated_predicate):
            return [
                self.unify_builtin_substitution_equality(
                    evaluated_predicate
                )
            ]
        elif self.is_containment_of_symbol_in_constant(evaluated_predicate):
            return self.unify_builtin_substitution_containment(
                evaluated_predicate
            )
        else:
            return []

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
    def is_containment_of_symbol_in_constant(predicate):
        return (
            isinstance(predicate, FunctionApplication) and
            isinstance(predicate.functor, Constant) and
            predicate.functor.value is contains and
            isinstance(predicate.args[0], Constant[Iterable]) and
            isinstance(predicate.args[1], Symbol)
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

    @staticmethod
    def unify_builtin_substitution_containment(evaluated_predicate):
        set_value = evaluated_predicate.args[0].value
        symbol = evaluated_predicate.args[1]
        if len(set_value) == 0:
            return []
        elif isinstance(next(iter(set_value)), Constant):
            return [
                {symbol: v}
                for v in set_value
            ]
        else:
            el_type = ChaseGeneral.infer_iterable_subtype(evaluated_predicate)
            return [
                {
                    symbol:
                    Constant[el_type](
                        v,
                        auto_infer_type=False, verify_type=False
                    )
                }
                for v in set_value
            ]

    @staticmethod
    def infer_iterable_subtype(evaluated_predicate):
        iterable_subtype = evaluated_predicate.args[0].type
        type_args = get_args(iterable_subtype)
        el_type = type_args[0]
        if is_leq_informative(iterable_subtype, Tuple):
            for another_type in type_args[1:]:
                try:
                    el_type = unify_types(el_type, another_type)
                except NeuroLangTypeException:
                    el_type = Unknown
                    break
        return el_type

    def extract_rule_predicates(
        self, rule, instance, restriction_instance=None
    ):
        if restriction_instance is None:
            restriction_instance = dict()

        rule_predicates = extract_logic_predicates(rule.antecedent)
        restricted_predicates = []
        nonrestricted_predicates = []
        builtin_predicates = []
        for predicate in rule_predicates:
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

        row_type = None
        tuples = []
        for substitution in substitutions:
            tuple_, tuple_type = self.compute_new_tuple_and_type(
                rule,
                substitution
            )
            tuples.append(tuple_)
            row_type = self.aggregate_tuple_type_into_row_type(
                row_type,
                tuple_type
            )
        new_tuples = self.datalog_program.new_set(tuples, row_type=row_type)
        return self.compute_instance_update(
            rule, new_tuples, instance, restriction_instance
        )

    def aggregate_tuple_type_into_row_type(self, row_type, tuple_type):
        if row_type is None:
            row_type = tuple_type
        elif row_type is not tuple_type:
            row_type = unify_types(row_type, tuple_type)
        return row_type

    def compute_new_tuple_and_type(self, rule, substitution):
        tuple_ = tuple()
        type_ = tuple()
        for el in apply_substitution_arguments(
            rule.consequent.args, substitution
        ):
            tuple_ += (el.value,)
            type_ += (el.type,)
        tuple_type = Tuple[type_]
        return tuple_, tuple_type

    def compute_instance_update(
        self, rule, new_tuples, instance, restriction_instance
    ):
        instance_update = MapInstance({rule.consequent.functor: new_tuples})
        instance_update -= instance
        instance_update -= restriction_instance
        return instance_update

    def build_chase_tree(self, chase_set=chase_step):
        root = ChaseNode(
            MapInstance(self.datalog_program.extensional_database()),
            dict()
        )
        rules = []
        for union in self.datalog_program.intensional_database(
        ).values():
            for rule in union.formulas:
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
            new_instance = node.instance | instance_update
            new_node = ChaseNode(new_instance, dict())
            node.children[rule] = new_node
            return new_node
        else:
            return None

    def eliminate_already_computed(self, consequent, instance, substitutions):
        return substitutions

    def build_chase_solution(self):
        instance = MapInstance()
        instance_update = MapInstance(
            self.datalog_program.extensional_database()
        )
        self.check_constraints(instance_update)
        instance = self.execute_chase(self.rules, instance_update, instance)
        return instance


class ChaseNonRecursive:
    """Chase class for non-recursive programs.
    """

    def execute_chase(self, rules, instance_update, instance):
        rules_seen = set()
        while rules:
            rule = rules.pop(0)
            functor = rule.consequent.functor
            functor_ix = self._dependency_matrix_symbols.index(functor)
            if any(
                self._dependency_matrix_symbols[dep_index] not in rules_seen
                for dep_index in
                self._dependency_matrix[functor_ix].nonzero()[0]
            ):
                rules.append(rule)
                continue
            rules_seen.add(functor)

            with log_performance(LOG, 'Evaluating rule %s', (rule,)):
                instance_update |= self.chase_step(
                    instance, rule, restriction_instance=instance_update
                )
        return instance_update

    def check_constraints(self, instance_update):
        super().check_constraints(instance_update)
        self._dependency_matrix_symbols, self._dependency_matrix = \
            dependency_matrix(
                self.datalog_program, rules=self.rules
            )
        if program_has_loops(self._dependency_matrix):
            raise NeuroLangProgramHasLoopsException(
                "Use a different resolution algorithm"
            )


class ChaseNaive:
    """Chase implementation using the naive algorithm.
    """
    def execute_chase(self, rules, instance_update, instance):
        while len(instance_update) > 0:
            instance |= instance_update
            new_update = MapInstance()
            for rule in rules:
                with log_performance(LOG, 'Evaluating rule %s', (rule,)):
                    upd = self.chase_step(
                        instance | instance_update, rule,
                    )
                new_update |= upd
            instance_update = new_update
        return instance


class ChaseSemiNaive:
    """Chase implementation using the semi-naive algorithm.
    This algorithm will not work if there are non-linear rules.
       """

    def execute_chase(self, rules, instance_update, instance):
        continue_chase = len(instance_update) > 0
        while continue_chase:
            instance |= instance_update
            instance_update = MapInstance()
            continue_chase = False
            for rule in rules:
                with log_performance(LOG, 'Evaluating rule %s', (rule,)):
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
            instance_update |= new_instance_update
        return instance_update

    def check_constraints(self, instance_update):
        super().check_constraints(instance_update)
        for rule in self.rules:
            if not is_linear_rule(rule):
                raise NeuroLangNonLinearProgramException(
                    f"Rule {rule} is non-linear. "
                    "Use a different resolution algorithm"
                )
