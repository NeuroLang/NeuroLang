from collections import namedtuple
from itertools import chain, tee
from operator import eq
from typing import AbstractSet

from ..expressions import Constant, Symbol, FunctionApplication
from ..unification import (
    apply_substitution, apply_substitution_arguments, compose_substitutions,
    most_general_unifier_arguments
)
from ..relational_algebra import (
    Column, Selection, Product, Projection, eq_,
    RelationalAlgebraOptimiser, RelationalAlgebraSolver
)

from . import extract_datalog_predicates


ChaseNode = namedtuple('ChaseNode', 'instance children')


class DatalogChaseGeneral():
    def __init__(self, datalog_program, rules=None):
        self.datalog_program = datalog_program
        self._set_rules(rules)

        self.builtins = datalog_program.builtins()

    def _set_rules(self, rules):
        self.rules = []
        if rules is None:
            for expression_block in \
                    self.datalog_program.intensional_database().values():
                self.rules += expression_block.expressions
        else:
            self.rules += rules.expressions

    def build_chase_solution(self):
        instance = dict()
        instance_update = self.datalog_program.extensional_database()
        self.check_constraints(instance_update)
        while len(instance_update) > 0:
            instance = self.merge_instances(instance, instance_update)
            instance_update = self.merge_instances(
                *(
                    self.chase_step(
                        instance, rule, restriction_instance=instance_update
                    ) for rule in self.rules
                )
            )

        return instance

    def check_constraints(self, instance_update):
        pass

    def chase_step(self, instance, rule, restriction_instance=None):
        if restriction_instance is None:
            restriction_instance = dict()

        rule_predicates = self.extract_rule_predicates(
            rule, instance, restriction_instance=restriction_instance
        )

        if all(len(predicate_list) == 0 for predicate_list in rule_predicates):
            return dict()

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

        substitutions = self.evaluate_builtins(
            builtin_predicates, substitutions
        )

        return self.compute_result_set(
            rule, substitutions, instance, restriction_instance
        )

    def get_args_to_project(self, rule, builtin_predicates_):
        args_to_project = self.extract_variable_arguments(rule.consequent)
        for predicate, _ in builtin_predicates_:
            args_to_project += self.extract_variable_arguments(predicate)
        new_args_to_project = tuple()
        for i, a in enumerate(args_to_project):
            if a not in args_to_project[:i]:
                new_args_to_project += (a,)
        args_to_project = new_args_to_project
        return args_to_project

    @staticmethod
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

    def evaluate_builtins(self, builtin_predicates, substitutions):
        new_substitutions = []
        predicates = [p for p, _ in builtin_predicates]
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

        head_functor = rule.consequent.functor
        rule_predicates = extract_datalog_predicates(rule.antecedent)
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
            restriction_instance = dict()

        new_tuples = self.datalog_program.new_set(
            Constant(
                apply_substitution_arguments(
                    rule.consequent.args, substitution
                )
            ) for substitution in substitutions
        )

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
        for expression_block in self.datalog_program.intensional_database(
        ).values():
            for rule in expression_block.expressions:
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


class DatalogChaseRelationalAlgebraMixin:
    def obtain_substitutions(self, args_to_project, rule_predicates_iterator):
        ra_code, projected_var_names = self.translate_to_ra_plus(
            args_to_project,
            rule_predicates_iterator
        )
        ra_code_opt = RelationalAlgebraOptimiser().walk(ra_code)
        if not isinstance(ra_code_opt, Constant) or len(ra_code_opt.value) > 0:
            result = RelationalAlgebraSolver().walk(ra_code_opt)
        else:
            return [{}]

        substitutions = self.compute_substitutions(result, projected_var_names)

        return substitutions

    def translate_to_ra_plus(
        self,
        args_to_project,
        rule_predicates_iterator
    ):
        self.seen_vars = dict()
        self.selections = []
        self.projections = tuple()
        self.projected_var_names = dict()
        column = 0
        new_ra_expressions = tuple()
        rule_predicates_iterator = list(rule_predicates_iterator)
        for _, pred_ra in enumerate(rule_predicates_iterator):
            ra_expression_arity = pred_ra[1].arity
            new_ra_expression = self.translate_predicate(
                pred_ra, column, args_to_project
            )
            new_ra_expressions += (new_ra_expression,)
            column += ra_expression_arity
        if len(new_ra_expressions) > 0:
            if len(new_ra_expressions) == 1:
                relation = new_ra_expressions[0]
            else:
                relation = Product(new_ra_expressions)
            for s1, s2 in self.selections:
                relation = Selection(relation, eq_(s1, s2))
            relation = Projection(relation, self.projections)
        else:
            relation = Constant[AbstractSet](self.datalog_program.new_set())
        projected_var_names = self.projected_var_names
        del self.seen_vars
        del self.selections
        del self.projections
        del self.projected_var_names
        return relation, projected_var_names

    def translate_predicate(self, pred_ra, column, args_to_project):
        predicate, ra_expression = pred_ra
        local_selections = []
        for i, arg in enumerate(predicate.args):
            c = Constant[Column](Column(column + i))
            local_column = Constant[Column](Column(i))
            self.translate_predicate_process_argument(
                arg, local_selections, local_column, c, args_to_project
            )
        new_ra_expression = Constant[AbstractSet](ra_expression)
        for s1, s2 in local_selections:
            new_ra_expression = Selection(new_ra_expression, eq_(s1, s2))
        return new_ra_expression

    def translate_predicate_process_argument(
        self, arg, local_selections, local_column,
        global_column, args_to_project
    ):
        if isinstance(arg, Constant):
            local_selections.append((local_column, arg))
        elif isinstance(arg, Symbol):
            if arg in self.seen_vars:
                self.selections.append((self.seen_vars[arg], global_column))
            else:
                if arg in args_to_project:
                    self.projected_var_names[arg] = len(self.projections)
                    self.projections += (global_column,)
                self.seen_vars[arg] = global_column

    def compute_substitutions(self, result, projected_var_names):
        substitutions = []
        for tuple_ in result.value:
            subs = {
                var: tuple_.value[col]
                for var, col in projected_var_names.items()
            }
            substitutions.append(subs)
        return substitutions


class DatalogChaseMGUMixin:
    @staticmethod
    def obtain_substitutions(args_to_project, rule_predicates_iterator):
        substitutions = [{}]
        for predicate, representation in rule_predicates_iterator:
            new_substitutions = []
            for substitution in substitutions:
                new_substitutions += DatalogChaseMGUMixin.unify_substitution(
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


class DatalogChase(DatalogChaseGeneral, DatalogChaseRelationalAlgebraMixin):
    pass
