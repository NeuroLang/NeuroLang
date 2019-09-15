
from typing import AbstractSet

from ...expressions import Constant, Symbol
from ...relational_algebra import (Column, Product, Projection,
                                   RelationalAlgebraOptimiser,
                                   RelationalAlgebraSolver, Selection, eq_)
from ..expressions import Conjunction
from ..translate_to_named_ra import TranslateToNamedRA


class ChaseRelationalAlgebraMixin:
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
        for pred_ra in rule_predicates_iterator:
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


class ChaseNamedRelationalAlgebraMixin:
    def obtain_substitutions(self, args_to_project, rule_predicates_iterator):
        symbol_table = {}
        predicates = tuple()
        for predicate, set_ in rule_predicates_iterator:
            type_ = AbstractSet[set_.row_type]
            set_ = Constant[type_](set_, verify_type=False)
            symbol_table[predicate.functor] = set_
            predicates += (predicate,)

        if len(predicates) == 0:
            return [{}]

        traslator_to_named_ra = TranslateToNamedRA(symbol_table)
        ra_code = traslator_to_named_ra.walk(
            Conjunction(predicates)
        )
        result = RelationalAlgebraSolver().walk(ra_code)

        substitutions = self.compute_substitutions(result)

        return substitutions

    def compute_substitutions(self, result):
        substitutions = []
        for tuple_ in result.value:
            subs = {
                k: Constant(v)
                for k, v in tuple_._asdict().items()
            }
            substitutions.append(subs)
        return substitutions
