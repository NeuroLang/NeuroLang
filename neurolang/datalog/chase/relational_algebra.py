import logging
import operator
from collections import defaultdict
from functools import lru_cache
from typing import AbstractSet, Callable

from ...expressions import Constant, FunctionApplication, Symbol
from ...expression_walker import ExpressionWalker, ReplaceSymbolWalker
from ...logic.unification import apply_substitution_arguments
from ...relational_algebra import (ColumnInt, Product, Projection,
                                   RelationalAlgebraOptimiser,
                                   RelationalAlgebraPushInSelections,
                                   RelationalAlgebraSolver, Selection, eq_)
from ...type_system import Unknown, is_leq_informative
from ...utils import NamedRelationalAlgebraFrozenSet
from ..expression_processing import extract_logic_free_variables
from ..expressions import Conjunction, Implication
from ..instance import MapInstance
from ..translate_to_named_ra import TranslateToNamedRA
from ..wrapped_collections import (WrappedNamedRelationalAlgebraFrozenSet,
                                   WrappedRelationalAlgebraSet)


LOG = logging.getLogger(__name__)


invert = Constant(operator.invert)
eq = Constant(operator.eq)


class ChaseRelationalAlgebraPlusCeriMixin:
    """
    Conjunctive query solving using Ceri et al [1]_ algorithm for unnamed
    positive relational algebra.

    .. [1] S. Ceri, G. Gottlob, L. Lavazza, in Proceedings of the 12th
       International Conference on Very Large Data Bases
       (Morgan Kaufmann Publishers Inc.,
       San Francisco, CA, USA, 1986;
       http://dl.acm.org/citation.cfm?id=645913.671468), VLDB ’86, pp. 395–402.
    """
    def obtain_substitutions(self, args_to_project, rule_predicates_iterator):
        ra_code, projected_var_names = self.translate_to_ra_plus(
            args_to_project,
            rule_predicates_iterator
        )
        ra_code_opt = RelationalAlgebraOptimiser().walk(ra_code)
        if not isinstance(ra_code_opt, Constant) or len(ra_code_opt.value) > 0:
            LOG.info('About to execute RA query %s', ra_code)
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
            c = Constant[ColumnInt](ColumnInt(column + i))
            local_column = Constant[ColumnInt](ColumnInt(i))
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
            self.translate_predicate_process_argument_symbol(
                arg, global_column, args_to_project
            )

    def translate_predicate_process_argument_symbol(
        self, arg, global_column, args_to_project
    ):
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


class NamedRelationalAlgebraOptimiser(
    RelationalAlgebraPushInSelections,
    ExpressionWalker
):
    pass


class ChaseNamedRelationalAlgebraMixin:
    """
    Conjunctive query solving using the algorithm 5.4.8 from Abiteboul et al
    [1]_ algorithm for named relational algebra.

    ..[1] S. Abiteboul, R. Hull, V. Vianu, Foundations of databases
      (Addison Wesley, 1995), Addison-Wesley.

    """
    def chase_step(self, instance, rule, restriction_instance=None):
        if restriction_instance is None:
            restriction_instance = MapInstance()

        rule = self.rewrite_constants_in_consequent(rule)
        rule = self.rewrite_antecedent_equalities(rule)

        consequent = rule.consequent

        if isinstance(rule.antecedent, Conjunction):
            predicates = rule.antecedent.formulas
        else:
            predicates = (rule.antecedent,)

        substitutions = self.obtain_substitutions(
            predicates, instance, restriction_instance
        )

        if consequent.functor in instance:
            substitutions = self.eliminate_already_computed(
                consequent, instance, substitutions
            )
        if consequent.functor in restriction_instance:
            substitutions = self.eliminate_already_computed(
                consequent, restriction_instance, substitutions
            )

        return self.compute_result_set(
            rule, substitutions, instance, restriction_instance
        )

    @lru_cache(1024)
    def rewrite_constants_in_consequent(self, rule):
        new_equalities = []
        new_args = tuple()
        for arg in rule.consequent.args:
            if isinstance(arg, Constant):
                fresh = Symbol[arg.type].fresh()
                new_equalities.append(eq_(fresh, arg))
                arg = fresh
            new_args += (arg,)
        if len(new_equalities) > 0:
            rule = self.rewrite_rule_consequent_constants_to_equalities(
                rule, new_args, new_equalities
            )
        return rule

    @lru_cache(1024)
    def rewrite_antecedent_equalities(self, rule):
        if not isinstance(rule.antecedent, Conjunction):
            return rule

        free_variables = (
            extract_logic_free_variables(rule.antecedent) |
            extract_logic_free_variables(rule.consequent)
        )
        cont = True
        while cont:
            cont = False
            for pos, predicate in enumerate(rule.antecedent.formulas):
                if (
                    isinstance(predicate, FunctionApplication) and
                    predicate.functor == eq and
                    predicate.args[0] in free_variables and
                    predicate.args[1] in free_variables
                ):
                    equality = predicate
                    new_antecedent_preds = (
                        rule.antecedent.formulas[:pos] +
                        rule.antecedent.formulas[pos + 1:]
                    )
                    cont = len(new_antecedent_preds) > 1
                    new_antecedent = Conjunction(new_antecedent_preds)
                    rule = Implication(rule.consequent, new_antecedent)
                    rule = ReplaceSymbolWalker(
                        {equality.args[0]: equality.args[1]}
                    ).walk(rule)
        return rule

    @staticmethod
    def rewrite_rule_consequent_constants_to_equalities(
        rule, new_args, new_equalities
    ):
        consequent = rule.consequent.functor(*new_args)
        if isinstance(rule.antecedent, Conjunction):
            antecedent_formulas = rule.antecedent.formulas
        else:
            antecedent_formulas = (rule.antecedent,)
        antecedent = Conjunction(
            antecedent_formulas + tuple(new_equalities)
        )
        rule = Implication(consequent, antecedent)
        return rule

    def eliminate_already_computed(self, consequent, instance, substitutions):
        substitutions_columns = set(substitutions.columns)
        if substitutions_columns.isdisjoint(consequent.args):
            return substitutions

        args = []
        args_to_project = []
        for i, arg in enumerate(consequent.args):
            if not isinstance(arg, Symbol) or arg.name in args:
                continue
            args.append(arg.name)
            args_to_project.append(i)

        already_computed = NamedRelationalAlgebraFrozenSet(
            args,
            (
                instance[consequent.functor]
                .value.unwrap()
                .projection(*args_to_project)
            )
        )
        if substitutions_columns.symmetric_difference(args):
            already_computed = (
                already_computed
                .naturaljoin(substitutions)
                .projection(*substitutions.columns)
            )
        substitutions_unwrapped = substitutions.unwrap()
        substitutions_unwrapped = substitutions_unwrapped - already_computed

        res = WrappedNamedRelationalAlgebraFrozenSet(
            substitutions_unwrapped.columns,
            substitutions_unwrapped,
            row_type=substitutions.row_type,
            verify_row_type=False
        )

        return res

    def obtain_substitutions(
        self, rule_predicates_iterator, instance, restriction_instance
    ):
        symbol_table = defaultdict(
            lambda: Constant[AbstractSet](WrappedRelationalAlgebraSet())
        )
        symbol_table.update(instance)
        for k, v in restriction_instance.items():
            if k in symbol_table:
                s = symbol_table[k]
                v = v.apply(s.value | v.value)
            symbol_table[k] = v
        predicates = tuple(rule_predicates_iterator)

        if len(predicates) == 0:
            return [{}]

        ra_code = self.translate_conjunction_to_named_ra(
            Conjunction(predicates)
        )

        LOG.info('About to execute RA query %s', ra_code)
        result = RelationalAlgebraSolver(symbol_table).walk(ra_code)

        result_value = result.value
        substitutions = WrappedNamedRelationalAlgebraFrozenSet(
            result_value.columns,
            result_value
        )

        return substitutions

    @lru_cache(1024)
    def translate_conjunction_to_named_ra(self, conjunction):
        builtin_symbols = {
            k: v
            for k, v in self.datalog_program.symbol_table.items()
            if (
                v.type is not Unknown and
                is_leq_informative(v.type, Callable)
            )
        }
        rsw = ReplaceSymbolWalker(builtin_symbols)
        conjunction = rsw.walk(conjunction)
        traslator_to_named_ra = TranslateToNamedRA()
        LOG.info(f"Translating and optimising CQ {conjunction} to RA")
        ra_code = traslator_to_named_ra.walk(conjunction)
        ra_code = NamedRelationalAlgebraOptimiser().walk(ra_code)
        return ra_code

    def compute_result_set(
        self, rule, substitutions, instance, restriction_instance=None
    ):
        if restriction_instance is None:
            restriction_instance = MapInstance

        if isinstance(substitutions, NamedRelationalAlgebraFrozenSet):
            new_tuples = substitutions.projection_to_unnamed(
                *(
                    arg.name for arg in rule.consequent.args
                    if arg in substitutions.columns
                )
            )
            new_tuples = WrappedRelationalAlgebraSet(new_tuples)
        else:
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

        return self.compute_instance_update(
            rule, new_tuples, instance, restriction_instance
        )
