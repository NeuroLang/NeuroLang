from collections import defaultdict
from functools import lru_cache
from typing import AbstractSet, Callable, Sequence
from uuid import uuid1

from ...expressions import Constant, Definition, Symbol
from ...relational_algebra import (ColumnInt, Product, Projection,
                                   RelationalAlgebraOptimiser,
                                   RelationalAlgebraSolver, Selection, eq_)
from ...type_system import is_leq_informative
from ...unification import apply_substitution_arguments
from ...utils import NamedRelationalAlgebraFrozenSet
from ..expression_processing import (extract_datalog_free_variables,
                                     extract_datalog_predicates)
from ..expressions import Conjunction, Implication
from ..translate_to_named_ra import TranslateToNamedRA
from ..wrapped_collections import WrappedRelationalAlgebraSet


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


class ChaseNamedRelationalAlgebraMixin:
    """
    Conjunctive query solving using the algorithm 5.4.8 from Abiteboul et al
    [1]_ algorithm for named relational algebra.

    ..[1] S. Abiteboul, R. Hull, V. Vianu, Foundations of databases
      (Addison Wesley, 1995), Addison-Wesley.

    """
    def chase_step(self, instance, rule, restriction_instance=None):
        if restriction_instance is None:
            restriction_instance = dict()

        rule = self.rewrite_constants_in_consequent(rule)

        consequent = rule.consequent
        rule_predicates = self.extract_rule_predicates(
            rule, instance, restriction_instance=restriction_instance
        )

        if all(len(predicate_list) == 0 for predicate_list in rule_predicates):
            return dict()

        rule_predicates_iterator, builtin_predicates = rule_predicates

        substitutions = self.obtain_substitutions(
            rule_predicates_iterator, instance, restriction_instance
        )

        if consequent.functor in instance:
            substitutions = self.eliminate_already_computed(
                consequent, instance, substitutions
            )
        if consequent.functor in restriction_instance:
            substitutions = self.eliminate_already_computed(
                consequent, restriction_instance, substitutions
            )
        substitutions = self.evaluate_builtins(
            builtin_predicates, substitutions
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
                fresh = Symbol[arg.type](str(uuid1()).replace('-', '_'))
                new_equalities.append(eq_(fresh, arg))
                arg = fresh
            new_args += (arg,)
        if len(new_equalities) > 0:
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
        if len(consequent.args) > substitutions.arity:
            return substitutions

        args = tuple(
            arg.name for arg in consequent.args
            if isinstance(arg, Symbol)
        )
        already_computed = NamedRAFSTupleIterAdapter(
            args,
            instance[consequent.functor].value
        )
        if set(substitutions.columns).issuperset(already_computed.columns):
            already_computed = substitutions.naturaljoin(already_computed)
        substitutions = substitutions - already_computed
        if not isinstance(substitutions, NamedRAFSTupleIterAdapter):
            substitutions = (
                sorted(substitutions.columns),
                substitutions
            )
        return substitutions

    def obtain_substitutions(
        self, rule_predicates_iterator, instance, restriction_instance
    ):
        symbol_table = defaultdict(
            lambda: Constant[AbstractSet](WrappedRelationalAlgebraSet())
        )
        symbol_table.update(instance)
        symbol_table.update(restriction_instance)
        predicates = tuple(rule_predicates_iterator)

        if len(predicates) == 0:
            return [{}]

        ra_code = self.translate_conjunction_to_named_ra(
            Conjunction(predicates)
        )

        result = RelationalAlgebraSolver(symbol_table).walk(ra_code)

        result_value = result.value
        substitutions = NamedRAFSTupleIterAdapter(
            sorted(result_value.columns),
            result_value
        )

        return substitutions

    @lru_cache(1024)
    def translate_conjunction_to_named_ra(self, conjunction):
        traslator_to_named_ra = TranslateToNamedRA()
        return traslator_to_named_ra.walk(conjunction)

    def extract_rule_predicates(
        self, rule, instance, restriction_instance=None
    ):
        if restriction_instance is None:
            restriction_instance = dict()

        rule_predicates = extract_datalog_predicates(rule.antecedent)
        builtin_predicates, edb_idb_predicates, cq_free_vars = \
            self.split_predicates(rule_predicates)

        builtin_predicates = self.process_builtins(
            builtin_predicates, edb_idb_predicates, cq_free_vars
        )

        return edb_idb_predicates, builtin_predicates

    def split_predicates(self, rule_predicates):
        edb_idb_predicates = []
        builtin_predicates = []
        cq_free_vars = set()
        for predicate in rule_predicates:
            functor = predicate.functor
            if functor in self.idb_edb_symbols:
                edb_idb_predicates.append(predicate)
                cq_free_vars |= extract_datalog_free_variables(predicate)
            elif functor in self.builtins:
                builtin_predicates.append(
                    (predicate, self.builtins[functor])
                )
            elif isinstance(functor, Constant):
                builtin_predicates.append((predicate, functor))
            else:
                edb_idb_predicates = []
                builtin_predicates = []
                break
        return builtin_predicates, edb_idb_predicates, cq_free_vars

    def process_builtins(
        self, builtin_predicates,
        edb_idb_predicates, cq_free_vars
    ):
        new_builtin_predicates = []
        builtin_vectorized_predicates = []
        for pred, functor in builtin_predicates:
            if (
                ChaseNamedRelationalAlgebraMixin.
                is_eq_expressible_as_ra(functor, pred, cq_free_vars)
            ):
                edb_idb_predicates.append(pred)
            elif (
                isinstance(functor.type, Callable) and
                is_leq_informative(Sequence[bool], functor.type.__args__[-1])
            ):
                builtin_vectorized_predicates.append((pred, functor))
            else:
                new_builtin_predicates.append((pred, functor))
        builtin_predicates = new_builtin_predicates
        return builtin_predicates

    @staticmethod
    def is_eq_expressible_as_ra(functor, pred, cq_free_vars):
        return (
            functor == eq_ and
            not any(isinstance(arg, Definition) for arg in pred.args) and
            any(
                isinstance(arg, Constant) or arg in cq_free_vars
                for arg in pred.args
            )
        )

    def compute_result_set(
        self, rule, substitutions, instance, restriction_instance=None
    ):
        if restriction_instance is None:
            restriction_instance = dict()

        if isinstance(substitutions, NamedRelationalAlgebraFrozenSet):
            new_tuples = substitutions.projection(
                *(arg.name for arg in rule.consequent.args)
            )
            new_tuples = WrappedRelationalAlgebraSet(new_tuples.to_unnamed())
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


class NamedRAFSTupleIterAdapter(NamedRelationalAlgebraFrozenSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._row_types = dict()
        self.row_types

    @property
    def row_types(self):
        if (
            len(self._row_types) == 0 and
            self.arity > 0 and len(self) > 0
        ):
            element = next(super().__iter__())
            self._row_types = {
                c: Constant(getattr(element, c)).type
                for c in self.columns
            }

        return self._row_types

    def __iter__(self):
        if self.arity > 0:
            row_types = self.row_types
            for row in super().__iter__():
                yield {
                    f: Constant[row_types[f]](
                        v, verify_type=False
                    )
                    for f, v in zip(row._fields, row)
                }
        else:
            for _ in range(len(self)):
                yield dict()
