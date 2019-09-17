from collections import defaultdict
from functools import lru_cache
from typing import AbstractSet

from ...expressions import Constant, Symbol
from ...relational_algebra import (Column, Product, Projection,
                                   RelationalAlgebraOptimiser,
                                   RelationalAlgebraSolver, Selection, eq_)
from ...utils import NamedRelationalAlgebraFrozenSet
from ..expressions import Conjunction
from ..translate_to_named_ra import TranslateToNamedRA


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
    """
    Conjunctive query solving using the algorithm 5.4.8 from Abiteboul et al
    [1]_ algorithm for named relational algebra.

    ..[1] S. Abiteboul, R. Hull, V. Vianu, Foundations of databases
      (Addison Wesley, 1995), Addison-Wesley.

    """
    def obtain_substitutions(self, args_to_project, rule_predicates_iterator):
        symbol_table = defaultdict(
            default_factory=lambda: NamedRAFSTupleIterAdapter([], set())
        )
        predicates = tuple()
        for predicate, set_ in rule_predicates_iterator:
            type_ = AbstractSet[set_.row_type]
            set_ = Constant[type_](set_, verify_type=False)
            symbol_table[predicate.functor] = set_
            predicates += (predicate,)

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


class NamedRAFSTupleIterAdapter(NamedRelationalAlgebraFrozenSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if len(self) > 0 and self.arity > 0:
            element = next(super().__iter__())
            self._row_types = {
                c: Constant(getattr(element, c)).type
                for c in self.columns
            }
        else:
            self._row_types = dict()

    @property
    def row_types(self):
        return self._row_types

    def __iter__(self):
        if self.arity > 0:
            row_types = self.row_types
            for row in super().__iter__():
                yield {
                    f: Constant[row_types[f]](v)
                    for f, v in zip(row._fields, row)
                }
        else:
            for _ in range(len(self)):
                yield dict()
