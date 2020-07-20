"""
    Implementation of the weighted model counting approach through knowledge
    compilation.
"""
import logging
import operator as op

import numpy as np
import pandas as pd
from pysdd import sdd

from ..datalog.expression_processing import (extract_logic_predicates,
                                             flatten_query)
from ..datalog.translate_to_named_ra import TranslateToNamedRA
from ..expression_walker import ExpressionWalker, PatternWalker, add_match
from ..expressions import (Constant, ExpressionBlock, FunctionApplication,
                           Symbol, sure_is_not_pattern)
from ..logic import Implication
from ..relational_algebra import (ColumnInt, ColumnStr, ExtendedProjection,
                                  ExtendedProjectionListMember, NameColumns,
                                  Projection, RelationalAlgebraOperation,
                                  RelationalAlgebraPushInSelections,
                                  RelationalAlgebraStringExpression,
                                  RenameColumns, str2columnstr_constant)
from ..relational_algebra_provenance import (
    ProvenanceAlgebraSet, RelationalAlgebraProvenanceExpressionSemringSolver)
from ..utils.relational_algebra_set import (RelationalAlgebraColumnInt,
                                            RelationalAlgebraColumnStr)

LOG = logging.getLogger(__name__)


class SemiRingRAPToSDD(PatternWalker):
    def __init__(self, var_count, symbols_to_literals=None):
        self._init_manager(var_count)
        if symbols_to_literals is None:
            self.symbols_to_literals = dict()
        else:
            self.symbols_to_literals = symbols_to_literals
        self.literals_to_symbols = dict()

    def _init_manager(self, var_count):
        if isinstance(var_count, sdd.SddManager):
            self.manager = var_count.copy(list(var_count.vars))
        if isinstance(var_count, sdd.Vtree):
            self.manager = sdd.SddManager.from_vtree(var_count)
        else:
            self.manager = sdd.SddManager(var_count=var_count)

    @add_match(Symbol)
    def symbol(self, expression):
        if expression not in self.symbols_to_literals:
            var_num = len(self.symbols_to_literals) + 1
            self.symbols_to_literals[expression] = var_num
            self.literals_to_symbols[var_num] = expression

        lit = self.symbols_to_literals[expression]
        sdd_literal = self.manager.literal(lit)
        return sdd_literal

    @add_match(FunctionApplication(Constant(op.mul), ...))
    def mul(self, expression):
        args = self.walk(expression.args)
        exp = args[0]
        for arg in args[1:]:
            exp = exp & arg
        return exp

    @add_match(FunctionApplication(Constant(op.add), ...))
    def add(self, expression):
        args = self.walk(expression.args)
        exp = args[0]
        for arg in args[1:]:
            exp = exp | arg
        return exp

    @add_match(FunctionApplication(Constant(op.neg), ...))
    def neg(self, expression):
        arg = self.walk(expression.args[0])
        return ~arg

    @add_match(FunctionApplication(Constant(op.eq), ...))
    def eq(self, expression):
        args = self.walk(expression.args)
        return args[0].equiv(args[1])

    @add_match(FunctionApplication(Constant(op.or_), ...))
    def or_(self, expression):
        args = self.walk(expression.args)
        ret = args[0].condition(args[1])
        return ret

    @add_match(ExpressionBlock)
    def expression_block(self, expression):
        res = tuple(
            exp
            for exp in self.walk(expression.expressions)
        )
        for s in res:
            s.ref()
        return res


class ProbabilisticFactSet(RelationalAlgebraOperation):
    def __init__(self, relation, probability_column):
        self.relation = relation
        self.probability_column = probability_column


class ProbabilisticChoiceSet(RelationalAlgebraOperation):
    def __init__(self, relation, probability_column):
        self.relation = relation
        self.probability_column = probability_column


class DeterministicFactSet(RelationalAlgebraOperation):
    def __init__(self, relation):
        self.relation = relation


class WMCSemiRingSolver(RelationalAlgebraProvenanceExpressionSemringSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.translated_probfact_sets = dict()
        self.tagged_sets = []

    @add_match(DeterministicFactSet(Symbol))
    def deterministic_fact_set(self, deterministic_set):
        relation_symbol = deterministic_set.relation
        if relation_symbol in self.translated_probfact_sets:
            return self.translated_probfact_sets[relation_symbol]

        relation = self.walk(relation_symbol)
        tagged_relation, rap_column = self._add_tag_column(relation)

        self.tagged_sets.append(self.walk(
            ExtendedProjection(
                tagged_relation,
                [
                    ExtendedProjectionListMember(
                        rap_column, str2columnstr_constant('id')
                    ),
                    ExtendedProjectionListMember(
                        Constant[float](1.), str2columnstr_constant('prob')
                    )
                ]
            )
        ))

        prov_set = self._generate_provenance_set(
            tagged_relation, Constant[int](-1), rap_column
        )

        self.translated_probfact_sets[relation_symbol] = prov_set
        return prov_set

    @add_match(ProbabilisticFactSet(Symbol, Constant))
    def probabilistic_fact_set(self, prob_fact_set):
        relation_symbol = prob_fact_set.relation
        if relation_symbol in self.translated_probfact_sets:
            return self.translated_probfact_sets[relation_symbol]

        relation = self.walk(relation_symbol)
        tagged_relation, rap_column = self._add_tag_column(relation)

        self._generate_tag_probability_set(
            rap_column, prob_fact_set, tagged_relation
        )

        prov_set = self._generate_provenance_set(
            tagged_relation, prob_fact_set.probability_column, rap_column
        )

        self.translated_probfact_sets[relation_symbol] = prov_set
        return prov_set

    @add_match(ProbabilisticChoiceSet(Symbol, Constant))
    def probabilistic_choice_set(self, prob_fact_set):
        relation_symbol = prob_fact_set.relation
        if relation_symbol in self.translated_probfact_sets:
            return self.translated_probfact_sets[relation_symbol]

        relation = self.walk(relation_symbol)
        tagged_relation, rap_column = self._add_tag_column(relation)

        self._generate_choice_tag_probability_set(
            rap_column, prob_fact_set, tagged_relation
        )

        prov_set = self._generate_choice_provenance_set(
            tagged_relation, prob_fact_set.probability_column, rap_column
        )

        self.translated_probfact_sets[relation_symbol] = prov_set
        return prov_set

    def _generate_tag_probability_set(
        self, rap_column, prob_fact_set, tagged_relation
    ):
        columns_for_tagged_set = (
            rap_column,
            Constant[ColumnInt](ColumnInt(
                prob_fact_set.probability_column.value
            ))
        )

        self.tagged_sets.append(self.walk(
            ExtendedProjection(
                tagged_relation,
                [
                    ExtendedProjectionListMember(
                        rap_column, str2columnstr_constant('id')
                    ),
                    ExtendedProjectionListMember(
                        columns_for_tagged_set[1],
                        str2columnstr_constant('prob')
                    ),
                ]
            )
        ))

    def _generate_choice_tag_probability_set(
        self, rap_column, prob_fact_set, tagged_relation
    ):
        columns_for_tagged_set = (
            rap_column,
            Constant[ColumnInt](ColumnInt(
                prob_fact_set.probability_column.value
            ))
        )

        self.tagged_sets.append(self.walk(
            ExtendedProjection(
                tagged_relation,
                [
                    ExtendedProjectionListMember(
                        rap_column, str2columnstr_constant('id')
                    ),
                    ExtendedProjectionListMember(
                        columns_for_tagged_set[1],
                        str2columnstr_constant('prob')
                    ),
                    ExtendedProjectionListMember(
                        Constant[float](1.),
                        str2columnstr_constant('nprob')
                    ),

                ]
            )
        ))

    def _generate_provenance_set(
        self, tagged_relation,
        prob_column, rap_column
    ):
        out_columns = tuple(
            Constant[ColumnInt](ColumnInt(c))
            for c in tagged_relation.value.columns
            if (
                c != rap_column and
                int(c) != prob_column.value
            )

        ) + (rap_column,)

        prov_set = (
            RenameColumns(
                Projection(tagged_relation, out_columns),
                tuple(
                    (c, Constant[ColumnInt](ColumnInt(i)))
                    for i, c in enumerate(out_columns[:-1])
                )
            )
        )
        prov_set = ProvenanceAlgebraSet(
            self.walk(prov_set).value,
            rap_column.value
        )
        return prov_set

    def _generate_choice_provenance_set(
        self, tagged_relation,
        prob_column, rap_column
    ):
        out_columns = tuple(
            Constant[ColumnInt](ColumnInt(c))
            for c in tagged_relation.value.columns
            if (
                c != rap_column and
                int(c) != prob_column.value
            )
        )

        symbols = tagged_relation.value.as_pandas_dataframe()[rap_column.value]
        add = Constant(op.add)
        with sure_is_not_pattern():
            all_ = FunctionApplication(
                add,
                tuple(symbols.values)
            )

        def get_mutually_exclusive_formula_(v):
            with sure_is_not_pattern():
                ret = (v * (-(all_ | -v)))
            return ret

        get_mutually_exclusive_formula = Constant(
            get_mutually_exclusive_formula_
        )

        prov_set = ExtendedProjection(
            tagged_relation,
            tuple(
                ExtendedProjectionListMember(
                    c,
                    Constant[ColumnInt](ColumnInt(i))
                )
                for i, c in enumerate(out_columns)
            ) + (
                ExtendedProjectionListMember(
                    get_mutually_exclusive_formula(rap_column),
                    rap_column
                ),
            )
        )

        prov_set = ProvenanceAlgebraSet(
            self.walk(prov_set).value,
            rap_column.value
        )
        return prov_set

    def _add_tag_column(self, relation):
        new_columns = tuple(
            str2columnstr_constant(str(i))
            for i in relation.value.columns
        )

        relation = NameColumns(relation, new_columns)

        def get_fresh_symbol():
            return Symbol.fresh()
        get_fresh_symbol_functor = Constant(get_fresh_symbol)

        rap_column = str2columnstr_constant(Symbol.fresh().name)
        projection_list = [
            ExtendedProjectionListMember(
                Constant[RelationalAlgebraStringExpression](
                    RelationalAlgebraStringExpression(c.value),
                    verify_type=False
                ),
                c
            )
            for c in new_columns
        ] + [
            ExtendedProjectionListMember(
                FunctionApplication(
                    get_fresh_symbol_functor,
                    tuple()
                ),
                rap_column
            )
        ]
        expression = RenameColumns(
            ExtendedProjection(
                relation, projection_list
            ),
            tuple(
                (c, Constant[ColumnInt](ColumnInt(c.value)))
                for i, c in enumerate(new_columns)
            )
        )

        res = self.walk(expression)
        return res, rap_column

    @add_match(ProbabilisticFactSet)
    def probabilistic_fact_set_invalid(self, prob_fact_set):
        raise NotImplementedError()


def generate_weights(symbol_probs, literals_to_symbols, extras=0):
    ixs = pd.Series(literals_to_symbols)
    probs = symbol_probs.loc[ixs, 'prob'].values
    nprobs = symbol_probs.loc[ixs, 'nprob'].values
    probs = np.r_[nprobs[::-1], [0] * extras, [1] * extras, probs]
    return probs


class RAQueryOptimiser(
    RelationalAlgebraPushInSelections,
    ExpressionWalker
):
    pass


def solve_succ_query(query_predicate, cpl_program):
    """
    Obtain the solution of a SUCC query on a CP-Logic program.

    The SUCC query must take the form

        SUCC[ P(x) ]
    """
    if isinstance(query_predicate, Implication):
        query_antecedent = flatten_query(
            query_predicate.antecedent, cpl_program
        )
        ra_query = TranslateToNamedRA().walk(query_antecedent)
        ra_query = Projection(
            ra_query,
            tuple(
                str2columnstr_constant(s.name)
                for s in query_predicate.consequent.args
                if isinstance(s, Symbol)
            )
        )
    else:
        query_predicate_orig = query_predicate
        query_predicate = flatten_query(query_predicate, cpl_program)
        ra_query = TranslateToNamedRA().walk(query_predicate)
        ra_query = Projection(
            ra_query,
            tuple(
                str2columnstr_constant(s.name)
                for s in query_predicate_orig._symbols
                if s not in (
                    p.functor for p in
                    extract_logic_predicates(query_predicate_orig)
                )
            )
        )

    ra_query = RAQueryOptimiser().walk(ra_query)

    symbol_table = _generate_symbol_table(cpl_program, query_predicate)

    solver = WMCSemiRingSolver(symbol_table)
    prob_set_result = solver.walk(ra_query)

    sdd_compiler, sdd_program, prob_set_program = \
        sdd_compilation(prob_set_result)

    res, provenance_column = perform_wmc(
        solver, sdd_compiler, sdd_program,
        prob_set_program, prob_set_result
    )

    return ProvenanceAlgebraSet(
        res, ColumnStr(provenance_column)
    )


def _generate_symbol_table(cpl_program, query_predicate):
    symbol_table = dict()
    for predicate_symbol, facts in cpl_program.probabilistic_facts().items():
        if predicate_symbol not in query_predicate._symbols:
            continue

        fresh_symbol = Symbol.fresh()
        symbol_table[predicate_symbol] = ProbabilisticFactSet(
            fresh_symbol,
            Constant[ColumnInt](ColumnInt(0))
        )
        symbol_table[fresh_symbol] = facts

    for predicate_symbol, facts in cpl_program.probabilistic_choices().items():
        if predicate_symbol not in query_predicate._symbols:
            continue

        fresh_symbol = Symbol.fresh()
        symbol_table[predicate_symbol] = ProbabilisticChoiceSet(
            fresh_symbol,
            Constant[ColumnInt](ColumnInt(0))
        )
        symbol_table[fresh_symbol] = facts

    for predicate_symbol, facts in cpl_program.extensional_database().items():
        if predicate_symbol not in query_predicate._symbols:
            continue

        fresh_symbol = Symbol.fresh()
        symbol_table[predicate_symbol] = DeterministicFactSet(
            fresh_symbol
        )
        symbol_table[fresh_symbol] = facts

    return symbol_table


def sdd_compilation(prob_set_result):
    prob_set_program = ExpressionBlock(
        tuple(
            t[0]
            for t in (
                prob_set_result
                .relations
                .projection(prob_set_result.provenance_column)
            )
        )
    )
    sdd_compiler = SemiRingRAPToSDD(len(prob_set_program._symbols))
    sdd_program = sdd_compiler.walk(prob_set_program)
    return sdd_compiler, sdd_program, prob_set_program


def perform_wmc(
    solver, sdd_compiler, sdd_program,
    prob_set_program, prob_set_result
):
    ids_with_probs = generate_probability_table(solver)

    weights = generate_weights(
        ids_with_probs,
        sdd_compiler.literals_to_symbols
    )

    probs = dict()
    for i, sdd_exp in enumerate(sdd_program):
        wmc = sdd_exp.wmc(log_mode=True)
        wmc.set_literal_weights_from_array(np.log(weights))
        probs[prob_set_program.expressions[i]] = np.exp(wmc.propagate())

    provenance_column = 'prob'
    while provenance_column in prob_set_result.value.columns:
        provenance_column += '_'

    res = prob_set_result.value.extended_projection(
        dict(
            [
                (c, RelationalAlgebraStringExpression(c))
                for c in prob_set_result.value.columns
                if c != prob_set_result.provenance_column
            ] +
            [(
                provenance_column,
                lambda x: probs[x[prob_set_result.provenance_column]]
            )]
        )
    )
    return res, provenance_column


def generate_probability_table(solver):
    ids_with_probs = []
    for ts in solver.tagged_sets:
        ts = ts.value.as_pandas_dataframe()
        if 'nprob' not in ts:
            ts = ts.eval('nprob = 1 - prob')
        ids_with_probs.append(ts)
    ids_with_probs = pd.concat(ids_with_probs, axis=0).set_index('id')
    return ids_with_probs
