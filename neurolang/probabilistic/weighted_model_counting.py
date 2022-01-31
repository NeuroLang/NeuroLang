"""
    Implementation of the weighted model counting approach through knowledge
    compilation.
"""
import logging
import operator as op
from typing import AbstractSet

import numpy as np
import pandas as pd
from pysdd import sdd

from ..config import config
from ..datalog.expression_processing import (
    extract_logic_predicates,
    flatten_query
)
from ..datalog.translate_to_named_ra import TranslateToNamedRA
from ..exceptions import NeuroLangException
from ..expression_walker import ExpressionWalker, PatternWalker, add_match
from ..expressions import (
    Constant,
    ExpressionBlock,
    FunctionApplication,
    Symbol,
    sure_is_not_pattern
)
from ..logic import FALSE, Implication
from ..relational_algebra import (
    ColumnInt,
    ColumnStr,
    ExtendedProjection,
    FunctionApplicationListMember,
    NameColumns,
    NumberColumns,
    Projection,
    PushInSelections,
    RelationalAlgebraStringExpression,
    RenameOptimizations,
    int2columnint_constant,
    str2columnstr_constant
)
from ..relational_algebra_provenance import (
    ProvenanceAlgebraSet,
    RelationalAlgebraProvenanceExpressionSemringSolver
)
from ..utils import log_performance
from ..utils.relational_algebra_set import NamedRelationalAlgebraFrozenSet
from .expression_processing import lift_optimization_for_choice_predicates
from .probabilistic_ra_utils import (
    DeterministicFactSet,
    ProbabilisticChoiceSet,
    ProbabilisticFactSet,
    generate_probabilistic_symbol_table_for_query
)
from .query_resolution import lift_solve_marg_query as _solve_marg_query

LOG = logging.getLogger(__name__)

ADD = Constant(op.add)
MUL = Constant(op.mul)
NEG = Constant(op.neg)


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
        args = list(expression.args)
        exp = self.manager.true()
        while len(args) > 0:
            arg = args.pop()
            if (
                isinstance(arg, FunctionApplication) and
                arg.functor == MUL
            ):
                args += arg.args
            else:
                arg = self.walk(arg)
                exp = exp & arg
        return exp

    @add_match(FunctionApplication(Constant(op.add), ...))
    def add(self, expression):
        args = self.walk(expression.args)
        args = list(expression.args)
        exp = self.manager.false()
        while len(args) > 0:
            arg = args.pop()
            if (
                isinstance(arg, FunctionApplication) and
                arg.functor == ADD
            ):
                args += arg.args
            else:
                arg = self.walk(arg)
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


class EliminateSuperfluousProjectionMixin(PatternWalker):
    @add_match(
        Projection,
        lambda exp: (
            isinstance(
                exp.relation,
                (
                    DeterministicFactSet,
                    ProbabilisticFactSet,
                    ProbabilisticChoiceSet
                )
            )
            and all(att.type is ColumnInt for att in exp.attributes)
        )
    )
    def eliminate_superfluous_projection(self, expression):
        relation = self.walk(expression.relation)
        return relation


class DeterministicFactSetTranslation(PatternWalker):
    @add_match(
        DeterministicFactSet(Constant),
        lambda e: e.relation.value.is_empty()
    )
    def deterministic_fact_set_constant(self, deterministic_set):
        return ProvenanceAlgebraSet(
            deterministic_set.relation,
            int2columnint_constant(0)
        )


class WMCSemiRingSolver(
    EliminateSuperfluousProjectionMixin,
    DeterministicFactSetTranslation,
    RelationalAlgebraProvenanceExpressionSemringSolver,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.translated_probfact_sets = dict()
        self.tagged_sets = []

    def _semiring_mul(self, left, right):
        return FunctionApplication(Constant(self._internal_mul), (left, right))

    @staticmethod
    def _internal_mul(left, right):
        r = left * right
        return r

    @add_match(DeterministicFactSet(Symbol))
    def deterministic_fact_set(self, deterministic_set):
        relation_symbol = deterministic_set.relation
        if relation_symbol in self.translated_probfact_sets:
            return self.translated_probfact_sets[relation_symbol]

        with log_performance(
            LOG, f"Build deterministic fact set {relation_symbol}"
        ):
            relation = self.walk(relation_symbol)
            df = relation.value.as_pandas_dataframe()
            tags = [Symbol.fresh() for _ in range(len(df))]
            probs = [1.0 for _ in range(len(df))]
            tag_set = list(zip(tags, probs))
            prob_column = Symbol.fresh().name
            return self._build_tag_and_prov_sets(
                relation, relation_symbol, df, prob_column, tag_set, tags
            )

    @add_match(ProbabilisticFactSet(Symbol, Constant))
    def probabilistic_fact_set(self, prob_fact_set):
        relation_symbol = prob_fact_set.relation
        prob_column = prob_fact_set.probability_column.value
        if relation_symbol in self.translated_probfact_sets:
            return self.translated_probfact_sets[relation_symbol]

        with log_performance(
            LOG, f"Build probabilistic fact set {relation_symbol}"
        ):
            relation = self.walk(relation_symbol)
            df = relation.value.as_pandas_dataframe()
            tags = [Symbol.fresh() for _ in range(len(df))]
            probs = df[prob_column]
            tag_set = list(zip(tags, probs))
            return self._build_tag_and_prov_sets(
                relation, relation_symbol, df, prob_column, tag_set, tags
            )

    @add_match(ProbabilisticChoiceSet(Symbol, Constant))
    def probabilistic_choice_set(self, prob_choice_set):
        relation_symbol = prob_choice_set.relation
        prob_column = prob_choice_set.probability_column.value
        if relation_symbol in self.translated_probfact_sets:
            return self.translated_probfact_sets[relation_symbol]

        with log_performance(
            LOG, f"Build Probabilistic choice set {relation_symbol}"
        ):
            relation = self.walk(relation_symbol)
            df = relation.value.as_pandas_dataframe()
            previous = tuple()
            previous_probability = 0
            prov_column = []
            tag_set = []
            with sure_is_not_pattern():
                for probability in df[prob_column]:
                    adjusted_probability = (
                        probability / (1 - previous_probability)
                    )
                    previous_probability += probability
                    previous = self._generate_tag_choice_expression(
                        previous, adjusted_probability, prov_column, tag_set
                    )

            return self._build_tag_and_prov_sets(
                relation,
                relation_symbol,
                df,
                prob_column,
                tag_set,
                prov_column,
            )

    def _build_tag_and_prov_sets(
        self, relation, relation_symbol, df, prob_column, tag_set, prov_column
    ):
        tag_set = self._build_relation_constant(
            NamedRelationalAlgebraFrozenSet(("id", "prob"), tag_set)
        )
        self.tagged_sets.append(tag_set)
        tagged_df = df.copy()
        new_columns = tuple(f"col_{c}" for c in relation.value.columns)
        if prob_column not in df.columns:
            new_columns = new_columns + (f"col_{prob_column}",)
        tagged_df[prob_column] = prov_column
        tagged_ras_set = NamedRelationalAlgebraFrozenSet(
            new_columns, tagged_df
        )
        prov_set = ProvenanceAlgebraSet(
            Constant[AbstractSet](tagged_ras_set),
            str2columnstr_constant(f"col_{prob_column}")
        )
        self.translated_probfact_sets[relation_symbol] = prov_set
        return prov_set

    def _generate_tag_choice_expression(
        self, previous, adjusted_probability, prov_column, tag_set
    ):
        symbol = Symbol.fresh()
        neg_symbol = FunctionApplication(
            NEG, (symbol,), validate_arguments=False
        )
        if len(previous) == 0:
            tag_expression = symbol
            previous = (neg_symbol,)
        else:
            args = (symbol,) + previous
            tag_expression = FunctionApplication(
                MUL, args, validate_arguments=False
            )
        previous = (neg_symbol,) + previous
        prov_column.append(tag_expression)
        tag_set.append((symbol, adjusted_probability))
        return previous

    @add_match(ProbabilisticFactSet)
    def probabilistic_fact_set_invalid(self, prob_fact_set):
        raise NotImplementedError()


class SDDWMCSemiRingSolver(
    EliminateSuperfluousProjectionMixin,
    RenameOptimizations,
    DeterministicFactSetTranslation,
    RelationalAlgebraProvenanceExpressionSemringSolver,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.translated_probfact_sets = dict()
        self.tagged_sets = []
        self.var_count = 0
        symbols_seen = set()
        for v in self.symbol_table.values():
            if not isinstance(
                v,
                (
                    DeterministicFactSet,
                    ProbabilisticFactSet,
                    ProbabilisticChoiceSet
                )
            ):
                continue
            symbol = v.relation
            if symbol in symbols_seen:
                continue
            n = len(self.symbol_table[symbol].value)
            self.var_count += n
            symbols_seen.add(symbol)
        self.manager = sdd.SddManager(
            var_count=self.var_count,
            auto_gc_and_minimize=False
        )
        self.symbols_to_literals = dict()
        self.literals_to_symbos = dict()
        self.var_count = self.manager.var_count()
        self.positive_weights = []
        self._current_var = 1

    def wmc_weights(self):
        if len(self.positive_weights) != self.var_count:
            raise NeuroLangException("Not all SDD variables have been used")
        pos_weights = np.array(self.positive_weights)
        return np.r_[
            1 - pos_weights[::-1],
            pos_weights
        ]

    def get_new_bernoulli_variable(self, probability):
        literal = self.manager.literal(self._current_var)
        self._current_var += 1
        self.positive_weights.append(probability)
        return literal

    def _semiring_agg_sum(self, args):
        return FunctionApplication(
            Constant(self._internal_sum),
            args,
            validate_arguments=False,
            verify_type=False,
        )

    def _internal_sum(self, x):
        sum_ = self.manager.false()
        for el in x:
            el.ref()
            sum_ = sum_ | el
            el.deref()
        sum_.ref()
        return sum_

    def _semiring_mul(self, left, right):
        return FunctionApplication(
            Constant(self._internal_mul),
            (left, right)
        )

    @staticmethod
    def _internal_mul(left, right):
        left.ref()
        right.ref()
        r = left & right
        r.ref()
        left.deref()
        right.deref()

        return r

    @add_match(DeterministicFactSet(Symbol))
    def deterministic_fact_set(self, deterministic_set):
        relation_symbol = deterministic_set.relation
        if relation_symbol in self.translated_probfact_sets:
            return self.translated_probfact_sets[relation_symbol]

        relation = self.walk(relation_symbol)
        rap_column = Symbol.fresh().name

        new_columns = tuple(
            str2columnstr_constant(str(i))
            for i in relation.value.columns
        )

        relation = NameColumns(relation, new_columns)

        rap_column = str2columnstr_constant(Symbol.fresh().name)
        projection_list = tuple(
            FunctionApplicationListMember(
                Constant[RelationalAlgebraStringExpression](
                    RelationalAlgebraStringExpression(c.value),
                    verify_type=False
                ),
                c
            )
            for c in new_columns
        )
        deterministic_tag_function = FunctionApplication(
            Constant(lambda: self.get_new_bernoulli_variable(1.)),
            tuple()
        )

        new_relation = NumberColumns(
            ExtendedProjection(
                relation,
                projection_list + (
                    FunctionApplicationListMember(
                        deterministic_tag_function,
                        rap_column
                    ),
                )
            ),
            (rap_column,) + new_columns
        )

        tagged_relation = self.walk(new_relation)
        self.tagged_sets.append(tagged_relation)

        prov_set = ProvenanceAlgebraSet(
            tagged_relation, int2columnint_constant(0)
        )

        self.translated_probfact_sets[relation_symbol] = prov_set
        return prov_set

    @add_match(ProbabilisticFactSet(Symbol, Constant))
    def probabilistic_fact_set(self, prob_fact_set):
        relation_symbol = prob_fact_set.relation
        if relation_symbol in self.translated_probfact_sets:
            return self.translated_probfact_sets[relation_symbol]

        with log_performance(
            LOG, f"Build probabilistic fact set {relation_symbol}"
        ):
            relation = self.walk(relation_symbol)
            named_columns = tuple(
                str2columnstr_constant(f'col_{i}')
                for i in relation.value.columns
            )
            new_columns = tuple(
                c for i, c in enumerate(named_columns)
                if i != prob_fact_set.probability_column.value
            )

            relation = NameColumns(
                relation, named_columns
            )

            rap_column = str2columnstr_constant(Symbol.fresh().name)
            projection_list = [
                FunctionApplicationListMember(
                    Constant[RelationalAlgebraStringExpression](
                        RelationalAlgebraStringExpression(c.value),
                        verify_type=False
                    ),
                    c
                )
                for c in new_columns
            ]
            probfact_tag_function = FunctionApplication(
                Constant(self.get_new_bernoulli_variable),
                (named_columns[prob_fact_set.probability_column.value],)
            )

            tagged_relation = self.walk(
                NumberColumns(
                    ExtendedProjection(
                        relation,
                        projection_list + [
                            FunctionApplicationListMember(
                                probfact_tag_function,
                                rap_column
                            )
                        ]
                    ),
                    (rap_column,) + new_columns
                )
            )
            self.tagged_sets.append(tagged_relation)

            prov_set = ProvenanceAlgebraSet(
                tagged_relation, int2columnint_constant(0)
            )

            self.translated_probfact_sets[relation_symbol] = prov_set
        return prov_set

    @add_match(ProbabilisticChoiceSet(Symbol, Constant))
    def probabilistic_choice_set(self, prob_fact_set):
        relation_symbol = prob_fact_set.relation
        prob_column = prob_fact_set.probability_column.value
        if relation_symbol in self.translated_probfact_sets:
            return self.translated_probfact_sets[relation_symbol]

        with log_performance(
            LOG, f"Build Probabilistic choice set {relation_symbol}"
        ):
            relation = self.walk(relation_symbol)
            df = relation.value.as_pandas_dataframe()
            previous_probability = 0
            previous_expression = self.manager.true()
            tag_expressions = []
            for probability in df[prob_column]:
                adjusted_probability = probability / (1 - previous_probability)
                previous_probability += probability
                previous_expression = self.generate_sdd_expression(
                    adjusted_probability, tag_expressions, previous_expression
                )

            tagged_df = df.copy()
            tagged_df[prob_column] = tag_expressions
            new_columns = tuple(f'col_{c}' for c in relation.value.columns)
            tagged_ras_set = NamedRelationalAlgebraFrozenSet(
                new_columns, tagged_df
            )
        return ProvenanceAlgebraSet(
            NumberColumns(
                Constant[AbstractSet](tagged_ras_set, verify_type=False),
                tuple(str2columnstr_constant(c) for c in new_columns)
            ),
            int2columnint_constant(prob_column)
        )

    def generate_sdd_expression(
        self, adjusted_probability, tag_expressions, previous_expression
    ):
        bv = self.get_new_bernoulli_variable(adjusted_probability)
        previous_expression.ref()
        exp = bv & previous_expression
        exp.ref()
        previous_expression &= ~bv
        previous_expression.ref()
        tag_expressions.append(exp)
        return previous_expression

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
    PushInSelections,
    ExpressionWalker
):
    pass


def _build_empty_result_set(variables_to_project):
    cols = tuple(v.value for v in variables_to_project)
    prov_col = ColumnStr(Symbol.fresh().name)
    cols += (prov_col,)
    return ProvenanceAlgebraSet(
        Constant[AbstractSet](NamedRelationalAlgebraFrozenSet(
            iterable=[], columns=cols)
        ),
        str2columnstr_constant(prov_col),
    )


def solve_succ_query_boolean_diagram(
    query_predicate, cpl_program, run_relational_algebra_solver=False
):
    """
    Obtain the solution of a SUCC query on a CP-Logic program.

    The SUCC query must take the form

        SUCC[ P(x) ]
    """
    flat_query, ra_query = _prepare_and_translate_query(
        query_predicate, cpl_program
    )

    if flat_query == FALSE:
        return ra_query

    with log_performance(LOG, "Run RAP query"):
        symbol_table = generate_probabilistic_symbol_table_for_query(
            cpl_program, flat_query
        )

        solver = WMCSemiRingSolver(symbol_table)
        prob_set_result = solver.walk(ra_query)

    with log_performance(LOG, "SDD Model count"):
        res, provenance_column = sdd_compilation_and_wmc(
            prob_set_result, solver
        )

    return ProvenanceAlgebraSet(
        Constant[AbstractSet](res),
        str2columnstr_constant(provenance_column)
    )


def sdd_compilation_and_wmc(prob_set_result, solver):
    sdd_compiler, sdd_program, prob_set_program = \
        sdd_compilation(prob_set_result)

    if len(prob_set_program._symbols) > 0:
        res, provenance_column = perform_wmc(
            solver, sdd_compiler, sdd_program,
            prob_set_program, prob_set_result
        )
    else:
        provenance_column = ColumnStr('prob')
        res = NamedRelationalAlgebraFrozenSet(
            columns=(str(provenance_column),)
        )

    return res, provenance_column


def solve_succ_query_sdd_direct(
    query_predicate, cpl_program,
    per_row_model=True, run_relational_algebra_solver=True
):
    """
    Obtain the solution of a SUCC query on a CP-Logic program.

    The SUCC query must take the form

        SUCC[ P(x) ]
    """
    flat_query, ra_query = _prepare_and_translate_query(
        query_predicate, cpl_program
    )

    if flat_query == FALSE:
        return ra_query

    with log_performance(LOG, "Run RAP query"):
        symbol_table = generate_probabilistic_symbol_table_for_query(
            cpl_program, flat_query
        )
        solver = SDDWMCSemiRingSolver(symbol_table)
        prob_set_result = solver.walk(ra_query)

    df = prob_set_result.relation.value.as_pandas_dataframe()
    if per_row_model:
        probabilities = sdd_solver_per_individual_row(
            solver, df[prob_set_result.provenance_column.value]
        )
    else:
        probabilities = sdd_solver_global_model(
            solver, df[prob_set_result.provenance_column.value]
        )

    df[prob_set_result.provenance_column.value] = probabilities

    new_ras = type(prob_set_result.relation.value)(
        prob_set_result.relation.value.columns,
        df
    )

    return ProvenanceAlgebraSet(
        Constant[AbstractSet](new_ras),
        prob_set_result.provenance_column
    )


def _prepare_and_translate_query(query_predicate, cpl_program):
    with log_performance(LOG, 'Preparing query'):
        conjunctive_query, variables_to_project = prepare_initial_query(
            query_predicate
        )

        flat_query = flatten_query(conjunctive_query, cpl_program)
        if len(cpl_program.pchoice_pred_symbs) > 0:
            flat_query = lift_optimization_for_choice_predicates(
                flat_query, cpl_program
            )

        if flat_query == FALSE:
            ra_query = _build_empty_result_set(variables_to_project)
        else:
            with log_performance(LOG, "Translation and lifted optimisation"):
                ra_query = TranslateToNamedRA().walk(flat_query)
                ra_query = Projection(ra_query, variables_to_project)
                ra_query = RAQueryOptimiser().walk(ra_query)
    return flat_query, ra_query


def sdd_solver_global_model(solver, set_probabilities):
    with log_performance(LOG, "Build global SAT problem"):
        rows, exclusive_clause, initial_var_count = \
            build_global_sdd_model_rows(
                solver,
                set_probabilities
            )

        while len(rows) > 1:
            new_rows = []
            node_sizes = 0
            nodes_processed = 0
            solver.manager.minimize()
            for i in range(0, len(rows) - 1, 2):
                node_sizes += rows[i].size() + rows[i + 1].size()
                nodes_processed += 2
                new_rows.append(rows[i].conjoin(rows[i + 1]))
                rows[i].deref()
                rows[i + 1].deref()
                new_rows[-1].ref()
            new_rows += rows[i + 2:]
            rows = new_rows
        model = rows[0].conjoin(exclusive_clause)
    with log_performance(LOG, "Minimize manager"):
        model.ref()
        solver.manager.minimize()

    with log_performance(LOG, "Model count"):
        probabilities = model_count_and_per_row_probability(
            model, solver, initial_var_count, set_probabilities.shape[0]
        )
    return probabilities


def sdd_solver_per_individual_row(solver, set_probabilities):
    probabilities = np.empty(set_probabilities.shape[0])
    new_rows = []
    with log_performance(LOG, "Minimize manager"):
        new_rows = set_probabilities
        solver.manager.minimize()
    with log_performance(LOG, "Model Count"):
        weights = solver.wmc_weights()
        for i, row in enumerate(new_rows):
            tt = [row]
            nm = solver.manager.copy(tt)
            tt[0].ref()
            nm.minimize()
            wmc = row.wmc(log_mode=False)
            wmc.set_literal_weights_from_array(weights)
            probabilities[i] = wmc.propagate()
    return probabilities


def prepare_initial_query(query_predicate):
    if isinstance(query_predicate, Implication):
        conjunctive_query = query_predicate.antecedent
        variables_to_project = tuple(
            str2columnstr_constant(s.name)
            for s in query_predicate.consequent.args
            if isinstance(s, Symbol)
        )
    else:
        conjunctive_query = query_predicate
        variables_to_project = tuple(
            str2columnstr_constant(s.name)
            for s in query_predicate._symbols
            if s not in (
                p.functor for p in
                extract_logic_predicates(query_predicate)
            )
        )
    return conjunctive_query, variables_to_project


def model_count_and_per_row_probability(model, solver, initial_var_count, n):
    probabilities = np.empty(n)
    wmc = model.wmc(log_mode=False)
    pos_weights = np.r_[
        solver.positive_weights,
    ]
    extra_ones = np.ones(solver.manager.var_count() - initial_var_count)
    weights = np.r_[
        extra_ones,
        1 - pos_weights[::-1],
        pos_weights,
        extra_ones
    ]
    wmc.set_literal_weights_from_array(weights)
    for i in range(len(probabilities)):
        lit = solver.manager.literal(
            initial_var_count + i + 1
        )
        probabilities[i] = wmc.literal_derivative(lit)

    return probabilities


def build_global_sdd_model_rows(solver, literal_probabilities):
    rows = []
    neg = solver.manager.true()
    exclusive_clause = solver.manager.false()
    initial_var_count = solver.manager.var_count()
    for sdd_ in literal_probabilities:
        solver.manager.add_var_after_last()
        new_literal = solver.manager.literal(solver.manager.var_count())
        clause = (new_literal).equiv(sdd_)
        sdd_.deref()
        clause.ref()
        rows.append(clause)
        exclusive_clause |= new_literal  # & exclusive_clause
        old_neg = neg
        neg = neg & ~new_literal
        neg.ref()
        old_neg.deref()
    exclusive_clause.ref()
    return rows, exclusive_clause, initial_var_count


def sdd_compilation(prob_set_result):
    result_symbols = (
        prob_set_result
        .relation
        .value
        .projection(prob_set_result.provenance_column.value)
    )
    if result_symbols.is_empty():
        result_symbols = tuple()
    else:
        result_symbols = tuple(
            result_symbols.
            as_pandas_dataframe()
            .iloc[:, 0]
        )

    prob_set_program = ExpressionBlock(result_symbols)
    if len(prob_set_program._symbols) > 0:
        sdd_compiler = SemiRingRAPToSDD(len(prob_set_program._symbols))
        sdd_program = sdd_compiler.walk(prob_set_program)
    else:
        sdd_compiler = None
        sdd_program = None
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
        wmc = sdd_exp.wmc(log_mode=False)
        wmc.set_literal_weights_from_array(weights)
        prob = wmc.propagate()
        probs[prob_set_program.expressions[i]] = prob

    provenance_column = 'prob'
    while str2columnstr_constant(provenance_column) in prob_set_result.columns():
        provenance_column += '_'
    provenance_column = ColumnStr(provenance_column)

    extended_projections = dict(
        [
            (c.value, RelationalAlgebraStringExpression(c.value))
            for c in prob_set_result.columns()
            if c != prob_set_result.provenance_column
        ] +
        [(
            provenance_column,
            lambda x: probs[x[prob_set_result.provenance_column.value]]
        )]
    )
    res = prob_set_result.relation.value.extended_projection(
        extended_projections
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


if config["RAS"].get("backend", "pandas") == "dask":
    solve_succ_query = solve_succ_query_boolean_diagram
else:
    solve_succ_query = solve_succ_query_sdd_direct


def solve_marg_query(rule, cpl):
    """
    Solve a MARG query on a CP-Logic program.

    Parameters
    ----------
    query : Implication
        Consequent must be of type `Condition`.
        MARG query of the form `ans(x) :- P(x)`.
    cpl_program : CPLogicProgram
        CP-Logic program on which the query should be solved.

    Returns
    -------
    ProvenanceAlgebraSet
        Provenance set labelled with probabilities for each tuple in the result
        set.

    """
    return _solve_marg_query(rule, cpl, solve_succ_query)
