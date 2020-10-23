"""
    Implementation of the weighted model counting approach through knowledge
    compilation.
"""
import logging
import operator as op

import numpy as np
import pandas as pd
from pysdd import sdd

from ..datalog.expression_processing import (
    extract_logic_predicates,
    flatten_query,
)
from ..datalog.translate_to_named_ra import TranslateToNamedRA
from ..expression_walker import (
    ExpressionWalker,
    PatternWalker,
    add_match
)
from ..exceptions import NeuroLangException
from ..expressions import (
    Constant,
    ExpressionBlock,
    FunctionApplication,
    Symbol,
    sure_is_not_pattern
)
from ..logic import Conjunction, Implication
from ..logic.expression_processing import (
    extract_logic_free_variables,
)
from ..relational_algebra import (
    ColumnInt,
    ColumnStr,
    ExtendedProjection,
    ExtendedProjectionListMember,
    NameColumns,
    Projection,
    RelationalAlgebraPushInSelections,
    RelationalAlgebraStringExpression,
    RenameColumns,
    str2columnstr_constant
)
from ..relational_algebra_provenance import (
    NaturalJoinInverse,
    ProvenanceAlgebraSet,
    RelationalAlgebraProvenanceCountingSolver,
    RelationalAlgebraProvenanceExpressionSemringSolver
)
from ..utils.relational_algebra_set.pandas import (
    NamedRelationalAlgebraFrozenSet
)
from ..utils import log_performance

from .expression_processing import lift_optimization_for_choice_predicates
from .probabilistic_ra_utils import (
    DeterministicFactSet,
    ProbabilisticChoiceSet,
    ProbabilisticFactSet,
    generate_probabilistic_symbol_table_for_query
)


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
    def probabilistic_choice_set(self, prob_choice_set):
        relation_symbol = prob_choice_set.relation
        prob_column = prob_choice_set.probability_column.value
        if relation_symbol in self.translated_probfact_sets:
            return self.translated_probfact_sets[relation_symbol]

        with log_performance(LOG, "building probabilistic choice set"):
            relation = self.walk(relation_symbol)
            df, prov_column = self._generate_tag_choice_set(
                relation, prob_column
            )

            prov_set = self._generate_choice_provenance_set(
                relation, prob_column, df, prov_column
            )

        self.translated_probfact_sets[relation_symbol] = prov_set
        return prov_set

    def _generate_choice_provenance_set(
        self, relation, prob_column, df, prov_column
    ):
        out_columns = tuple(
            c
            for c in relation.value.columns
            if int(c) != prob_column
        )
        prov_set = df[list(out_columns)]
        prov_column_name = Symbol.fresh().name
        prov_set[prov_column_name] = prov_column
        renamed_out_columns = tuple(
            range(len(out_columns))
        ) + (prov_column_name,)
        prov_set.columns = renamed_out_columns
        prov_set = NamedRelationalAlgebraFrozenSet(
            renamed_out_columns, prov_set
        )
        prov_set = ProvenanceAlgebraSet(prov_set, ColumnStr(prov_column_name))
        return prov_set

    def _generate_tag_choice_set(self, relation, prob_column):
        previous = tuple()
        previous_probability = 0
        df = relation.value.as_pandas_dataframe()
        prov_column = []
        tag_set = []
        with sure_is_not_pattern():
            for probability in df.iloc[:, prob_column]:
                tag_expression, symbol, previous = \
                    self._generate_tag_choice_expression(previous)

                adjusted_probability = probability / (1 - previous_probability)
                previous_probability += probability

                prov_column.append(tag_expression)
                tag_set.append((symbol, adjusted_probability))

        tag_set = self._build_relation_constant(
            NamedRelationalAlgebraFrozenSet(('id', 'prob'), tag_set)
        )
        self.tagged_sets.append(tag_set)
        return df, prov_column

    def _generate_tag_choice_expression(self, previous):
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
        return tag_expression, symbol, previous

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

    def _add_tag_column(self, relation):
        new_columns = tuple(
            str2columnstr_constant(str(i))
            for i in relation.value.columns
        )

        relation = NameColumns(relation, new_columns)

        def get_fresh_symbol():
            s = Symbol.fresh()
            return s

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


class SDDWMCSemiRingSolver(RelationalAlgebraProvenanceExpressionSemringSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.translated_probfact_sets = dict()
        self.tagged_sets = []
        self.var_count = 0
        symbols_seen = set()
        for k, v in self.symbol_table.items():
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

    def _semiring_agg_sum(self, x):
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
            # and (len(exp.attributes) == len(exp.relation.relation.columns))
        )
    )
    def eliminate_superfluous_projection(self, expression):
        return self.walk(expression.relation)

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
        projection_list = [
            ExtendedProjectionListMember(
                Constant[RelationalAlgebraStringExpression](
                    RelationalAlgebraStringExpression(c.value),
                    verify_type=False
                ),
                c
            )
            for c in new_columns
        ]
        deterministic_tag_function = FunctionApplication(
            Constant(lambda: self.get_new_bernoulli_variable(1.)),
            tuple()
        )

        tagged_relation = self.walk(
            ExtendedProjection(
                relation,
                projection_list + [
                    ExtendedProjectionListMember(
                        deterministic_tag_function,
                        rap_column
                    )
                ]
            )
        )
        self.tagged_sets.append(tagged_relation)

        prov_set = ProvenanceAlgebraSet(
            tagged_relation.value, rap_column.value
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
                ExtendedProjectionListMember(
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
                ExtendedProjection(
                    relation,
                    projection_list + [
                        ExtendedProjectionListMember(
                            probfact_tag_function,
                            rap_column
                        )
                    ]
                )
            )
            self.tagged_sets.append(tagged_relation)

            prov_set = ProvenanceAlgebraSet(
                tagged_relation.value, rap_column.value
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
            tagged_ras_set,
            ColumnStr(f'col_{prob_column}')
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
    RelationalAlgebraPushInSelections,
    ExpressionWalker
):
    pass


def solve_succ_query_boolean_diagram(query_predicate, cpl_program):
    """
    Obtain the solution of a SUCC query on a CP-Logic program.

    The SUCC query must take the form

        SUCC[ P(x) ]
    """
    with log_performance(LOG, 'Preparing query'):
        conjunctive_query, variables_to_project = prepare_initial_query(
            query_predicate
        )

        flat_query = flatten_query(conjunctive_query, cpl_program)

    with log_performance(LOG, "Translation and lifted optimisation"):
        if len(cpl_program.pchoice_pred_symbs) > 0:
            flat_query = lift_optimization_for_choice_predicates(
                flat_query, cpl_program
            )

        ra_query = TranslateToNamedRA().walk(flat_query)
        ra_query = Projection(ra_query, variables_to_project)
        ra_query = RAQueryOptimiser().walk(ra_query)

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
        res, ColumnStr(provenance_column)
    )


def sdd_compilation_and_wmc(prob_set_result, solver):
    sdd_compiler, sdd_program, prob_set_program = \
        sdd_compilation(prob_set_result)

    res, provenance_column = perform_wmc(
        solver, sdd_compiler, sdd_program,
        prob_set_program, prob_set_result
    )

    return res, provenance_column


def solve_succ_query_sdd_direct(
    query_predicate, cpl_program, per_row_model=True
):
    """
    Obtain the solution of a SUCC query on a CP-Logic program.

    The SUCC query must take the form

        SUCC[ P(x) ]
    """
    with log_performance(LOG, 'Preparing query'):
        conjunctive_query, variables_to_project = prepare_initial_query(
            query_predicate
        )

        flat_query = flatten_query(conjunctive_query, cpl_program)

    with log_performance(LOG, "Translation and lifted optimisation"):
        if len(cpl_program.pchoice_pred_symbs) > 0:
            flat_query = lift_optimization_for_choice_predicates(
                flat_query, cpl_program
            )

        ra_query = TranslateToNamedRA().walk(flat_query)
        ra_query = Projection(ra_query, variables_to_project)
        ra_query = RAQueryOptimiser().walk(ra_query)

    with log_performance(LOG, "Run RAP query"):
        symbol_table = generate_probabilistic_symbol_table_for_query(
            cpl_program, flat_query
        )
        solver = SDDWMCSemiRingSolver(symbol_table)
        prob_set_result = solver.walk(ra_query)

    df = prob_set_result.relations.as_pandas_dataframe()
    if per_row_model:
        probabilities = sdd_solver_per_individual_row(
            solver, df[prob_set_result.provenance_column]
        )
    else:
        probabilities = sdd_solver_global_model(
            solver, df[prob_set_result.provenance_column]
        )

    df[prob_set_result.provenance_column] = probabilities
    return ProvenanceAlgebraSet(
        type(prob_set_result.relations)(
            prob_set_result.relations.columns,
            df
        ),
        prob_set_result.provenance_column
    )


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
    for i, sdd_ in enumerate(literal_probabilities):
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
    prob_set_program = ExpressionBlock(
        tuple(
            prob_set_result
            .relations
            .projection(prob_set_result.provenance_column)
            .as_pandas_dataframe()
            .iloc[:, 0]
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
        wmc = sdd_exp.wmc(log_mode=False)
        wmc.set_literal_weights_from_array(weights)
        prob = wmc.propagate()
        probs[prob_set_program.expressions[i]] = prob

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
    res_args = tuple(
        s
        for s in rule.consequent.args
        if isinstance(s, Symbol)
    )

    joint_antecedent = Conjunction(
        tuple(
            extract_logic_predicates(rule.antecedent.conditioned) |
            extract_logic_predicates(rule.antecedent.conditioning)
        )
    )
    joint_logic_variables = extract_logic_free_variables(
        joint_antecedent
    ) & res_args
    joint_rule = Implication(
        Symbol.fresh()(*joint_logic_variables), joint_antecedent
    )
    joint_provset = solve_succ_query(joint_rule, cpl)

    denominator_antecedent = rule.antecedent.conditioning
    denominator_logic_variables = extract_logic_free_variables(
        denominator_antecedent
    ) & res_args
    denominator_rule = Implication(
        Symbol.fresh()(*denominator_logic_variables),
        denominator_antecedent
    )
    denominator_provset = solve_succ_query(denominator_rule, cpl)
    rapcs = RelationalAlgebraProvenanceCountingSolver()
    provset = rapcs.walk(
        Projection(
            NaturalJoinInverse(joint_provset, denominator_provset),
            tuple(str2columnstr_constant(s.name) for s in res_args)
        )
    )
    return provset
