"""
    Implementation of the weighted model counting approach through knowledge
    compilation.
"""
import logging
import operator as op

import numpy as np
import pandas as pd
from numpy.lib.shape_base import column_stack
from pysdd import sdd

from neurolang.utils.relational_algebra_set.pandas import (
    NamedRelationalAlgebraFrozenSet
)

from ..datalog.expression_processing import (
    extract_logic_predicates,
    flatten_query
)
from ..datalog.translate_to_named_ra import TranslateToNamedRA
from ..expression_walker import (
    ExpressionWalker,
    PatternWalker,
    SuccintRepr,
    add_match
)
from ..expressions import (
    Constant,
    ExpressionBlock,
    FunctionApplication,
    Symbol,
    sure_is_not_pattern
)
from ..logic import Conjunction, Implication
from ..relational_algebra import (
    ColumnInt,
    ColumnStr,
    ExtendedProjection,
    ExtendedProjectionListMember,
    NameColumns,
    Projection,
    RelationalAlgebraOperation,
    RelationalAlgebraPushInSelections,
    RelationalAlgebraStringExpression,
    RenameColumns,
    str2columnstr_constant
)
from ..relational_algebra_provenance import (
    ProvenanceAlgebraSet,
    RelationalAlgebraProvenanceExpressionSemringSolver
)
from .expression_processing import get_probchoice_variable_equalities

LOG = logging.getLogger(__name__)

SR = SuccintRepr()


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
        prob_column = prob_fact_set.probability_column.value
        if relation_symbol in self.translated_probfact_sets:
            return self.translated_probfact_sets[relation_symbol]

        relation = self.walk(relation_symbol)
        df, prov_column = self._generate_tag_choice_set(relation, prob_column)

        prov_set = self._generate_choice_provenance_set(
            relation, prob_column, df, prov_column
        )

        self.translated_probfact_sets[relation_symbol] = prov_set
        return prov_set

    def _generate_choice_provenance_set(
        elf, relation, prob_column, df, prov_column
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
        previous = None
        previous_probability = 0
        df = relation.value.as_pandas_dataframe()
        prov_column = []
        tag_set = []
        probabilities = []
        mul = Constant(op.mul)
        for i, probability in (
            enumerate(df.iloc[:, prob_column])
        ):
            symbol = Symbol.fresh()
            if i == 0:
                prov_column.append(symbol)
                previous = (-symbol,)
            else:
                args = (symbol,) + previous
                with sure_is_not_pattern():
                    mul_expr = FunctionApplication(
                        mul, args, validate_arguments=False
                    )
                prov_column.append(mul_expr)
                previous = (-symbol,) + previous

            probabilities.append(probability / (1 - previous_probability))
            tag_set.append((symbol, probability / (1 - previous_probability)))
            previous_probability += probability

        tag_set = self._build_relation_constant(
            NamedRelationalAlgebraFrozenSet(('id', 'prob'), tag_set)
        )
        self.tagged_sets.append(tag_set)
        return df, prov_column

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

    flat_query = flatten_query(conjunctive_query, cpl_program)
    if len(cpl_program.pchoice_pred_symbs) > 0:
        eq = Constant(op.eq)
        added_equalities = []
        for x, y in get_probchoice_variable_equalities(
            flat_query.formulas, cpl_program.pchoice_pred_symbs
        ):
            added_equalities.append(eq(x, y))
        if len(added_equalities) > 0:
            flat_query = Conjunction(
                flat_query.formulas + tuple(added_equalities)
            )

    ra_query = TranslateToNamedRA().walk(flat_query)
    ra_query = Projection(ra_query, variables_to_project)
    ra_query = RAQueryOptimiser().walk(ra_query)

    symbol_table = _generate_symbol_table(cpl_program, flat_query)

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
