from ..expression_walker import (
    ExpressionWalker,
    PatternWalker,
    ResolveSymbolMixin,
    add_match
)
from ..expressions import Constant, Symbol
from ..relational_algebra import (
    ColumnStr,
    ExtendedProjection,
    FunctionApplicationListMember,
    NameColumns,
    NumberColumns,
    Projection,
    RelationalAlgebraSolver,
    int2columnint_constant,
    str2columnstr_constant
)
from ..relational_algebra_provenance import (
    BuildProvenanceAlgebraSetWalkIntoMixin,
    ProvenanceAlgebraSet,
    RelationalAlgebraProvenanceCountingSolverMixin
)
from .probabilistic_ra_utils import (
    DeterministicFactSet,
    ProbabilisticChoiceSet,
    ProbabilisticFactSet
)


class ProbSemiringSolverMixin(
    BuildProvenanceAlgebraSetWalkIntoMixin,
    RelationalAlgebraProvenanceCountingSolverMixin,
    PatternWalker
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.translated_probfact_sets = dict()

    @add_match(
        Projection,
        lambda exp: (
            isinstance(
                exp.relation,
                (
                    ProbabilisticFactSet,
                    ProbabilisticChoiceSet,
                ),
            )
        ) and (
            exp.relation.probability_column == int2columnint_constant(0)
        )
    )
    def eliminate_superfluous_projection(self, expression):
        new_relation = expression.relation.apply(
            Projection(
                expression.relation.relation,
                (expression.relation.probability_column,) +
                tuple(
                    int2columnint_constant(a.value + 1)
                    for a in expression.attributes
                )
            ),
            expression.relation.probability_column
        )
        return self.walk(new_relation)

    @add_match(Projection(DeterministicFactSet, ...))
    def push_projection_in_deterministic(self, expression):
        return self.walk(
            DeterministicFactSet(
                Projection(
                    expression.relation.relation,
                    expression.attributes
                ))
        )

    @add_match(
        DeterministicFactSet(Constant),
        lambda e: e.relation.value.is_empty()
    )
    def empty_deterministic_fact_set(self, deterministic_set):
        provenance_column = ColumnStr(Symbol.fresh().name)
        return ProvenanceAlgebraSet(
            deterministic_set.relation,
            str2columnstr_constant(provenance_column)
        )

    @add_match(DeterministicFactSet(Symbol))
    def deterministic_fact_set(self, deterministic_set):
        relation_symbol = deterministic_set.relation
        if relation_symbol in self.translated_probfact_sets:
            return self.translated_probfact_sets[relation_symbol]

        relation = self.walk(relation_symbol)
        if isinstance(relation, Constant):
            columns = relation.value.columns
        else:
            columns = relation.columns()
        named_columns = tuple(
            str2columnstr_constant(f"col_{i}") for i in range(len(columns))
        )
        projection_list = [
            FunctionApplicationListMember(c, c)
            for c in named_columns
        ]

        prov_column = str2columnstr_constant(Symbol.fresh().name)
        provenance_set = NumberColumns(
            ExtendedProjection(
                NameColumns(relation, named_columns),
                tuple(projection_list)
                + (
                    FunctionApplicationListMember(
                        Constant[float](1.0),
                        prov_column,
                    ),
                ),
            ),
            (prov_column,) + named_columns
        )

        self.translated_probfact_sets[relation_symbol] = \
            ProvenanceAlgebraSet(
                provenance_set, int2columnint_constant(0)
            )
        return self.translated_probfact_sets[relation_symbol]

    @add_match(ProbabilisticFactSet(Symbol, ...))
    def probabilistic_fact_set(self, prob_fact_set):
        relation_symbol = prob_fact_set.relation
        if relation_symbol in self.translated_probfact_sets:
            return self.translated_probfact_sets[relation_symbol]

        relation = self.walk(relation_symbol)
        self.translated_probfact_sets[relation_symbol] = \
            ProvenanceAlgebraSet(
                relation, prob_fact_set.probability_column
            )
        return self.translated_probfact_sets[relation_symbol]

    @add_match(ProbabilisticChoiceSet(Symbol, ...))
    def probabilistic_choice_set(self, prob_choice_set):
        return self.probabilistic_fact_set(prob_choice_set)

    @add_match(DeterministicFactSet)
    def deterministic_fact_set_general(self, det_fact_set):
        new_symbol = Symbol.fresh()
        self.symbol_table[new_symbol] = det_fact_set.relation
        return self.walk(DeterministicFactSet(new_symbol))

    @add_match(ProbabilisticFactSet)
    def probabilistic_fact_set_invalid(self, prob_fact_set):
        new_symbol = Symbol.fresh()
        self.symbol_table[new_symbol] = prob_fact_set.relation
        return self.walk(ProbabilisticFactSet(
            new_symbol, prob_fact_set.probability_column
        ))

    @add_match(ProbabilisticChoiceSet)
    def probabilistic_choice_set_to_symbol(self, prob_choice_set):
        new_symbol = Symbol.fresh()
        self.symbol_table[new_symbol] = prob_choice_set.relation
        return self.walk(ProbabilisticChoiceSet(
            new_symbol, prob_choice_set.probability_column
        ))

    @staticmethod
    def _check_prov_col_not_in_proj_list(provset, proj_list):
        if any(
            member.dst_column.value == provset.provenance_column
            for member in proj_list
        ):
            raise ValueError(
                "Cannot project on provenance column: "
                f"{provset.provenance_column}"
            )

    @staticmethod
    def _check_all_non_prov_cols_in_proj_list(provset, proj_list):
        if provset.value.is_empty():
            return
        non_prov_cols = set(provset.non_provenance_columns)
        found_cols = set(
            member.dst_column.value
            for member in proj_list
            if member.dst_column.value in non_prov_cols
            and member.fun_exp == member.dst_column
        )
        if non_prov_cols.symmetric_difference(found_cols):
            raise ValueError(
                "All non-provenance columns must be part of the extended "
                "projection as {c: c} projection list member."
            )


class ProbSemiringSolver(
    ProbSemiringSolverMixin,
    RelationalAlgebraSolver
):
    pass


class ProbSemiringToRelationalAlgebraSolver(
    ProbSemiringSolverMixin,
    ResolveSymbolMixin,
    ExpressionWalker
):
    def __init__(self, *args, symbol_table=None, **kwargs):
        self.symbol_table = symbol_table
        super().__init__(*args, **kwargs)
