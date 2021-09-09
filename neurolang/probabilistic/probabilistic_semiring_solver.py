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
    Projection,
    RelationalAlgebraSolver,
    RelationalAlgebraStringExpression,
    str2columnstr_constant
)
from ..relational_algebra_provenance import (
    ProvenanceAlgebraSet,
    BuildProvenanceAlgebraSetWalkIntoMixin,
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
                    DeterministicFactSet,
                    ProbabilisticFactSet,
                    ProbabilisticChoiceSet,
                ),
            )
        ),
    )
    def eliminate_superfluous_projection(self, expression):
        return self.walk(expression.relation)

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
        named_columns = tuple(
            str2columnstr_constant(f"col_{i}") for i in relation.value.columns
        )
        projection_list = [
            FunctionApplicationListMember(
                Constant[RelationalAlgebraStringExpression](
                    RelationalAlgebraStringExpression(c.value),
                    verify_type=False,
                ),
                c,
            )
            for c in named_columns
        ]

        prov_column = str2columnstr_constant(Symbol.fresh().name)
        provenance_set = ExtendedProjection(
            NameColumns(relation, named_columns),
            tuple(projection_list)
            + (
                FunctionApplicationListMember(
                    Constant[float](1.0),
                    prov_column,
                ),
            ),
        )

        self.translated_probfact_sets[relation_symbol] = \
            ProvenanceAlgebraSet(
                provenance_set, prov_column
            )
        return self.translated_probfact_sets[relation_symbol]

    @add_match(ProbabilisticFactSet(Symbol, ...))
    def probabilistic_fact_set(self, prob_fact_set):
        relation_symbol = prob_fact_set.relation
        if relation_symbol in self.translated_probfact_sets:
            return self.translated_probfact_sets[relation_symbol]

        relation = self.walk(relation_symbol)
        if isinstance(relation, Constant):
            named_columns = tuple(
                str2columnstr_constant(f"col_{i}") for i in relation.value.columns
            )
        else:
            named_columns = tuple(
                str2columnstr_constant(f"col_{i.value}") for i in relation.columns()
            )
        relation = NameColumns(relation, named_columns)
        if len(named_columns) > 0:
            rap_column = named_columns[prob_fact_set.probability_column.value]
        else:
            rap_column = str2columnstr_constant(Symbol.fresh().name)

        self.translated_probfact_sets[relation_symbol] = \
            ProvenanceAlgebraSet(
                relation, rap_column
            )
        return self.translated_probfact_sets[relation_symbol]

    @add_match(ProbabilisticChoiceSet(Symbol, ...))
    def probabilistic_choice_set(self, prob_choice_set):
        return self.probabilistic_fact_set(prob_choice_set)

    @add_match(ProbabilisticFactSet)
    def probabilistic_fact_set_invalid(self, prob_fact_set):
        raise NotImplementedError()

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
