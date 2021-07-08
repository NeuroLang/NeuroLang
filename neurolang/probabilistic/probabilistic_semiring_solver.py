import typing

from ..expression_walker import PatternWalker, add_match
from ..expressions import Constant, Symbol
from ..relational_algebra import (
    ColumnStr,
    ExtendedProjection,
    FunctionApplicationListMember,
    NameColumns,
    Projection,
    RelationalAlgebraStringExpression,
    str2columnstr_constant,
)
from ..relational_algebra_provenance import (
    ProvenanceAlgebraSet,
    ProvenanceExtendedProjectionMixin,
    RelationalAlgebraProvenanceExpressionSemringSolver,
)
from .probabilistic_ra_utils import (
    DeterministicFactSet,
    ProbabilisticChoiceSet,
    ProbabilisticFactSet,
)


class RemoveSuperfluousProjectionMixin(PatternWalker):
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
        Projection(ProvenanceAlgebraSet, ...),
        lambda projection: (
            set(projection.relation.non_provenance_columns)
            == set(col.value for col in projection.attributes)
        )
    )
    def projection_on_all_non_provenance_columns(self, proj_op):
        return self.walk(proj_op.relation)


class ProbSemiringSolver(
    ProvenanceExtendedProjectionMixin,
    RelationalAlgebraProvenanceExpressionSemringSolver,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.translated_probfact_sets = dict()

    @add_match(
        DeterministicFactSet(Constant),
        lambda e: e.relation.value.is_empty()
    )
    def empty_deterministic_fact_set(self, deterministic_set):
        provenance_column = ColumnStr(Symbol.fresh().name)
        return ProvenanceAlgebraSet(
            deterministic_set.relation.value,
            provenance_column
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

        prov_column = ColumnStr(Symbol.fresh().name)
        provenance_set = self.walk(
            ExtendedProjection(
                NameColumns(relation, named_columns),
                tuple(projection_list)
                + (
                    FunctionApplicationListMember(
                        Constant[float](1.0),
                        str2columnstr_constant(prov_column),
                    ),
                ),
            )
        )

        self.translated_probfact_sets[relation_symbol] = ProvenanceAlgebraSet(
            provenance_set.value, prov_column
        )
        return self.translated_probfact_sets[relation_symbol]

    @add_match(ProbabilisticFactSet(Symbol, ...))
    def probabilistic_fact_set(self, prob_fact_set):
        relation_symbol = prob_fact_set.relation
        if relation_symbol in self.translated_probfact_sets:
            return self.translated_probfact_sets[relation_symbol]

        relation = self.walk(relation_symbol)
        named_columns = tuple(
            str2columnstr_constant(f"col_{i}") for i in relation.value.columns
        )
        relation = NameColumns(relation, named_columns)
        relation = self.walk(relation)
        if len(relation.value.columns) > 0:
            rap_column = ColumnStr(
                relation.value.columns[prob_fact_set.probability_column.value]
            )
        else:
            rap_column = ColumnStr('p')

        self.translated_probfact_sets[relation_symbol] = ProvenanceAlgebraSet(
            relation.value, rap_column
        )
        return self.translated_probfact_sets[relation_symbol]

    @add_match(ProbabilisticChoiceSet(Symbol, ...))
    def probabilistic_choice_set(self, prob_choice_set):
        return self.probabilistic_fact_set(prob_choice_set)

    @add_match(ProbabilisticFactSet)
    def probabilistic_fact_set_invalid(self, prob_fact_set):
        raise NotImplementedError()
