import typing

from ..expression_walker import add_match
from ..expressions import Constant, Symbol
from ..relational_algebra import (
    ColumnStr,
    ExtendedProjection,
    ExtendedProjectionListMember,
    NameColumns,
    Projection,
    RelationalAlgebraStringExpression,
    str2columnstr_constant
)
from ..relational_algebra_provenance import (
    ProvenanceAlgebraSet,
    RelationalAlgebraProvenanceExpressionSemringSolver
)
from .probabilistic_ra_utils import (
    DeterministicFactSet,
    ProbabilisticChoiceSet,
    ProbabilisticFactSet
)


class ProbSemiringSolver(RelationalAlgebraProvenanceExpressionSemringSolver):
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
            ExtendedProjectionListMember(
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
                    ExtendedProjectionListMember(
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

    @add_match(ExtendedProjection(ProvenanceAlgebraSet, ...))
    def extended_projection(self, proj_op):
        provset = self.walk(proj_op.relation)
        self._check_prov_col_not_in_proj_list(provset, proj_op.projection_list)
        self._check_all_non_prov_cols_in_proj_list(
            provset, proj_op.projection_list
        )
        relation = Constant[typing.AbstractSet](provset.relations)
        prov_col = str2columnstr_constant(provset.provenance_column)
        new_prov_col = str2columnstr_constant(Symbol.fresh().name)
        proj_list_with_prov_col = proj_op.projection_list + (
            ExtendedProjectionListMember(prov_col, new_prov_col),
        )
        ra_op = ExtendedProjection(relation, proj_list_with_prov_col)
        new_relation = self.walk(ra_op)
        new_provset = ProvenanceAlgebraSet(
            new_relation.value, new_prov_col.value
        )
        return new_provset

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
