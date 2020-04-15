import typing

from ...expression_pattern_matching import add_match
from ...expressions import Constant
from ...relational_algebra import Projection
from ...relational_algebra_provenance import (
    ProvenanceAlgebraSet,
    RelationalAlgebraProvenanceCountingSolver,
)


class NoisyORProbabilityProvenanceSolver(
    RelationalAlgebraProvenanceCountingSolver
):
    @add_match(Projection)
    def projection(self, projection_op):
        prov_cst_relation = self.walk(projection_op.relation)
        prov_col = prov_cst_relation.provenance_column
        cst_relation = prov_cst_relation.value
        group_cols = list(col.value for col in projection_op.attributes)
        aggregations = {prov_col: noisy_or_aggregation}
        new_relation = cst_relation.value
        new_relation = new_relation.aggregate(group_cols, aggregations)
        proj_cols = [prov_col] + group_cols
        new_relation = new_relation.projection(*proj_cols)
        return ProvenanceAlgebraSet(new_relation, prov_col)


def noisy_or_aggregation(probs_series):
    return 1 - (1 - probs_series).prod()
