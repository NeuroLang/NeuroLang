from ...expression_pattern_matching import add_match
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
        prov_set = self.walk(projection_op.relation)
        prov_col = prov_set.provenance_column
        group_cols = list(col.value for col in projection_op.attributes)
        aggregations = {prov_col: lambda s: 1 - (1 - s).prod()}
        new_relation = prov_set.value
        new_relation = new_relation.aggregate(group_cols, aggregations)
        proj_cols = [prov_col] + group_cols
        new_relation = new_relation.projection(*proj_cols)
        return ProvenanceAlgebraSet(new_relation, prov_col)
