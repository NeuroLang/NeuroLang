import itertools
import operator
import typing

from ..expression_pattern_matching import add_match
from ..expression_walker import PatternWalker
from ..expressions import Constant, FunctionApplication, Symbol
from ..relational_algebra import ColumnStr, str2columnstr_constant
from ..relational_algebra_provenance import (
    ProvenanceAlgebraSet,
    RelationalAlgebraProvenanceCountingSolver,
)


class WhySemiringCompiler(PatternWalker):
    @add_match(FunctionApplication(Constant(operator.add), ...))
    def add(self, function_application):
        args = self.walk(function_application.args)
        return args[0].union(args[1])

    @add_match(FunctionApplication(Constant(operator.mul), ...))
    def mul(self, function_application):
        args = self.walk(function_application.args)
        return frozenset(
            x.union(y) for x, y in itertools.product(args[0], args[1])
        )

    @add_match(Symbol)
    def symbol(self, symbol):
        return frozenset([frozenset([symbol])])

    @add_match(ProvenanceAlgebraSet)
    def provset(self, provset):
        new_prov_col = ColumnStr(Symbol.fresh().name)
        prov_col_idx = list(provset.relations.columns).index(
            provset.provenance_column
        )
        projections = {col: col for col in provset.non_provenance_columns}
        projections[new_prov_col] = lambda tupl: self.walk(tupl[prov_col_idx])
        relation = provset.relations.extended_projection(projections)
        return ProvenanceAlgebraSet(relation, new_prov_col)
