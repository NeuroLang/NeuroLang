from ..expression_walker import PatternWalker, add_match
from .expressions import Implication
from ..expressions import Query, Symbol
from .expression_processing import reachable_code


class ExtractReachableRules(PatternWalker):
    @add_match(Query)
    def process_union(self, expression):
        return reachable_code(
            Implication(
                Symbol.fresh()(*expression.head),
                expression.body
            ),
            self.datalog
        )
