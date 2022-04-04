


from neurolang.expression_pattern_matching import add_match
from neurolang.expression_walker import ExpressionWalker
from neurolang.probabilistic.exceptions import MalformedCausalOperatorError
from ..expressions import Condition, Symbol
from ...logic import Conjunction

DO = Symbol("DO")

class CausalIntervention(Conjunction):
    def __init__(self, formulas):
        self.formulas = tuple(formulas)

        self._symbols = set()
        for formula in self.formulas:
            self._symbols |= formula._symbols

    def __repr__(self):
        return f"DO{self.formulas}"



class CausalInterventionWalker(ExpressionWalker):

    @add_match(CausalIntervention)
    def match_do_operator(self, intervention):
        pass

    @add_match(Condition)
    def valid_intervention(self, condition):
        n_interventions = [
            isinstance(formula, CausalIntervention)
            for formula in condition.conditioning.formulas
        ]
        if sum(n_interventions) != 1:
            raise MalformedCausalOperatorError('''The use of more than one DO operator
            is not allowed. All interventions must be combined in a single DO operator.
            ''')