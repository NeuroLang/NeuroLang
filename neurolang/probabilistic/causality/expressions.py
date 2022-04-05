from ...expression_pattern_matching import add_match
from ...expression_walker import ExpressionWalker
from ...logic import Conjunction
from ..exceptions import MalformedCausalOperatorError
from ..expressions import Condition
from ...expressions import Symbol, Constant

DO = Symbol("DO")

class CausalIntervention(Conjunction):
    def __init__(self, formulas):
        self.formulas = tuple(formulas)

        self._symbols = set()
        for formula in self.formulas:
            if any([not isinstance(arg, Constant) for arg in formula.args]):
                raise MalformedCausalOperatorError('''The atoms intervened by the
                operator DO can only contain constants''')
            self._symbols |= formula._symbols

    def __repr__(self):
        return f"DO{self.formulas}"



class CausalInterventionWalker(ExpressionWalker):

    @add_match(Condition)
    def valid_intervention(self, condition):
        n_interventions = [
            formula
            for formula in condition.conditioning.formulas
            if isinstance(formula, CausalIntervention)
        ]
        if len(n_interventions) != 1:
            raise MalformedCausalOperatorError('''The use of more than one DO operator
            is not allowed. All interventions must be combined in a single DO operator.
            ''')

        return n_interventions[0]
