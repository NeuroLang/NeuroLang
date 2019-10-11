from collections import defaultdict

from ..expression_pattern_matching import add_match
from ..expression_walker import ExpressionWalker
from ..expressions import Expression, Symbol
from ..exceptions import NeuroLangException
from .probdatalog import ProbFact


class BayesianNetwork(Expression):
    def __init__(self):
        self.rv_to_cpt = dict()
        self.edges = defaultdict(set)

    @property
    def random_variables(self):
        return self.rv_to_cpt.keys()


def _add_parent_to_deterministic_or_cpt(cpt, parent_rv, parent_rv_values):
    pass


def repr_ground_atom(ground_atom):
    return '{}{}'.format(
        pfact.consequent.functor.name,
        '({})'.format(', '.join([arg.value for arg in pfact.consequent.args]))
    )


class TranslatorGroundedProbDatalogToBN(ExpressionWalker):
    '''
    Translate a grounded Prob(Data)Log program to a bayesian network (BN).
    '''
    def _add_choice_variable(self, rv_name, cpt):
        if not hasattr(self, '_bayesian_network'):
            self._bayesian_network = BayesianNetwork()
        if rv_name in self._bayesian_network.random_variables:
            raise NeuroLangException(
                f'Choice variable {rv_name} already in bayesian network'
            )
        self._bayesian_network.rv_to_cpt[rv_name] = cpt

    def _add_atom_variable(self, rv_name, cpt):
        if not hasattr(self, '_bayesian_network'):
            self._bayesian_network = BayesianNetwork()
        if rv_name in self._bayesian_network.random_variables:
            self._bayesian_network.rv_to_cpt[rv_name] = (
                _add_parent_to_deterministic_or_cpt(
                    self._bayesian_network.rv_to_cpt[rv_name], rv_name, cpt
                )
            )
        else:
            self._bayesian_network.rv_to_cpt[rv_name] = cpt

    def _get_choice_var_count(self):
        if not hasattr(self, '_choice_variable_count'):
            self._choice_var_count = 0
        self._choice_var_count += 1
        return self._choice_var_count

    @add_match(ProbFact)
    def probfact(self, pfact):
        '''
        Translate a probabilistic fact to (1) a choice variable and (2) a
        random variable for its atom (if it is not already present in the
        bayesian network).

        A probabilistic fact can be seen as a CP-Logic rule with an empty body
        and a single head atom, e.g. the Prob(Data)Log probabilistic fact `0.2
        :: person('john').` can be seen as the CP-Logic rule `person('john') :
        0.2 <- true.`.

        '''
        choice_rv_name = Symbol('c_{}'.format(self._get_choice_var_count()))
        choice_rv_cpt = {
            0: 1.0 - pfact.probability.value,
            1: pfact.probability.value,
        }
        self._add_choice_variable(choice_rv_name, choice_rv_cpt)
        atom_rv_name = Symbol(repr_ground_atom(pfact.consequent))
        atom_rv_cpt = {}
        self._add_atom_variable(atom_rv_name, atom_rv_cpt)
