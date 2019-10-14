from collections import defaultdict

from ..expression_pattern_matching import add_match
from ..expression_walker import ExpressionBasicEvaluator
from ..expressions import (
    Expression, Symbol, ExpressionBlock, FunctionApplication, Constant
)
from ..datalog.expressions import Implication
from ..datalog.expression_processing import extract_datalog_predicates
from ..exceptions import NeuroLangException
from .probdatalog import ProbFact
from .distributions import TableDistribution


class BayesianNetwork(Expression):
    def __init__(self, edges, rv_to_cpd_functor):
        self.rv_to_cpd_functor = rv_to_cpd_functor
        self.edges = edges

    @property
    def random_variables(self):
        return self.rv_to_cpd_functor.keys()


def _repr_ground_atom(ground_atom):
    return '{}{}'.format(
        ground_atom.functor.name,
        '({})'.format(', '.join([arg.value for arg in ground_atom.args]))
    )


def deterministic_or_cpd_functor(parent_values):
    if any(parent_values.values()):
        return TableDistribution({0: 0.0, 1: 1.0})
    else:
        return TableDistribution({0: 1.0, 1: 0.0})


def pfact_cpd_functor(pfact):
    def cpd_functor(parent_values):
        if parent_values:
            raise NeuroLangException(
                'No parent expected for probabilistic fact choice variable'
            )
        return TableDistribution({
            0: 1 - pfact.probability.value,
            1: pfact.probability.value,
        })

    return cpd_functor


class TranslatorGroundedProbDatalogToBN(ExpressionBasicEvaluator):
    '''
    Translate a grounded Prob(Data)Log program to a bayesian network (BN).
    '''
    def _add_choice_variable(self, rv_name, cpt):
        if rv_name in self._rv_to_cpd_functor:
            raise NeuroLangException(
                f'Choice variable {rv_name} already in bayesian network'
            )
        self._rv_to_cpd_functor[rv_name] = cpt

    def _add_atom_variable(self, atom, parents):
        rv_name = Symbol(_repr_ground_atom(atom))
        if rv_name not in self._rv_to_cpd_functor:
            self._rv_to_cpd_functor[rv_name] = deterministic_or_cpd_functor
        self._edges[rv_name] |= parents
        return rv_name

    def _get_choice_var_count(self):
        if not hasattr(self, '_choice_var_count'):
            self._choice_var_count = 0
        self._choice_var_count += 1
        return self._choice_var_count

    @add_match(ExpressionBlock)
    def program_code(self, program_code):
        self._edges = defaultdict(set)
        self._rv_to_cpd_functor = dict()
        for expression in program_code.expressions:
            self.walk(expression)
        return BayesianNetwork(self._edges, self._rv_to_cpd_functor)

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
        choice_var_name = Symbol('c_{}'.format(self._get_choice_var_count()))
        self._add_choice_variable(choice_var_name, pfact_cpd_functor(pfact))
        self._add_atom_variable(pfact.consequent, {choice_var_name})

    @add_match(
        Implication(FunctionApplication[bool](Symbol, ...), Expression),
        lambda exp: exp.antecedent != Constant[bool](True)
    )
    def statement_intensional(self, expression):
        parents = set()
        for atom in extract_datalog_predicates(expression.antecedent):
            parents.add(self._add_atom_variable(atom, set()))
        self._add_atom_variable(expression.consequent, parents)
