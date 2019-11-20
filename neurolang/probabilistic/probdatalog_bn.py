from collections import defaultdict
from typing import Mapping, AbstractSet, Callable

from ..expression_pattern_matching import add_match
from ..expression_walker import ExpressionBasicEvaluator
from ..expressions import (
    Expression,
    Definition,
    Symbol,
    ExpressionBlock,
    FunctionApplication,
    Constant,
)
from ..logic import Implication
from ..datalog.expression_processing import extract_logic_predicates
from ..exceptions import NeuroLangException
from .expressions import TableDistribution
from .probdatalog import (
    is_probabilistic_fact,
    is_existential_probabilistic_fact,
)

CPDFactory = Callable[
    [Constant[Mapping[Symbol, Constant]]], Callable[[Constant], float]
]


class BayesianNetwork(Definition):
    def __init__(self, edges, rv_to_cpd_factory):
        self.rv_to_cpd_factory = rv_to_cpd_factory
        self.edges = edges

    @property
    def random_variables(self):
        return Constant[AbstractSet[Symbol]](
            frozenset(self.rv_to_cpd_factory.value.keys())
        )


def _repr_ground_atom(ground_atom):
    return "{}{}".format(
        ground_atom.functor.name,
        "({})".format(", ".join([arg.value for arg in ground_atom.args])),
    )


def deterministic_or_cpd_factory(parent_values):
    if any(cst.value for cst in parent_values.value.values()):
        return TableDistribution(
            Constant[Mapping](
                {
                    Constant[int](0): Constant[float](0.0),
                    Constant[int](1): Constant[float](1.0),
                }
            )
        )
    else:
        return TableDistribution(
            Constant[Mapping](
                {
                    Constant[int](0): Constant[float](1.0),
                    Constant[int](1): Constant[float](0.0),
                }
            )
        )


def pfact_cpd_factory(pfact):
    def cpd_factory(parent_values):
        if parent_values.value:
            raise NeuroLangException(
                "No parent expected for probabilistic fact choice variable"
            )
        return TableDistribution(
            Constant[Mapping](
                {
                    Constant[int](0): Constant[float](1.0)
                    - pfact.consequent.probability,
                    Constant[int](1): pfact.consequent.probability,
                }
            )
        )

    return cpd_factory


class TranslatorGroundedProbDatalogToBN(ExpressionBasicEvaluator):
    """
    Translate a grounded Prob(Data)Log program to a bayesian network (BN).
    """

    def _add_choice_variable(self, rv_symbol, cpd_factory):
        if rv_symbol in self._rv_to_cpd_factory:
            raise NeuroLangException(
                f"Choice variable {rv_symbol} already in bayesian network"
            )
        self._rv_to_cpd_factory[rv_symbol] = cpd_factory

    def _add_atom_variable(self, atom, parents):
        rv_symbol = Symbol(_repr_ground_atom(atom))
        if rv_symbol not in self._rv_to_cpd_factory:
            self._rv_to_cpd_factory[rv_symbol] = deterministic_or_cpd_factory
        self._edges[rv_symbol] |= parents
        return rv_symbol

    def _get_choice_var_count(self):
        if not hasattr(self, "_choice_var_count"):
            self._choice_var_count = 0
        self._choice_var_count += 1
        return self._choice_var_count

    @add_match(ExpressionBlock)
    def program_code(self, program_code):
        self._edges = defaultdict(set)
        self._rv_to_cpd_factory = dict()
        for expression in program_code.expressions:
            self.walk(expression)
        edges = Constant[Mapping](
            {
                rv_symbol: Constant[AbstractSet](frozenset(parents))
                for rv_symbol, parents in self._edges.items()
            }
        )
        rv_to_cpd_factory = Constant[Mapping](
            {
                rv_symbol: cpd_factory
                for rv_symbol, cpd_factory in self._rv_to_cpd_factory.items()
            }
        )
        return BayesianNetwork(edges, rv_to_cpd_factory)

    @add_match(
        Implication,
        lambda exp: is_probabilistic_fact(exp)
        or is_existential_probabilistic_fact(exp),
    )
    def probfact(self, pfact):
        """
        Translate a probabilistic fact to (1) a choice variable and (2) a
        random variable for its atom (if it is not already present in the
        bayesian network).

        A probabilistic fact can be seen as a CP-Logic rule with an empty body
        and a single head atom, e.g. the Prob(Data)Log probabilistic fact `0.2
        :: person('john').` can be seen as the CP-Logic rule `person('john') :
        0.2 <- true.`.

        """
        choice_var_name = Symbol("c_{}".format(self._get_choice_var_count()))
        self._add_choice_variable(choice_var_name, pfact_cpd_factory(pfact))
        self._add_atom_variable(pfact.consequent.body, {choice_var_name})

    @add_match(
        Implication(FunctionApplication[bool](Symbol, ...), Expression),
        lambda exp: exp.antecedent != Constant[bool](True),
    )
    def statement_intensional(self, expression):
        parents = set()
        for atom in extract_logic_predicates(expression.antecedent):
            parents.add(self._add_atom_variable(atom, set()))
        self._add_atom_variable(expression.consequent, parents)
