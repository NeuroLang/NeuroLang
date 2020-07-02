import collections
from typing import AbstractSet, Iterable

import numpy

from ..datalog import WrappedRelationalAlgebraSet
from ..datalog.expression_processing import (
    extract_logic_predicates,
    reachable_code,
)
from ..exceptions import NeuroLangFrontendException, UnexpectedExpressionError
from ..expressions import Constant, Expression, FunctionApplication
from ..logic import Implication, Union
from .exceptions import DistributionDoesNotSumToOneError
from .expressions import ProbabilisticPredicate


def is_probabilistic_fact(expression):
    r"""
    Whether the expression is a probabilistic fact.

    Notes
    -----
    In CP-Logic [1]_, a probabilistic fact is seen as a CP-event

    .. math:: \left( \alpha \text{P}(x_1, \dots, x_n)  \right) \gets \top

    with a single atom in its head which is true with probability
    :math:`\alpha`.

    .. [1] Vennekens, Joost, Marc Denecker, and Maurice Bruynooghe. "CP-Logic:
       A Language of Causal Probabilistic Events and Its Relation to Logic
       Programming." Theory and Practice of Logic Programming 9, no. 3 (May
       2009): 245–308. https://doi.org/10.1017/S1471068409003767.

    """
    return (
        isinstance(expression, Implication)
        and isinstance(expression.consequent, ProbabilisticPredicate)
        and isinstance(expression.consequent.body, FunctionApplication)
        and expression.antecedent == Constant[bool](True)
    )


def group_probabilistic_facts_by_pred_symb(union):
    probfacts = collections.defaultdict(list)
    non_probfacts = list()
    for expression in union.formulas:
        if is_probabilistic_fact(expression):
            probfacts[expression.consequent.body.functor].append(expression)
        else:
            non_probfacts.append(expression)
    return probfacts, non_probfacts


def const_or_symb_as_python_type(exp):
    if isinstance(exp, Constant):
        return exp.value
    else:
        return exp.name


def build_probabilistic_fact_set(pred_symb, pfacts):
    iterable = [
        (const_or_symb_as_python_type(pf.consequent.probability),)
        + tuple(
            const_or_symb_as_python_type(arg)
            for arg in pf.consequent.body.args
        )
        for pf in pfacts
    ]
    return Constant[AbstractSet](WrappedRelationalAlgebraSet(iterable))


def check_probabilistic_choice_set_probabilities_sum_to_one(ra_set):
    probs_sum = sum(v.value[0].value for v in ra_set.value)
    if not numpy.isclose(probs_sum, 1.0):
        raise DistributionDoesNotSumToOneError(
            "Probability labels of a probabilistic choice should sum to 1. "
            f"Got {probs_sum} instead."
        )


def add_to_union(union, to_add):
    """
    Extend `Union` with another `Union` or an iterable of `Expression`.

    Parameters
    ----------
    union: Union
        The initial `Union` to which expressions will be added.
    to_add: Unino or Expression iterable
        `Expression`s to be added to the `Union`.

    Returns
    -------
    new_union: Union
        A new `Union` containing the new expressions.

    """
    if isinstance(to_add, Union):
        return Union(union.formulas + to_add.formulas)
    if isinstance(to_add, Iterable):
        if not all(isinstance(item, Expression) for item in to_add):
            raise UnexpectedExpressionError("Expected Expression")
        return Union(union.formulas + tuple(to_add))
    raise UnexpectedExpressionError("Expected Union or Expression iterable")


def union_contains_probabilistic_facts(union):
    return any(is_probabilistic_fact(exp) for exp in union.formulas)


def separate_deterministic_probabilistic_code(
    solver, query_pred=None, det_symbols=None, prob_symbols=None
):
    if det_symbols is None:
        det_symbols = set()
    if prob_symbols is None:
        prob_symbols = set()
    if query_pred is None:
        formulas = tuple()
        for union in solver.intensional_database().values():
            formulas += union.formulas
        query_reachable_code = Union(formulas)
    else:
        query_reachable_code = reachable_code(query_pred, solver)

    if hasattr(solver, "constraints"):
        constraints_symbols = set(
            [ri.consequent.functor for ri in solver.constraints().formulas]
        )
    else:
        constraints_symbols = set()

    deterministic_symbols = (
        set(solver.extensional_database().keys())
        | set(det_symbols)
        | set(solver.builtins().keys())
        | constraints_symbols
    )
    deterministic_program = list()

    if hasattr(solver, "pchoice_pred_symbs"):
        probabilistic_symbols = (
            solver.pfact_pred_symbs
            | solver.pchoice_pred_symbs
            | set(prob_symbols)
        )
    else:
        probabilistic_symbols = set(prob_symbols)

    probabilistic_program = list()
    unclassified_code = list(query_reachable_code.formulas)
    unclassified = 0
    initial_unclassified_length = len(unclassified_code) + 1
    while (
        len(unclassified_code) > 0
        and unclassified <= initial_unclassified_length
    ):
        pred = unclassified_code.pop(0)
        preds_antecedent = set(
            p.functor
            for p in extract_logic_predicates(pred.antecedent)
            if p.functor != pred.consequent.functor
            and not is_builtin(p, solver.builtins().keys())
        )

        if not probabilistic_symbols.isdisjoint(preds_antecedent):
            probabilistic_symbols.add(pred.consequent.functor)
            probabilistic_program.append(pred)
            unclassified = 0
            initial_unclassified_length = len(unclassified_code) + 1
        elif deterministic_symbols.issuperset(preds_antecedent):
            deterministic_symbols.add(pred.consequent.functor)
            deterministic_program.append(pred)
            unclassified = 0
            initial_unclassified_length = len(unclassified_code) + 1
        else:
            unclassified_code.append(pred)
            unclassified += 1
    if not probabilistic_symbols.isdisjoint(deterministic_symbols):
        raise NeuroLangFrontendException(
            "An atom was defined as both deterministic and probabilistic"
        )
    if len(unclassified_code) > 0:
        raise NeuroLangFrontendException("There are unclassified atoms")

    return Union(deterministic_program), Union(probabilistic_program)


def is_builtin(pred, known_builtins=None):
    if known_builtins is None:
        known_builtins = set()

    return isinstance(pred.functor, Constant) or pred.functor in known_builtins
