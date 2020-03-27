import typing
import collections

import numpy

from ..exceptions import NeuroLangException
from ..expressions import (
    Expression,
    Constant,
    Symbol,
    FunctionApplication,
    ExpressionBlock,
)
from ..logic import Implication, ExistentialPredicate
from ..logic.expression_processing import (
    extract_logic_predicates,
    extract_logic_free_variables,
)
from ..datalog import WrappedRelationalAlgebraSet
from .expressions import ProbabilisticPredicate, DeltaTerm


def is_probabilistic_fact(expression):
    """
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


def is_existential_probabilistic_fact(expression):
    return (
        isinstance(expression, Implication)
        and isinstance(expression.consequent, ExistentialPredicate)
        and isinstance(expression.consequent.body, ProbabilisticPredicate)
        and isinstance(expression.consequent.body.body, FunctionApplication)
        and expression.antecedent == Constant[bool](True)
    )


def is_existential_predicate(expression):
    free_vars = extract_logic_free_variables(expression)
    if len(free_vars) > 0:
        return True
    return False


def get_rule_pfact_pred_symbs(rule, pfact_pred_symbs):
    return set(
        p.functor
        for p in extract_logic_predicates(rule.antecedent)
        if p.functor in pfact_pred_symbs
    )


def put_probfacts_in_front(code_block):
    probfacts = []
    non_probfacts = []
    for expression in code_block.expressions:
        if is_probabilistic_fact(
            expression
        ) or is_existential_probabilistic_fact(expression):
            probfacts.append(expression)
        else:
            non_probfacts.append(expression)
    return ExpressionBlock(probfacts + non_probfacts)


def group_probfacts_by_pred_symb(code_block):
    probfacts = collections.defaultdict(list)
    non_probfacts = list()
    for expression in code_block.expressions:
        if is_probabilistic_fact(expression):
            probfacts[expression.consequent.body.functor].append(expression)
        else:
            non_probfacts.append(expression)
    return probfacts, non_probfacts


def check_existential_probfact_validity(expression):
    qvar = expression.consequent.head
    if qvar in expression.consequent.body.body._symbols:
        raise NeuroLangException(
            "Existentially quantified variable can only be used as the "
            "probability of the probability fact"
        )


def extract_probfact_or_eprobfact_pred_symb(expression):
    if is_existential_probabilistic_fact(expression):
        return expression.consequent.body.body.functor
    else:
        return expression.consequent.body.functor


def get_pfact_var_idxs(pfact):
    if is_probabilistic_fact(pfact):
        atom = pfact.consequent.body
    else:
        atom = pfact.consequent.body.body
    return {i for i, arg in enumerate(atom.args) if isinstance(arg, Symbol)}


def const_or_symb_as_python_type(exp):
    if isinstance(exp, Constant):
        return exp.value
    else:
        return exp.name


def build_pfact_set(pred_symb, pfacts):
    iterable = [
        (const_or_symb_as_python_type(pf.consequent.probability),)
        + tuple(
            const_or_symb_as_python_type(arg)
            for arg in pf.consequent.body.args
        )
        for pf in pfacts
    ]
    return Constant[typing.AbstractSet](WrappedRelationalAlgebraSet(iterable))


def check_probchoice_probs_sum_to_one(ra_set):
    probs_sum = sum(v.value[0].value for v in ra_set.value)
    if not numpy.isclose(probs_sum, 1.0):
        raise NeuroLangException(
            "Probabilities of probabilistic choice should sum to 1"
        )


def get_dterm(datom):
    return next(arg for arg in datom.args if isinstance(arg, DeltaTerm))


def get_dterm_index(datom):
    return next(
        i for i, arg in enumerate(datom.args) if isinstance(arg, DeltaTerm)
    )


def concatenate_to_expression_block(block, to_add):
    """
    Extend `ExpressionBlock` with another `ExpressionBlock` or
    an iterable of `Expression`.

    Parameters
    ----------
    block: ExpressionBlock
        The initial `ExpressionBlock` to which expressions will be added.
    to_add: ExpressionBlock or Expression iterable
        `Expression`s to be added to the `ExpressionBlock`.

    Returns
    -------
    new_block: ExpressionBlock
        A new `ExpressionBlock` containing the new expressions.

    """
    if isinstance(to_add, ExpressionBlock):
        return ExpressionBlock(block.expressions + to_add.expressions)
    if isinstance(to_add, typing.Iterable):
        if not all(isinstance(item, Expression) for item in to_add):
            raise NeuroLangException("Expected iterable on expressions")
        return ExpressionBlock(block.expressions + tuple(to_add))
    raise NeuroLangException("Expected ExpressionBlock or Expression iterable")


def get_antecedent_constant_indexes(rule):
    """Get indexes of constants occurring in antecedent predicates."""
    constant_indexes = dict()
    for antecedent_atom in extract_logic_predicates(rule.antecedent):
        predicate = antecedent_atom.functor.name
        indexes = {
            i
            for i, arg in enumerate(antecedent_atom.args)
            if isinstance(arg, Constant)
        }
        if len(indexes) > 0:
            constant_indexes[predicate] = indexes
    return constant_indexes


def get_predicate_probabilistic_rules(ppdl, predicate):
    if predicate not in ppdl.symbol_table:
        return set()
    return set(
        rule
        for rule in ppdl.symbol_table[predicate].formulas
        if is_ppdl_rule(rule)
    )
