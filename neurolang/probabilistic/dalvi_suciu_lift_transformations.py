
from neurolang.logic.transformations import GuaranteeConjunction, PushExistentialsDown, RemoveTrivialOperations
from neurolang.expression_walker import ChainedWalker
from neurolang.probabilistic.transforms import RTO, add_existentials_except, minimize_component_conjunction
from neurolang.expressions import FunctionApplication
from neurolang.logic import Conjunction, Disjunction, ExistentialPredicate, NaryLogicOperator
from neurolang.logic.expression_processing import extract_logic_atoms, extract_logic_free_variables

import numpy as np

PED = PushExistentialsDown()


def convert_rule_to_components_dnf(implication):
    """Convert datalog rule to CCQ.
    CCQ's are defined in Dalvi and Suciu - 2012 -
    The dichotomy of probabilistic inference for union.

    Parameters
    ----------
    expression : Implication
        Datalog rule.

    Returns
    -------
    LogicExpression
       CCQ with the same ground set as the
       input datalog rule.
    """
    implication = RTO.walk(implication)
    consequent, antecedent = implication.unapply()
    head_vars = set(consequent.args)
    existential_vars = (
        extract_logic_free_variables(antecedent) -
        set(head_vars)
    )

    ccq = []
    expression = Conjunction(plain_expression(antecedent))
    scc = args_connected_components(expression)
    for component in scc:
        for free_var in existential_vars:
            if free_var in get_component_args(component):
                ccq.append(ExistentialPredicate(free_var, component))
            else:
                ccq.append(component)

    new_ant = Disjunction(tuple(ccq))

    return RTO.walk(new_ant)

def convert_rule_to_components_cnf(implication):
    """Convert datalog rule to CCQ.
    CCQ's are defined in Dalvi and Suciu - 2012 -
    The dichotomy of probabilistic inference for union.

    Parameters
    ----------
    expression : Implication
        Datalog rule.

    Returns
    -------
    LogicExpression
       CCQ with the same ground set as the
       input datalog rule.
    """
    implication = RTO.walk(implication)
    consequent, antecedent = implication.unapply()
    head_vars = set(consequent.args)
    existential_vars = (
        extract_logic_free_variables(antecedent) -
        set(head_vars)
    )

    ccq = []
    expression = Disjunction(plain_expression(antecedent))

    scc = args_connected_components(expression)
    for component in scc:
        if len(existential_vars) > 1:
            ccq.append(match_existentials(component, existential_vars))
        else:
            ccq.append(component)

    new_ant = Conjunction(tuple(ccq))

    return RTO.walk(new_ant)

def match_existentials(component, existential_vars):
    c_args = get_component_args(component)
    free_vars = set(c_args) & set(existential_vars)
    if len(free_vars) > 0:
        for fv in free_vars:
            component = PED.walk(ExistentialPredicate(fv, component))

    return component


def minimize_component_query(query):
    head_variables = extract_logic_free_variables(query)
    if isinstance(query, NaryLogicOperator):
        cq_d_min = Conjunction(tuple(
            minimize_component_conjunction(c)
            for c in query.formulas
        ))
    else:
        cq_d_min = minimize_component_conjunction(query)

    simplify = ChainedWalker(
        PushExistentialsDown,
        RemoveTrivialOperations,
        GuaranteeConjunction,
    )

    cq_min = minimize_component_conjunction(cq_d_min)
    cq_min = add_existentials_except(cq_min, head_variables)
    return simplify.walk(cq_min)

def get_component_args(q):
    args = ()
    if isinstance(q, NaryLogicOperator):
        formulas = q.formulas
        for f in formulas:
            args = args + f.args
    else:
        args = q.args

    return args

def plain_expression(expression):
    if isinstance(expression, FunctionApplication):
        return (expression,)

    atoms = ()
    for f in expression.formulas:
        atom = plain_expression(f)
        atoms = atoms  + atom

    return atoms

def args_connected_components(expression):
    if isinstance(expression, FunctionApplication):
        return [expression]

    c_matrix = np.zeros((len(expression.formulas),) * 2)
    for i, formula in enumerate(expression.formulas):
        f_args = set(b for a in extract_logic_atoms(formula) for b in a.args)
        for j, formula_ in enumerate(expression.formulas[i + 1:]):
            f_args_ = set(b for a in extract_logic_atoms(formula_) for b in a.args)
            if not f_args.isdisjoint(f_args_):
                c_matrix[i, i + 1 + j] = 1
                c_matrix[i + 1 + j, i] = 1

    components = connected_components(c_matrix)

    operation = type(expression)
    return [
        operation(tuple(expression.formulas[i] for i in component))
        for component in components
    ]

def connected_components(adjacency_matrix):
    """Connected components of an undirected graph.

    Parameters
    ----------
    adjacency_matrix : numpy.ndarray
        squared array representing the adjacency
        matrix of an undirected graph.

    Returns
    -------
    list of integer sets
        connected components of the graph.
    """
    node_idxs = set(range(adjacency_matrix.shape[0]))
    components = []
    while node_idxs:
        idx = node_idxs.pop()
        component = {idx}
        component_follow = [idx]
        while component_follow:
            idx = component_follow.pop()
            idxs = set(adjacency_matrix[idx].nonzero()[0]) - component
            component |= idxs
            component_follow += idxs
        components.append(component)
        node_idxs -= component
    return components