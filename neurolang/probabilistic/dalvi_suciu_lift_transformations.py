
from neurolang.logic.transformations import (
    GuaranteeConjunction, GuaranteeDisjunction,
    PushExistentialsDown, RemoveTrivialOperations,
)
from neurolang.expression_walker import ChainedWalker
from neurolang.probabilistic.transforms import (
    RTO, add_existentials_except,
    minimize_component_conjunction, minimize_component_disjunction,
)
from neurolang.expressions import FunctionApplication
from neurolang.logic import (
    Conjunction, Disjunction,
    ExistentialPredicate, NaryLogicOperator,
)
from neurolang.logic.expression_processing import (
    extract_logic_atoms, extract_logic_free_variables
)

import numpy as np

PED = PushExistentialsDown()


def match_existentials(component, existential_vars):
    c_args = get_component_args(component)
    free_vars = set(c_args) & set(existential_vars)
    if len(free_vars) > 0:
        for fv in free_vars:
            component = PED.walk(ExistentialPredicate(fv, component))

    return component


def minimize_component_conj(query):
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


    cq_min = simplify.walk(cq_min)

    return args_connected_components_cnf(cq_min)

    #return simplify.walk(cq_min)

def minimize_component_disj(query):
    head_variables = extract_logic_free_variables(query)
    if isinstance(query, NaryLogicOperator):
        cq_d_min = Disjunction(tuple(
            minimize_component_conjunction(c)
            for c in query.formulas
        ))
    else:
        cq_d_min = minimize_component_conjunction(query)

    simplify = ChainedWalker(
        PushExistentialsDown,
        RemoveTrivialOperations,
        GuaranteeDisjunction,
    )

    cq_min = minimize_component_disjunction(cq_d_min)
    cq_min = add_existentials_except(cq_min, head_variables)
    cq_min = simplify.walk(cq_min)

    return args_connected_components_dnf(cq_min)

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

def args_connected_components_cnf(expression):
    if not isinstance(expression, NaryLogicOperator):
        return expression

    c_matrix = args_co_occurence_graph(expression)
    components = connected_components(c_matrix)

    operation = Disjunction
    res = [
        operation(tuple(expression.formulas[i] for i in component))
        for component in components
    ]

    return Conjunction(res)

def args_connected_components_dnf(expression):
    if not isinstance(expression, NaryLogicOperator):
        return expression

    c_matrix = args_co_occurence_graph(expression)
    components = connected_components(c_matrix)

    operation = Disjunction
    res = [
        operation(tuple(expression.formulas[i] for i in component))
        for component in components
    ]

    return Disjunction(res)

def args_co_occurence_graph(expression):
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

    #components = connected_components(c_matrix)

    #return [
    #    operation(tuple(expression.formulas[i] for i in component))
    #    for component in components
    #]

    return c_matrix

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