
from neurolang.logic.transformations import (
    GuaranteeConjunction, GuaranteeDisjunction,
    PushExistentialsDown, RemoveTrivialOperations,
)
from neurolang.expression_walker import ChainedWalker
from neurolang.probabilistic.transforms import (
    add_existentials_except,
    minimize_component_conjunction,
    minimize_component_disjunction,
)
from neurolang.logic import (
    Conjunction, Disjunction, NaryLogicOperator,
)
from neurolang.logic.expression_processing import (
    extract_logic_atoms, extract_logic_free_variables
)

import numpy as np

PED = PushExistentialsDown()


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

    return cq_min

def minimize_component_disj(query):
    head_variables = extract_logic_free_variables(query)
    if isinstance(query, NaryLogicOperator):
        cq_d_min = Disjunction(tuple(
            minimize_component_disjunction(c)
            for c in query.formulas
        ))
    else:
        cq_d_min = minimize_component_disjunction(query)

    simplify = ChainedWalker(
        PushExistentialsDown,
        RemoveTrivialOperations,
        GuaranteeDisjunction,
    )

    cq_min = minimize_component_disjunction(cq_d_min)
    cq_min = add_existentials_except(cq_min, head_variables)
    cq_min = simplify.walk(cq_min)

    return cq_min

def symbol_co_occurence_graph(expression):
    """Symbol co-ocurrence graph expressed as
    an adjacency matrix.

    Parameters
    ----------
    expression : NAryLogicExpression
        logic expression for which the adjacency matrix is computed.

    Returns
    -------
    numpy.ndarray
        squared binary array where a component is 1 if there is a
        shared predicate symbol between two subformulas of the
        logic expression.
    """
    c_matrix = np.zeros((len(expression.formulas),) * 2)
    for i, formula in enumerate(expression.formulas):
        atom_symbols = set(a.functor for a in extract_logic_atoms(formula))
        for j, formula_ in enumerate(expression.formulas[i + 1:]):
            atom_symbols_ = set(
                a.functor for a in extract_logic_atoms(formula_)
            )
            if not atom_symbols.isdisjoint(atom_symbols_):
                c_matrix[i, i + 1 + j] = 1
                c_matrix[i + 1 + j, i] = 1
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

def symbol_connected_components_cnf(expression):
    if not isinstance(expression, NaryLogicOperator):
        raise ValueError(
            "Connected components can only be computed "
            "for n-ary logic operators."
        )
    c_matrix = symbol_co_occurence_graph(expression)
    components = connected_components(c_matrix)

    operation = Disjunction
    return [
        operation(tuple(expression.formulas[i] for i in component))
        for component in components
    ]

def symbol_connected_components_dnf(expression):
    if not isinstance(expression, NaryLogicOperator):
        raise ValueError(
            "Connected components can only be computed "
            "for n-ary logic operators."
        )
    c_matrix = symbol_co_occurence_graph(expression)
    components = connected_components(c_matrix)

    operation = Conjunction
    return [
        operation(tuple(expression.formulas[i] for i in component))
        for component in components
    ]