import logging
from functools import reduce

import numpy as np

from ..expression_walker import ChainedWalker, ReplaceExpressionWalker
from ..expressions import Symbol
from ..logic import (
    Conjunction,
    Disjunction,
    ExistentialPredicate,
    NaryLogicOperator,
    Negation,
)
from ..logic.expression_processing import (
    extract_logic_atoms,
    extract_logic_free_variables,
    extract_logic_predicates,
)
from ..logic.transformations import (
    CollapseConjunctions,
    CollapseDisjunctions,
    DistributeConjunctions,
    DistributeDisjunctions,
    ExtractConjunctiveQueryWithNegation,
    GuaranteeConjunction,
    GuaranteeDisjunction,
    MoveNegationsToAtomsInFONegE,
    PushExistentialsDown,
    RemoveDuplicatedConjunctsDisjuncts,
    RemoveTrivialOperations,
    RemoveUniversalPredicates,
    convert_to_pnf_with_dnf_matrix,
)
from ..logic.unification import compose_substitutions, most_general_unifier
from .containment import is_contained

LOG = logging.getLogger(__name__)


GC = GuaranteeConjunction()
GD = GuaranteeDisjunction()
MNA = MoveNegationsToAtomsInFONegE()
PED = PushExistentialsDown()
RTO = RemoveTrivialOperations()


def minimize_ucq_in_cnf(query):
    """Convert UCQ to CNF form
    and minimise.

    Parameters
    ----------
    query : LogicExpression.
        query in UCQ semantics.

    Returns
    -------
    LogicExpression
        minimised query in UCQ semantics.
    """
    query = convert_to_cnf_ucq(query)
    head_variables = extract_logic_free_variables(query)
    cq_d_min = Conjunction(
        tuple(
            minimize_component_disjunction(c, head_variables)
            for c in query.formulas
        )
    )

    simplify = ChainedWalker(
        MoveNegationsToAtomsInFONegE,
        PushExistentialsDown,
        RemoveDuplicatedConjunctsDisjuncts,
        RemoveTrivialOperations,
        GuaranteeConjunction,
    )

    cq_min = minimize_component_conjunction(cq_d_min, head_variables)
    cq_min = add_existentials_except(cq_min, head_variables)
    return simplify.walk(cq_min)


def minimize_ucq_in_dnf(query):
    """Convert UCQ to DNF form
    and minimise.

    Parameters
    ----------
    query : LogicExpression.
        query in UCQ semantics.

    Returns
    -------
    LogicExpression
        minimised query in UCQ semantics.
    """
    query = convert_to_dnf_ucq(query)
    head_variables = extract_logic_free_variables(query)
    cq_d_min = Disjunction(
        tuple(
            minimize_component_conjunction(c, head_variables)
            for c in query.formulas
        )
    )

    simplify = ChainedWalker(
        MoveNegationsToAtomsInFONegE,
        PushExistentialsDown,
        RemoveDuplicatedConjunctsDisjuncts,
        RemoveTrivialOperations,
        GuaranteeDisjunction,
    )

    cq_min = minimize_component_disjunction(cq_d_min, head_variables)
    cq_min = add_existentials_except(cq_min, head_variables)
    return simplify.walk(cq_min)


def convert_rule_to_ucq(implication):
    """Convert datalog rule to logic UCQ.
    A UCQ is defined by a logic expression in
    where the only quantifier is existential.

    Parameters
    ----------
    expression : Implication
        Datalog rule.

    Returns
    -------
    LogicExpression
       UCQ with the same ground set as the
       input datalog rule.
    """
    implication = RTO.walk(implication)
    consequent, antecedent = implication.unapply()
    antecedent = MNA.walk(antecedent)
    head_vars = set(consequent.args)
    existential_vars = extract_logic_free_variables(antecedent) - set(
        head_vars
    )
    for a in existential_vars:
        antecedent = ExistentialPredicate(a, antecedent)
    return RTO.walk(PED.walk(antecedent))


def convert_to_cnf_ucq(expression):
    """Convert logic UCQ to
    conjunctive normal from (CNF).

    Parameters
    ----------
    expression : LogicExpression
        UCQ.

    Returns
    -------
    LogicExpression
       equivalent UCQ in CNF form.
    """
    expression = RTO.walk(expression)
    expression = Conjunction((expression,))
    c = ChainedWalker(
        MoveNegationsToAtomsInFONegE,
        PushExistentialsDown,
        DistributeDisjunctions,
        CollapseConjunctions,
        CollapseDisjunctions,
    )
    return c.walk(expression)


def convert_to_dnf_ucq(expression):
    """Convert logic UCQ to
    disjunctive normal from (DNF).

    Parameters
    ----------
    expression : LogicExpression
        UCQ.

    Returns
    -------
    LogicExpression
       equivalent UCQ in DNF form.
    """
    expression = RTO.walk(expression)
    expression = Disjunction((expression,))
    c = ChainedWalker(
        MoveNegationsToAtomsInFONegE,
        PushExistentialsDown,
        DistributeConjunctions,
        CollapseDisjunctions,
        CollapseConjunctions,
    )
    return c.walk(expression)


def minimize_component_disjunction(disjunction, head_vars=None):
    """Given a disjunction of queries Q1  ∨ ... ∨ Qn
    remove each query Qi such that exists Qj and
    Qi → Qj.

    Parameters
    ----------
    disjunction : Disjunction
        Disjunction of logical formulas to minimise.

    head_vars: set
        variables to be considered as constants

    Returns
    -------
    Disjunction
        Minimised disjunction.
    """

    if head_vars is None:
        head_vars = {}

    if not isinstance(disjunction, Disjunction):
        return disjunction
    positive_formulas, negative_formulas = split_positive_negative_formulas(
        disjunction
    )
    keep = minimise_formulas_containment(
        positive_formulas, is_contained, head_vars
    ) + tuple(negative_formulas)

    return GD.walk(RTO.walk(Disjunction(keep)))


def minimize_component_conjunction(conjunction, head_vars=None):
    """Given a conjunction of queries Q1 ∧ ... ∧ Qn
    remove each query Qi such that exists Qj and
    Qj → Qi.

    Parameters
    ----------
    conjunction : Conjunction
        conjunction of logical formulas to minimise.

    head_vars: set
        variables to be considered as constants

    Returns
    -------
    Conjunction
        minimised conjunction.
    """

    LOG.debug("About to minimize conjunction %s", conjunction)

    if head_vars is None:
        head_vars = {}

    if not isinstance(conjunction, Conjunction):
        return conjunction
    positive_formulas, negative_formulas = split_positive_negative_formulas(
        conjunction
    )
    keep = minimise_formulas_containment(
        positive_formulas, lambda x, y: is_contained(y, x), head_vars
    ) + tuple(negative_formulas)

    return GC.walk(RTO.walk(Conjunction(keep)))


def split_positive_negative_formulas(nary_logic_operation):
    """Split formulas of the n_ary_logic operation in those
    containing a negated predicate and those not.

    Parameters
    ----------
    nary_logic_operation : NAryLogicOperation
        Operation whose formulas are going to be split

    Returns
    -------
    positive, negative
        two Iterable[Union[LogicOperation, FunctionApplication]] objects
        containing the positive and negative formulas
    """

    formulas = nary_logic_operation.formulas
    positive_formulas = []
    negative_formulas = []
    for formula in formulas:
        if any(
            isinstance(sub_formula, Negation)
            for sub_formula in extract_logic_predicates(formula)
        ):
            negative_formulas.append(formula)
        else:
            positive_formulas.append(formula)
    return positive_formulas, negative_formulas


def minimise_formulas_containment(components, containment_op, head_vars):
    components_fv = [extract_logic_free_variables(c) for c in components]
    keep = tuple()
    containments = {}
    for i, c in enumerate(components):
        for j, c_ in enumerate(components):
            if i == j:
                continue
            c_fv = components_fv[i] & components_fv[j]
            q = add_existentials_except(c, c_fv | head_vars)
            q_ = add_existentials_except(c_, c_fv | head_vars)
            is_contained = containments.setdefault(
                (i, j), containment_op(q_, q)
            )
            LOG.debug("Checking containment %s <= %s: %s", q, q_, is_contained)
            if is_contained and not (
                j < i and containments.get((j, i), False)
            ):
                break
        else:
            keep += (c,)
    return keep


def add_existentials_except(query, variables):
    """Existentially-quantify each free variable in query
    except for those in variables

    Parameters
    ----------
    query : LogicExpression
        logic expression to add the existential quantifiers to.
    variables : Iterable of Symbol
        variables to exclude from existential quantification.

    Returns
    -------
    LogicExpression
        logic expression with existentially-quantified variables
        added.
    """
    fv = extract_logic_free_variables(query) - variables
    for v in fv:
        query = ExistentialPredicate(v, query)
    return query


def unify_existential_variables(query):
    """Reduce the number of existentially-quantified variables.
    Specifically if query is an UCQ and can be rewritten in DNF such that
    Q = ∃x.Q1 ∨ ∃y.Q2 and x in Q1 unifies with y in
    Q2, then Q is transformed to Q = ∃f.(Q1[f/x] ∨ Q2[f/y])
    where f is possibly a fresh variable.

    Parameters
    ----------
    query : LogicExpression
        UCQ expression to unify existential variables if possible.

    Returns
    -------
    LogicExpression
        logic expression with existential variables unifies
    """
    original_query = query
    query = convert_to_pnf_with_dnf_matrix(query)
    query = RTO.walk(RemoveUniversalPredicates().walk(query))
    variables_to_project = extract_logic_free_variables(query)
    while isinstance(query, ExistentialPredicate):
        query = query.body
    if not isinstance(query, Disjunction):
        return original_query

    unifiers = []
    for i, clause in enumerate(query.formulas):
        atoms = extract_logic_atoms(clause)
        for clause_ in query.formulas[i + 1 :]:
            atoms_ = extract_logic_atoms(clause_)
            unifiers += [
                most_general_unifier(a, a_) for a in atoms for a_ in atoms_
            ]
    unifiers = reduce(
        compose_substitutions, (u[0] for u in unifiers if u is not None), {}
    )
    unifiers = {
        k: v
        for k, v in unifiers.items()
        if variables_to_project.isdisjoint((k, v))
    }
    for i in range(len(unifiers)):
        query = ReplaceExpressionWalker(unifiers).walk(query)
    query = add_existentials_except(query, variables_to_project)
    return query


def convert_ucq_to_ccq(rule, transformation="CNF"):
    """Given a UCQ expression, this function transforms it into a connected
    component expression following the definition provided by Dalvi Suciu[1]
    in section 2.6 Special Query Expressions. The transformation can be
    parameterized to apply a CNF or DNF transformation, as needed.

    [1] Dalvi, N. & Suciu, D. The dichotomy of probabilistic inference
    for unions of conjunctive queries. J. ACM 59, 1–87 (2012).

    Parameters
    ----------
    rule : Definition
        UCQ expression

    Returns
    -------
    NAryLogicOperation
        Transformation of the initial expression
        in a connected component query.
    """
    rule = PED.walk(rule)
    free_vars = extract_logic_free_variables(rule)
    existential_vars = set()
    for atom in extract_logic_atoms(rule):
        existential_vars.update(set(atom.args) - set(free_vars))

    conjunctions = ExtractConjunctiveQueryWithNegation().walk(rule)
    dic_components = extract_connected_components(
        conjunctions, existential_vars
    )

    fresh_symbols_expression = ReplaceExpressionWalker(dic_components).walk(
        rule
    )
    if transformation == "CNF":
        fresh_symbols_expression = convert_to_cnf_ucq(fresh_symbols_expression)
        minimize = minimize_ucq_in_cnf
        gcd = GuaranteeConjunction()
    elif transformation == "DNF":
        fresh_symbols_expression = convert_to_dnf_ucq(fresh_symbols_expression)
        minimize = minimize_ucq_in_dnf
        gcd = GuaranteeDisjunction()
    else:
        raise ValueError(f"Invalid transformation type: {transformation}")

    final_expression = ReplaceExpressionWalker(
        {v: k for k, v in dic_components.items()}
    ).walk(fresh_symbols_expression)
    final_expression = minimize(final_expression)

    return gcd.walk(final_expression)


def extract_connected_components(list_of_conjunctions, existential_vars):
    """Given a list of conjunctions, this function is in charge of calculating
    the connected components of each one. As a result, it returns a dictionary
    with the necessary transformations so that these conjunctions are replaced
    by symbols that preserve their free variables.

    Parameters
    ----------
    list_of_conjunctions : list
        List of conjunctions for which we want to
        calculate the connected components

    existential_vars : set
        Set of variables associated with existentials
        in the expressions that compose the list of conjunctions.

    Returns
    -------
    dict
        Dictionary of transformations where the keys are the expressions that
        compose the list of conjunctions and the values are the components
        by which they must be replaced
    """
    transformations = {}
    for f in list_of_conjunctions:
        c_matrix = args_co_occurence_graph(f, existential_vars)
        components = connected_components(c_matrix)

        if isinstance(f, ExistentialPredicate):
            transformations[f] = calc_new_fresh_symbol(f, existential_vars)
            continue

        for c in components:
            form = [f.formulas[i] for i in c]
            if len(form) > 1:
                form = Conjunction(form)
                transformations[form] = calc_new_fresh_symbol(
                    form, existential_vars
                )
            else:
                conj = Conjunction(form)
                transformations[form[0]] = calc_new_fresh_symbol(
                    conj, existential_vars
                )

    return transformations


def calc_new_fresh_symbol(formula, existential_vars):
    """Given a formula and a list of variables, this function creates a new
    symbol containing only the unbound variables of the formula.

    Parameters
    ----------
    formula : Definition
        Formula to be transformed.

    existential_vars : set
        Set of variables associated with existentials.

    Returns
    -------
    Symbol
        New symbol containing only the unbound variables of the formula.

    """
    cvars = extract_logic_free_variables(formula) - existential_vars

    fresh_symbol = Symbol.fresh()
    new_symbol = fresh_symbol(tuple(cvars))

    return new_symbol


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


def symbol_connected_components(expression):
    if not isinstance(expression, NaryLogicOperator):
        raise ValueError(
            "Connected components can only be computed "
            "for n-ary logic operators."
        )
    c_matrix = symbol_co_occurence_graph(expression)
    components = connected_components(c_matrix)

    operation = type(expression)
    return [
        operation(tuple(expression.formulas[i] for i in component))
        for component in components
    ]


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
        for j, formula_ in enumerate(expression.formulas[i + 1 :]):
            atom_symbols_ = set(
                a.functor for a in extract_logic_atoms(formula_)
            )
            if not atom_symbols.isdisjoint(atom_symbols_):
                c_matrix[i, i + 1 + j] = 1
                c_matrix[i + 1 + j, i] = 1
    return c_matrix


def args_co_occurence_graph(expression, variable_to_use=None):
    """Arguments co-ocurrence graph expressed as
    an adjacency matrix.

    Parameters
    ----------
    expression : LogicOperator
        logic expression for which the adjacency matrix is computed.

    Returns
    -------
    numpy.ndarray
        squared binary array where a component is 1 if there is a
        shared argument between two formulas of the
        logic expression.
    """

    if isinstance(expression, ExistentialPredicate):
        return np.ones((1,))

    c_matrix = np.zeros((len(expression.formulas),) * 2)
    for i, formula in enumerate(expression.formulas):
        f_args = set(b for a in extract_logic_atoms(formula) for b in a.args)
        if variable_to_use is not None:
            f_args &= variable_to_use
        for j, formula_ in enumerate(expression.formulas[i + 1 :]):
            f_args_ = set(
                b for a in extract_logic_atoms(formula_) for b in a.args
            )
            if variable_to_use is not None:
                f_args_ &= variable_to_use
            if not f_args.isdisjoint(f_args_):
                c_matrix[i, i + 1 + j] = 1
                c_matrix[i + 1 + j, i] = 1

    return c_matrix
