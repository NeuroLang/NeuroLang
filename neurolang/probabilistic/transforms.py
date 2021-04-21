from functools import reduce

from ..expression_walker import ChainedWalker, ReplaceExpressionWalker
from ..logic import Conjunction, Disjunction, ExistentialPredicate
from ..logic.expression_processing import (
    extract_logic_atoms,
    extract_logic_free_variables
)
from ..logic.transformations import (
    CollapseConjunctions,
    CollapseDisjunctions,
    DistributeConjunctions,
    DistributeDisjunctions,
    GuaranteeConjunction,
    GuaranteeDisjunction,
    PushExistentialsDown,
    RemoveTrivialOperations,
    RemoveUniversalPredicates,
    convert_to_pnf_with_dnf_matrix
)
from ..logic.unification import compose_substitutions, most_general_unifier
from .containment import is_contained

GC = GuaranteeConjunction()
GD = GuaranteeDisjunction()
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
    cq_d_min = Conjunction(tuple(
        minimize_component_disjunction(c)
        for c in query.formulas
    ))

    simplify = ChainedWalker(
        PushExistentialsDown,
        RemoveTrivialOperations,
        GuaranteeConjunction,
    )

    cq_min = minimize_component_conjunction(cq_d_min)
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
    cq_d_min = Disjunction(tuple(
        minimize_component_conjunction(c)
        for c in query.formulas
    ))

    simplify = ChainedWalker(
        PushExistentialsDown,
        RemoveTrivialOperations,
        GuaranteeDisjunction
    )

    cq_min = minimize_component_disjunction(cq_d_min)
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
    head_vars = set(consequent.args)
    existential_vars = (
        extract_logic_free_variables(antecedent) -
        set(head_vars)
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
        PushExistentialsDown,
        DistributeConjunctions,
        CollapseDisjunctions,
        CollapseConjunctions,
    )
    return c.walk(expression)


def minimize_component_disjunction(disjunction):
    """Given a disjunction of queries Q1  ∨ ... ∨ Qn
    remove each query Qi such that exists Qj and
    Qi → Qj.

    Parameters
    ----------
    disjunction : Disjunction
        Disjunction of logical formulas to minimise.

    Returns
    -------
    Conjunction
        Minimised conjunction.
    """
    if not isinstance(disjunction, Disjunction):
        return disjunction
    keep = minimise_formulas_containment(
        disjunction.formulas,
        is_contained
    )

    return GD.walk(RTO.walk(Disjunction(keep)))


def minimize_component_conjunction(conjunction):
    """Given a conjunction of queries Q1 ∧ ... ∧ Qn
    remove each query Qi such that exists Qj and
    Qj → Qi.

    Parameters
    ----------
    conjunction : Conjunction
        conjunction of logical formulas to minimise.

    Returns
    -------
    Conjunction
        minimised conjunction.
    """
    if not isinstance(conjunction, Conjunction):
        return conjunction
    keep = minimise_formulas_containment(
        conjunction.formulas,
        lambda x, y: is_contained(y, x)
    )

    return GC.walk(RTO.walk(Conjunction(keep)))


def minimise_formulas_containment(components, containment_op):
    components_fv = [
        extract_logic_free_variables(c)
        for c in components
    ]
    keep = tuple()
    containments = {}
    for i, c in enumerate(components):
        for j, c_ in enumerate(components):
            if i == j:
                continue
            c_fv = components_fv[i] & components_fv[j]
            q = add_existentials_except(c, c_fv)
            q_ = add_existentials_except(c_, c_fv)
            is_contained = containments.setdefault(
                (i, j), containment_op(q_, q)
            )
            if (
                is_contained and
                not (
                    j < i and
                    containments[(j, i)]
                )
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
    query = RemoveUniversalPredicates().walk(query)
    variables_to_project = extract_logic_free_variables(query)
    while isinstance(query, ExistentialPredicate):
        query = query.body
    if not isinstance(query, Disjunction):
        return original_query

    unifiers = []
    for i, clause in enumerate(query.formulas):
        atoms = extract_logic_atoms(clause)
        for clause_ in query.formulas[i + 1:]:
            atoms_ = extract_logic_atoms(clause_)
            unifiers += [
                most_general_unifier(a, a_)
                for a in atoms
                for a_ in atoms_
            ]
    unifiers = reduce(
        compose_substitutions,
        (u[0] for u in unifiers if u is not None),
        {}
    )
    unifiers = {
        k: v for k, v in unifiers.items()
        if variables_to_project.isdisjoint((k, v))
    }
    for i in range(len(unifiers)):
        query = ReplaceExpressionWalker(unifiers).walk(query)
    query = add_existentials_except(query, variables_to_project)
    return query
