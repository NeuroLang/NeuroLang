from functools import lru_cache, reduce
from itertools import chain, combinations

import numpy as np

from .. import relational_algebra_provenance as rap
from ..datalog.translate_to_named_ra import TranslateToNamedRA
from ..expression_walker import (
    ChainedWalker,
    PatternWalker,
    ReplaceExpressionWalker,
    add_match
)
from ..expressions import FunctionApplication, Symbol
from ..logic import (
    Conjunction,
    Disjunction,
    ExistentialPredicate,
    Implication,
    LogicOperator,
    NaryLogicOperator
)
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
    MakeExistentialsImplicit,
    PushExistentialsDown,
    RemoveTrivialOperations,
    convert_to_pnf_with_dnf_matrix
)
from ..logic.unification import compose_substitutions, most_general_unifier
from ..relational_algebra import (
    BinaryRelationalAlgebraOperation,
    NAryRelationalAlgebraOperation,
    RelationalAlgebraOperation,
    UnaryRelationalAlgebraOperation
)
from .containment import is_contained

__all__ = [
    "dalvi_suciu_lift",
]


def dalvi_suciu_lift(rule):
    if isinstance(rule, Implication):
        rule = convert_rule_to_e_query(rule)
    rule = RemoveTrivialOperations().walk(rule)
    if isinstance(rule, FunctionApplication):
        return TranslateToNamedRA().walk(rule)

    rule_cnf = minimize_rule_in_cnf(rule)
    connected_components = symbol_connected_components(rule_cnf)
    if len(connected_components) > 1:
        return components_plan(connected_components, rap.NaturalJoin)
    elif len(rule_cnf.formulas) > 1:
        return inclusion_exclusion_conjunction(rule_cnf)

    rule_dnf = minimize_rule_in_dnf(rule)
    connected_components = symbol_connected_components(rule_dnf)
    if len(connected_components) > 1:
        return components_plan(connected_components, rap.Union)
    elif has_separator_variables(rule_dnf):
        return separator_variable_plan(rule_dnf)

    return NonLiftable(rule)


def has_separator_variables(query):
    '''
    Returns true if `query` has a separator variable.

    According to Dalvi and Suciu [1]_ if `query` is in DNF,
    a variable z is called a separator variable if Q starts with ∃z,
    that is, Q = ∃z.Q1, for some query expression Q1, and (a) z
    is a root variable (i.e. it appears in every atom),
    (b) for every relation symbol R, there exists an attribute (R, iR)
    such that every atom with symbol R has z in position iR. This is
    equivalent, in datalog syntax, to Q ≡ Q0←∃z.Q1.

    Also, according to Suciu [2]_ the dual is also true,
    if `query` is in CNF i.e. the separation variable z needs to
    be universally quantified, that is Q = ∀x.Q1. But this is not
    implemented.

    [1] Dalvi, N. & Suciu, D. The dichotomy of probabilistic inference
    for unions of conjunctive queries. J. ACM 59, 1–87 (2012).
    [2] Suciu, D. Probabilistic Databases for All. in Proceedings of the
    39th ACM SIGMOD-SIGACT-SIGAI Symposium on Principles of Database Systems
    19–31 (ACM, 2020).
    '''

    return len(find_separator_variables(query)[0]) > 0


class NonLiftable(RelationalAlgebraOperation):
    def __init__(self, non_liftable_query):
        self.non_liftable_query = non_liftable_query

    def __repr__(self):
        return (
            "NonLiftable"
            f"({self.non_liftable_query})"
        )


def minimize_rule_in_cnf(query):
    query = convert_to_cnf_rule(query)
    head_variables = extract_logic_free_variables(query)
    query = MakeExistentialsImplicit().walk(query)
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

    for v in extract_logic_free_variables(cq_min) - head_variables:
        cq_min = ExistentialPredicate(v, cq_min)
    return simplify.walk(cq_min)


def minimize_rule_in_dnf(query):
    query = convert_to_dnf_rule(query)
    head_variables = extract_logic_free_variables(query)
    query = MakeExistentialsImplicit().walk(query)
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
    for v in extract_logic_free_variables(cq_min) - head_variables:
        cq_min = ExistentialPredicate(v, cq_min)
    return simplify.walk(cq_min)


def mobius_weights(formula_containments):
    _mobius_weights = {}
    for formula in formula_containments:
        _mobius_weights[formula] = mobius_function(
            formula, formula_containments, _mobius_weights
        )
    return _mobius_weights


def mobius_function(formula, formula_containments, known_weights=None):
    if known_weights is None:
        known_weights = dict()
    if formula in known_weights:
        return known_weights[formula]
    res = -sum(
        (
            known_weights.setdefault(
                f,
                mobius_function(f, formula_containments)
            )
            for f in formula_containments[formula]
            if f != formula
        ),
        -1
    )
    return res


def convert_rule_to_e_query(implication):
    implication = RemoveTrivialOperations().walk(implication)
    consequent, antecedent = implication.unapply()
    head_vars = set(consequent.args)
    existential_vars = (
        extract_logic_free_variables(antecedent) -
        set(head_vars)
    )
    for a in existential_vars:
        antecedent = ExistentialPredicate(a, antecedent)
    return PushExistentialsDown().walk(antecedent)


def convert_to_cnf_rule(expression):
    expression = RemoveTrivialOperations().walk(expression)
    expression = Conjunction((expression,))
    c = ChainedWalker(
        PushExistentialsDown,
        DistributeDisjunctions,
        CollapseConjunctions,
        CollapseDisjunctions,
    )
    return c.walk(expression)


def convert_to_dnf_rule(expression):
    expression = RemoveTrivialOperations().walk(expression)
    expression = Disjunction((expression,))
    c = ChainedWalker(
        PushExistentialsDown,
        DistributeConjunctions,
        CollapseDisjunctions,
        CollapseConjunctions,
    )
    return c.walk(expression)


def minimize_component_disjunction(disjunction):
    if not isinstance(disjunction, Disjunction):
        return disjunction
    components = disjunction.formulas
    components_fv = [
        extract_logic_free_variables(c)
        for c in components
    ]
    s = Symbol.fresh()
    keep = tuple()
    for i, c in enumerate(components):
        c_fv = components_fv[i]
        for j, c_ in enumerate(components):
            if i == j:
                continue
            head = s(*(c_fv & components_fv[j]))
            q = Implication(head, c)
            q_ = Implication(head, c_)
            if is_contained(q, q_):
                break
        else:
            keep += (c,)

    return Disjunction(keep)


def minimize_component_conjunction(conjunction):
    if not isinstance(conjunction, Conjunction):
        return conjunction
    components = conjunction.formulas
    components_fv = [
        extract_logic_free_variables(c)
        for c in components
    ]
    s = Symbol.fresh()
    keep = tuple()
    for i, c in enumerate(components):
        c_fv = components_fv[i]
        for j, c_ in enumerate(components):
            if i == j:
                continue
            head = s(*(c_fv & components_fv[j]))
            q = Implication(head, c)
            q_ = Implication(head, c_)
            if is_contained(q_, q):
                break
        else:
            keep += (c,)

    return Conjunction(keep)


def inclusion_exclusion_formulas(query):
    query_powerset = powerset(query.formulas)
    clean_powerset = set(
        set(q) if len(q) > 1 else set((q,))
        for q in query_powerset
    )
    result = tuple(
        Conjunction(tuple(q)) if len(q) > 1 else q.pop()
        for q in clean_powerset
    )
    return result


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


@lru_cache
def find_separator_variables(query):
    '''
    According to Dalvi and Suciu [1]_ if `query` is in DNF,
    a variable z is called a separator variable if Q starts with ∃z,
    that is, Q = ∃z.Q1, for some query expression Q1, and (a) z
    is a root variable (i.e. it appears in every atom),
    (b) for every relation symbol R, there exists an attribute (R, iR)
    such that every atom with symbol R has z in position iR. This is
    equivalent, in datalog syntax, to Q ≡ Q0←∃z.Q1.

    Also, according to Suciu [2]_ the dual is also true,
    if `query` is in CNF i.e. the separation variable z needs to
    be universally quantified, that is Q = ∀x.Q1.

    This algorithm assumes that Q1 can't be splitted into independent
    formulas.

    [1] Dalvi, N. & Suciu, D. The dichotomy of probabilistic inference
    for unions of conjunctive queries. J. ACM 59, 1–87 (2012).
    [2] Suciu, D. Probabilistic Databases for All. in Proceedings of the
    39th ACM SIGMOD-SIGACT-SIGAI Symposium on Principles of Database Systems
    19–31 (ACM, 2020).
    '''
    exclude_variables = extract_logic_free_variables(query)
    query = convert_to_pnf_with_dnf_matrix(query)
    query = _unify_existential_variables(query)
    query = MakeExistentialsImplicit().walk(query)

    if isinstance(query, NaryLogicOperator):
        formulas = query.formulas
    else:
        formulas = [query]

    candidates = None
    all_atoms = set()
    for formula in formulas:
        atoms = extract_logic_atoms(formula)
        all_atoms |= atoms
        root_variables = reduce(
            lambda y, x: set(x.args) & y,
            atoms[1:],
            set(atoms[0].args)
        )
        if candidates is None:
            candidates = root_variables
        else:
            candidates &= root_variables

    separator_variables = set()
    for var in candidates:
        atom_positions = {}
        for atom in all_atoms:
            functor = atom.functor
            pos_ = {i for i, v in enumerate(atom.args) if v == var}
            if any(
                pos_.isdisjoint(pos)
                for pos in atom_positions.setdefault(functor, [])
            ):
                break
            atom_positions[functor].append(pos_)
        else:
            separator_variables.add(var)

    return separator_variables - exclude_variables, query


def has_existential_quantifiers(query):
    return HasExistentialPredicates().walk(query)


class HasExistentialPredicates(PatternWalker):
    @add_match(FunctionApplication)
    def function_application(self, expression):
        return False

    @add_match(ExistentialPredicate)
    def existential_predicate(self, expression):
        return True

    @add_match(NaryLogicOperator)
    def nary(self, expression):
        return any(
            self.walk(f)
            for f in expression.formulas
        )

    @add_match(LogicOperator)
    def operator(self, expression):
        return any(
            self.walk(f)
            for f in expression.unapply()
        )


def _unify_existential_variables(query):
    original_query = query
    exclude_variables = extract_logic_free_variables(query)
    query = PushExistentialsDown().walk(query)
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
        if exclude_variables.isdisjoint((k, v))
    }
    for i in range(len(unifiers)):
        query = ReplaceExpressionWalker(unifiers).walk(query)
    new_formulas = tuple()
    for formula in query.formulas:
        while isinstance(formula, ExistentialPredicate):
            formula = formula.body
        new_formulas += (formula,)
    query = Disjunction(new_formulas)
    return query


class IsPureLiftedPlan(PatternWalker):
    @add_match(NonLiftable)
    def non_liftable(self, expression):
        return False

    @add_match(NAryRelationalAlgebraOperation)
    def nary(self, expression):
        return all(
            self.walk(relation)
            for relation in expression.relations
        )

    @add_match(BinaryRelationalAlgebraOperation)
    def binary(self, expression):
        return (
            self.walk(expression.relation_left) &
            self.walk(expression.relation_right)
        )

    @add_match(UnaryRelationalAlgebraOperation)
    def unary(self, expression):
        return self.walk(expression.relation)

    @add_match(...)
    def other(self, expression):
        return True


def is_pure_lifted_plan(query):
    return IsPureLiftedPlan().walk(query)


def separator_variable_plan(expression):
    variables_to_project = extract_logic_free_variables(expression)
    svs, expression = find_separator_variables(expression)
    expression = MakeExistentialsImplicit().walk(expression)
    existentials_to_add = (
        extract_logic_free_variables(expression) -
        variables_to_project -
        svs
    )
    for v in existentials_to_add:
        expression = ExistentialPredicate(v, expression)
    return rap.Projection(
        rap.Projection(
            dalvi_suciu_lift(expression),
            tuple(variables_to_project | svs)
        ),
        tuple(variables_to_project)
    )


def symbol_connected_components(expression):
    if not isinstance(expression, NaryLogicOperator):
        raise ValueError(
            "Connected components can only be computed "
            "for n-ary logic operators."
        )
    c_matrix = symbol_co_occurence_graph(expression)
    formula_idxs = set(range(len(expression.formulas)))
    components = []
    while formula_idxs:
        idx = formula_idxs.pop()
        component = {idx}
        component_follow = [idx]
        while component_follow:
            idx = component_follow.pop()
            idxs = set(c_matrix[idx].nonzero()[0]) - component
            component |= idxs
            component_follow += idxs
        components.append(component)
        formula_idxs -= component

    operation = type(expression)
    return [
        operation(tuple(expression.formulas[i] for i in component))
        for component in components
    ]


def symbol_co_occurence_graph(expression):
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


def components_plan(components, operation):
    formulas = []
    for component in components:
        formulas.append(dalvi_suciu_lift(component))
    return reduce(operation, formulas[1:], formulas[0])


def inclusion_exclusion_conjunction(expression):
    formula_powerset = []
    for formula in powerset(expression.formulas):
        if len(formula) == 0:
            continue
        elif len(formula) == 1:
            formula_powerset.append(formula[0])
        else:
            formula_powerset.append(Disjunction(tuple(formula)))
    formulas_weights = _formulas_weights(formula_powerset)
    new_formulas, weights = zip(*(
        (dalvi_suciu_lift(formula), weight)
        for formula, weight in formulas_weights.items()
        if weight != 0
    ))

    return rap.WeightedNaturalJoin(tuple(new_formulas), weights)


def _formulas_weights(formula_powerset):
    formula_containments = {
        formula: set()
        for formula in formula_powerset
    }
    for i, f0 in enumerate(formula_powerset):
        for f1 in formula_powerset[i + 1:]:
            for c0, c1 in ((f0, f1), (f1, f0)):
                if (
                    (c1 not in formula_containments[f0]) &
                    is_contained(c0, c1)
                ):
                    formula_containments[c0].add(c1)
                    formula_containments[c0] |= (
                        formula_containments[c1] -
                        {c0}
                    )
                    break

    formulas_weights = mobius_weights(formula_containments)
    return formulas_weights
