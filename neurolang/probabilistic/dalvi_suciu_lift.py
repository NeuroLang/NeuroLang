import logging
from functools import reduce
from itertools import chain, combinations

import numpy as np
from typing_inspect import NEW_TYPING

from .. import relational_algebra_provenance as rap
from ..datalog.expression_processing import (
    UnifyVariableEqualities,
    flatten_query,
)
from ..datalog.translate_to_named_ra import TranslateToNamedRA
from ..exceptions import NonLiftableException
from ..expression_walker import (
    PatternWalker,
    ReplaceExpressionWalker,
    add_match,
)
from ..expressions import Constant, FunctionApplication, Symbol
from ..logic import (
    FALSE,
    Conjunction,
    Disjunction,
    ExistentialPredicate,
    Implication,
    NaryLogicOperator,
)
from ..logic.expression_processing import (
    ExtractFreeVariablesWalker,
    extract_logic_atoms,
    extract_logic_free_variables,
)
from ..logic.transformations import (
    CollapseConjunctions,
    GuaranteeDisjunction,
    IdentifyPureConjunctions,
    MakeExistentialsImplicit,
    PushExistentialsDown,
    RemoveExistentialPredicates,
    RemoveTrivialOperations,
    GuaranteeConjunction,
)
from ..relational_algebra import (
    BinaryRelationalAlgebraOperation,
    ColumnStr,
    NamedRelationalAlgebraFrozenSet,
    NAryRelationalAlgebraOperation,
    Projection,
    UnaryRelationalAlgebraOperation,
    str2columnstr_constant,
)
from ..relational_algebra_provenance import ProvenanceAlgebraSet
from ..utils import OrderedSet, log_performance
from .containment import is_contained
from .small_dichotomy_theorem_based_solver import (
    RAQueryOptimiser,
    lift_optimization_for_choice_predicates,
)
from .exceptions import NotEasilyShatterableError
from .probabilistic_ra_utils import (
    DeterministicFactSet,
    NonLiftable,
    ProbabilisticFactSet,
    generate_probabilistic_symbol_table_for_query,
)
from .probabilistic_semiring_solver import ProbSemiringSolver
from .query_resolution import lift_solve_marg_query
from .shattering import shatter_easy_probfacts
from .transforms import (
    convert_rule_to_ucq,
    convert_to_cnf_ucq,
    convert_to_dnf_ucq,
    minimize_ucq_in_cnf,
    minimize_ucq_in_dnf,
    unify_existential_variables,
)
from .small_dichotomy_theorem_based_solver import (
    _project_on_query_head,
    _maybe_reintroduce_head_variables,
)
from neurolang.logic import transformations

LOG = logging.getLogger(__name__)


__all__ = [
    "dalvi_suciu_lift",
    "solve_succ_query",
    "solve_marg_query",
]

RTO = RemoveTrivialOperations()
PED = PushExistentialsDown()


def solve_succ_query(query, cpl_program):
    """
    Solve a SUCC query on a CP-Logic program.

    Parameters
    ----------
    query : Implication
        SUCC query of the form `ans(x) :- P(x)`.
    cpl_program : CPLogicProgram
        CP-Logic program on which the query should be solved.

    Returns
    -------
    ProvenanceAlgebraSet
        Provenance set labelled with probabilities for each tuple in the result
        set.

    """
    with log_performance(
        LOG,
        "Preparing query %s",
        init_args=(query.consequent.functor.name,),
    ):
        flat_query_body = flatten_query(query.antecedent, cpl_program)

    if flat_query_body == FALSE or (
        isinstance(flat_query_body, Conjunction)
        and any(conjunct == FALSE for conjunct in flat_query_body.formulas)
    ):
        head_var_names = tuple(
            term.name for term in query.consequent.args
            if isinstance(term, Symbol)
        )
        return ProvenanceAlgebraSet(
            NamedRelationalAlgebraFrozenSet(("_p_",) + head_var_names),
            ColumnStr("_p_"),
        )

    with log_performance(LOG, "Translation and lifted optimisation"):
        flat_query_body = GuaranteeConjunction().walk(
            lift_optimization_for_choice_predicates(
                flat_query_body, cpl_program
            )
        )
        flat_query = Implication(query.consequent, flat_query_body)
        symbol_table = generate_probabilistic_symbol_table_for_query(
            cpl_program, flat_query_body
        )
        unified_query = UnifyVariableEqualities().walk(flat_query)
        try:
            shattered_query = symbolic_shattering(unified_query, symbol_table)
        except NotEasilyShatterableError:
            shattered_query = unified_query

        ra_query = dalvi_suciu_lift(shattered_query, symbol_table)
        if not is_pure_lifted_plan(ra_query):
            LOG.info(
                "Query not liftable %s",
                shattered_query
            )
            raise NonLiftableException(
                "Query %s not liftable, algorithm can't be applied",
                query
            )
        # project on query's head variables
        ra_query = _project_on_query_head(ra_query, shattered_query)
        # re-introduce head variables potentially removed by unification
        ra_query = _maybe_reintroduce_head_variables(
            ra_query, flat_query, unified_query
        )
        ra_query = RAQueryOptimiser().walk(ra_query)

    with log_performance(LOG, "Run RAP query"):
        solver = ProbSemiringSolver(symbol_table)
        prob_set_result = solver.walk(ra_query)

    return prob_set_result


def solve_marg_query(rule, cpl):
    return lift_solve_marg_query(rule, cpl, solve_succ_query)


def dalvi_suciu_lift(rule, symbol_table):
    '''
    Translation from a datalog rule which allows disjunctions in the body
    to a safe plan according to [1]_. Non-liftable segments are identified
    by the `NonLiftable` expression.
    [1] Dalvi, N. & Suciu, D. The dichotomy of probabilistic inference
    for unions of conjunctive queries. J. ACM 59, 1–87 (2012).
    '''
    if isinstance(rule, Implication):
        # Tomar variables libres en el metodo y agregar transformacion a UCQ
        rule = convert_ucq_to_ccq(rule)
    rule = RTO.walk(rule)
    if (
        isinstance(rule, FunctionApplication) or
        all(
            is_atom_a_deterministic_relation(atom, symbol_table)
            for atom in extract_logic_atoms(rule)
        )
    ):
        free_vars = extract_logic_free_variables(rule)
        rule = MakeExistentialsImplicit().walk(rule)
        result = TranslateToNamedRA().walk(rule)
        proj_cols = tuple(Constant(ColumnStr(v.name)) for v in free_vars)
        return Projection(result, proj_cols)

    rule_cnf = minimize_ucq_in_cnf(rule)
    connected_components = symbol_connected_components(rule_cnf)
    if len(connected_components) > 1:
        return components_plan(
            connected_components, rap.NaturalJoin, symbol_table
        )
    elif len(rule_cnf.formulas) > 1:
        return inclusion_exclusion_conjunction(rule_cnf, symbol_table)

    rule_dnf = minimize_ucq_in_dnf(rule)
    connected_components = symbol_connected_components(rule_dnf)
    if len(connected_components) > 1:
        return components_plan(connected_components, rap.Union, symbol_table)
    else:
        has_svs, plan = has_separator_variables(rule_dnf, symbol_table)
        if has_svs:
            return plan

    return NonLiftable(rule)


def convert_ucq_to_ccq(rule, transformation='CNF'):
    implication = RTO.walk(rule)
    consequent, antecedent = implication.unapply()
    head_vars = set(consequent.args)
    existential_vars = set(
        extract_logic_free_variables(antecedent) -
        set(head_vars)
    )

    for a in existential_vars:
        antecedent = ExistentialPredicate(a, antecedent)
    expression = PED.walk(RTO.walk(antecedent))

    conjunctions = IdentifyPureConjunctions().walk(expression)
    dic_components = extract_connected_components(conjunctions, existential_vars)

    fresh_symbols_expression = ReplaceExpressionWalker(dic_components).walk(expression)
    if transformation == 'CNF':
        fresh_symbols_expression = convert_to_cnf_ucq(fresh_symbols_expression)
        GCD = GuaranteeConjunction()
    elif transformation == 'DNF':
        GCD = GuaranteeDisjunction()
        fresh_symbols_expression = convert_to_dnf_ucq(fresh_symbols_expression)
    else:
        raise ValueError('Invalid transformation type')

    final_expression = fresh_symbols_to_components(dic_components, fresh_symbols_expression)
    final_expression = GCD.walk(PED.walk(final_expression))

    return final_expression


def extract_connected_components(list_of_conjunctions, existential_vars):
    transformations = {}
    for f in list_of_conjunctions:
        c_matrix = args_co_occurence_graph(f)
        components = connected_components(c_matrix)

        if isinstance(f, ExistentialPredicate):
            transformations[f] = calc_new_fresh_symbol(f, existential_vars)
            continue

        for c in components:
            form = [f.formulas[i] for i in c]
            if len(form) > 1:
                form = Conjunction(form)
                transformations[form] = calc_new_fresh_symbol(form, existential_vars)
            else:
                conj = Conjunction(form)
                transformations[form[0]] = calc_new_fresh_symbol(conj, existential_vars)

    return transformations

def calc_new_fresh_symbol(formula, existential_vars):
    cvars = extract_logic_free_variables(formula) - existential_vars

    fresh_symbol = Symbol.fresh()
    new_symbol = fresh_symbol(tuple(cvars))

    return new_symbol

def fresh_symbols_to_components(dic_replacements, expression):
    dic_replacements = {v: k for k, v in dic_replacements.items()}
    expression = ReplaceExpressionWalker(dic_replacements).walk(expression)

    return expression


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
        for j, formula_ in enumerate(expression.formulas[i + 1:]):
            atom_symbols_ = set(
                a.functor for a in extract_logic_atoms(formula_)
            )
            if not atom_symbols.isdisjoint(atom_symbols_):
                c_matrix[i, i + 1 + j] = 1
                c_matrix[i + 1 + j, i] = 1
    return c_matrix


def args_co_occurence_graph(expression):
    """Arguments co-ocurrence graph expressed as
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
    if isinstance(expression, ExistentialPredicate):
        return np.ones((1,))

    c_matrix = np.zeros((len(expression.formulas),) * 2)
    for i, formula in enumerate(expression.formulas):
        #f_args = ExtractFreeVariables()
        f_args = set(b for a in extract_logic_atoms(formula) for b in a.args)
        for j, formula_ in enumerate(expression.formulas[i + 1:]):
            f_args_ = set(b for a in extract_logic_atoms(formula_) for b in a.args)
            if not f_args.isdisjoint(f_args_):
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


def has_separator_variables(query, symbol_table):
    """Returns true if `query` has a separator variable and the plan.

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

    Parameters
    ----------
    query : LogicExpression
        UCQ to check if it can have a plan based on a separator variable
        strategy.
    symbol_table : dict
        dictionary of symbols and probabilistic/deterministic fact sets.

    Returns
    -------
    boolean, Expression
        Returns true and the plan if the query has separation variables,
        if not, False and None.
    """
    svs, expression = find_separator_variables(query, symbol_table)
    if len(svs) > 0:
        return True, separator_variable_plan(
            expression, svs, symbol_table
        )
    else:
        return False, None


def find_separator_variables(query, symbol_table):
    '''
    According to Dalvi and Suciu [1]_ if `query` is rewritten in prenex
    normal form (PNF) with a DNF matrix, then a variable z is called a
    separator variable if Q starts with ∃z that is, Q = ∃z.Q1, for some
    query expression Q1, and:
      a. z is a root variable (i.e. it appears in every atom); and
      b. for every relation symbol R in Q1, there exists an attribute (R, iR)
      such that every atom with symbol R has z in position iR.

    This algorithm assumes that Q1 can't be splitted into independent
    formulas.

    .. [1] Dalvi, N. & Suciu, D. The dichotomy of probabilistic inference
    for unions of conjunctive queries. J. ACM 59, 1–87 (2012).
    .. [2] Suciu, D. Probabilistic Databases for All. in Proceedings of the
    39th ACM SIGMOD-SIGACT-SIGAI Symposium on Principles of Database Systems
    19–31 (ACM, 2020).
    '''
    exclude_variables = extract_logic_free_variables(query)
    query = unify_existential_variables(query)

    if isinstance(query, NaryLogicOperator):
        formulas = query.formulas
    else:
        formulas = [query]

    candidates = extract_probabilistic_root_variables(formulas, symbol_table)
    all_atoms = extract_logic_atoms(query)

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


def extract_probabilistic_root_variables(formulas, symbol_table):
    candidates = None
    for formula in formulas:
        probabilistic_atoms = OrderedSet(
            atom
            for atom in extract_logic_atoms(formula)
            if not is_atom_a_deterministic_relation(atom, symbol_table)
        )
        if len(probabilistic_atoms) == 0:
            continue
        root_variables = reduce(
            lambda y, x: set(x.args) & y,
            probabilistic_atoms[1:],
            set(probabilistic_atoms[0].args)
        )
        if candidates is None:
            candidates = root_variables
        else:
            candidates &= root_variables
    if candidates is None:
        candidates = set()
    return candidates


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


def separator_variable_plan(expression, separator_variables, symbol_table):
    """Generate RAP plan for the logic query expression assuming it has
    the given separator varialbes.

    Parameters
    ----------
    expression : LogicExpression
        expression to generate the plan for.
    separator_variables : Set[Symbol]
        separator variables for expression.
    symbol_table : mapping of symbol to probabilistic table.
        mapping used to figure out specific strategies according to
        the probabilistic table type.

    Returns
    -------
    RelationalAlgebraOperation
        plan for the logic expression.
    """
    variables_to_project = extract_logic_free_variables(expression)
    expression = MakeExistentialsImplicit().walk(expression)
    existentials_to_add = (
        extract_logic_free_variables(expression) -
        variables_to_project -
        separator_variables
    )
    for v in existentials_to_add:
        expression = ExistentialPredicate(v, expression)
    return rap.Projection(
        dalvi_suciu_lift(expression, symbol_table),
        tuple(
            str2columnstr_constant(v.name)
            for v in variables_to_project
        )
    )


def variable_co_occurrence_graph(expression):
    """Free variable co-ocurrence graph expressed as
    an adjacency matrix.

    Parameters
    ----------
    expression : NAryLogicExpression
        logic expression for which the adjacency matrixis computed.

    Returns
    -------
    numpy.ndarray
        squared binary array where a component is 1 if there is a
        shared free variable between two subformulas of the logic expression.
    """
    c_matrix = np.zeros((len(expression.formulas),) * 2)
    for i, formula in enumerate(expression.formulas):
        free_variables = extract_logic_free_variables(formula)
        for j, formula_ in enumerate(expression.formulas[i + 1:]):
            free_variables_ = extract_logic_free_variables(formula_)
            if not free_variables.isdisjoint(free_variables_):
                c_matrix[i, i + 1 + j] = 1
                c_matrix[i + 1 + j, i] = 1
    return c_matrix


def components_plan(components, operation, symbol_table):
    formulas = []
    for component in components:
        formulas.append(dalvi_suciu_lift(component, symbol_table))
    return reduce(operation, formulas[1:], formulas[0])


def inclusion_exclusion_conjunction(expression, symbol_table):
    """Produce a RAP query plan for the conjunction logic query
    based on the inclusion exclusion formula.

    Parameters
    ----------
    expression : Conjunction
        logic expression to produce a plan for.
    symbol_table : mapping of symbol to probabilistic table.
        mapping used to figure out specific strategies according to
        the probabilistic table type.

    Returns
    -------
    RelationalAlgebraOperation
        plan for the logic expression.
    """
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
        (dalvi_suciu_lift(formula, symbol_table), weight)
        for formula, weight in formulas_weights.items()
        if weight != 0
    ))

    return rap.WeightedNaturalJoin(
        tuple(new_formulas),
        tuple(Constant[int](w) for w in weights)
    )


def _formulas_weights(formula_powerset):
    # Using list set instead of a dictionary
    # due to a strange bug in dictionary lookups with Expressions
    # as keys
    formula_containments = [
        set()
        for formula in formula_powerset
    ]

    for i, f0 in enumerate(formula_powerset):
        for j, f1 in enumerate(formula_powerset[i + 1:], start=i + 1):
            for i0, c0, i1, c1 in ((i, f0, j, f1), (j, f1, i, f0)):
                if (
                    c0 not in formula_containments[i1] and
                    is_contained(c0, c1)
                ):
                    formula_containments[i0] |= (
                        {c1} | formula_containments[i1] - {c0}
                    )

    fcs = {
        formula: containment
        for formula, containment in
        zip(formula_powerset, formula_containments)
    }
    formulas_weights = mobius_weights(fcs)
    return formulas_weights


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
    res = -1
    for f in formula_containments[formula]:
        weight = known_weights.setdefault(
            f,
            mobius_function(
                f,
                formula_containments, known_weights=known_weights
            )
        )
        res += weight
    return -res


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def symbolic_shattering(unified_query, symbol_table):
    shattered_query = shatter_easy_probfacts(unified_query, symbol_table)
    inverted_symbol_table = {v: k for k, v in symbol_table.items()}
    for atom in extract_logic_atoms(shattered_query):
        functor = atom.functor
        if isinstance(functor, ProbabilisticFactSet):
            if functor not in inverted_symbol_table:
                s = Symbol.fresh()
                inverted_symbol_table[functor] = s
                symbol_table[s] = functor

    shattered_query = (
            ReplaceExpressionWalker(inverted_symbol_table)
            .walk(shattered_query)
    )
    return shattered_query


def is_atom_a_deterministic_relation(atom, symbol_table):
    return isinstance(
        symbol_table.get(atom.functor, None),
        DeterministicFactSet
    )
