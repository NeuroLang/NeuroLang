import logging
from functools import lru_cache, reduce
from itertools import chain, combinations

import numpy as np

from neurolang.exceptions import NeuroLangException

from .. import relational_algebra_provenance as rap
from ..datalog.expression_processing import (
    UnifyVariableEqualities,
    enforce_conjunction,
    flatten_query
)
from ..datalog.translate_to_named_ra import TranslateToNamedRA
from ..expression_walker import (
    PatternWalker,
    ReplaceExpressionWalker,
    add_match
)
from ..expressions import FunctionApplication, Symbol
from ..logic import (
    FALSE,
    Conjunction,
    Disjunction,
    ExistentialPredicate,
    Implication,
    NaryLogicOperator
)
from ..logic.expression_processing import (
    extract_logic_atoms,
    extract_logic_free_variables
)
from ..logic.transformations import (
    GuaranteeConjunction,
    GuaranteeDisjunction,
    MakeExistentialsImplicit,
    PushExistentialsDown,
    RemoveTrivialOperations
)
from ..relational_algebra import (
    BinaryRelationalAlgebraOperation,
    ColumnStr,
    NamedRelationalAlgebraFrozenSet,
    NAryRelationalAlgebraOperation,
    RelationalAlgebraOperation,
    UnaryRelationalAlgebraOperation,
    str2columnstr_constant
)
from ..relational_algebra_provenance import ProvenanceAlgebraSet
from ..utils import log_performance, OrderedSet
from .containment import is_contained
from .dichotomy_theorem_based_solver import (
    RAQueryOptimiser,
    lift_optimization_for_choice_predicates,
    shatter_easy_probfacts
)
from .probabilistic_ra_utils import (
    DeterministicFactSet,
    ProbabilisticFactSet,
    generate_probabilistic_symbol_table_for_query
)
from .probabilistic_semiring_solver import ProbSemiringSolver
from .transforms import (
    convert_rule_to_ucq,
    minimize_ucq_in_cnf,
    minimize_ucq_in_dnf,
    unify_existential_variables
)

LOG = logging.getLogger(__name__)


__all__ = [
    "dalvi_suciu_lift",
]

GC = GuaranteeConjunction()
GD = GuaranteeDisjunction()
PED = PushExistentialsDown()
RTO = RemoveTrivialOperations()


def is_deterministic(atom, symbol_table):
    return isinstance(
        symbol_table.get(atom.functor, None),
        DeterministicFactSet
    )


def dalvi_suciu_lift(rule, symbol_table):
    '''
    Translation from a datalog rule which allows disjunctions in the body
    to a safe plan according to [1]_. Non-liftable segments are identified
    by the `NonLiftable` expression.

    [1] Dalvi, N. & Suciu, D. The dichotomy of probabilistic inference
    for unions of conjunctive queries. J. ACM 59, 1–87 (2012).
    '''
    if isinstance(rule, Implication):
        rule = convert_rule_to_ucq(rule)
    rule = RTO.walk(rule)
    if (
        isinstance(rule, FunctionApplication) or
        all(
            is_deterministic(atom, symbol_table)
            for atom in extract_logic_atoms(rule)
        )
    ):
        return TranslateToNamedRA().walk(rule)

    rule_cnf = minimize_ucq_in_cnf(rule)
    connected_components = symbol_connected_components(rule_cnf)
    if len(connected_components) > 1:
        return components_plan(connected_components, rap.NaturalJoin, symbol_table)
    elif len(rule_cnf.formulas) > 1:
        return inclusion_exclusion_conjunction(rule_cnf, symbol_table)

    rule_dnf = minimize_ucq_in_dnf(rule)
    connected_components = symbol_connected_components(rule_dnf)
    if len(connected_components) > 1:
        return components_plan(connected_components, rap.Union, symbol_table)
    elif has_separator_variables(rule_dnf, symbol_table):
        return separator_variable_plan(rule_dnf, symbol_table)

    return NonLiftable(rule)


def has_separator_variables(query, symbol_table):
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

    return len(find_separator_variables(query, symbol_table)[0]) > 0


class NonLiftable(RelationalAlgebraOperation):
    def __init__(self, non_liftable_query):
        self.non_liftable_query = non_liftable_query

    def __repr__(self):
        return (
            "NonLiftable"
            f"({self.non_liftable_query})"
        )


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


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


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
        atoms = OrderedSet(
            atom
            for atom in extract_logic_atoms(formula)
            if not is_deterministic(atom, symbol_table)
        )
        if len(atoms) == 0:
            continue
        root_variables = reduce(
            lambda y, x: set(x.args) & y,
            atoms[1:],
            set(atoms[0].args)
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


def separator_variable_plan(expression, symbol_table):
    variables_to_project = extract_logic_free_variables(expression)
    svs, expression = find_separator_variables(expression, symbol_table)
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
            dalvi_suciu_lift(expression, symbol_table),
            tuple(
                str2columnstr_constant(v.name)
                for v in (variables_to_project | svs)
            )
        ),
        tuple(
            str2columnstr_constant(v.name)
            for v in variables_to_project
        )
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


def components_plan(components, operation, symbol_table):
    formulas = []
    for component in components:
        formulas.append(dalvi_suciu_lift(component, symbol_table))
    return reduce(operation, formulas[1:], formulas[0])


def inclusion_exclusion_conjunction(expression, symbol_table):
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
        return ProvenanceAlgebraSet(
            NamedRelationalAlgebraFrozenSet(("_p_",)),
            ColumnStr("_p_"),
        )

    with log_performance(LOG, "Translation and lifted optimisation"):
        flat_query_body = enforce_conjunction(
            lift_optimization_for_choice_predicates(
                flat_query_body, cpl_program
            )
        )
        flat_query = Implication(query.consequent, flat_query_body)
        symbol_table = generate_probabilistic_symbol_table_for_query(
            cpl_program, flat_query_body
        )
        unified_query = UnifyVariableEqualities().walk(flat_query)
        shattered_query = symbolic_shattering(unified_query, symbol_table)
        ra_query = dalvi_suciu_lift(shattered_query, symbol_table)
        if not is_pure_lifted_plan(ra_query):
            LOG.info(
                "Query not liftable %s",
                shattered_query
            )
            raise NeuroLangException(
                "Query not hierarchical, algorithm can't be applied"
            )
        ra_query = RAQueryOptimiser().walk(ra_query)

    with log_performance(LOG, "Run RAP query"):
        solver = ProbSemiringSolver(symbol_table)
        prob_set_result = solver.walk(ra_query)

    return prob_set_result


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
