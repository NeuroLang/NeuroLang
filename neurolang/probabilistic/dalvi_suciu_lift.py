import logging
from functools import reduce
from itertools import chain, combinations
from typing import AbstractSet

import numpy as np

from .. import relational_algebra_provenance as rap
from ..config import config
from ..datalog.expression_processing import (
    UnifyVariableEqualities,
    flatten_query
)
from ..datalog.translate_to_named_ra import TranslateToNamedRA
from ..exceptions import (
    NeuroLangException,
    NonLiftableException,
    NotInFONegE,
    NotRankedException,
    NotUnateException,
    UnsupportedSolverError
)
from ..expression_walker import (
    ExpressionWalker,
    PatternWalker,
    ReplaceExpressionWalker,
    add_match,
    expression_iterator
)
from ..expressions import (
    Constant,
    FunctionApplication,
    Symbol,
    TypedSymbolTableMixin
)
from ..logic import (
    FALSE,
    Conjunction,
    Disjunction,
    ExistentialPredicate,
    Implication,
    NaryLogicOperator,
    Negation,
    Quantifier,
    Union,
    UniversalPredicate
)
from ..logic.expression_processing import (
    extract_logic_atoms,
    extract_logic_free_variables,
    extract_logic_predicates
)
from ..logic.transformations import (
    ExtractConjunctiveQueryWithNegation,
    GuaranteeConjunction,
    GuaranteeDisjunction,
    MakeExistentialsImplicit,
    MakeUniversalsImplicit,
    MoveNegationsToAtoms,
    MoveQuantifiersUp,
    PushQuantifiersDown,
    RemoveExistentialOnVariables,
    RemoveTrivialOperations,
    RemoveUniversalPredicates,
    convert_to_pnf_with_dnf_matrix
)
from ..relational_algebra import (
    BinaryRelationalAlgebraOperation,
    ColumnStr,
    NamedRelationalAlgebraFrozenSet,
    NAryRelationalAlgebraOperation,
    UnaryRelationalAlgebraOperation,
    str2columnstr_constant
)
from ..relational_algebra_provenance import ProvenanceAlgebraSet
from ..utils import OrderedSet, log_performance
from .containment import is_contained
from .expression_processing import (
    is_builtin,
    lift_optimization_for_choice_predicates
)
from .probabilistic_ra_utils import (
    NonLiftable,
    ProbabilisticChoiceSet,
    ProbabilisticFactSet,
    generate_probabilistic_symbol_table_for_query,
    is_atom_a_deterministic_relation,
    is_atom_a_probabilistic_choice_relation
)
from .probabilistic_semiring_solver import (
    ProbSemiringToRelationalAlgebraSolver
)
from .query_resolution import (
    compute_projections_needed_to_reintroduce_head_terms,
    generate_provenance_query_solver,
    lift_solve_marg_query
)
from .ranking import partially_rank_query, verify_that_the_query_is_ranked
from .shattering import shatter_constants
from .transforms import (
    add_existentials_except,
    convert_rule_to_ucq,
    convert_to_cnf_existential_ucq,
    convert_to_dnf_existential_ucq,
    minimize_ucq_in_cnf,
    minimize_ucq_in_dnf,
    unify_existential_variables
)

LOG = logging.getLogger(__name__)


__all__ = [
    "dalvi_suciu_lift",
    "solve_succ_query",
    "solve_marg_query",
]


GC = GuaranteeConjunction()
MNTA = MoveNegationsToAtoms()
MQU = MoveQuantifiersUp()
PQD = PushQuantifiersDown()
RTO = RemoveTrivialOperations()
RUP = RemoveUniversalPredicates()


class ExtendedRAPToRAWalker(
    rap.IndependentDisjointProjectionsAndUnionMixin,
    rap.WeightedNaturalJoinSolverMixin,
    ProbSemiringToRelationalAlgebraSolver,
):
    pass


class ReplaceAllSymbolsButConstantSets(ExpressionWalker):
    def __init__(self, symbol_table):
        self.symbol_table = symbol_table

    @add_match(Symbol)
    def replace_symbol(self, expression):
        res = expression
        if expression in self.symbol_table:
            value = self.symbol_table[expression]
            if not isinstance(value, Constant[AbstractSet]):
                res = value
        return res


def solve_succ_query(query, cpl_program, run_relational_algebra_solver=True):
    """
    Solve a SUCC query on a CP-Logic program.

    Parameters
    ----------
    query : Implication
        SUCC query of the form `ans(x) :- P(x)`.
    cpl_program : CPLogicProgram
        CP-Logic program on which the query should be solved.
    run_relational_algebra_solver: bool
        When true the result's `relation` attribute is a
        NamedRelationalAlgebraFrozenSet,
        when false the attribute is the relational algebra expression that
        produces the such set.

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
            Constant[AbstractSet](NamedRelationalAlgebraFrozenSet(
                ("_p_",) + head_var_names
            )),
            str2columnstr_constant("_p_"),
        )

    with log_performance(LOG, "Translation to extensional plan"):
        flat_query = Implication(query.consequent, flat_query_body)
        shattered_query, symbol_table = \
            _prepare_and_optimise_query(flat_query, cpl_program)
        ucq_shattered_query = convert_rule_to_ucq(shattered_query)
        ucq_shattered_query = MNTA.walk(
            MQU.walk(ucq_shattered_query)
        )
        ra_query = dalvi_suciu_lift(ucq_shattered_query, symbol_table)
        if not is_pure_lifted_plan(ra_query):
            LOG.info(
                "Query not liftable %s",
                shattered_query
            )
            raise NonLiftableException(
                "Query %s not liftable, algorithm can't be applied",
                query
            )
        ra_query = (
            ReplaceAllSymbolsButConstantSets(symbol_table)
            .walk(ra_query)
        )

    needed_projections = compute_projections_needed_to_reintroduce_head_terms(
        ra_query, flat_query, shattered_query
    )

    query_solver = generate_provenance_query_solver(
        symbol_table, run_relational_algebra_solver,
        needed_projections=needed_projections,
        solver_class=ExtendedRAPToRAWalker
    )

    with log_performance(LOG, "Run RAP query %s", init_args=(ra_query,)):
        prob_set_result = query_solver.walk(ra_query)

    return prob_set_result


def _prepare_and_optimise_query(flat_query, cpl_program):
    flat_query_body = convert_to_dnf_existential_ucq(flat_query.antecedent)
    flat_query_body = RTO.walk(Disjunction(tuple(
        lift_optimization_for_choice_predicates(f, cpl_program)
        for f in flat_query_body.formulas
    )))
    flat_query = Implication(flat_query.consequent, flat_query_body)
    unified_query = UnifyVariableEqualities().walk(flat_query)
    if config.get_probabilistic_solver_check_unate():
        _verify_that_the_query_has_one_quantifier(flat_query)
    symbol_table = generate_probabilistic_symbol_table_for_query(
        cpl_program, unified_query.antecedent
    )
    shattered_query_body = shatter_constants(
        convert_rule_to_ucq(unified_query),
        symbol_table
    )
    shattered_query_body = partially_rank_query(
        shattered_query_body, symbol_table
    )
    shattered_query = Implication(
        unified_query.consequent,
        shattered_query_body
    )
    _verify_that_the_query_has_no_builtins(shattered_query, cpl_program)
    if config.get_probabilistic_solver_check_unate():
        _verify_that_the_query_is_unate(shattered_query, symbol_table)
    if not verify_that_the_query_is_ranked(
        convert_rule_to_ucq(shattered_query)
    ):
        raise NotRankedException(f"Query {flat_query} is not ranked")

    return shattered_query, symbol_table


def _verify_that_the_query_has_no_builtins(shattered_query, cpl_program):
    if any(
        is_builtin(atom, known_builtins=cpl_program.builtins())
        for atom in extract_logic_atoms(shattered_query)
    ):
        raise UnsupportedSolverError(
            "Builtins not allowed for Dalvi-Suciu lifting"
        )


def _verify_that_the_query_is_unate(query, symbol_table):
    positive_relational_symbols = set()
    negative_relational_symbols = set()

    query = convert_rule_to_ucq(query)
    query = convert_to_pnf_with_dnf_matrix(query)
    for predicate in extract_logic_predicates(query):
        if isinstance(predicate, Negation):
            atom = predicate.formula
        else:
            atom = predicate
        if is_atom_a_deterministic_relation(atom, symbol_table):
            continue
        if isinstance(predicate, Negation):
            while isinstance(predicate, Negation):
                predicate = predicate.formula
            negative_relational_symbols.add(predicate.functor)
        else:
            positive_relational_symbols.add(predicate.functor)

    if not positive_relational_symbols.isdisjoint(negative_relational_symbols):
        raise NotUnateException(f"Query {query} is not unate")


def _verify_that_the_query_has_one_quantifier(query):
    query = convert_rule_to_ucq(query)
    query = convert_to_pnf_with_dnf_matrix(query)
    if not isinstance(query, Quantifier):
        return

    first_quantifier = type(query)
    query_act = query.body
    while isinstance(query_act, Quantifier):
        if not isinstance(query_act, first_quantifier):
            raise NotUnateException(
                f"Query {query} uses two types of quantifiers"
            )
        query_act = query_act.body


def solve_marg_query(rule, cpl):
    return lift_solve_marg_query(rule, cpl, solve_succ_query)


def dalvi_suciu_lift(rule, symbol_table):
    '''
    Translation from a datalog rule which allows disjunctions in the body
    to a safe plan according to [1]. Non-liftable segments are identified
    by the `NonLiftable` expression.
    [1] Dalvi, N. & Suciu, D. The dichotomy of probabilistic inference
    for unions of conjunctive queries. J. ACM 59, 1–87 (2012).
    '''
    #  rule = RemoveUniversalPredicates().walk(rule)
    rule = RTO.walk(rule)

    has_safe_plan, res = symbol_or_deterministic_plan(rule, symbol_table)
    if has_safe_plan:
        return res

    rule_cnf = convert_ucq_to_ccq(rule, transformation='CNF')
    connected_components = symbol_connected_components(rule_cnf)
    if len(connected_components) > 1:
        return components_plan(
            connected_components, rap.NaturalJoin, symbol_table,
            negative_operation=rap.Difference
        )

    rule_dnf = convert_ucq_to_ccq(rule, transformation='DNF')
    connected_components = symbol_connected_components(rule_dnf)
    if len(connected_components) > 1:
        return components_plan(
            connected_components, rap.Union, symbol_table
        )

    rule_pdnf = convert_to_pnf_with_dnf_matrix(rule_dnf)
    has_svs, plan = has_separator_variables(rule_pdnf, symbol_table)
    if has_svs:
        return plan

    has_safe_plan, plan = disjoint_project(rule_dnf, symbol_table)
    if has_safe_plan:
        return plan

    if len(rule_cnf.formulas) > 1:
        return inclusion_exclusion_conjunction(rule_cnf, symbol_table)

    return NonLiftable(rule)


def symbol_or_deterministic_plan(rule, symbol_table):
    """If the rule is a single relational symbol or a
    fully deterministic subquery, then return the safe plan.

    Parameters
    ----------
    rule : Expression
        rule to lift
    symbol_table : Mapping of Symbol to Expression
        symbol table for the relational symbols

    Returns
    -------
        Tuple[bool, Union[Expression, NoneType]]
        if the rule is liftable then the boolean will be True and
        the second element of the tuple will be the lifted plan.
    """

    has_safe_plan = False
    res = None
    if isinstance(rule, FunctionApplication):
        has_safe_plan = True
        res = TranslateToNamedRA().walk(rule)
    elif (
        all(
            is_atom_a_deterministic_relation(atom, symbol_table)
            for atom in extract_logic_atoms(rule)
        )
    ):
        free_vars = extract_logic_free_variables(rule)
        rule = MakeExistentialsImplicit().walk(rule)
        result = TranslateToNamedRA().walk(rule)
        proj_cols = tuple(Constant(ColumnStr(v.name)) for v in free_vars)
        has_safe_plan = True
        res = rap.Projection(result, proj_cols)
    return has_safe_plan, res


def convert_ucq_to_ccq(rule, transformation='CNF'):
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
    rule = PQD.walk(RUP.walk(PQD.walk(MNTA.walk(rule))))
    free_vars = extract_logic_free_variables(rule)
    existential_vars = set()
    for atom in extract_logic_atoms(rule):
        existential_vars.update(set(atom.args) - set(free_vars))

    conjunctions = ExtractConjunctiveQueryWithNegation().walk(rule)
    dic_components = extract_connected_components(
        conjunctions, existential_vars
    )

    fresh_symbols_expression = (
        ReplaceExpressionWalker(dic_components)
        .walk(rule)
    )
    if transformation == 'CNF':
        fresh_symbols_expression = convert_to_cnf_existential_ucq(
            fresh_symbols_expression
        )
        minimize = minimize_ucq_in_cnf
        gcd = GuaranteeConjunction()
    elif transformation == 'DNF':
        fresh_symbols_expression = convert_to_dnf_existential_ucq(
            fresh_symbols_expression
        )
        minimize = minimize_ucq_in_dnf
        gcd = GuaranteeDisjunction()
    else:
        raise ValueError(f'Invalid transformation type: {transformation}')

    final_expression = ReplaceExpressionWalker(
        {v: k for k, v in dic_components.items()}
    ).walk(fresh_symbols_expression)

    try:
        final_expression = minimize(final_expression)
    except NotInFONegE:
        pass

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

        if isinstance(f, Quantifier):
            transformations[f] = calc_new_fresh_symbol(f, existential_vars)
            continue

        for c in components:
            form = [f.formulas[i] for i in c]
            if len(form) > 1:
                form = Conjunction(form)
                transformations[form] = \
                    calc_new_fresh_symbol(form, existential_vars)
            else:
                conj = Conjunction(form)
                transformations[form[0]] = \
                    calc_new_fresh_symbol(conj, existential_vars)

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

    if isinstance(expression, Quantifier):
        return np.ones((1,))

    c_matrix = np.zeros((len(expression.formulas),) * 2)
    for i, formula in enumerate(expression.formulas):
        f_args = set(b for a in extract_logic_atoms(formula) for b in a.args)
        if variable_to_use is not None:
            f_args &= variable_to_use
        for j, formula_ in enumerate(expression.formulas[i + 1:]):
            f_args_ = set(
                b for a in extract_logic_atoms(formula_) for b in a.args
            )
            if variable_to_use is not None:
                f_args_ &= variable_to_use
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


def disjoint_project(rule_dnf, symbol_table):
    """
    Rule that extends the lifted query processing algorithm to handle
    Block-Independent Disjoint (BID) tables which encode mutual exclusivity
    assumptions on top of the tuple-independent assumption of probabilistic
    tables.

    Modifications to the lifted query processing algorithm that are necessary
    to extend it to BID tables are detailed in section 4.3.1 of [1]_.

    Two variants of the rule exist: one for conjunctive queries, and one for
    disjunctive queries.

    [1] Suciu, Dan, Dan Olteanu, Christopher Ré, and Christoph Koch, eds.
    Probabilistic Databases. Synthesis Lectures on Data Management 16. San
    Rafael, Calif.: Morgan & Claypool Publ, 2011.

    """
    if len(rule_dnf.formulas) == 1:
        conjunctive_query = rule_dnf.formulas[0]
        return disjoint_project_conjunctive_query(
            conjunctive_query, symbol_table
        )
    return disjoint_project_disjunctive_query(rule_dnf, symbol_table)


def disjoint_project_conjunctive_query(conjunctive_query, symbol_table):
    """
    First variant of the disjoint project on a CQ in CNF.

    A disjoint project operator is applied whenever any of the atoms in the CQ
    has only constants in its key positions and has at least one variable in a
    non-key position.

    Note: as variables are removed through shattering, this only applies to
    all probabilistic choice variables.

    """
    free_variables = extract_logic_free_variables(conjunctive_query)
    atoms_with_constants_in_all_key_positions = set(
        atom
        for atom in extract_logic_atoms(conjunctive_query)
        if is_probabilistic_atom_with_constants_in_all_key_positions(
            atom, symbol_table
        )
    )
    if not atoms_with_constants_in_all_key_positions:
        return False, None
    nonkey_variables = set.union(
        *(
            extract_nonkey_variables(atom, symbol_table)
            for atom in atoms_with_constants_in_all_key_positions
        )
    )
    for atom in atoms_with_constants_in_all_key_positions:
        if not isinstance(symbol_table[atom.functor], ProbabilisticChoiceSet):
            raise NeuroLangException(
                "Any atom with constants in all its key positions should be "
                "a probabilistic choice atom"
            )
    conjunctive_query = (
        RemoveExistentialOnVariables(nonkey_variables)
        .walk(conjunctive_query)
    )
    plan = dalvi_suciu_lift(conjunctive_query, symbol_table)
    attributes = tuple(
        str2columnstr_constant(v.name)
        for v in free_variables
    )
    plan = rap.DisjointProjection(plan, attributes)
    return True, plan


def disjoint_project_disjunctive_query(disjunctive_query, symbol_table):
    """
    Second variant of the disjoint project on a UCQ in DNF.

    This rule applies whenever the given query Q can be written as Q = Q1 v Q',
    where the conjunctive query Q1 has an atom where all key attribute are
    constants, and Q' is any other UCQ.

    Then we return P(Q) = P(Q1) + P(Q') - P(Q1 ∧ Q') with subsequent recursive
    calls to the resolution algorithms. Note that a disjoint project should be
    applied during the calculation of P(Q1) and P(Q1 ∧ Q').

    """
    matching_disjuncts = (
        _get_disjuncts_containing_atom_with_all_key_attributes(
            disjunctive_query, symbol_table
        )
    )
    for disjunct in matching_disjuncts:
        has_safe_plan, plan = _apply_disjoint_project_ucq_rule(
            disjunctive_query, disjunct, symbol_table
        )
        if has_safe_plan:
            return has_safe_plan, plan
    return False, None


def _get_disjuncts_containing_atom_with_all_key_attributes(ucq, symbol_table):
    matching_disjuncts = set()
    for disjunct in ucq.formulas:
        if any(
            is_probabilistic_atom_with_constants_in_all_key_positions(
                atom, symbol_table
            )
            for atom in extract_logic_atoms(disjunct)
        ):
            matching_disjuncts.add(disjunct)
    return matching_disjuncts


def _apply_disjoint_project_ucq_rule(
    disjunctive_query: Union,
    disjunct: Conjunction,
    symbol_table: TypedSymbolTableMixin,
):
    free_vars = extract_logic_free_variables(disjunctive_query)
    head = add_existentials_except(disjunct, free_vars)
    head_plan = dalvi_suciu_lift(head, symbol_table)
    if not is_pure_lifted_plan(head_plan):
        return False, None
    tail = set(disjunctive_query.formulas) - {disjunct}
    tail = add_existentials_except(
        Conjunction(tuple(tail)),
        free_vars,
    )
    tail_plan = dalvi_suciu_lift(tail, symbol_table)
    if not is_pure_lifted_plan(tail_plan):
        return False, None
    head_and_tail = add_existentials_except(
        Conjunction(disjunctive_query.formulas), free_vars
    )
    head_and_tail_plan = dalvi_suciu_lift(head_and_tail, symbol_table)
    if not is_pure_lifted_plan(head_and_tail_plan):
        return False, None
    return True, rap.WeightedNaturalJoin(
        (head_plan, tail_plan, head_and_tail_plan),
        (Constant(1), Constant(1), Constant(-1)),
    )


def extract_nonkey_variables(atom, symbol_table):
    """
    Get all variables in the atom that occur on non-key attributes.

    Makes the assumption that the atom is probabilistic.

    As we only support probabilistic choices and not all BID tables, this can
    only be a variable occurring in a probabilistic choice.

    """
    if is_atom_a_probabilistic_choice_relation(atom, symbol_table):
        return {arg for arg in atom.args if isinstance(arg, Symbol)}
    return set()


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

    query = convert_to_pnf_with_dnf_matrix(query)
    query = PQD.walk(query)
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
        pchoice_atoms = OrderedSet(
            atom
            for atom in probabilistic_atoms
            if is_atom_a_probabilistic_choice_relation(atom, symbol_table)
        )
        # all variables occurring in probabilistic choices cannot occur in key
        # positions, as probabilistic choice relations have no key attribute
        # (making their respective tuples mutually exclusive)
        nonkey_variables = set().union(*(set(a.args) for a in pchoice_atoms))
        # variables occurring in non-key positions cannot be root variables
        # because root variables must occur in every atom in a key position
        root_variables -= nonkey_variables
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
    if any(
        isinstance(expression, ExistentialPredicate)
        for _, expression in expression_iterator(expression)
    ):
        make_implicit = MakeExistentialsImplicit()
        quantifier_class = ExistentialPredicate
        projection_class = rap.IndependentProjection
    else:
        make_implicit = MakeUniversalsImplicit()
        quantifier_class = UniversalPredicate
        projection_class = rap.IndependentProjectionUniversal

    expression = make_implicit.walk(expression)

    quantified_to_add = (
        extract_logic_free_variables(expression) -
        variables_to_project -
        separator_variables
    )

    for v in quantified_to_add:
        expression = quantifier_class(v, expression)
    return projection_class(
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


def components_plan(
    components, operation, symbol_table,
    negative_operation=None,
    unary_negative_operation=None,
):
    positive_formulas = []
    negative_formulas = []
    for component in components:
        component = RTO.walk(component)
        if isinstance(component, Negation):
            formula = dalvi_suciu_lift(component.formula, symbol_table)
            negative_formulas.append(formula)
        else:

            formula = dalvi_suciu_lift(component, symbol_table)
            positive_formulas.append(formula)

    if len(positive_formulas) > 0:
        output = reduce(operation, positive_formulas[1:], positive_formulas[0])

    if (
        len(positive_formulas) > 0 and
        len(negative_formulas) > 0
    ):
        if negative_operation is None:
            raise ValueError(
                "If negative components are included,"
                " a negative operation should be provided"
            )
        output = reduce(negative_operation, negative_formulas, output)
    elif len(negative_formulas) > 0:
        if unary_negative_operation is None:
            raise ValueError(
                "If negative components are included,"
                " and no positive formulas exist,"
                " a unary negative operation should be provided"
            )
        complemented_negative_formulas = [
            unary_negative_operation(f) for f in negative_formulas
        ]
        output = reduce(
            operation,
            complemented_negative_formulas[1:],
            complemented_negative_formulas[0]
        )
    return output


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

        formula = RTO.walk(convert_ucq_to_ccq(
            Disjunction(tuple(formula)), transformation="DNF")
        )
        formula_powerset.append(formula)
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


def is_probabilistic_atom_with_constants_in_all_key_positions(
    atom, symbol_table
):
    """
    As we only handle probabilistic choice relations (and not more general BID
    tables), which are relations with no key attribute, there are only two
    cases:
    (1) if the atom is a probabilistic choice, then it validates the
    requirement of having constants in all key positions (it has no key
    attribute), or
    (2) if the atom is a tuple-independent relation, all its attributes are key
    attributes, and so all its terms must be constants for the requirement to
    be validated.

    """
    return is_atom_a_probabilistic_choice_relation(atom, symbol_table) or (
        not is_atom_a_deterministic_relation(atom, symbol_table)
        and (
            isinstance(atom.functor, Symbol)
            and atom.functor in symbol_table
            and isinstance(symbol_table[atom.functor], ProbabilisticFactSet)
            and all(isinstance(arg, Constant) for arg in atom.args)
        )
    )
