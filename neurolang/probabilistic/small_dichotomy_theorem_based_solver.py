"""
Implementation of probabilistic query resolution for
hierarchical queries [^1]. Using this we apply the small dichotomy
theorem [^1, ^2]:
Let Q be a conjunctive query without self-joins or a non-repeating relational
algebra expression. Then:
* If Q is hierarchical, then P(Q) is in polynomial time, and can be computed
using only the lifted inference rules for join, negation, union, and
existential quantifier.

* If Q is not hierarchical, then P(Q) is #P-hard in the size of the database.

[^1]: Robert Fink and Dan Olteanu. A dichotomy for non-repeating queries with
negation in probabilistic databases. In Proceedings of the 33rd ACM
SIGMOD-SIGACT-SIGART Symposium on Principles of Database Systems, PODS ’14,
pages 144–155, New York, NY, USA, 2014. ACM.

[^2]: Nilesh N. Dalvi and Dan Suciu. Efficient query evaluation on
probabilistic databases. VLDB J., 16(4):523–544, 2007.
"""

import logging
from collections import defaultdict
from typing import AbstractSet

from ..datalog.expression_processing import (
    EQ,
    UnifyVariableEqualities,
    extract_logic_atoms,
    extract_logic_free_variables,
    extract_logic_predicates,
    flatten_query,
    remove_conjunction_duplicates
)
from ..datalog.translate_to_named_ra import TranslateToNamedRA
from ..exceptions import UnsupportedSolverError
from ..expression_walker import ReplaceExpressionWalker
from ..expressions import Constant, Symbol
from ..logic import FALSE, Conjunction, Disjunction, Implication, Negation
from ..logic.transformations import (
    GuaranteeConjunction,
    MakeExistentialsImplicit,
    RemoveTrivialOperations,
    convert_to_pnf_with_dnf_matrix
)
from ..relational_algebra import (
    NamedRelationalAlgebraFrozenSet,
    Projection,
    str2columnstr_constant
)
from ..relational_algebra_provenance import ProvenanceAlgebraSet
from ..utils import log_performance
from ..utils.orderedset import OrderedSet
from .exceptions import NotHierarchicalQueryException
from .expression_processing import lift_optimization_for_choice_predicates
from .probabilistic_ra_utils import (
    generate_probabilistic_symbol_table_for_query,
    is_atom_a_probabilistic_choice_relation,
    is_atom_a_probabilistic_fact_relation
)
from .query_resolution import (
    generate_provenance_query_solver,
    lift_solve_marg_query,
    compute_projections_needed_to_reintroduce_head_terms
)
from .shattering import shatter_constants

LOG = logging.getLogger(__name__)


def is_hierarchical_without_self_joins(query):
    """
    Let Q be first-order formula. For each variable x denote at(x) the
    set of atoms that contain the variable x. We say that Q is hierarchical
    if forall x, y one of the following holds:
    at(x) ⊆ at(y) or at(x) ⊇ at(y) or at(x) ∩ at(y) = ∅.
    """
    query = convert_to_pnf_with_dnf_matrix(query)
    antecedent = MakeExistentialsImplicit().walk(query)
    if isinstance(antecedent, Disjunction):
        return False

    has_self_joins, atom_set = extract_atom_sets_and_detect_self_joins(query)
    has_negative_atoms = any(
        isinstance(predicate, Negation)
        for predicate in extract_logic_predicates(query)
    )

    if has_self_joins or has_negative_atoms:
        return False

    variables = list(atom_set)
    for i, v in enumerate(variables):
        at_v = atom_set[v]
        for v2 in variables[i + 1:]:
            at_v2 = atom_set[v2]
            if not (at_v <= at_v2 or at_v2 <= at_v or at_v.isdisjoint(at_v2)):
                LOG.info(
                    "Not hierarchical on variables %s %s", v.name, v2.name
                )
                return False

    return True


def extract_atom_sets_and_detect_self_joins(query):
    has_self_joins = False
    predicates = extract_logic_atoms(query)
    predicates = set(pred for pred in predicates if not pred.functor == EQ)
    seen_predicate_functor = set()
    atom_set = defaultdict(set)
    for predicate in predicates:
        functor = predicate.functor
        if functor in seen_predicate_functor:
            LOG.info("Not hierarchical self join on variables %s", functor)
            has_self_joins = True
        seen_predicate_functor.add(functor)
        for variable in predicate.args:
            if not isinstance(variable, Symbol):
                continue
            atom_set[variable].add(functor)
    return has_self_joins, atom_set


def solve_succ_query(query, cpl_program, run_relational_algebra_solver=True):
    """
    Solve a SUCC query on a CP-Logic program.

    Parameters
    ----------
    query : Implication
        SUCC query of the form `ans(x) :- P(x)`.
    cpl_program : CPLogicProgram
        CP-Logic program on which the query should be solved.
    run_relational_algebra_solver: bool, default True
        When true the result's `relation` attribute is
        a NamedRelationalAlgebraFrozenSet,
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
                ("_p_",) + head_var_names)
            ),
            str2columnstr_constant("_p_"),
        )

    if isinstance(flat_query_body, Conjunction):
        flat_query_body = remove_conjunction_duplicates(flat_query_body)

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
        _verify_fact_and_choice_e_quantification(symbol_table, unified_query)
        original_symbols = set(symbol_table)
        shattered_query_antecedent = shatter_constants(
            unified_query.antecedent, symbol_table
        )
        shattered_symbols = set(symbol_table) - original_symbols

        shattered_query_antecedent = convert_to_pnf_with_dnf_matrix(
            shattered_query_antecedent
        )
        shattered_query_antecedent = (
            RemoveTrivialOperations()
            .walk(MakeExistentialsImplicit().walk(shattered_query_antecedent))
        )
        if isinstance(shattered_query_antecedent, Disjunction):
            raise UnsupportedSolverError("Query not conjunctive")

        shattered_query = Implication(
            unified_query.consequent,
            shattered_query_antecedent
        )
        probabilistic_predicates = \
            _extract_antecedent_probabilistic_predicates(
                shattered_query,
                symbol_table
            )
        shattered_query_probabilistic_body = Conjunction(
            tuple(probabilistic_predicates)
        )
        if not is_hierarchical_without_self_joins(
            shattered_query_probabilistic_body
        ):
            LOG.info(
                "Query with conjunctions %s not hierarchical",
                shattered_query_probabilistic_body.formulas,
            )
            raise NotHierarchicalQueryException(
                "Query not hierarchical, algorithm can't be applied"
            )

        ra_query = TranslateToNamedRA().walk(shattered_query.antecedent)
        replace_shattered_symbols = ReplaceExpressionWalker(
            {k: symbol_table[k] for k in shattered_symbols}
        )
        ra_query = replace_shattered_symbols.walk(ra_query)
        # project on query's head variables
        ra_query = _project_on_query_head(ra_query, shattered_query)
        # re-introduce head variables potentially removed by unification
        needed_projections = compute_projections_needed_to_reintroduce_head_terms(
            ra_query, flat_query, unified_query
        )

    query_solver = generate_provenance_query_solver(
        symbol_table, run_relational_algebra_solver,
        needed_projections=needed_projections
    )

    with log_performance(LOG, "Run RAP query %s", init_args=(ra_query,)):
        prob_set_result = query_solver.walk(ra_query)

    return prob_set_result


def _verify_fact_and_choice_e_quantification(symbol_table, shattered_query):
    e_vars = (
        extract_logic_free_variables(shattered_query.antecedent) -
        extract_logic_free_variables(shattered_query.consequent)
    )
    e_vars_choice = {
        var for var in e_vars
        if any(
            var in set(atom.args)
            for atom in extract_logic_atoms(shattered_query.antecedent)
            if is_atom_a_probabilistic_choice_relation(atom, symbol_table)
        )
    }
    if any(
        (e_vars & atom.args) and
        is_atom_a_probabilistic_fact_relation(atom, symbol_table) and
        (e_vars & atom.args) > e_vars_choice
        for atom in extract_logic_atoms(shattered_query.antecedent)
    ):
        raise UnsupportedSolverError(
                "Cannot solve queries with existentially quantified variable "
                "occurring in probabilistic fact but not in any probabilistic "
                "choice, or disjunctive queries"
            )


def _extract_antecedent_probabilistic_predicates(datalog_rule, symbol_table):
    probabilistic_predicates = []
    for predicate in extract_logic_predicates(datalog_rule):
        atom = predicate
        while isinstance(atom, Negation):
            atom = atom.formula
        if isinstance(atom.functor, Symbol) and (
            is_atom_a_probabilistic_choice_relation(atom, symbol_table)
            or
            is_atom_a_probabilistic_fact_relation(atom, symbol_table)
        ):
            probabilistic_predicates.append(predicate)
    return probabilistic_predicates


def _project_on_query_head(provset, query):
    proj_cols = tuple(
        OrderedSet(
            str2columnstr_constant(arg.name)
            for arg in query.consequent.args
            if isinstance(arg, Symbol)
        )
    )
    return Projection(provset, proj_cols)


def solve_marg_query(rule, cpl):
    """
    Solve a MARG query on a CP-Logic program.
    Parameters
    ----------
    query : Implication
        Consequent must be of type `Condition`.
        MARG query of the form `ans(x) :- P(x)`.
    cpl_program : CPLogicProgram
        CP-Logic program on which the query should be solved.
    Returns
    -------
    ProvenanceAlgebraSet
        Provenance set labelled with probabilities for each tuple in the result
        set.
    """
    return lift_solve_marg_query(rule, cpl, solve_succ_query)
