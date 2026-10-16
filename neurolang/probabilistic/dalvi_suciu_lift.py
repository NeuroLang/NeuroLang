import logging
import threading
from collections import OrderedDict
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
    AbsorbAssertedConjunctsInNegations,
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
    convert_to_pnf_with_dnf_matrix,
    standardise_apart
)
from ..relational_algebra import (
    BinaryRelationalAlgebraOperation,
    ColumnStr,
    Materialise,
    NamedRelationalAlgebraFrozenSet,
    NAryRelationalAlgebraOperation,
    UnaryRelationalAlgebraOperation,
    str2columnstr_constant
)
from ..relational_algebra_provenance import ProvenanceAlgebraSet
from ..utils import OrderedSet, log_performance
from .containment import is_contained
from .deterministic_subqueries import replace_deterministic_negations
from .expression_processing import (
    STRICT_COMPARISONS,
    is_builtin,
    is_comparison,
    lift_optimization_for_choice_predicates
)
from .probabilistic_ra_utils import (
    DeterministicFactSet,
    GetProbabilisticSetAtom,
    NonLiftable,
    ProbabilisticChoiceSet,
    ProbabilisticFactSet,
    classify_and_wrap_symbols,
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
from .non_repeating_lift import columns_comparison, non_repeating_plan
from .ranking import partially_rank_query, verify_that_the_query_is_ranked
from .shattering import NE, shatter_constants
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
    "clear_cache",
]


# Bound cache to avoid unbounded memory growth in long-running processes.
_MAX_DALVI_SUCIU_CACHE_SIZE = config.getint(
    "probabilistic.cache", "dalvi_suciu_max_size", fallback=4096
)

_dalvi_suciu_lift_cache = OrderedDict()
_dalvi_suciu_lift_cache_lock = threading.Lock()


def clear_cache():
    """Clear the internal cache used by `dalvi_suciu_lift`."""
    with _dalvi_suciu_lift_cache_lock:
        _dalvi_suciu_lift_cache.clear()


def _lift_cache_key(rule, symbol_table):
    """
    Key a lifted plan by the rule and the kind of each of its symbols.

    The plan only refers to the relational symbols of the rule, and only
    depends on the symbol table through whether each symbol is a
    deterministic, probabilistic fact or probabilistic choice relation.
    Keying on that, rather than on the tables themselves, avoids hashing
    the data and lets queries with fresh table symbols share plans.
    """
    gpsa = GetProbabilisticSetAtom(symbol_table)
    kinds = frozenset(
        (symbol, type(symbol_table[symbol]), type(gpsa.walk(symbol)))
        for symbol in rule._symbols
        if symbol in symbol_table
    )
    return rule, kinds


AACN = AbsorbAssertedConjunctsInNegations()
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
            if not isinstance(value, (Constant[AbstractSet], Materialise)):
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
        # The lifting rules identify variables by their names, which
        # standardising apart makes unambiguous.
        flat_query_body, deterministic_relations = \
            replace_deterministic_negations(
                AACN.walk(standardise_apart(flat_query_body)), cpl_program
            )
        flat_query = Implication(query.consequent, flat_query_body)
        non_repeating = _non_repeating_query_plan(
            flat_query, cpl_program, deterministic_relations
        )
        if non_repeating is not None:
            ra_query, symbol_table, shattered_query = non_repeating
        else:
            shattered_query, symbol_table = _prepare_and_optimise_query(
                flat_query, cpl_program, deterministic_relations
            )
            ra_query = lift_with_dependent_negations(
                shattered_query, symbol_table
            )
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


def _non_repeating_query_plan(
    flat_query, cpl_program, deterministic_relations=None
):
    """Plan of a non-repeating query with negations, see
    `non_repeating_plan`, with its symbol table and the query planned,
    or None if the query is not such a hierarchical query.
    """
    antecedent = flat_query.antecedent
    if isinstance(antecedent, Conjunction):
        antecedent = lift_optimization_for_choice_predicates(
            antecedent, cpl_program
        )
    unified_query = UnifyVariableEqualities().walk(
        Implication(flat_query.consequent, antecedent)
    )
    symbol_table = generate_probabilistic_symbol_table_for_query(
        cpl_program, unified_query.antecedent
    )
    classify_and_wrap_symbols(
        deterministic_relations or {}, unified_query.antecedent,
        symbol_table, lambda s, _: DeterministicFactSet(s)
    )
    head_variables = {
        arg for arg in unified_query.consequent.args
        if isinstance(arg, Symbol)
    }
    plan = non_repeating_plan(
        add_existentials_except(unified_query.antecedent, head_variables),
        symbol_table
    )
    if plan is None:
        return None
    return plan, symbol_table, unified_query


def _prepare_and_optimise_query(
    flat_query, cpl_program, deterministic_relations=None
):
    flat_query_body = convert_to_dnf_existential_ucq(flat_query.antecedent)
    flat_query_body = RTO.walk(Disjunction(tuple(
        lift_optimization_for_choice_predicates(f, cpl_program)
        for f in flat_query_body.formulas
    )))
    flat_query = Implication(flat_query.consequent, flat_query_body)
    unified_query = UnifyVariableEqualities().walk(flat_query)
    symbol_table = generate_probabilistic_symbol_table_for_query(
        cpl_program, unified_query.antecedent
    )
    classify_and_wrap_symbols(
        deterministic_relations or {}, unified_query.antecedent,
        symbol_table, lambda s, _: DeterministicFactSet(s)
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
    if (
        config.get_probabilistic_solver_check_unate() and
        _split_dependent_conjunct(
            RTO.walk(shattered_query.antecedent), Negation, symbol_table
        ) is None
    ):
        # Otherwise the checks are applied by
        # `lift_with_dependent_negations` to each query it lifts.
        _verify_that_the_query_has_one_quantifier(flat_query)
        _verify_that_the_query_is_unate(shattered_query, symbol_table)
    if not verify_that_the_query_is_ranked(
        convert_rule_to_ucq(shattered_query)
    ):
        raise NotRankedException(f"Query {flat_query} is not ranked")

    return shattered_query, symbol_table


def _verify_that_the_query_has_no_builtins(shattered_query, cpl_program):
    """Builtins are only allowed as comparisons within existential
    conjuncts, possibly negated, see `comparison_conjunct_plan`.
    """
    antecedent = RTO.walk(shattered_query.antecedent)
    if isinstance(antecedent, Conjunction):
        formulas = antecedent.formulas
    else:
        formulas = (antecedent,)
    if any(
        is_builtin(atom, known_builtins=cpl_program.builtins()) and not (
            is_comparison(atom) and _is_existential_conjunct(formula)
        )
        for formula in formulas
        for atom in extract_logic_atoms(formula)
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
        if (
            is_comparison(atom) or
            is_atom_a_deterministic_relation(atom, symbol_table)
        ):
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


def lift_with_dependent_negations(rule, symbol_table, check=False):
    """Lift a shattered rule whose negated conjuncts can share
    probabilistic symbols with the rest of the rule.

    A negated conjunct `¬N` sharing no probabilistic symbol with the rest
    `A` of the rule is left to `dalvi_suciu_lift`, which lifts the
    query as P[A] (1 - P[N]), as `A` and `N` are independent: the query
    is liftable when `A` and `N` are, see Fink & Olteanu [1] for queries
    with negation without repeated relations. Otherwise, if the shared
    relations are tuple-independent, `free_variable_shattering_plan`
    and `negation_subtraction_plan` are tried in turn. They are sound
    but not known to be complete. Shared probabilistic choices are not
    supported, as the queries these rules build would conjoin atoms of
    the same choice relation, which the choice optimisations applied
    before shattering expect to have unified. Comparisons are supported
    within existential conjuncts, possibly negated, by
    `comparison_conjunct_plan`.

    [1] Fink, R. & Olteanu, D. Dichotomies for queries with negation in
    probabilistic databases. ACM TODS 41, 1–47 (2016).

    Parameters
    ----------
    rule : Implication
        shattered and ranked rule.
    symbol_table : Mapping of Symbol to Expression
        symbol table for the relational symbols
    check : bool
        whether to apply the unate check, if enabled, before lifting
        a query with `dalvi_suciu_lift`.

    Returns
    -------
    RelationalAlgebraOperation
        lifted plan, possibly containing `NonLiftable` nodes.
    """
    consequent, antecedent = rule.unapply()
    antecedent = RTO.walk(antecedent)
    if any(is_comparison(atom) for atom in extract_logic_atoms(antecedent)):
        split = _split_comparison_conjunct(antecedent)
        if split is not None:
            plan = comparison_conjunct_plan(
                consequent, *split, symbol_table
            )
            if plan is not None:
                return plan
        return NonLiftable(antecedent)

    split = _split_dependent_conjunct(antecedent, Negation, symbol_table)
    if split is None:
        rule = Implication(consequent, antecedent)
        if check and config.get_probabilistic_solver_check_unate():
            _verify_that_the_query_has_one_quantifier(rule)
            _verify_that_the_query_is_unate(rule, symbol_table)
        ucq_rule = MNTA.walk(MQU.walk(convert_rule_to_ucq(rule)))
        plan = dalvi_suciu_lift(ucq_rule, symbol_table)
        if not is_pure_lifted_plan(plan):
            shattered_plan = existential_shattering_plan(
                consequent, antecedent, symbol_table
            )
            if shattered_plan is not None:
                plan = shattered_plan
        return plan

    remainder, negation = split
    negated_formula = negation.formula
    if _shares_a_choice_relation(negated_formula, remainder, symbol_table):
        return NonLiftable(antecedent)
    for plan_rule in (
        free_variable_shattering_plan, negation_subtraction_plan
    ):
        plan = plan_rule(consequent, remainder, negated_formula, symbol_table)
        if plan is not None:
            return plan
    return NonLiftable(antecedent)


def existential_shattering_plan(consequent, antecedent, symbol_table):
    """Plan for a conjunction `A ∧ ∃t ψ(t)` where `∃t ψ(t)` shares
    probabilistic symbols with `A`, as P[A] - P[A ∧ ¬∃t ψ(t)], which
    is sound for any `A` and `ψ`. The latter is planned with
    `free_variable_shattering_plan`.

    Parameters
    ----------
    consequent : FunctionApplication
        head of the rule.
    antecedent : LogicExpression
        body of the rule.
    symbol_table : Mapping of Symbol to Expression
        symbol table for the relational symbols

    Returns
    -------
    Optional[RelationalAlgebraOperation]
        the plan, or None if the rule does not apply.
    """
    split = _split_dependent_conjunct(
        antecedent, ExistentialPredicate, symbol_table
    )
    if split is None:
        return None
    remainder, existential = split
    if _shares_a_choice_relation(existential, remainder, symbol_table):
        return None
    negation_plan = free_variable_shattering_plan(
        consequent, remainder, existential, symbol_table
    )
    if negation_plan is None:
        return None
    plans = _lift_pure((Implication(consequent, remainder),), symbol_table)
    if plans is None:
        return None
    return rap.Subtraction(plans[0], negation_plan)


def negation_subtraction_plan(
    consequent, remainder, negated_formula, symbol_table
):
    """Plan for `A ∧ ¬N` as P[A] - P[A ∧ N], which is sound for any `A`
    and `N`, as `A ∧ N` implies `A`.

    Parameters
    ----------
    consequent : FunctionApplication
        head of the rule.
    remainder : Conjunction
        the conjunction `A`.
    negated_formula : LogicExpression
        the formula `N`, whose free variables are those of `A` or a
        subset.
    symbol_table : Mapping of Symbol to Expression
        symbol table for the relational symbols

    Returns
    -------
    Optional[RelationalAlgebraOperation]
        the plan, or None if `A` or `A ∧ N` is not liftable.
    """
    plans = _lift_pure(
        (
            Implication(consequent, remainder),
            Implication(
                consequent,
                Conjunction(remainder.formulas + (negated_formula,))
            ),
        ),
        symbol_table
    )
    if plans is None:
        return None
    return rap.Subtraction(*plans)


def free_variable_shattering_plan(
    consequent, remainder, negated_formula, symbol_table
):
    """Plan for `A ∧ ¬∃t ψ(t)` when the dependency between `A` and `ψ`
    comes from atoms in which a free variable `x` of `A` takes the place
    of `t`, such as `R(x) ∧ ¬∃t (R(t) ∧ S(x, t))` or
    `R(x, y) ∧ ¬∃t (R(x, t) ∧ T(t))`. As

        ∃t ψ(t) ≡ ψ(x) ∨ ∃t (t ≠ x ∧ ψ(t))

    the plan is, with `L = lift(A ∧ ¬ψ(x))`,

        Difference(L, ind-π σ_{t ≠ x}(lift(ψ(t)) ⋈ support(L)))

    where joining with the tuples of `L` provides the values of `x`
    when it is not a free variable of `ψ`.

    which is sound when

    * `t` is a separator variable of `ψ`, so that the events ψ(t) for
      different `t` are independent;
    * when t ≠ x, every probabilistic atom of `ψ(t)` is a different
      tuple from the atoms of the same relation in `A` and `ψ(x)`: it
      has `t` in a position where they have `x`.

    Parameters
    ----------
    consequent : FunctionApplication
        head of the rule.
    remainder : Conjunction
        the conjunction `A`.
    negated_formula : LogicExpression
        the formula `∃t ψ(t)`, whose free variables are those of `A` or
        a subset.
    symbol_table : Mapping of Symbol to Expression
        symbol table for the relational symbols

    Returns
    -------
    Optional[RelationalAlgebraOperation]
        the plan, or None if the rule does not apply.
    """
    free_variables = extract_logic_free_variables(negated_formula)
    body = _existential_body(negated_formula)
    remainder_variables = extract_logic_free_variables(remainder)
    for t in extract_logic_free_variables(body) - free_variables:
        psi_t = add_existentials_except(body, free_variables | {t})
        if not _is_separator_variable(t, psi_t, symbol_table):
            continue
        for x in sorted(remainder_variables, key=lambda v: v.name):
            psi_x = ReplaceExpressionWalker({t: x}).walk(psi_t)
            if not _are_different_tuples_when_different(
                psi_t, t, Conjunction(remainder.formulas + (psi_x,)), {x},
                symbol_table
            ):
                continue
            plans = _lift_pure(
                (Implication(
                    consequent,
                    Conjunction(remainder.formulas + (Negation(psi_x),))
                ),),
                symbol_table
            )
            if plans is None:
                continue
            separated_plan = _separated_existential_plan(
                t, psi_t, (NE(t, x),), plans[0], remainder_variables,
                symbol_table
            )
            if separated_plan is not None:
                return rap.Difference(plans[0], separated_plan)
    return None


def comparison_conjunct_plan(consequent, remainder, conjunct, symbol_table):
    """Plan for `A ∧ ∃t (ψ(t) ∧ C)` or `A ∧ ¬∃t (ψ(t) ∧ C)`, where `C`
    is a conjunction of comparisons between `t`, the free variables and
    constants, e.g. `P(x, y) ∧ ¬∃w (P(w, y) ∧ w ≠ x)`. Following
    Olteanu & Huang [1], with `L = lift(A)`, the event
    `E = ∃t (ψ(t) ∧ C)` is planned as

        ind-π σ_C(lift(ψ(t)) ⋈ support(L))

    and the plan is `L ⋈ E`, or `Difference(L, E)` if negated. This is
    sound when `t` is a separator variable of `ψ`, and when every
    probabilistic atom of `ψ(t)` is a different tuple from the atoms
    of the same relation in `A`, because it has `t` in a position where
    they have a term from which `C` makes `t` different.

    [1] Olteanu, D. & Huang, J. Secondary-storage confidence computation
    for conjunctive queries with inequalities. SIGMOD 2009, 389–402.

    Parameters
    ----------
    consequent : FunctionApplication
        head of the rule.
    remainder : Conjunction
        the conjunction `A`.
    conjunct : LogicExpression
        the existential conjunct, possibly negated.
    symbol_table : Mapping of Symbol to Expression
        symbol table for the relational symbols

    Returns
    -------
    Optional[RelationalAlgebraOperation]
        the plan, or None if the rule does not apply.
    """
    negated = isinstance(conjunct, Negation)
    existential = conjunct.formula if negated else conjunct
    free_variables = extract_logic_free_variables(existential)
    body = _existential_body(existential)
    formulas = body.formulas if isinstance(body, Conjunction) else (body,)
    comparisons = tuple(f for f in formulas if is_comparison(f))
    psi_body = Conjunction(tuple(
        f for f in formulas if not is_comparison(f)
    ))
    if _shares_a_choice_relation(psi_body, remainder, symbol_table):
        return None
    remainder_variables = extract_logic_free_variables(remainder)
    for t in extract_logic_free_variables(body) - free_variables:
        if not all(
            extract_logic_free_variables(comparison) <= free_variables | {t}
            for comparison in comparisons
        ):
            continue
        psi_t = add_existentials_except(psi_body, free_variables | {t})
        if not (
            _is_separator_variable(t, psi_t, symbol_table) and
            _are_different_tuples_when_different(
                psi_t, t, remainder,
                _terms_different_from(t, comparisons), symbol_table
            )
        ):
            continue
        plans = _lift_pure(
            (Implication(consequent, remainder),), symbol_table
        )
        if plans is None:
            return None
        separated_plan = _separated_existential_plan(
            t, psi_t, comparisons, plans[0], remainder_variables,
            symbol_table
        )
        if separated_plan is None:
            continue
        if negated:
            return rap.Difference(plans[0], separated_plan)
        return rap.NaturalJoin(plans[0], separated_plan)
    return None


def _separated_existential_plan(
    t, psi_t, comparisons, remainder_plan, remainder_variables,
    symbol_table
):
    """Plan for `∃t (ψ(t) ∧ C)`, `t` being a separator variable of `ψ`,
    for each tuple of `remainder_plan`, which provides the values of the
    free variables of `C` which are not free in `ψ`:

        ind-π σ_C(lift(ψ(t)) ⋈ support(remainder_plan))
    """
    plans = _lift_pure(
        (Implication(
            Symbol.fresh()(*extract_logic_free_variables(psi_t)), psi_t
        ),),
        symbol_table
    )
    if plans is None:
        return None
    plan = rap.NaturalJoin(plans[0], rap.Support(remainder_plan))
    for comparison in comparisons:
        plan = rap.Selection(plan, columns_comparison(comparison))
    return rap.IndependentProjection(
        plan,
        tuple(str2columnstr_constant(v.name) for v in remainder_variables)
    )


def _are_different_tuples_when_different(
    formula, t, other_formula, different_terms, symbol_table
):
    """Whether, when `t` is different from `different_terms`, every
    probabilistic atom of `formula` is a different tuple from the atoms of
    the same relation in `other_formula`, because it has `t` in a
    position where they have one of `different_terms`.
    """
    deterministic_symbols = _deterministic_symbols(formula, symbol_table)
    other_atoms = extract_logic_atoms(other_formula)
    return all(
        any(
            arg == t and other_arg in different_terms
            for arg, other_arg in zip(atom.args, other_atom.args)
        )
        for atom in extract_logic_atoms(formula)
        if atom.functor not in deterministic_symbols
        for other_atom in other_atoms
        if other_atom.functor == atom.functor
    )


def _is_separator_variable(t, formula, symbol_table):
    separator_variables, _ = find_separator_variables(
        ExistentialPredicate(t, formula), symbol_table
    )
    return t in separator_variables


def _existential_body(formula):
    while isinstance(formula, ExistentialPredicate):
        formula = formula.body
    return formula


def _is_existential_conjunct(formula):
    if isinstance(formula, Negation):
        formula = formula.formula
    return isinstance(formula, ExistentialPredicate)


def _terms_different_from(t, comparisons):
    """Terms from which the comparisons make `t` different."""
    terms = set()
    for comparison in comparisons:
        if comparison.functor.value in STRICT_COMPARISONS:
            left, right = comparison.args
            if left == t:
                terms.add(right)
            elif right == t:
                terms.add(left)
    return terms


def _lift_pure(rules, symbol_table):
    """Lift each rule, returning the plans if all of them are pure,
    None otherwise.
    """
    plans = tuple(
        lift_with_dependent_negations(rule, symbol_table, check=True)
        for rule in rules
    )
    if all(is_pure_lifted_plan(plan) for plan in plans):
        return plans
    return None


def _split_comparison_conjunct(antecedent):
    """Find an existential conjunct, possibly negated, with comparisons
    and whose free variables are among those of the rest.

    Returns
    -------
    Optional[Tuple[Conjunction, LogicExpression]]
        the conjunction without the conjunct and the conjunct, or None if
        there is no such conjunct.
    """
    if not isinstance(antecedent, Conjunction):
        return None
    for i, formula in enumerate(antecedent.formulas):
        remainder = Conjunction(
            antecedent.formulas[:i] + antecedent.formulas[i + 1:]
        )
        if (
            _is_existential_conjunct(formula) and
            any(is_comparison(a) for a in extract_logic_atoms(formula)) and
            extract_logic_free_variables(formula) <=
            extract_logic_free_variables(remainder)
        ):
            return remainder, formula
    return None


def _split_dependent_conjunct(antecedent, conjunct_class, symbol_table):
    """Find a conjunct of class `conjunct_class` sharing probabilistic
    symbols with the rest of the conjunction, and whose free variables
    are among those of the rest.

    Returns
    -------
    Optional[Tuple[Conjunction, LogicExpression]]
        the conjunction without the conjunct and the conjunct, or None if
        there is no such conjunct.
    """
    if not isinstance(antecedent, Conjunction):
        return None
    deterministic_symbols = _deterministic_symbols(antecedent, symbol_table)
    for i, formula in enumerate(antecedent.formulas):
        if not isinstance(formula, conjunct_class):
            continue
        remainder = Conjunction(
            antecedent.formulas[:i] + antecedent.formulas[i + 1:]
        )
        if (
            extract_logic_free_variables(formula) <=
            extract_logic_free_variables(remainder) and
            (_functors(formula) & _functors(remainder)) -
            deterministic_symbols
        ):
            return remainder, formula
    return None


def _shares_a_choice_relation(formula, other_formula, symbol_table):
    return any(
        is_atom_a_probabilistic_choice_relation(atom, symbol_table)
        for atom in extract_logic_atoms(formula)
        if atom.functor in _functors(other_formula)
    )


def solve_marg_query(rule, cpl):
    return lift_solve_marg_query(rule, cpl, solve_succ_query)


def dalvi_suciu_lift(rule, symbol_table):
    '''
    Translation from a datalog rule which allows disjunctions in the body
    to a safe plan according to [1]. Non-liftable segments are identified
    by the `NonLiftable` expression. Plans are cached, see `clear_cache`.
    [1] Dalvi, N. & Suciu, D. The dichotomy of probabilistic inference
    for unions of conjunctive queries. J. ACM 59, 1–87 (2012).
    '''
    key = _lift_cache_key(rule, symbol_table)
    with _dalvi_suciu_lift_cache_lock:
        if key in _dalvi_suciu_lift_cache:
            _dalvi_suciu_lift_cache.move_to_end(key)
            return _dalvi_suciu_lift_cache[key]

    plan = _dalvi_suciu_lift(rule, symbol_table)

    with _dalvi_suciu_lift_cache_lock:
        _dalvi_suciu_lift_cache[key] = plan
        if len(_dalvi_suciu_lift_cache) > _MAX_DALVI_SUCIU_CACHE_SIZE:
            _dalvi_suciu_lift_cache.popitem(last=False)
    return plan


def _dalvi_suciu_lift(rule, symbol_table):
    #  rule = RemoveUniversalPredicates().walk(rule)
    rule = RTO.walk(rule)

    has_safe_plan, res = symbol_or_deterministic_plan(rule, symbol_table)
    if has_safe_plan:
        return res

    deterministic_symbols = _deterministic_symbols(rule, symbol_table)
    rule_cnf = convert_ucq_to_ccq(rule, transformation='CNF')
    if len(extract_logic_atoms(rule_cnf)) < len(extract_logic_atoms(rule)):
        # Minimisation removed redundant atoms: the base cases
        # might apply to the minimised rule.
        return dalvi_suciu_lift(RTO.walk(rule_cnf), symbol_table)
    connected_components = symbol_connected_components(
        rule_cnf, ignore_symbols=deterministic_symbols
    )
    if len(connected_components) > 1:
        return components_plan(
            connected_components, rap.NaturalJoin, symbol_table,
            negative_operation=rap.Difference
        )

    rule_dnf = convert_ucq_to_ccq(rule, transformation='DNF')
    connected_components = symbol_connected_components(
        rule_dnf, ignore_symbols=deterministic_symbols
    )
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

    if (
        len(rule_cnf.formulas) > 1 and
        _have_the_same_free_variables(rule_cnf.formulas)
    ):
        return inclusion_exclusion_conjunction(rule_cnf, symbol_table)

    return NonLiftable(rule)


def _functors(formula):
    return set(atom.functor for atom in extract_logic_atoms(formula))


def _deterministic_symbols(rule, symbol_table):
    return frozenset(
        atom.functor for atom in extract_logic_atoms(rule)
        if is_atom_a_deterministic_relation(atom, symbol_table)
    )


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


def symbol_connected_components(expression, ignore_symbols=frozenset()):
    if not isinstance(expression, NaryLogicOperator):
        raise ValueError(
            "Connected components can only be computed "
            "for n-ary logic operators."
        )
    c_matrix = symbol_co_occurence_graph(
        expression, ignore_symbols=ignore_symbols
    )
    components = connected_components(c_matrix)

    operation = type(expression)
    return [
        operation(tuple(expression.formulas[i] for i in component))
        for component in components
    ]


def symbol_co_occurence_graph(expression, ignore_symbols=frozenset()):
    """Symbol co-ocurrence graph expressed as
    an adjacency matrix.

    Parameters
    ----------
    expression : NAryLogicExpression
        logic expression for which the adjacency matrix is computed.
    ignore_symbols : AbstractSet[Symbol]
        predicate symbols which do not make two subformulas adjacent,
        e.g. deterministic relations, whose tuples are not random
        and therefore do not introduce dependencies.

    Returns
    -------
    numpy.ndarray
        squared binary array where a component is 1 if there is a
        shared predicate symbol between two subformulas of the
        logic expression.
    """
    c_matrix = np.zeros((len(expression.formulas),) * 2)
    for i, formula in enumerate(expression.formulas):
        atom_symbols = _functors(formula) - ignore_symbols
        for j, formula_ in enumerate(expression.formulas[i + 1:]):
            atom_symbols_ = _functors(formula_) - ignore_symbols
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
            idxs = set(np.atleast_1d(adjacency_matrix[idx]).nonzero()[0]) - component
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

    This algorithm assumes that Q1 can't be split into independent
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
    the given separator variables.

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


def _have_the_same_free_variables(formulas):
    """Inclusion-exclusion is only sound when all conjuncts have the same
    free variables: the disjunctions in the expansion are otherwise
    not range-restricted and the containment test used to compute the
    Möbius weights does not apply.
    """
    free_variables = extract_logic_free_variables(formulas[0])
    return all(
        extract_logic_free_variables(formula) == free_variables
        for formula in formulas[1:]
    )


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
