"""
Show-RA support for the frontends.

Machinery behind ``show_ra=True`` (``--show-ra`` in the CLI): instead of
reporting a query's answer, print the named relational algebra plan each
rule is evaluated with, stratum by stratum.
"""
from dataclasses import dataclass
from functools import lru_cache

from ..datalog.instance import MapInstance
from ..exceptions import UnsupportedSolverError
from ..logic import Conjunction, Union
from ..logic.transformations import RemoveDuplicatedConjunctsDisjuncts
from ..probabilistic.expression_processing import (
    is_within_language_prob_query,
)
from ..probabilistic.expressions import Condition
from ..probabilistic.query_resolution import (
    _build_probabilistic_program,
    lift_solve_marg_query,
    within_language_succ_query_to_intensional_rule,
)
from ..relational_algebra.optimisers import (
    DegenerateNaturalJoinToProduct,
    GreedyJoinOrdering,
    RelationalAlgebraOptimiser,
)
from ..relational_algebra.pretty_printer import (
    build_name_map_from_conjunction,
    pretty_repr,
)
from .datalog.pretty_printer import DatalogPrettyPrinter
from .stratum_solver import StratumSolver

__all__ = [
    "ShowRAChaseMixin",
    "ShowRAOptions",
    "ShowRAStratumSolver",
    "print_deterministic_ra_plan",
    "print_rule_ra_plan",
    "show_ra_chase_class",
    "show_ra_options",
]


@dataclass(frozen=True)
class ShowRAOptions:
    """
    Options of the relational algebra plan display (``show_ra``).

    Parameters
    ----------
    collapse : bool
        If True (the default), trivial wrappers are collapsed for
        readability, e.g. ``ρ[x, y](π[0, 1](edge))`` is printed as
        ``edge(x, y)``. If False, the plans are printed in full.
    analyze : bool
        If True, the query is run and every node of each plan is
        annotated with the number of rows it produced, as SQL's
        ``EXPLAIN ANALYZE`` does.

    """

    collapse: bool = True
    analyze: bool = False


def show_ra_options(show_ra):
    """
    Return the :class:`ShowRAOptions` requested by ``show_ra``.

    ``show_ra`` is a boolean or a :class:`ShowRAOptions`: ``True`` stands
    for the default options and a false value for no plan display, in
    which case None is returned.
    """
    if isinstance(show_ra, ShowRAOptions):
        return show_ra
    if show_ra:
        return ShowRAOptions()
    return None


def print_rule_ra_plan(rule, conjunction, symbol_table, build_ra_code):
    """
    Print the ``--show-ra`` block of a rule.

    The block is the rule header (skipped when ``rule`` is None) followed
    by the pretty-printed RA plan returned by ``build_ra_code``, whose
    tables are named after the predicates of ``conjunction``.
    """
    if rule is not None:
        rule_str = DatalogPrettyPrinter().walk(rule)
        print(f"── rule {rule_str} ──")
    name_map = build_name_map_from_conjunction(conjunction, symbol_table)
    print(pretty_repr(build_ra_code(), name_map=name_map))


@lru_cache(maxsize=None)
def show_ra_chase_class(chase_class):
    """Return ``chase_class`` extended with :class:`ShowRAChaseMixin`."""
    return type(
        f"ShowRA{chase_class.__name__}",
        (ShowRAChaseMixin, chase_class),
        {},
    )


def print_deterministic_ra_plan(chase_class, program, rules, options=None):
    """
    Print the RA plans of the deterministic stratum ``rules``.

    The stratum is run with ``chase_class`` extended with
    :class:`ShowRAChaseMixin`, displaying the plans with the
    :class:`ShowRAOptions` ``options`` (the default ones when None), and
    the resulting chase solution is returned.
    """
    print("── deterministic stratum ──")
    chase = show_ra_chase(chase_class, program, rules, options)
    return chase.build_chase_solution()


def show_ra_chase(chase_class, program, rules, options=None):
    """
    Build a show-ra chase of ``rules`` over ``program``.

    The chase is an instance of ``chase_class`` extended with
    :class:`ShowRAChaseMixin`, which displays the plans with the
    :class:`ShowRAOptions` ``options`` (the default ones when None).
    """
    chase = show_ra_chase_class(chase_class)(program, rules=rules)
    chase.show_ra_options = options or ShowRAOptions()
    return chase


class ShowRAChaseMixin:
    """
    Chase mixin printing the named RA expression of each rule.

    Before each rule is evaluated, the named RA expression it is
    evaluated with is printed. Errors raised while evaluating a rule
    (e.g. a predicate used with the wrong arity) reach the caller, as
    they do without show-ra. The plans are displayed with the
    :class:`ShowRAOptions` ``show_ra_options``.
    """

    show_ra_options = ShowRAOptions()

    def __init__(self, *args, **kwargs):
        """Initialise the chase with no current rule."""
        super().__init__(*args, **kwargs)
        self._show_ra_current_rule = None

    def chase_step(self, instance, rule, restriction_instance=None):
        """Record ``rule`` as the current rule, then run the step."""
        self._show_ra_current_rule = rule
        return super().chase_step(instance, rule, restriction_instance)

    def obtain_substitutions(
        self, rule_predicates_iterator, instance, restriction_instance
    ):
        """Print the RA plan of the current rule, then evaluate it."""
        predicates = tuple(rule_predicates_iterator)
        if not predicates:
            return []
        conjunction = Conjunction(predicates)
        ra_code = self.translate_conjunction_to_named_ra(conjunction)
        print_rule_ra_plan(
            getattr(self, "_show_ra_current_rule", None),
            conjunction,
            getattr(self, "datalog_program", None),
            lambda: ra_code,
        )
        return super().obtain_substitutions(
            iter(predicates), instance, restriction_instance
        )

    def pick_chase_instance_for_stratum(self, stratum, instance_update):
        """Wrap the inner per-stratum chase so its rules print RA too."""
        chase_instance = super().pick_chase_instance_for_stratum(
            stratum, instance_update
        )
        if chase_instance is None:
            return None
        wrapped = show_ra_chase(
            chase_instance.__class__,
            chase_instance.datalog_program,
            Union(tuple(chase_instance.rules)),
            self.show_ra_options,
        )
        wrapped.check_constraints(instance_update)
        return wrapped


class ShowRAStratumSolver(StratumSolver):
    """
    Print the RA plans of each stratum instead of evaluating it.

    The deterministic stratum is run with a show-ra chase, so that the
    probabilistic plans can be built over its solution. The probabilistic
    and post-probabilistic strata are only printed, and an empty
    ``MapInstance`` is returned for them.

    Parameters
    ----------
    engine : NeurolangPDL
        engine whose program is displayed.
    options : ShowRAOptions, optional
        how the plans are displayed, the default options when None.

    """

    def __init__(self, engine, options=None):
        """Store the engine and the display options."""
        super().__init__(engine)
        self.options = options or ShowRAOptions()

    def deterministic(self, det_idb):
        """Print the RA plan of every deterministic rule."""
        return print_deterministic_ra_plan(
            self.engine.chase_class, self.engine.program_ir, det_idb,
            self.options,
        )

    def probabilistic(self, solution, prob_idb):
        """Print the optimised RA plan of every probabilistic rule."""
        print("── probabilistic stratum ──")
        engine = self.engine
        cpl, prob_idb = _build_probabilistic_program(
            solution,
            engine.program_ir.probabilistic_facts(),
            engine.program_ir.probabilistic_choices(),
            prob_idb,
            engine.check_qbased_pfact_tuple_unicity,
        )
        dedup = RemoveDuplicatedConjunctsDisjuncts()
        prob_idb = Union(tuple(dedup.walk(r) for r in prob_idb.formulas))
        for rule in prob_idb.formulas:
            print_rule_ra_plan(
                rule,
                rule.antecedent,
                engine.program_ir.symbol_table,
                lambda rule=rule: self.optimise_plan(
                    self.rule_provenance_set(rule, cpl).relation
                ),
            )
        return MapInstance()

    def postprobabilistic(self, solution, postprob_idb):
        """Print the RA plan of every post-probabilistic rule."""
        print("── post-probabilistic stratum ──")
        solver = self.engine._build_postprobabilistic_solver(
            solution, postprob_idb
        )
        chase = show_ra_chase(
            self.engine.chase_class, solver, postprob_idb, self.options
        )
        chase.build_chase_solution()
        return MapInstance()

    def rule_provenance_set(self, rule, cpl):
        """
        Build the provenance set of a probabilistic rule without solving it.

        The relational algebra solver is not run, so the returned set holds
        the unevaluated RA plan.
        """
        query = rule
        solve = self._solve_unevaluated_succ_query
        if is_within_language_prob_query(rule):
            query = within_language_succ_query_to_intensional_rule(rule)
            if isinstance(rule.antecedent, Condition):
                # As in the solver (_solve_within_language_prob_query), a
                # conditional query goes through the MARG solver, which
                # splits it into joint and conditioning SUCC queries and
                # does not run the RA solver on them.
                solve = self._solve_unevaluated_marg_query
        solvers = self.engine.probabilistic_solvers
        for succ_solver in solvers:
            try:
                return solve(query, cpl, succ_solver)
            except UnsupportedSolverError:
                if succ_solver == solvers[-1]:
                    raise

    @staticmethod
    def _solve_unevaluated_succ_query(query, cpl, succ_solver):
        """Build the provenance set of a SUCC query without evaluating it."""
        return succ_solver(query, cpl, run_relational_algebra_solver=False)

    @staticmethod
    def _solve_unevaluated_marg_query(rule, cpl, succ_solver):
        """Build the provenance set of a MARG query without evaluating it."""
        return lift_solve_marg_query(rule, cpl, succ_solver)

    @staticmethod
    def optimise_plan(ra_code):
        """
        Optimise an RA plan for display.

        Apply the RA optimiser together with greedy join ordering, then run
        a second optimisation pass after the reordering.
        """
        class ShowRAOptimiser(
            RelationalAlgebraOptimiser,
            DegenerateNaturalJoinToProduct,
            GreedyJoinOrdering,
        ):
            pass
        ra_code = ShowRAOptimiser().walk(ra_code)
        # Second RA optimisation pass after GOO reordering
        return RelationalAlgebraOptimiser().walk(ra_code)
