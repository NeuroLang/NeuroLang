r"""
Probabilistic Frontend
======================
Complements QueryBuilderDatalog class with probabilistic capabilities
1- add extensional probabilistic facts and choices
2- solve probabilistic queries
"""
import collections
import functools
import typing
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)
from uuid import uuid1

import pandas as pd
from nl_expr.type_system import get_args, replace_type_variable_fix_python36_37

from nl_expr import expressions as ir
from nl_core.datalog.aggregation import (
    BuiltinAggregationMixin,
    DatalogWithAggregationMixin,
    TranslateToLogicWithAggregation,
)
from nl_core.datalog.chase import Chase
from nl_core.datalog.constraints_representation import (
    DatalogConstraintsProgram,
)
from nl_core.datalog.exceptions import InvalidMagicSetError
from nl_expr.exceptions import SymbolNotFoundError
from nl_core.datalog.expression_processing import (
    EqualitySymbolLeftHandSideNormaliseMixin,
    InlineEqualityConstantsMixin,
)
from nl_core.datalog.magic_sets import magic_rewrite
from nl_core.datalog.negation import DatalogProgramNegationMixin
from nl_core.datalog.ontologies_parser import OntologyParser
from nl_core.datalog.ontologies_rewriter import OntologyRewriter

from nl_expr.exceptions import (
    UnsupportedQueryError,
    UnsupportedSolverError,
    UnsupportedProgramError,
)
from nl_expr.expression_walker import (
    ExpressionBasicEvaluator,
    TypedSymbolTableMixin,
)
from nl_expr.logic import Implication, Union, Symbol
from nl_expr.logic.expression_processing import extract_logic_free_variables
from nl_core.probabilistic import (
    dalvi_suciu_lift,
    small_dichotomy_theorem_based_solver,
)
from nl_core.probabilistic.cplogic.program import CPLogicMixin
from nl_core.probabilistic.expressions import PROB, Condition
from .resampling import (
    ResampledProgram,
    resampling_relation,
    rewrite_resample_markers,
)
from nl_core.probabilistic.expression_processing import (
    is_probabilistic_predicate_symbol,
    is_within_language_prob_query,
)
from nl_core.probabilistic.magic_sets_processing import (
    probabilistic_postprocess_magic_rules,
)
from nl_core.probabilistic.query_resolution import (
    QueryBasedProbFactToDetRule,
    compute_probabilistic_solution,
)
from nl_core.probabilistic.stratification import (
    get_list_of_intensional_rules,
    stratify_program,
)
from nl_core.probabilistic.weighted_model_counting import (
    solve_marg_query as wmc_solve_marg_query,
)
from nl_core.probabilistic.weighted_model_counting import (
    solve_succ_query as wmc_solve_succ_query,
)
from ..region_solver import RegionSolver
from nl_core.relational_algebra import (
    NamedRelationalAlgebraFrozenSet,
    RelationalAlgebraColumnStr,
)
from nl_core.datalog.wrapped_collections import (
    WrappedRelationalAlgebraFrozenSet,
)
from ..commands import CommandsMixin
from nl_core.datalog.basic_representation import UnionOfConjunctiveQueries
from . import query_resolution_expressions as fe
from .datalog.sugar import (
    TranslateProbabilisticQueryMixin,
    TranslateQueryBasedProbabilisticFactMixin,
)
from .datalog.sugar.spatial import (
    TranslateEuclideanDistanceBoundMatrixMixin,
    TranslateRegionDestroy,
)
from .datalog.syntax_preprocessing import ProbFol2DatalogMixin
from .type_resolution import ProbabilisticTypeResolutionMixin
from .datalog.squall import ResolveInvertedFunctionApplicationMixin, StripDimensionTypePredicatesMixin
from .frontend_extensions import NumpyFunctionsMixin
from .query_resolution_datalog import QueryBuilderDatalog
from .show_ra import (
    ShowRAStratumSolver,
    generated_query_symbol,
    show_ra_options,
)
from .stratum_solver import StratumSolver, with_empty_relations


def _symbol_names(symbols):
    """Return the names of ``symbols``."""
    return tuple(symbol.name for symbol in symbols)


def instance_lru_cache(key_fn, maxsize=128):
    """LRU cache decorator for instance methods with non-hashable arguments.

    key_fn(*args, **kwargs) must return a hashable cache key.
    The cache is stored per-instance to avoid cross-instance sharing.
    Call wrapper.cache_clear(instance) to invalidate.
    """
    def decorator(method):
        cache_attr = f'_lru_cache_{method.__name__}'

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            cache = getattr(self, cache_attr, None)
            if cache is None:
                cache = collections.OrderedDict()
                setattr(self, cache_attr, cache)
            key = key_fn(*args, **kwargs)
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            result = method(self, *args, **kwargs)
            cache[key] = result
            if maxsize is not None and len(cache) > maxsize:
                cache.popitem(last=False)
            return result

        def cache_clear(instance):
            setattr(instance, cache_attr, collections.OrderedDict())

        wrapper.cache_clear = cache_clear
        return wrapper
    return decorator


class RegionFrontendCPLogicSolver(
    EqualitySymbolLeftHandSideNormaliseMixin,
    InlineEqualityConstantsMixin,
    TranslateProbabilisticQueryMixin,
    ResolveInvertedFunctionApplicationMixin,
    StripDimensionTypePredicatesMixin,
    TranslateToLogicWithAggregation,
    TranslateQueryBasedProbabilisticFactMixin,
    TranslateEuclideanDistanceBoundMatrixMixin,
    QueryBasedProbFactToDetRule,
    ProbFol2DatalogMixin,
    ProbabilisticTypeResolutionMixin,
    TranslateRegionDestroy,
    RegionSolver,
    CommandsMixin,
    NumpyFunctionsMixin,
    CPLogicMixin,
    DatalogWithAggregationMixin,
    BuiltinAggregationMixin,
    DatalogProgramNegationMixin,
    DatalogConstraintsProgram,
    TypedSymbolTableMixin,
    ExpressionBasicEvaluator,
):
    pass


class NeurolangPDL(QueryBuilderDatalog):
    """
    Probabilistic Datalog frontend.

    Complements :class:`QueryBuilderDatalog` with probabilistic
    capabilities:

    1. add extensional probabilistic facts and choices;
    2. solve probabilistic queries.
    """

    def __init__(
        self,
        chase_class: Type[Chase] = Chase,
        probabilistic_solvers: Tuple[Callable] = (
            small_dichotomy_theorem_based_solver.solve_succ_query,
            dalvi_suciu_lift.solve_succ_query,
            wmc_solve_succ_query,
        ),
        probabilistic_marg_solvers: Tuple[Callable] = (
            small_dichotomy_theorem_based_solver.solve_marg_query,
            dalvi_suciu_lift.solve_marg_query,
            wmc_solve_marg_query,
        ),
        check_qbased_pfact_tuple_unicity=False,
        solver_class: Optional[Type] = None,
    ) -> "NeurolangPDL":
        """
        Query builder with probabilistic capabilities

        Parameters
        ----------
        chase_class : Type[Chase], optional
            used to compute deterministic solutions, by default Chase
        probabilistic_solvers : Tuple[Callable], optional
            used to compute probabilistic solutions,
            the order of the elements indicates the priority
            of usage of the solver.
            by default (lifted_solve_succ_query, wmc_solve_succ_query)

        probabilistic_marg_solvers : Tuple[Callable], optional
            used to compute probabilistic solutions,
            by default (lifted_solve_marg_query, wmc_solve_marg_query)

        solver_class : Type, optional
            Custom solver class to use instead of RegionFrontendCPLogicSolver.
            Allows injecting custom mixins (e.g. with dimension_type_predicate_names
            tailored from engine YAML config).


        Returns
        -------
        NeurolangPDL
            see description
        """
        if solver_class is not None:
            solver = solver_class()
        else:
            solver = RegionFrontendCPLogicSolver()
        super().__init__(
            solver, chase_class=chase_class
        )
        if len(probabilistic_solvers) != len(probabilistic_marg_solvers):
            raise ValueError(
                "probabilistic solve and marg "
                "solvers should be the same length"
            )
        self.probabilistic_solvers = probabilistic_solvers
        self.probabilistic_marg_solvers = probabilistic_marg_solvers
        self.current_program_rewritten = None
        self.check_qbased_pfact_tuple_unicity = (
            check_qbased_pfact_tuple_unicity
        )
        self.resamplings = {}

    def load_ontology(
        self,
        paths: typing.Union[str, List[str]],
        connector_symbol_name=None,
        format=None,
        individuals=False,
        names="local",
    ) -> None:
        """
        Loads and parses ontology stored at the specified paths, and
        store them into attributes

        Parameters
        ----------
        paths : typing.Union[str, List[str]]
            where the ontology files are stored
        connector_symbol_name : str
            name to be used in the connector_symbol.
            if None, the name is randomly generated
        format : str or list of str
            rdflib format of the files (``xml``, ``turtle``, ``nt``…),
            guessed from their extensions when None.
        individuals : bool
            also load every triple as a fact: ``s p o`` is ``p(s, o)``
            and ``s rdf:type C`` is also ``C(s)``, IRIs being the
            strings of their full IRI. The subclasses of the ontology
            apply to these facts: ``Super(x)`` holds for an instance of
            a subclass.
        names : str
            ``local`` names classes and properties by their local name
            (``Dean``, ``rdf-schema:label``); ``prefixed`` by the
            prefixes the files declare (``pv:name``, ``rdfs:label``),
            else by their full IRI.

        Returns
        -------
        Expression
            the new connector_symbol, to be used in
            the query program.
        """
        if connector_symbol_name is None:
            self.connector_symbol = self.new_symbol()
        else:
            self.connector_symbol = self.new_symbol(name=connector_symbol_name)
        onto = OntologyParser(
            paths, connector_symbol=self.connector_symbol.expression,
            load_format=format, individuals=individuals, names=names,
        )
        constraints, est_knowledge, entity_rules = onto.parse_ontology()
        self.program_ir.set_constraints(constraints)
        for name, expressions in constraints.items():
            symbol = Symbol(name=name).cast(UnionOfConjunctiveQueries)
            self.program_ir.symbol_table[symbol] = Union(expressions)
        for symbol, expressions in est_knowledge.items():
            iterable = [exp.args for exp in expressions]
            self.program_ir.add_extensional_predicate_from_tuples(
                symbol, iterable
            )
        self.program_ir.add_existential_rules(onto.existential_rules)
        for symbol, expressions in entity_rules.items():
            for e in expressions:
                self.program_ir.walk(e)
        if individuals:
            for name, rows in onto.parse_individuals().items():
                self.add_tuple_set(rows, name=name)

        self._rewrite_det_idb.cache_clear(self)
        return self.connector_symbol

    @property
    def current_program(self) -> List[fe.Expression]:
        """
        Returns the list of Front End Expressions that have
        currently been declared in the program, or through
        the program's constraints

        Returns
        -------
        List[fe.Expression]
            see description

        Examples
        --------
        >>> nl = NeurolangPDL()
        >>> P = nl.add_uniform_probabilistic_choice_over_set(
        ...     [("a",), ("b",), ("c",)], name="P"
        ... )
        >>> Q = nl.add_uniform_probabilistic_choice_over_set(
        ...     [("a",), ("d",), ("c",)], name="Q"
        ... )
        >>> with nl.scope as e:
        ...     e.Z[e.x, e.PROB[e.x]] = P[e.x] & Q[e.x]
        ...     cp = nl.current_program
        >>> cp
        [
            Z(x, ( PROB(x) )) ← ( P(x) ) ∧ ( Q(x) )
        ]
        """
        cp = []
        for constraint in self.program_ir.constraints().formulas:
            cp.append(self.frontend_translator.walk(constraint))
        cp += super().current_program
        return cp

    def _execute_query(
        self,
        head: typing.Union[fe.Symbol, Tuple[fe.Expression, ...]],
        predicate: fe.Expression,
        show_rewritten: bool = False,
        dry_run: bool = False,
        show_ra=False,
    ) -> Tuple[AbstractSet, Optional[ir.Symbol]]:
        """
        [Internal usage - documentation for developers]

        Performs an inferential query: will return as first output
        an AbstractSet with as many elements as solutions
        of the predicate query. AbstractSet's columns correspond to the
        expressions in the head.
        Typically, probabilities are abstracted and processed similar
        to symbols, though of different nature (see examples)
        If head expressions are arguments of a functor, the latter will
        be returned as the second output, defaulted as None

        Parameters
        ----------
        head : typing.Union[
            fe.Symbol,
            Tuple[fe.Expression, ...],
        ]
            see description
        predicate : Expression
            see description
        show_rewritten : bool
            If True, print the Datalog program after magic-sets rewriting.
        dry_run : bool
            If True, print the rewritten program and return immediately
            without running the chase.
        show_ra : bool or ShowRAOptions
            If True, print the named relational algebra expression
            of each rule instead of evaluating it. A
            :class:`~neurolang.frontend.show_ra.ShowRAOptions` selects
            how the plans are displayed.

        Returns
        -------
        Tuple[AbstractSet, Optional[ir.Symbol]]
            see description

        Examples
        --------
        Note: example ran with pandas backend
        >>> nl = NeurolangPDL()
        >>> P = nl.add_uniform_probabilistic_choice_over_set(
        ...     [("a",), ("b",), ("c",)], name="P"
        ... )
        >>> Q = nl.add_uniform_probabilistic_choice_over_set(
        ...     [("a",), ("d",), ("c",)], name="Q"
        ... )
        >>> with nl.scope as e:
        ...     e.Z[e.x, e.PROB[e.x]] = P[e.x] & Q[e.x]
        ...     s1 = nl._execute_query(tuple(), e.Z[e.x, e.p])
        ...     s2 = nl._execute_query((e.x, ), e.Z[e.x, e.p])
        ...     s3 = nl._execute_query(e.Z[e.x, e.p], e.Z[e.x, e.p])
        >>> s1
        (
            C{
                Empty DataFrame
                Columns: []
                Index: [0, 1]
                : typing.AbstractSet
            },
            None
        )
        >>> s2
        (
            C{
                    x
                0   a
                1   c
                : typing.AbstractSet
            },
            None
        )
        >>> s3
        (
            C{
                    x   p
                0   a   0.111111
                1   c   0.111111
                : typing.AbstractSet
            },
            S{Z: Unknown}
        )
        """
        if _has_probability(head):
            return self._execute_probabilistic_head_query(
                head, predicate, show_rewritten=show_rewritten,
                dry_run=dry_run, show_ra=show_ra,
            )
        query_pred_symb = predicate.expression.functor
        query_pred_args = predicate.expression.args
        if is_probabilistic_predicate_symbol(query_pred_symb, self.program_ir):
            raise UnsupportedQueryError(
                "Queries on probabilistic predicates are not supported"
            )

        # The probabilistic query path (below) assumes the predicate functor
        # maps to a rule entry (Union.formulas) in the program_ir symbol
        # table.  This holds when querying an IDB predicate defined by a
        # rule, but fails for:
        #   1. EDB predicates — symbol entry is a Constant (no .formulas).
        #   2. Conjunction queries (p(x), q(y)) — functor is `and_`, which
        #      is not in the symbol table at all.
        #   3. Heads computing a term, such as count(s): the path below
        #      only restricts the predicate's relation to its variables.
        # Conjunction queries, and EDB predicates when the program has an
        # ontology's constraints, are queried through a fresh rule defining
        # them, which takes the probabilistic path. The other cases fall
        # through to the base QueryBuilderDatalog implementation, which
        # creates a fresh query rule and only runs the deterministic chase.
        symbol_entry = self.program_ir.symbol_table.get(query_pred_symb, None)
        if _solved_as_query_rule(head, symbol_entry):
            is_edb = symbol_entry is None or isinstance(
                symbol_entry, ir.Constant
            )
            if is_edb and (
                "__constraints__" in self.symbol_table or (
                    symbol_entry is None and
                    self._depends_on_probabilistic_queries(predicate)
                )
            ):
                # The deterministic chase of the base implementation
                # neither solves the probabilistic queries a conjunction
                # may depend on nor applies the ontology's constraints.
                return self._execute_query_through_rule(
                    head, predicate, show_rewritten, dry_run, show_ra
                )
            if not is_edb and self._depends_on_probabilistic_queries(
                predicate
            ):
                return self._execute_query_computing_terms(
                    head, predicate, show_rewritten, dry_run, show_ra
                )
            return super()._execute_query(
                head, predicate, show_rewritten=show_rewritten, dry_run=dry_run,
                show_ra=show_ra,
            )

        with self.scope:
            query, query_pred_args = self._query_rule(
                symbol_entry, predicate.expression
            )
            return self._solve_query_rule(
                head, query, query_pred_args, show_rewritten, dry_run, show_ra
            )

    def _query_rule(self, symbol_entry, query_atom):
        """
        Return the rule to solve for ``query_atom`` and its head's arguments.

        When the arguments of ``query_atom`` are variables and constants,
        not all distinct variables, the rule is a fresh one,
        ``q(v) :- query_atom`` with ``v`` the variables of ``query_atom``,
        which are returned, added to the program: magic sets then restrict
        the solved rules to the constants and repeated variables of
        ``query_atom``. Otherwise the rule is the first of
        ``symbol_entry``, the rules of the query predicate, and the
        arguments are those of ``query_atom``.
        """
        args = query_atom.args
        variables = tuple(
            dict.fromkeys(arg for arg in args if isinstance(arg, ir.Symbol))
        )
        if len(variables) == len(args) or not all(
            isinstance(arg, (ir.Symbol, ir.Constant)) for arg in args
        ):
            return symbol_entry.formulas[0], args
        goal = ir.Symbol.fresh()
        self.program_ir.walk(Implication(goal(*variables), query_atom))
        return self.program_ir.symbol_table[goal].formulas[0], variables

    def _solve_query_rule(
        self, head, query, query_pred_args, show_rewritten, dry_run, show_ra
    ):
        """
        Solve the rule ``query`` and restrict its solution to ``head``.

        ``query_pred_args`` are the arguments of the query predicate, the
        consequent of ``query``; the other parameters and the result are
        those of :meth:`_execute_query`.
        """
        options = show_ra_options(show_ra)
        if options is not None:
            return self._show_ra_query(
                query, options, show_rewritten, dry_run,
                (self._query_head_symbols(head)[0], query_pred_args),
            )

        solved = self._solve_rewritten(query, show_rewritten, dry_run)
        if solved is None:
            return ir.Constant(WrappedRelationalAlgebraFrozenSet()), None
        solution, query_pred_symb = solved

        head_symbols, functor_orig = self._query_head_symbols(head)
        solution = self._restrict_to_query_solution(
            head_symbols, query_pred_symb, query_pred_args, solution
        )
        if functor_orig is None:
            solution = solution.value
        return solution, functor_orig

    def _solve_rewritten(self, query, show_rewritten, dry_run):
        """
        Solve ``query``, rewritten with magic sets if it can be.

        The rewritten program is printed, within the scope holding the
        magic rules it needs, when ``show_rewritten`` or ``dry_run`` is
        set. When the query cannot be rewritten, ``query`` itself is
        printed and solved. Return the solution and the symbol of the
        solved query predicate, or None for a dry run.
        """
        try:
            magic_query, magic_rules = self._magic_rewrite_query(query)
            with self.scope:
                self.program_ir.walk(magic_rules)
                if self._handle_rewritten_output(
                    magic_query, show_rewritten, dry_run
                ):
                    return None
                solution = self._solve(magic_query)
                return solution, magic_query.consequent.functor
        except (InvalidMagicSetError, UnsupportedProgramError, SymbolNotFoundError):
            if self._handle_rewritten_output(
                query, show_rewritten, dry_run
            ):
                return None
            return self._solve(query), query.consequent.functor

    def _execute_probabilistic_head_query(self, head, predicate, **options):
        """
        Answer a query whose head has a probability, as ``ans(x, PROB(x))``.

        Only a rule computes a probability: the query becomes the rule
        ``q(x, PROB(x)) :- predicate`` of a fresh ``q``, which is queried
        in a scope closed on return, and whose columns are named after
        the query's head.
        """
        if isinstance(predicate, Condition):
            # a conditional body reaches the query as a Condition of
            # front-end expressions; '//' builds it as a rule body
            predicate = predicate.conditioned // predicate.conditioning
        variables = tuple(self.new_symbol() for _ in head)
        with self.scope:
            rule_symbol = self.new_symbol()
            self._declare_implication(rule_symbol(*head), predicate)
            solution, _ = self._execute_query(
                variables, rule_symbol(*variables), **options
            )
        if isinstance(solution, ir.Constant):
            # nothing was solved: the rules or their plans were printed
            return solution, None
        names = self._parameter_names(h.expression for h in head)
        return solution.rename_columns(
            {v.expression.name: name for v, name in zip(variables, names)}
        ), None

    def _depends_on_probabilistic_queries(self, predicate):
        """Whether the rules ``predicate`` depends on have probabilistic
        queries, which the deterministic chase does not solve."""
        with self.scope:
            try:
                idbs = stratify_program(
                    self._declare_fresh_rule(predicate), self.program_ir
                )
            except SymbolNotFoundError:
                # Left to the base implementation, which reports it.
                return False
        return bool(idbs.get("probabilistic", Union(tuple())).formulas)

    def _execute_query_through_rule(
        self, head, predicate, show_rewritten, dry_run, show_ra
    ):
        """
        Run the query of ``predicate`` through a fresh rule defining it.

        The rule ``fresh(x, ...) :- predicate``, over the free variables
        of ``predicate``, lives in a scope closed on return; querying it
        rewrites the program with the ontology's constraints.
        """
        with self.scope:
            rule = self._declare_fresh_rule(predicate)
            return self._execute_query(
                head, fe.Expression(self, rule.consequent),
                show_rewritten=show_rewritten, dry_run=dry_run,
                show_ra=show_ra,
            )

    def _execute_query_computing_terms(
        self, head, predicate, show_rewritten, dry_run, show_ra
    ):
        """
        Run a query whose head computes terms, such as the aggregate
        ``max(p)``, over probabilistic queries: the deterministic chase of
        the base implementation does not solve them, the rule
        ``fresh(head) :- predicate`` is solved as any other.
        """
        with self.scope:
            fresh = self.new_symbol()
            self._declare_query(fresh(*head), predicate)
            variables = tuple(self.new_symbol() for _ in head)
            solution, _ = self._execute_query(
                variables, fresh(*variables),
                show_rewritten=show_rewritten, dry_run=dry_run,
                show_ra=show_ra,
            )
        if isinstance(solution, ir.Constant):
            # Nothing was run (dry_run, show_ra).
            return solution, None
        names = self._parameter_names(h.expression for h in head)
        return solution.rename_columns(
            dict(zip(solution.columns, names))
        ), None

    def _declare_fresh_rule(self, predicate):
        """Declare, in the current scope, the rule ``fresh(x, ...) :-
        predicate`` over the free variables of ``predicate``, and return
        it as stored in the program."""
        variables = tuple(sorted(
            extract_logic_free_variables(predicate.expression),
            key=lambda symbol: symbol.name,
        ))
        fresh = ir.Symbol.fresh()
        self.program_ir.walk(Union((
            Implication(fresh(*variables), predicate.expression),
        )))
        return self.program_ir.symbol_table[fresh].formulas[0]

    @staticmethod
    def _query_head_symbols(head):
        """
        Return the symbols of a query's head and its functor, if any.

        ``head`` is a tuple of expressions, or a predicate such as
        ``r(x, y)``, whose functor is returned (None for a tuple).
        """
        if not isinstance(head, tuple):
            # assumes head is a predicate e.g. r(x, y)
            return tuple(head.expression.args), head.expression.functor
        return tuple(t.expression for t in head), None

    def _magic_rewrite_query(self, query):
        """
        Rewrite ``query`` with magic sets.

        The rewriting is done in a scope that is closed on return, so the
        program is left unchanged. Return the rewritten query and the
        magic rules it needs.
        """
        try:
            return self._magic_rewrite_query_seeded(query, True)
        except (
            InvalidMagicSetError, UnsupportedProgramError, SymbolNotFoundError
        ):
            return self._magic_rewrite_query_seeded(query, False)

    def _magic_rewrite_query_seeded(self, query, seed_from_edb):
        """
        Rewrite ``query`` with magic sets, the literals of the extensional
        database with a constant binding variables if ``seed_from_edb``
        (see :class:`~neurolang.datalog.magic_sets.LeftToRightSIPS`).
        """
        with self.scope:
            goal, magic_rules = magic_rewrite(
                query.consequent, self.program_ir, seed_from_edb
            )
            self.program_ir.walk(magic_rules)
            magic_query = self.program_ir.symbol_table[goal].formulas[0]
            return probabilistic_postprocess_magic_rules(
                self.program_ir, magic_query, magic_rules
            )

    def _show_ra_query(
        self, query, options, show_rewritten, dry_run, query_atom
    ):
        """
        Print the RA plans of ``query`` instead of solving it.

        As in :meth:`_execute_query`, the plans are those of the query
        rewritten with magic sets, or of ``query`` itself when the rewritten
        query cannot be solved, followed by the restriction of the query
        predicate's relation to the query, whose head symbols and query
        predicate arguments are ``query_atom``. Errors raised while
        building the plans of ``query`` reach the caller. ``options`` is
        the :class:`~neurolang.frontend.show_ra.ShowRAOptions` of the
        display. An empty set is returned.
        """
        stratum_solver = ShowRAStratumSolver(
            self, options, generated_query_symbol(query)
        )
        try:
            magic_query, magic_rules = self._magic_rewrite_query(query)
            query_pred_symb = stratum_solver.query_predicate(
                query, magic_query
            )
            with self.scope:
                self.program_ir.walk(magic_rules)
                self._handle_rewritten_output(
                    magic_query, show_rewritten, dry_run, show_ra=True
                )
                solution = self._solve(
                    magic_query, stratum_solver=stratum_solver
                )
        except (InvalidMagicSetError, UnsupportedProgramError, SymbolNotFoundError):
            self._handle_rewritten_output(
                query, show_rewritten, dry_run, show_ra=True
            )
            solution = self._solve(query, stratum_solver=stratum_solver)
            query_pred_symb = query.consequent.functor
        head_symbols, query_pred_args = query_atom
        stratum_solver.print_query(
            head_symbols, query_pred_symb(*query_pred_args), solution
        )
        return ir.Constant(WrappedRelationalAlgebraFrozenSet()), None

    def solve_all(self, show_ra=False) -> Dict[str, NamedRelationalAlgebraFrozenSet]:
        """
        Returns a dictionary of "predicate_name": "Content"
        for all elements in the solution of the Datalog program.
        Typically, probabilities are abstracted and processed similar
        to symbols, though of different nature (see examples)

        Parameters
        ----------
        show_ra : bool or ShowRAOptions, optional
            If True, print the named RA expression of each rule instead
            of evaluating it, and return an empty solution. A
            :class:`~neurolang.frontend.show_ra.ShowRAOptions` selects how
            the plans are displayed.

        Returns
        -------
        Dict[str, NamedRelationalAlgebraFrozenSet]
            extensional and intentional facts that have been derived
            through the current program, optionally with probabilities

        Examples
        --------
        Note: example ran with pandas backend
        >>> nl = NeurolangPDL()
        >>> P = nl.add_uniform_probabilistic_choice_over_set(
        ...     [("a",), ("b",), ("c",)], name="P"
        ... )
        >>> Q = nl.add_uniform_probabilistic_choice_over_set(
        ...     [("a",), ("d",), ("c",)], name="Q"
        ... )
        >>> with nl.scope as e:
        ...     e.Z[e.PROB[e.x], e.x] = P[e.x] & Q[e.x]
        ...     solution = nl.solve_all()
        """
        solution_ir = self._solve(show_ra=show_ra)
        return self._solution_to_named_sets(solution_ir)

    def _solve(self, query=None, show_ra=False, stratum_solver=None):
        """
        Solve the program, restricted to ``query`` when given.

        The strata are solved by ``stratum_solver``; when it is None, a
        :class:`StratumSolver` or, if ``show_ra`` requests it, a
        :class:`~neurolang.frontend.show_ra.ShowRAStratumSolver`.
        """
        idbs = stratify_program(query, self.program_ir)
        det_idb = idbs.get("deterministic", Union(tuple()))
        prob_idb = idbs.get("probabilistic", Union(tuple()))
        postprob_idb = idbs.get("post_probabilistic", Union(tuple()))
        if stratum_solver is None:
            stratum_solver = self._stratum_solver(show_ra_options(show_ra))
        solution = stratum_solver.deterministic(det_idb)
        if prob_idb.formulas:
            solution = stratum_solver.probabilistic(solution, prob_idb)
        if postprob_idb.formulas:
            solution = stratum_solver.postprobabilistic(
                solution, postprob_idb
            )
        return solution

    def _stratum_solver(self, show_ra_opts):
        """Return the stratum solver for the ShowRAOptions (or None)."""
        if show_ra_opts is None:
            return StratumSolver(self)
        return ShowRAStratumSolver(self, show_ra_opts)

    def _solve_deterministic_stratum(self, det_idb):
        '''Resolution of the deterministic stratum. In case there
        are entries in the symbol table under the key __constraints__,
        a rewrite is performed and the resulting program is assigned
        to the variable `current_program_rewritten` to provide a way
        to access this information.

        In case connector_symbol has been defined, all rules derived
        from the ontology parsing are included in the program

        Parameters
        ----------
        det_idb : typing.Union
            union of rules composing the deterministic stratum.

        Returns
        -------
        Result obtained after resolution of the deterministic stratum
        '''
        if "__constraints__" in self.symbol_table:
            det_idb = self._rewrite_det_idb(det_idb)
            if hasattr(self, 'connector_symbol'):
                connector_rules = tuple(
                    [q
                    for q in get_list_of_intensional_rules(self.program_ir)
                    if self.connector_symbol.expression in q.antecedent._symbols
                ])
                det_idb = Union(det_idb.formulas + connector_rules)
            self.current_program_rewritten = det_idb
        chase = self.chase_class(self.program_ir, rules=det_idb)
        solution = chase.build_chase_solution()
        return solution

    def _solve_probabilistic_stratum(self, solution, prob_idb):
        pfact_edb = self.program_ir.probabilistic_facts()
        pchoice_edb = self.program_ir.probabilistic_choices()
        det_solution = with_empty_relations(
            solution, prob_idb, self.program_ir
        )
        for i, (succ_solver, marg_solver) in enumerate(
            zip(self.probabilistic_solvers, self.probabilistic_marg_solvers)
        ):
            try:
                prob_solution = compute_probabilistic_solution(
                    det_solution,
                    pfact_edb,
                    pchoice_edb,
                    prob_idb,
                    succ_solver,
                    marg_solver,
                    self.check_qbased_pfact_tuple_unicity,
                )
            except UnsupportedSolverError:
                if i == len(self.probabilistic_solvers) - 1:
                    raise
            else:
                break

        wlq_symbs = set(
            rule.consequent.functor
            for rule in prob_idb.formulas
            if is_within_language_prob_query(rule)
        )
        solution.update(
            {
                pred_symb: relation
                for pred_symb, relation in prob_solution.items()
                if pred_symb in wlq_symbs
            }
        )
        return solution

    def _build_postprobabilistic_solver(self, solution, postprob_idb):
        """
        Build the solver of the post-probabilistic deterministic stratum.

        The relations of ``solution`` become extensional predicates, the
        builtins of the program are copied, and the rules of
        ``postprob_idb`` are added.
        """
        solver = RegionFrontendCPLogicSolver()
        for psymb, relation in solution.items():
            solver.add_extensional_predicate_from_tuples(
                psymb,
                relation.value,
            )
        for builtin_symb in self.program_ir.builtins():
            solver.symbol_table[builtin_symb] = self.program_ir.symbol_table[
                builtin_symb
            ]
        solver.walk(postprob_idb)
        return solver

    def _solve_postprobabilistic_deterministic_stratum(
        self, solution, postprob_idb, make_chase=None
    ):
        """
        Solve the post-probabilistic deterministic stratum.

        The rules of ``postprob_idb`` are run over ``solution`` by the
        chases ``make_chase(program, rules)`` returns, by default chases
        of the engine's chase class.
        """
        if make_chase is None:
            def make_chase(program, rules):
                return self.chase_class(program, rules=rules)
        solver = self._build_postprobabilistic_solver(solution, postprob_idb)
        chase = make_chase(solver, postprob_idb)
        solution = chase.build_chase_solution()
        return solution

    @staticmethod
    def _restrict_to_query_solution(
        head_symbols, pred_symb, pred_args, solution
    ):
        """
        Based on a solution instance and a query predicate symbol and args,
        retrieve the relation whose columns correspond to symbols in the head
        of the query.
        """
        # return an empty set when empty solution (reported in GH481)
        if pred_symb not in solution:
            return ir.Constant[AbstractSet](
                NamedRelationalAlgebraFrozenSet(
                    _symbol_names(head_symbols), []
                )
            )
        query_solution = solution[pred_symb].value.unwrap()
        query_row_type = solution[pred_symb].value.row_type
        constant_selection = {
            i: c.value for i, c in enumerate(pred_args)
            if isinstance(c, ir.Constant)
        }
        if constant_selection:
            query_solution = query_solution.selection(constant_selection)
        if len(pred_args) == len(head_symbols):
            cols = list(
                arg.name if isinstance(arg, ir.Symbol) else symb.name
                for arg, symb in zip(pred_args, head_symbols)
            )
        else:
            cols = list(
                arg.name if isinstance(arg, ir.Symbol)
                else ir.Symbol.fresh().name
                for arg in pred_args
            )
        query_solution = NamedRelationalAlgebraFrozenSet(cols, query_solution)
        query_solution = query_solution.projection(
            *(symb.name for symb in head_symbols)
        )
        type_args = get_args(query_row_type)
        proj_row_type = tuple(
            type_args[cols.index(symb.name)] for symb in head_symbols
        )
        query_solution.row_type = replace_type_variable_fix_python36_37(
            query_row_type, proj_row_type
        )
        return ir.Constant[AbstractSet](query_solution)

    @instance_lru_cache(
        key_fn=lambda det_idb: frozenset(str(r) for r in det_idb.formulas),
        maxsize=32,
    )
    def _rewrite_det_idb(self, det_idb):
        """Xrewrite det_idb against ontology constraints, LRU-cached per instance.

        Only the Xrewrite step is cached. The connector_rules scan that follows
        in _solve_deterministic_stratum accesses live program_ir and is NOT
        cached, so EDB added after load_ontology() remains visible.
        Cache is invalidated by load_ontology() on each call.
        """
        return self._rewrite_program_with_ontology(det_idb)

    def _rewrite_program_with_ontology(self, deterministic_program):
        orw = OntologyRewriter(
            deterministic_program,
            self.program_ir.get_constraints()
        )
        rewrite = orw.Xrewrite()

        return Union(rewrite)

    def add_probabilistic_facts_from_tuples(
        self,
        iterable: Iterable[Tuple[Any, ...]],
        type_: Type = ir.Unknown,
        name: Optional[str] = None,
    ) -> fe.Symbol:
        """
        Add probabilistic facts from tuples whose first element
        contains the probability label attached to that tuple.
        In the tuple (p, a, b, ...), p is the float probability
        of tuple (a, b, ...) to be True in any possible world.

        Note that each tuple from the iterable is independent
        from the others, meaning that multiple tuples can be True
        in the same possible world, contrary to a probabilistic choice.
        See example for details.

        Warnings
        --------
        Typing for the iterable is improper, true -but yet unsupported
        in Python typing- typing should be Iterable[Tuple[float, Any, ...]]
        See examples

        Parameters
        ----------
        iterable : Iterable[Tuple[Any, ...]]
            the first float number represents the probability
            of the tuple constituted of the remaining elements
        type_ : Type, optional
            type for resulting AbstractSet if None will be inferred
            from the data, by default ir.Unknown
        name : Optional[str], optional
            name for the resulting fe.Symbol, if None
            will be fresh, by default None

        Returns
        -------
        fe.Symbol
            see description

        Examples
        --------
        >>> nl = NeurolangPDL()
        >>> p = [(0.8, 'a', 'b'), (0.7, 'b', 'c')]
        >>> nl.add_probabilistic_facts_from_tuples(p, name="P")
        P: typing.AbstractSet[typing.Tuple[float, str]] = \
            [(0.8, 'a', 'b'), (0.7, 'b', 'c')]

        Adds the probabilistic facts:
            P((a, b)) : 0.8  <-  T
            P((b, c)) : 0.7  <-  T
        """
        return self._add_probabilistic_tuples(
            iterable,
            type_,
            name,
            self.program_ir.add_probabilistic_facts_from_tuples,
        )

    def add_probabilistic_choice_from_tuples(
        self,
        iterable: Iterable[Tuple[Any, ...]],
        type_: Type = ir.Unknown,
        name: Optional[str] = None,
    ) -> fe.Symbol:
        """
        Add probabilistic choice from tuples whose first element
        contains the probability label attached to that tuple.
        In the tuple (p, a, b, ...), p is the float probability
        of tuple (a, b, ...) to be True and all remaining tuples to
        be False in any possible world.

        Note that, contrary to a list of probabilistic facts, this represents
        a choice among possible values for the predicate, meaning that tuples
        in the set are mutually exclusive.
        See example for details.

        Warnings
        --------
        Typing for the iterable is improper, true -but yet unsupported-
        typing should be Iterable[Tuple[float, Any, ...]]
        See examples

        Parameters
        ----------
        iterable : Iterable[Tuple[Any, ...]]
            the first float number represents the probability
            of the tuple constituted of the remaining elements
            Note that the float probabilities must sum to 1.
        type_ : Type, optional
            type for resulting AbstractSet if None will be inferred
            from the data, by default ir.Unknown
        name : Optional[str], optional
            name for the resulting fe.Symbol, if None
            will be fresh, by default None

        Returns
        -------
        fe.Symbol
            see description

        Raises
        ------
        DistributionDoesNotSumToOneError
            if float probabilities do not sum to 1.

        Examples
        --------
        >>> nl = NeurolangPDL()
        >>> p = [(0.8, 'a', 'b'), (0.2, 'b', 'c')]
        >>> nl.add_probabilistic_choice_from_tuples(p, name="P")
        P: typing.AbstractSet[typing.Tuple[float, str, str]] = \
            [(0.8, 'a', 'b'), (0.2, 'b', 'c')]

        Adds the probabilistic choice:
            P((a, b)) : 0.8  v  P((b, c)) : 0.2  <-  T
        """
        return self._add_probabilistic_tuples(
            iterable,
            type_,
            name,
            self.program_ir.add_probabilistic_choice_from_tuples,
        )

    def _add_probabilistic_tuples(
        self, iterable, type_, name, solver_add_method
    ):
        if isinstance(iterable, pd.DataFrame):
            iterable = iterable.rename(
                columns={n: i for i, n in enumerate(iterable.columns)}
            )
        if name is None:
            name = str(uuid1())
        if isinstance(type_, tuple):
            type_ = Tuple[type_]
        symbol = ir.Symbol[AbstractSet[type_]](name)
        solver_add_method(symbol, iterable)
        return fe.Symbol(self, name)

    def add_resampling(
        self,
        name: str,
        choice: str,
        n: int = 1000,
        scheme: str = "bayesian",
        seed: int = 0,
    ) -> fe.Symbol:
        """
        Add *n* replicates of the probabilistic choice *choice*, resampling
        its outcomes, for confidence intervals on the probabilities over it
        and on the measures derived from them.

        The resampling is the relation ``name(b, c..., w)``: in replicate
        ``b`` the outcome ``c`` has ``w`` times its probability in the
        choice.  ``RESAMPLE[name](pred(x)) = b`` in a rule body is then
        ``pred(x)`` computed on replicate ``b``; every query over the same
        resampling sees the same replicates.

        Parameters
        ----------
        name : str
            name of the resampling relation.
        choice : str
            name of a probabilistic choice of the program.
        n : int, optional
            number of replicates, by default 1000.
        scheme : str, optional
            ``"bayesian"`` (Dirichlet weights, the Bayesian bootstrap),
            ``"multinomial"`` (the classical bootstrap) or ``"poisson"``,
            by default ``"bayesian"``.
        seed : int, optional
            seed of the random draws, by default 0.

        Returns
        -------
        fe.Symbol
            the resampling relation.

        Examples
        --------
        >>> nl = NeurolangPDL()
        >>> nl.add_tuple_set([('a', 'x'), ('b', 'x')], name="R")
        >>> nl.add_uniform_probabilistic_choice_over_set(
        ...     [('a',), ('b',)], name="C"
        ... )
        >>> nl.add_resampling("boot", "C", n=200)
        >>> nl.execute_datalog_program('''
        ... p(y, PROB) :- R(c, y) & C(c)
        ... p_rep(b, y, p) :- RESAMPLE[boot](p(y, p)) = b
        ... ans(y, percentile(p, 2.5), percentile(p, 97.5)) :- p_rep(b, y, p)
        ... ''')
        """
        choice_symbol = ir.Symbol(choice)
        choices = self.program_ir.probabilistic_choices()
        if choice_symbol not in choices:
            raise ValueError(f"{choice} is not a probabilistic choice")
        rows = choices[choice_symbol].value.as_pandas_dataframe()
        relation = resampling_relation(rows, n, scheme, seed)
        symbol = self.add_tuple_set(relation, name=name)
        self.resamplings[name] = choice_symbol
        return symbol

    def _expand_resamplings(self, rule):
        """*rule* with its ``RESAMPLE`` atoms rewritten over the replicates,
        adding the rewritten rules they depend on to the program."""
        programs = {}

        def resampled_predicate(name, functor):
            if name.name not in self.resamplings:
                raise ValueError(f"{name.name} is not a resampling")
            if name not in programs:
                choice = self.resamplings[name.name]
                arity = len(
                    self.program_ir.probabilistic_choices()[choice]
                    .value.columns
                ) - 1
                programs[name] = ResampledProgram(
                    self.program_ir.intensional_database(),
                    name, choice, name, arity,
                )
            return programs[name].predicate(functor)

        rule = rewrite_resample_markers(rule, resampled_predicate)
        for program in programs.values():
            new_rules = [
                r for r in program.rules
                if r.consequent.functor
                not in self.program_ir.intensional_database()
            ]
            self.program_ir.walk(Union(new_rules))
        return rule

    def add_uniform_probabilistic_choice_over_set(
        self,
        iterable: Iterable[Tuple[Any, ...]],
        type_: Type = ir.Unknown,
        name: Optional[str] = None,
    ) -> fe.Symbol:
        """
        Add uniform probabilistic choice over values in the iterable.

        Every tuple in the iterable will be assigned the same probability to
        be True, with all remaining tuples False, in any possible world.

        Note that, contrary to a list of probabilistic facts, this represents
        a choice among possible values for the predicate, meaning that tuples
        in the set are mutually exclusive.

        See example for details.

        Parameters
        ----------
        iterable : Iterable[Tuple[Any, ...]]
            typically List[Tuple[Any, ...]], other types of Iterable
            will be cast as lists
        type_ : Type, optional
            type for resulting AbstractSet if None will be inferred
            from the data, by default ir.Unknown
        name : Optional[str], optional
            name for the resulting fe.Symbol, if None
            will be fresh, by default None

        Returns
        -------
        fe.Symbol
            see description

        Examples
        --------
        >>> nl = NeurolangPDL()
        >>> p = [('a',), ('b',)]
        >>> nl.add_uniform_probabilistic_choice_over_set(p, name="P")
        P: typing.AbstractSet[typing.Tuple[float, str]] = \
            [(0.5, 'a'), (0.5, 'b')]

        Adds the probabilistic choice:
            P(a) : 0.5  v  P(b) : 0.5  <-  T
        """
        if name is None:
            name = str(uuid1())
        if isinstance(type_, tuple):
            type_ = Tuple[type_]
        if isinstance(iterable, pd.DataFrame):
            iterable = iterable.rename(
                columns={n: i for i, n in enumerate(iterable.columns)}
            )
            arity = len(iterable.columns)
        else:
            arity = len(next(iter(iterable)))
        symbol = ir.Symbol[AbstractSet[type_]](name)
        columns = tuple(ir.Symbol.fresh().name for _ in range(arity))
        ra_set = NamedRelationalAlgebraFrozenSet(columns, iterable)
        prob_col = ir.Symbol.fresh().name
        probability = 1 / len(ra_set)
        projections = collections.OrderedDict()
        projections[prob_col] = probability
        for col in columns:
            projections[col] = RelationalAlgebraColumnStr(col)
        ra_set = ra_set.extended_projection(projections)
        self.program_ir.add_probabilistic_choice_from_tuples(
            symbol,
            ra_set.projection_to_unnamed(*projections.keys())
        )
        return fe.Symbol(self, name)


def _has_probability(head) -> bool:
    """Return whether the query *head*, a tuple, has a ``PROB(…)``."""
    return isinstance(head, tuple) and any(
        getattr(getattr(h, "expression", None), "functor", None) == PROB
        for h in head
    )


def _solved_as_query_rule(head, symbol_entry) -> bool:
    """
    Return whether a query is solved as a rule of its own.

    It is when its predicate is not defined by rules (*symbol_entry* is
    not a union of rules) or when its *head* computes a term other than
    a probability, such as the aggregate ``count(s)``.
    """
    if not hasattr(symbol_entry, "formulas"):
        return True
    return isinstance(head, tuple) and any(
        isinstance(h.expression, ir.FunctionApplication)
        and h.expression.functor != PROB
        for h in head
    )
