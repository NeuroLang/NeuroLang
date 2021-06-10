r"""
Probabilistic Frontend
======================
Complements QueryBuilderDatalog class with probabilistic capabilities
1- add extensional probabilistic facts and choices
2- sove probabilistic queries
"""
import collections
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

from .. import expressions as ir
from ..datalog.aggregation import (
    BuiltinAggregationMixin,
    DatalogWithAggregationMixin,
    TranslateToLogicWithAggregation,
)
from ..datalog.chase import Chase
from ..datalog.constraints_representation import DatalogConstraintsProgram
from ..datalog.expression_processing import (
    EqualitySymbolLeftHandSideNormaliseMixin,
)
from ..datalog.negation import DatalogProgramNegationMixin
from ..datalog.ontologies_parser import OntologyParser
from ..datalog.ontologies_rewriter import OntologyRewriter
from ..exceptions import UnsupportedQueryError, UnsupportedSolverError
from ..expression_walker import ExpressionBasicEvaluator
from ..logic import Union
from ..probabilistic.cplogic.program import CPLogicMixin
from ..probabilistic import dichotomy_theorem_based_solver, dalvi_suciu_lift
from ..probabilistic.expression_processing import (
    is_probabilistic_predicate_symbol,
    is_within_language_prob_query,
)
from ..probabilistic.query_resolution import (
    QueryBasedProbFactToDetRule,
    compute_probabilistic_solution,
)
from ..probabilistic.stratification import stratify_program
from ..probabilistic.weighted_model_counting import (
    solve_marg_query as wmc_solve_marg_query,
)
from ..probabilistic.weighted_model_counting import (
    solve_succ_query as wmc_solve_succ_query,
)
from ..region_solver import RegionSolver
from ..relational_algebra import (
    NamedRelationalAlgebraFrozenSet,
    RelationalAlgebraColumnStr,
)
from . import query_resolution_expressions as fe
from .datalog.sugar import (
    TranslateProbabilisticQueryMixin,
    TranslateQueryBasedProbabilisticFactMixin,
)
from .datalog.sugar.spatial import TranslateEuclideanDistanceBoundMatrixMixin
from .datalog.syntax_preprocessing import ProbFol2DatalogMixin
from .query_resolution_datalog import QueryBuilderDatalog


class RegionFrontendCPLogicSolver(
    EqualitySymbolLeftHandSideNormaliseMixin,
    TranslateProbabilisticQueryMixin,
    TranslateToLogicWithAggregation,
    TranslateQueryBasedProbabilisticFactMixin,
    TranslateEuclideanDistanceBoundMatrixMixin,
    QueryBasedProbFactToDetRule,
    ProbFol2DatalogMixin,
    RegionSolver,
    CPLogicMixin,
    DatalogWithAggregationMixin,
    BuiltinAggregationMixin,
    DatalogProgramNegationMixin,
    DatalogConstraintsProgram,
    ExpressionBasicEvaluator,
):
    pass


class NeurolangPDL(QueryBuilderDatalog):
    """
    Complements QueryBuilderDatalog class with probabilistic capabilities
    1- add extensional probabilistic facts and choices
    2- sove probabilistic queries
    """

    def __init__(
        self,
        chase_class: Type[Chase] = Chase,
        probabilistic_solvers: Tuple[Callable] = (
            dalvi_suciu_lift.solve_succ_query,
            wmc_solve_succ_query,
        ),
        probabilistic_marg_solvers: Tuple[Callable] = (
            dalvi_suciu_lift.solve_marg_query,
            wmc_solve_marg_query,
        ),
        check_qbased_pfact_tuple_unicity=False,
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


        Returns
        -------
        NeurolangPDL
            see description
        """
        super().__init__(
            RegionFrontendCPLogicSolver(), chase_class=chase_class
        )
        if len(probabilistic_solvers) != len(probabilistic_marg_solvers):
            raise ValueError(
                "probabilistic solve and marg "
                "solvers should be the same length"
            )
        self.probabilistic_solvers = probabilistic_solvers
        self.probabilistic_marg_solvers = probabilistic_marg_solvers
        self.ontology_loaded = False
        self.check_qbased_pfact_tuple_unicity = (
            check_qbased_pfact_tuple_unicity
        )

    def load_ontology(
        self,
        paths: typing.Union[str, List[str]],
        load_format: typing.Union[str, List[str]] = "xml",
    ) -> None:
        """
        Loads and parses ontology stored at the specified paths, and
        store them into attributes

        Parameters
        ----------
        paths : typing.Union[str, List[str]]
            where the ontology files are stored
        load_format : typing.Union[str, List[str]], optional
            storage format, by default "xml"
        """
        onto = OntologyParser(paths, load_format)
        d_pred, u_constraints = onto.parse_ontology()
        self.program_ir.walk(u_constraints)
        self.program_ir.add_extensional_predicate_from_tuples(
            onto.get_triples_symbol(), d_pred[onto.get_triples_symbol()]
        )
        self.program_ir.add_extensional_predicate_from_tuples(
            onto.get_pointers_symbol(), d_pred[onto.get_pointers_symbol()]
        )

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

        Example
        -------
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
    ) -> Tuple[AbstractSet, Optional[ir.Symbol]]:
        """
        [Internal usage - documentation for developpers]

        Performs an inferential query: will return as first output
        an AbstractSet with as many elements as solutions
        of the predicate query. AbstractSet's columns correspond to
        the expressions in the head.
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
        query_pred_symb = predicate.expression.functor
        if is_probabilistic_predicate_symbol(query_pred_symb, self.program_ir):
            raise UnsupportedQueryError(
                "Queries on probabilistic predicates are not supported"
            )
        query = self.program_ir.symbol_table[query_pred_symb].formulas[0]
        solution = self._solve(query)
        if not isinstance(head, tuple):
            # assumes head is a predicate e.g. r(x, y)
            head_symbols = tuple(head.expression.args)
            functor_orig = head.expression.functor
        else:
            head_symbols = tuple(t.expression for t in head)
            functor_orig = None
        solution = self._restrict_to_query_solution(
            head_symbols, predicate, solution
        )
        if functor_orig is None:
            solution = solution.value
        return solution, functor_orig

    def solve_all(self) -> Dict[str, NamedRelationalAlgebraFrozenSet]:
        """
        Returns a dictionary of "predicate_name": "Content"
        for all elements in the solution of the Datalog program.
        Typically, probabilities are abstracted and processed similar
        to symbols, though of different nature (see examples)

        Returns
        -------
        Dict[str, NamedRelationalAlgebraFrozenSet]
            extensional and intentional facts that have been derived
            through the current program, optionally with probabilities

        Example
        -------
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
        >>> solution
        {
            'Z':
                PROB        x
            0   0.111111    a
            1   0.111111    c
        }
        """
        solution_ir = self._solve()
        solution = {}
        for k, v in solution_ir.items():
            solution[k.name] = NamedRelationalAlgebraFrozenSet(
                self.predicate_parameter_names(k.name), v.value.unwrap()
            )
            solution[k.name].row_type = v.value.row_type
        return solution

    def _solve(self, query=None):
        idbs = stratify_program(query, self.program_ir)
        det_idb = idbs.get("deterministic", Union(tuple()))
        prob_idb = idbs.get("probabilistic", Union(tuple()))
        postprob_idb = idbs.get("post_probabilistic", Union(tuple()))
        solution = self._solve_deterministic_stratum(det_idb)
        if prob_idb.formulas:
            solution = self._solve_probabilistic_stratum(solution, prob_idb)
        if postprob_idb.formulas:
            solution = self._solve_postprobabilistic_deterministic_stratum(
                solution, postprob_idb
            )
        return solution

    def _solve_deterministic_stratum(self, det_idb):
        if "__constraints__" in self.symbol_table:
            eB = self._rewrite_program_with_ontology(det_idb)
            det_idb = Union(det_idb.formulas + eB.formulas)
        chase = self.chase_class(self.program_ir, rules=det_idb)
        solution = chase.build_chase_solution()
        return solution

    def _solve_probabilistic_stratum(self, solution, prob_idb):
        pfact_edb = self.program_ir.probabilistic_facts()
        pchoice_edb = self.program_ir.probabilistic_choices()
        for i, (succ_solver, marg_solver) in enumerate(
            zip(self.probabilistic_solvers, self.probabilistic_marg_solvers)
        ):
            try:
                prob_solution = compute_probabilistic_solution(
                    solution,
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

    def _solve_postprobabilistic_deterministic_stratum(
        self, solution, postprob_idb
    ):
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
        chase = self.chase_class(solver, rules=postprob_idb)
        solution = chase.build_chase_solution()
        return solution

    @staticmethod
    def _restrict_to_query_solution(head_symbols, predicate, solution):
        """
        Based on a solution instance and a query predicate, retrieve the
        relation whose columns correspond to symbols in the head of the query.
        """
        pred_symb = predicate.expression.functor
        # return dum when empty solution (reported in GH481)
        if pred_symb not in solution:
            return ir.Constant[AbstractSet](
                NamedRelationalAlgebraFrozenSet.dum()
            )
        query_solution = solution[pred_symb].value.unwrap()
        cols = list(
            arg.name
            for arg in predicate.expression.args
            if isinstance(arg, ir.Symbol)
        )
        query_solution = NamedRelationalAlgebraFrozenSet(cols, query_solution)
        query_solution = query_solution.projection(
            *(symb.name for symb in head_symbols)
        )
        return ir.Constant[AbstractSet](query_solution)

    def _rewrite_program_with_ontology(self, deterministic_program):
        orw = OntologyRewriter(
            deterministic_program, self.program_ir.constraints()
        )
        rewrite = orw.Xrewrite()

        eB = ()
        for imp in rewrite:
            eB += (imp[0],)

        return Union(eB)

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

        Note that each tuple from the iterable is independant
        from the others, meaning that multiple tuples can be True
        in the same possible world, contrary to a probabilistic choice.
        See example for details.

        Warning
        -------
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

        Example
        -------
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

        Warning
        -------
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

        Example
        -------
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

    def add_uniform_probabilistic_choice_over_set(
        self,
        iterable: Iterable[Tuple[Any, ...]],
        type_: Type = ir.Unknown,
        name: Optional[str] = None,
    ) -> fe.Symbol:
        """
        Add uniform probabilistic choice over values in the iterable.

        Every tuple in the iterable will be assigned the same probability to
        be True, with all remaning tuples False, in any possible world.

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

        Example
        -------
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
        probability = 1 / len(iterable)
        projections = collections.OrderedDict()
        projections[prob_col] = probability
        for col in columns:
            projections[col] = RelationalAlgebraColumnStr(col)
        ra_set = ra_set.extended_projection(projections)
        self.program_ir.add_probabilistic_choice_from_tuples(symbol, ra_set)
        return fe.Symbol(self, name)
