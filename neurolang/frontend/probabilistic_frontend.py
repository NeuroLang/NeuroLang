import collections
from typing import (
    AbstractSet,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union as typingUnion,
)
from uuid import uuid1

from ..datalog.aggregation import (
    Chase,
    DatalogWithAggregationMixin,
    TranslateToLogicWithAggregation,
)
from ..datalog.constraints_representation import DatalogConstraintsProgram
from ..datalog.ontologies_parser import OntologyParser
from ..datalog.ontologies_rewriter import OntologyRewriter
from ..exceptions import UnsupportedQueryError
from ..expression_walker import ExpressionBasicEvaluator
from ..expressions import Constant, Symbol, Unknown
from ..logic import Union
from ..probabilistic.cplogic.program import (
    CPLogicMixin,
    TranslateProbabilisticQueryMixin,
)
from ..probabilistic.dichotomy_theorem_based_solver import (
    solve_succ_query as lifted_solve_succ_query,
)
from ..probabilistic.expression_processing import (
    is_probabilistic_predicate_symbol,
    is_within_language_succ_query,
)
from ..probabilistic.query_resolution import compute_probabilistic_solution
from ..probabilistic.stratification import stratify_program
from ..region_solver import RegionSolver
from ..relational_algebra import (
    NamedRelationalAlgebraFrozenSet,
    RelationalAlgebraStringExpression,
)
from . import QueryBuilderDatalog
from .query_resolution_expressions import Expression as FrontEndExpression
from .query_resolution_expressions import Symbol as FrontEndSymbol


class RegionFrontendCPLogicSolver(
    TranslateProbabilisticQueryMixin,
    TranslateToLogicWithAggregation,
    RegionSolver,
    CPLogicMixin,
    DatalogWithAggregationMixin,
    DatalogConstraintsProgram,
    ExpressionBasicEvaluator,
):
    pass


class ProbabilisticFrontend(QueryBuilderDatalog):
    def __init__(
        self, chase_class=Chase, probabilistic_solver=lifted_solve_succ_query
    ):
        super().__init__(
            RegionFrontendCPLogicSolver(), chase_class=chase_class
        )
        self.probabilistic_solver = probabilistic_solver
        self.ontology_loaded = False

    def load_ontology(
        self,
        paths: typingUnion[str, List[str]],
        load_format: typingUnion[str, List[str]] = "xml",
    ) -> None:
        """Loads and parses ontology stored at the specified paths, and
        store them into attributes

        Parameters
        ----------
        paths : typingUnion[str, List[str]]
            where the ontology files are stored
        load_format : typingUnion[str, List[str]], optional
            storage format, by default "xml"
        """
        onto = OntologyParser(paths, load_format)
        d_pred, u_constraints = onto.parse_ontology()
        self.solver.walk(u_constraints)
        self.solver.add_extensional_predicate_from_tuples(
            onto.get_triples_symbol(), d_pred[onto.get_triples_symbol()]
        )
        self.solver.add_extensional_predicate_from_tuples(
            onto.get_pointers_symbol(), d_pred[onto.get_pointers_symbol()]
        )

        self.ontology_loaded = True

    @property
    def current_program(self) -> List[FrontEndExpression]:
        """Returns the list of expressions that have currently been declared in the
        program, or through the program's constraints

        Returns
        -------
        List[FrontEndExpression]
            see description

        Example
        -------
        >>> nl = ProbabilisticFrontend()
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
        for constraint in self.solver.constraints().formulas:
            cp.append(self.frontend_translator.walk(constraint))
        cp += super().current_program
        return cp

    def execute_query(
        self,
        head: typingUnion[
            Symbol[Tuple[FrontEndExpression, ...]],
            Tuple[FrontEndExpression, ...],
        ],
        predicate: FrontEndExpression,
    ) -> Tuple[AbstractSet, Optional[Symbol]]:
        """Performs an inferential query: will return as first output
        an abstract set with as many elements as solutions
        of the predicate query,and columns corresponding to
        the expressions in the head.
        Typically, probabilities are encapsulated into predicates.
        If head expressions are arguments of a functor, the latter will
        be returned as the second output, defaulted as None

        Parameters
        ----------
        head : typingUnion[
            Symbol[Tuple[FrontEndExpression, ...]],
            Tuple[FrontEndExpression, ...],
        ]
            see description
        predicate : Expression
            see description

        Returns
        -------
        Tuple[AbstractSet, Optional[Symbol]]
            see description

        Examples
        --------
        Note: example ran with pandas backend
        >>> nl = ProbabilisticFrontend(...)
        >>> P = nl.add_uniform_probabilistic_choice_over_set(
        ...     [("a",), ("b",), ("c",)], name="P"
        ... )
        >>> Q = nl.add_uniform_probabilistic_choice_over_set(
        ...     [("a",), ("d",), ("c",)], name="Q"
        ... )
        >>> with nl.scope as e:
        ...     e.Z[e.x, e.PROB[e.x]] = P[e.x] & Q[e.x]
        ...     s1 = nl.execute_query(tuple(), e.Z[e.x, e.p])
        ...     s2 = nl.execute_query((e.x, ), e.Z[e.x, e.p])
        ...     s3 = nl.execute_query(e.Z[e.x, e.p], e.Z[e.x, e.p])
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
        if is_probabilistic_predicate_symbol(query_pred_symb, self.solver):
            raise UnsupportedQueryError(
                "Queries on probabilistic predicates are not supported"
            )
        query = self.solver.symbol_table[query_pred_symb].formulas[0]
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
        return solution, functor_orig

    def solve_all(self) -> Dict:
        """
        Returns a dictionary of "predicate_name": "Content"
        for all elements in the solution of the datalog program.
        Typically, probabilities are encapsulated into predicates.

        Returns
        -------
        Dict
            extensional and intentional facts that have been derived
            through the current program, optionally with probabilities

        Example
        -------
        Note: example ran with pandas backend
        >>> nl = ProbabilisticFrontend()
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
        solution = self._solve()
        solution_sets = dict()
        for pred_symb, relation in solution.items():
            solution_sets[pred_symb.name] = NamedRelationalAlgebraFrozenSet(
                self.predicate_parameter_names(pred_symb.name),
                relation.value.unwrap(),
            )
            solution_sets[pred_symb.name].row_type = relation.value.row_type
        return solution_sets

    def _solve(self, query=None):
        idbs = stratify_program(query, self.solver)
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
        if self.ontology_loaded:
            eB = self._rewrite_program_with_ontology(det_idb)
            det_idb = Union(det_idb.formulas + eB.formulas)
        chase = self.chase_class(self.solver, rules=det_idb)
        solution = chase.build_chase_solution()
        return solution

    def _solve_probabilistic_stratum(self, solution, prob_idb):
        pfact_edb = self.solver.probabilistic_facts()
        pchoice_edb = self.solver.probabilistic_choices()
        prob_solution = compute_probabilistic_solution(
            solution,
            pfact_edb,
            pchoice_edb,
            prob_idb,
            self.probabilistic_solver,
        )
        wlq_symbs = set(
            rule.consequent.functor
            for rule in prob_idb.formulas
            if is_within_language_succ_query(rule)
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
        for builtin_symb in self.solver.builtins():
            solver.symbol_table[builtin_symb] = self.solver.symbol_table[
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
            return Constant[AbstractSet](NamedRelationalAlgebraFrozenSet.dum())
        query_solution = solution[pred_symb].value.unwrap()
        cols = list(
            arg.name
            for arg in predicate.expression.args
            if isinstance(arg, Symbol)
        )
        query_solution = NamedRelationalAlgebraFrozenSet(cols, query_solution)
        query_solution = query_solution.projection(
            *(symb.name for symb in head_symbols)
        )
        return Constant[AbstractSet](query_solution)

    def _rewrite_program_with_ontology(self, deterministic_program):
        orw = OntologyRewriter(
            deterministic_program, self.solver.constraints()
        )
        rewrite = orw.Xrewrite()

        eB = ()
        for imp in rewrite:
            eB += (imp[0],)

        return Union(eB)

    def add_probabilistic_facts_from_tuples(
        self,
        iterable: Iterable[Tuple[float, Any]],
        type_: Type = Unknown,
        name: Optional[str] = None,
    ) -> FrontEndSymbol:
        """Add probabilistic facts from tuples whose first element
        contains the probability label attached to that tuple.
        Note that those facts are independant, contrary to
        a probabilistic choice.
        See example for details.

        Parameters
        ----------
        iterable : Iterable
            typically List[Tuple[float, Any]], other types of Iterable
            will be cast as lists.
        type_ : Type, optional
            type for resulting AbstractSet if None will be inferred
            from the data, by default Unknown
        name : Optional[str], optional
            name for the resulting FrontEndSYmbol, if None
            will be randomized, by default None

        Returns
        -------
        FrontEndSymbol
            see description

        Example
        -------
        >>> p = [(0.8, 'a'), (0.7, 'b')]
        >>> nl.add_probabilistic_facts_from_tuples(p, name="P")
        P: typing.AbstractSet[typing.Tuple[float, str]] = \
            [(0.8, 'a'), (0.7, 'b')]

        Adds the probabilistic facts:
            P(a) : 0.8  <-  T
            P(b) : 0.7  <-  T
        """
        return self._add_probabilistic_tuples(
            iterable,
            type_,
            name,
            self.solver.add_probabilistic_facts_from_tuples,
        )

    def add_probabilistic_choice_from_tuples(
        self,
        iterable: Iterable[Tuple[float, Any]],
        type_: Type = Unknown,
        name: Optional[str] = None,
    ) -> FrontEndSymbol:
        """Add probabilistic choice from tuples whose first element
        contains the probability label attached to that tuple.
        Contrary to a list of probabilistic facts, this represents a choice
        among possible values for the predicate.
        See example for details.

        Parameters
        ----------
        iterable : Iterable
            typically List[Tuple[float, Any]], other types of Iterable
            will be cast as lists
            Note that the float probabilities must sum to 1.
        type_ : Type, optional
            type for resulting AbstractSet if None will be inferred
            from the data, by default Unknown
        name : Optional[str], optional
            name for the resulting FrontEndSYmbol, if None
            will be randomized, by default None

        Returns
        -------
        FrontEndSymbol
            see description

        Raises
        ------
        DistributionDoesNotSumToOneError
            if float probabilities do not sum to 1.

        Example
        -------
        >>> p = [(0.8, 'a'), (0.2, 'b')]
        >>> nl.add_probabilistic_choice_from_tuples(p, name="P")
        P: typing.AbstractSet[typing.Tuple[float, str]] = \
            [(0.8, 'a'), (0.2, 'b')]

        Adds the probabilistic choice:
            P(a) : 0.8  v  P(b) : 0.2  <-  T
        """
        return self._add_probabilistic_tuples(
            iterable,
            type_,
            name,
            self.solver.add_probabilistic_choice_from_tuples,
        )

    def _add_probabilistic_tuples(
        self, iterable, type_, name, solver_add_method
    ):
        if name is None:
            name = str(uuid1())
        if isinstance(type_, tuple):
            type_ = Tuple[type_]
        symbol = Symbol[AbstractSet[type_]](name)
        solver_add_method(symbol, iterable)
        return FrontEndSymbol(self, name)

    def add_uniform_probabilistic_choice_over_set(
        self,
        iterable: Iterable[Tuple[float, Any]],
        type_: Type = Unknown,
        name: Optional[str] = None,
    ) -> FrontEndSymbol:
        """Add uniform probabilistic choice among values
        in the iterable.
        Contrary to a list of probabilistic facts, this represents a choice
        among possible values for the predicate.
        All probabilities will be equal.
        See example for details.

        Parameters
        ----------
        iterable : Iterable
            typically List[Any], other types of Iterable
            will be cast as lists
        type_ : Type, optional
            type for resulting AbstractSet if None will be inferred
            from the data, by default Unknown
        name : Optional[str], optional
            name for the resulting FrontEndSYmbol, if None
            will be randomized, by default None

        Returns
        -------
        FrontEndSymbol
            see description

        Example
        -------
        >>> p = ['a', 'b']
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
        symbol = Symbol[AbstractSet[type_]](name)
        arity = len(next(iter(iterable)))
        columns = tuple(Symbol.fresh().name for _ in range(arity))
        ra_set = NamedRelationalAlgebraFrozenSet(columns, iterable)
        prob_col = Symbol.fresh().name
        probability = 1 / len(iterable)
        projections = collections.OrderedDict()
        projections[prob_col] = probability
        for col in columns:
            projections[col] = RelationalAlgebraStringExpression(col)
        ra_set = ra_set.extended_projection(projections)
        self.solver.add_probabilistic_choice_from_tuples(symbol, ra_set)
        return FrontEndSymbol(self, name)
