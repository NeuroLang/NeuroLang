import collections
from typing import AbstractSet, Tuple
from uuid import uuid1

from ..datalog.aggregation import Chase, TranslateToLogicWithAggregation
from ..datalog.constraints_representation import DatalogConstraintsProgram
from ..datalog.ontologies_parser import OntologyParser
from ..datalog.ontologies_rewriter import OntologyRewriter
from ..expression_walker import ExpressionBasicEvaluator
from ..expressions import Constant, Symbol, Unknown
from ..logic import Union
from ..probabilistic.cplogic.program import (
    CPLogicMixin,
    CPLogicProgram,
    TranslateProbabilisticQueryMixin,
)
from ..probabilistic.dichotomy_theorem_based_solver import (
    solve_succ_query as lifted_solve_succ_query,
)
from ..probabilistic.expression_processing import (
    construct_within_language_succ_result,
    is_within_language_succ_query,
    within_language_succ_query_to_intensional_rule,
)
from ..probabilistic.stratification import stratify_program
from ..region_solver import RegionSolver
from ..relational_algebra import (
    NamedRelationalAlgebraFrozenSet,
    RelationalAlgebraStringExpression,
)
from . import QueryBuilderDatalog
from .query_resolution_expressions import Symbol as FrontEndSymbol


class RegionFrontendCPLogicSolver(
    TranslateProbabilisticQueryMixin,
    TranslateToLogicWithAggregation,
    RegionSolver,
    CPLogicMixin,
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

    def load_ontology(self, paths, load_format="xml"):
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

    def execute_query(self, head, predicate):
        query = self.solver.symbol_table[
            predicate.expression.functor
        ].formulas[0]
        solution = self._solve(query)
        return (
            self._restrict_to_query_solution(head, predicate, solution),
            None,
        )

    def solve_all(self):
        solution = self._solve()
        solution_sets = dict()
        for pred_symb, relation in solution.items():
            solution_sets[pred_symb.name] = relation.value
        return solution_sets

    def _solve(self, query=None):
        idbs = stratify_program(query, self.solver)
        det_idb = idbs.get("deterministic", Union(tuple()))
        prob_idb = idbs.get("probabilistic", Union(tuple()))
        ppq_det_idb = idbs.get("post_probabilistic", Union(tuple()))

        if self.ontology_loaded:
            eB = self._rewrite_program_with_ontology(det_idb)
            det_idb = Union(det_idb.formulas + eB.formulas)
        chase = self.chase_class(self.solver, rules=det_idb)
        solution = chase.build_chase_solution()
        if prob_idb.formulas:
            self._compute_probabilistic_solution(solution, prob_idb)
        if ppq_det_idb.formulas:
            solver = RegionFrontendCPLogicSolver()
            for psymb, relation in solution.items():
                solver.add_extensional_predicate_from_tuples(
                    psymb, relation.value,
                )
            solver.walk(ppq_det_idb)
            chase = self.chase_class(solver, rules=ppq_det_idb)
            solution = chase.build_chase_solution()
        return solution

    @staticmethod
    def _restrict_to_query_solution(head, predicate, solution):
        """
        Based on a solution instance and a query predicate, retrieve the
        relation whose columns correspond to symbols in `head`.

        """
        query_solution = solution[predicate.expression.functor].value.unwrap()
        cols = list(
            arg.name
            for arg in predicate.expression.args
            if isinstance(arg, Symbol)
        )
        query_solution = NamedRelationalAlgebraFrozenSet(cols, query_solution)
        query_solution = query_solution.projection(
            *(symb.expression.name for symb in head)
        )
        return Constant[AbstractSet](query_solution)

    def _compute_probabilistic_solution(self, solution, prob_idb):
        cpl = self._make_probabilistic_program_from_deterministic_solution(
            solution, prob_idb
        )
        for rule in prob_idb.formulas:
            if is_within_language_succ_query(rule):
                pred = within_language_succ_query_to_intensional_rule(
                    rule
                ).consequent
                provset = self.probabilistic_solver(pred, cpl)
                relation = construct_within_language_succ_result(provset, rule)
            else:
                pred = rule.consequent
                provset = self.probabilistic_solver(pred, cpl)
                relation = Constant[AbstractSet](
                    provset.value, auto_infer_type=False, verify_type=False,
                )
            solution[pred.functor] = Constant[AbstractSet](
                relation.value.to_unnamed()
            )
        return solution

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
        self, iterable, type_=Unknown, name=None
    ):
        return self._add_probabilistic_tuples(
            iterable,
            type_,
            name,
            self.solver.add_probabilistic_facts_from_tuples,
        )

    def add_probabilistic_choice_from_tuples(
        self, iterable, type_=Unknown, name=None
    ):
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
        self, iterable, type_=Unknown, name=None
    ):
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

    def _make_probabilistic_program_from_deterministic_solution(
        self, deterministic_solution, probabilistic_idb
    ):
        cpl = CPLogicProgram()
        for pred_symb, ra_set in deterministic_solution.items():
            cpl.add_extensional_predicate_from_tuples(
                pred_symb, ra_set.value.unwrap()
            )
        for pred_symb in self.solver.pfact_pred_symbs:
            cpl.add_probabilistic_facts_from_tuples(
                pred_symb, self.solver.symbol_table[pred_symb].value.unwrap()
            )
        for pred_symb in self.solver.pchoice_pred_symbs:
            cpl.add_probabilistic_choice_from_tuples(
                pred_symb, self.solver.symbol_table[pred_symb].value.unwrap()
            )
        cpl.walk(probabilistic_idb)
        return cpl
