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
from ..probabilistic.cplogic.problog_solver import (
    solve_succ_all as problog_solve_succ_all,
)
from ..probabilistic.cplogic.program import (
    CPLogicMixin,
    CPLogicProgram,
    TranslateProbabilisticQueryMixin,
)
from ..probabilistic.expression_processing import (
    separate_deterministic_probabilistic_code,
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
        self, chase_class=Chase, probabilistic_solver=problog_solve_succ_all
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
        pred_symb = predicate.expression.functor
        query = self.solver.symbol_table[pred_symb].formulas[0]
        det_idb, prob_idb, ppq_det_idb = stratify_program(query, self.solver)
        if self.ontology_loaded:
            eb = self._rewrite_program_with_ontology(det_idb)
            det_idb = Union(det_idb.formulas + eb.formulas)
        chase = self.chase_class(self.solver, rules=det_idb)
        det_solution = chase.build_chase_solution()
        cpl = self._make_probabilistic_program_from_deterministic_solution(
            det_solution, prob_idb
        )
        solution = self.probabilistic_solver(cpl)
        solver = RegionFrontendCPLogicSolver()
        for pred_symb, relation in solution.items():
            solver.add_extensional_predicate_from_tuples(
                pred_symb, relation.value
            )
        solver.walk(ppq_det_idb)
        chase = self.chase_class(solver, rules=ppq_det_idb)
        solution = chase.build_chase_solution()
        query_solution = solution[pred_symb].value.unwrap()
        cols = list(
            arg.name
            for arg in predicate.expression.args
            if isinstance(arg, Symbol)
        )
        query_solution = NamedRelationalAlgebraFrozenSet(cols, query_solution)
        query_solution = query_solution.projection(
            *(symb.expression.name for symb in head)
        )
        return Constant[AbstractSet](query_solution), None

    def solve_all(self):
        (
            deterministic_idb,
            probabilistic_idb,
        ) = separate_deterministic_probabilistic_code(self.solver)

        if self.ontology_loaded:
            eB = self._rewrite_program_with_ontology(deterministic_idb)
            deterministic_idb = Union(deterministic_idb.formulas + eB.formulas)

        solution = self.chase_class(
            self.solver, rules=deterministic_idb
        ).build_chase_solution()
        if (
            self.solver.pfact_pred_symbs
            or self.solver.pchoice_pred_symbs
            or probabilistic_idb.formulas
        ):
            cpl = self._make_probabilistic_program_from_deterministic_solution(
                solution, probabilistic_idb
            )
            solution = self.probabilistic_solver(cpl)
        solution_sets = dict()
        for pred_symb, relation in solution.items():
            solution_sets[pred_symb.name] = relation.value
        return solution_sets

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
        if name is None:
            name = str(uuid1())
        if isinstance(type_, tuple):
            type_ = Tuple[type_]
        symbol = Symbol[AbstractSet[type_]](name)
        self.solver.add_probabilistic_facts_from_tuples(symbol, iterable)
        return FrontEndSymbol(self, name)

    def add_probabilistic_choice_from_tuples(
        self, iterable, type_=Unknown, name=None
    ):
        if name is None:
            name = str(uuid1())
        if isinstance(type_, tuple):
            type_ = Tuple[type_]
        symbol = Symbol[AbstractSet[type_]](name)
        self.solver.add_probabilistic_choice_from_tuples(symbol, iterable)
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
