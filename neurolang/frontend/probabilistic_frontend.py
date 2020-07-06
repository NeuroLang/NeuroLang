import collections
from typing import AbstractSet, Tuple
from uuid import uuid1

from ..datalog.aggregation import Chase
from ..datalog.constraints_representation import DatalogConstraintsProgram
from ..datalog.ontologies_parser import OntologyParser
from ..datalog.ontologies_rewriter import OntologyRewriter
from ..expression_walker import ExpressionBasicEvaluator
from ..expressions import Symbol, Unknown
from ..logic import Union
from ..probabilistic.cplogic.problog_solver import (
    solve_succ_all as problog_solve_succ_all,
)
from ..probabilistic.cplogic.program import CPLogicMixin, CPLogicProgram
from ..probabilistic.expression_processing import (
    separate_deterministic_probabilistic_code,
)
from ..probabilistic.weighted_model_counting import solve_succ_query
from ..region_solver import RegionSolver
from ..relational_algebra import (
    NamedRelationalAlgebraFrozenSet,
    RelationalAlgebraStringExpression,
)
from ..relational_algebra_provenance import ProvenanceAlgebraSet
from . import QueryBuilderDatalog
from .query_resolution_expressions import Symbol as FrontEndSymbol


class RegionFrontendCPLogicSolver(
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

    def solve_all(self):
        (
            deterministic_idb,
            probabilistic_idb,
        ) = separate_deterministic_probabilistic_code(self.solver)

        if self.ontology_loaded:
            eB = self._rewrite_database_with_ontology(deterministic_idb)
            deterministic_idb = Union(deterministic_idb.formulas + eB.formulas)

        solution = self.chase_class(
            self.solver, rules=deterministic_idb
        ).build_chase_solution()
        if probabilistic_idb.formulas:
            cpl = self._make_probabilistic_program_from_deterministic_solution(
                solution, probabilistic_idb
            )
            solution = self.probabilistic_solver(cpl)
        solution_sets = dict()
        for pred_symb, relation in solution.items():
            if isinstance(relation, ProvenanceAlgebraSet):
                proj_cols = (relation.provenance_column.value,) + tuple(
                    c.value for c in relation.non_provenance_columns
                )
                ra_set = relation.value.projection(*proj_cols)
            else:
                ra_set = relation.value
            solution_sets[pred_symb.name] = ra_set
        return solution_sets

    def solve_query(self, query_pred):
        (
            deterministic_idb,
            probabilistic_idb,
        ) = separate_deterministic_probabilistic_code(self.solver)

        if self.ontology_loaded:
            eB = self._rewrite_database_with_ontology(deterministic_idb)
            self.solver.walk(eB)

        deterministic_solution = self.chase_class(
            self.solver
        ).build_chase_solution()
        if probabilistic_idb.formulas:
            cpl = self._make_probabilistic_program_from_deterministic_solution(
                deterministic_solution, probabilistic_idb
            )
            return solve_succ_query(query_pred.expression, cpl)
        return deterministic_solution

    def _rewrite_database_with_ontology(self, deterministic_program):
        orw = OntologyRewriter(
            deterministic_program, self.solver.constraints()
        )
        rewrite = orw.Xrewrite()

        eB = ()
        for imp in rewrite:
            eB += (imp[0],)

        return Union(eB)

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

    def add_probabilistic_facts_from_tuples(self, iterable, name=None):
        if name is None:
            name = str(uuid1())
        self.solver.add_probabilistic_facts_from_tuples(Symbol(name), iterable)

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
