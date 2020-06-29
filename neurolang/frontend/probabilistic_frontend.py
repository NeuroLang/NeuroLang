from uuid import uuid1

import pandas as pd

from ..datalog.aggregation import Chase
from ..datalog.constraints_representation import DatalogConstraintsProgram
from ..datalog.ontologies_parser import OntologyParser
from ..datalog.ontologies_rewriter import OntologyRewriter
from ..expression_walker import ExpressionBasicEvaluator
from ..expressions import Symbol
from ..logic import Union
from ..probabilistic.cplogic import solve_succ_all
from ..probabilistic.cplogic.program import CPLogicMixin, CPLogicProgram
from ..probabilistic.expression_processing import (
    separate_deterministic_probabilistic_code,
)
from ..region_solver import RegionSolver
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
    def __init__(self, probabilistic_solver="problog"):
        super().__init__(RegionFrontendCPLogicSolver(), chase_class=Chase)
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
            self.solver.walk(eB)

        deterministic_solution = self.chase_class(
            self.solver
        ).build_chase_solution()
        if probabilistic_idb.formulas:
            cpl = self._make_probabilistic_program_from_deterministic_solution(
                deterministic_solution, probabilistic_idb
            )
            return solve_succ_all(cpl, solver_name=self.probabilistic_solver)
        return deterministic_solution

    def _rewrite_database_with_ontology(self, deterministic_program):
        orw = OntologyRewriter(
            deterministic_program, self.solver.constraints()
        )
        rewrite = orw.Xrewrite()

        eB2 = ()
        for imp in rewrite:
            eB2 += (imp[0],)

        return Union(eB2)

    def add_uniform_probabilistic_choice_over_set(self, iterable, name=None):
        if name is None:
            name = str(uuid1())
        probability = 1 / len(iterable)
        if isinstance(iterable, pd.DataFrame):
            columns = iterable.columns
            prob_col = Symbol.fresh().name
            new_columns = (prob_col,) + tuple(columns)
            iterable = iterable.copy()
            iterable[prob_col] = probability
            iterable = iterable[new_columns]
        else:
            iterable = [(probability,) + tuple(tupl) for tupl in iterable]
        self.solver.add_probabilistic_choice_from_tuples(
            Symbol(name), iterable
        )
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
