from typing import AbstractSet, Iterable, Tuple
from uuid import uuid1

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, plotting
from nilearn.datasets import utils

from neurolang.datalog import DatalogProgram
from neurolang.datalog.aggregation import Chase, DatalogWithAggregationMixin
from neurolang.datalog.expression_processing import (
    extract_logic_predicates,
    reachable_code,
)
from neurolang.exceptions import NeuroLangFrontendException
from neurolang.expression_walker import ExpressionBasicEvaluator
from neurolang.expressions import Constant, Symbol, Unknown
from neurolang.frontend import QueryBuilderDatalog, RegionFrontendDatalogSolver
from neurolang.frontend.neurosynth_utils import NeuroSynthHandler
from neurolang.frontend.query_resolution_expressions import (
    Symbol as FrontEndSymbol,
)
from neurolang.logic import Implication, Union
from neurolang.probabilistic.cplogic.problog_solver import (
    solve_succ_all as problog_solve_succ_all,
)
from neurolang.probabilistic.cplogic.program import (
    CPLogicMixin,
    CPLogicProgram,
)
from neurolang.region_solver import RegionSolver
from neurolang.relational_algebra import (
    ConcatenateConstantColumn,
    NameColumns,
    Projection,
    RelationalAlgebraSet,
    RelationalAlgebraSolver,
    str2columnstr_constant,
)


class RegionFrontendCPLogicSolver(
    RegionSolver, CPLogicMixin, DatalogProgram, ExpressionBasicEvaluator,
):
    pass


class ProbabilisticFrontend(QueryBuilderDatalog):
    def __init__(self, probabilistic_solver=problog_solve_succ_all):
        super().__init__(
            RegionFrontendCPLogicSolver(), chase_class=Chase,
        )
        self.probabilistic_solver = probabilistic_solver

    def solve_all(self):
        (
            deterministic_idb,
            probabilistic_idb,
        ) = self._separate_deterministic_probabilistic_code()
        deterministic_solution = self.chase_class(
            self.solver, rules=deterministic_idb
        ).build_chase_solution()
        if probabilistic_idb.formulas:
            cpl = self._make_probabilistic_program_from_deterministic_solution(
                deterministic_solution, probabilistic_idb,
            )
            return self.probabilistic_solver(cpl)
        return deterministic_solution

    def add_uniform_probabilistic_choice_over_set(
        self, iterable, type_=Unknown, name=None
    ):
        if name is None:
            name = str(uuid1())
        if isinstance(type_, tuple):
            type_ = Tuple[type_]
        symbol = Symbol[AbstractSet[type_]](name)
        ra_set = Constant[AbstractSet[type_]](
            RelationalAlgebraSet(iterable),
            auto_infer_type=False,
            verify_type=False,
        )
        columns = tuple(
            str2columnstr_constant(Symbol.fresh().name)
            for _ in range(ra_set.value.arity)
        )
        ra_set = NameColumns(ra_set, columns)
        prob_col = str2columnstr_constant(Symbol.fresh().name)
        probability = Constant[float](
            1 / len(iterable), auto_infer_type=False, verify_type=False
        )
        ra_set = ConcatenateConstantColumn(ra_set, prob_col, probability)
        # ensure probability column is first
        # TODO: this does not respect column-order invariance of RA relations
        ra_set = Projection(ra_set, (prob_col,) + columns)
        solver = RelationalAlgebraSolver()
        ra_set = solver.walk(ra_set)
        self.solver.add_probabilistic_choice_from_tuples(symbol, ra_set.value)
        return FrontEndSymbol(self, name)

    def _make_probabilistic_program_from_deterministic_solution(
        self, deterministic_solution, probabilistic_idb,
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

    def _separate_deterministic_probabilistic_code(
        self, query_pred=None, det_symbols=None, prob_symbols=None
    ):
        if det_symbols is None:
            det_symbols = set()
        if prob_symbols is None:
            prob_symbols = set()
        if query_pred is None:
            query_reachable_code = self._union_of_idb()
        else:
            query_reachable_code = reachable_code(query_pred, self.solver)
        deterministic_symbols = (
            set(self.solver.extensional_database().keys())
            | set(det_symbols)
            | set(self.solver.builtins().keys())
        )
        deterministic_program = list()
        probabilistic_symbols = (
            self.solver.pfact_pred_symbs
            | self.solver.pchoice_pred_symbs
            | set(prob_symbols)
        )
        probabilistic_program = list()
        unclassified_code = list(query_reachable_code.formulas)
        unclassified = 0
        initial_unclassified_length = len(unclassified_code) + 1
        while (
            len(unclassified_code) > 0
            and unclassified <= initial_unclassified_length
        ):
            pred = unclassified_code.pop(0)
            preds_antecedent = set(
                p.functor
                for p in extract_logic_predicates(pred.antecedent)
                if p.functor != pred.consequent.functor
            )
            if not probabilistic_symbols.isdisjoint(preds_antecedent):
                probabilistic_symbols.add(pred.consequent.functor)
                probabilistic_program.append(pred)
                unclassified = 0
                initial_unclassified_length = len(unclassified_code) + 1
            elif deterministic_symbols.issuperset(preds_antecedent):
                deterministic_symbols.add(pred.consequent.functor)
                deterministic_program.append(pred)
                unclassified = 0
                initial_unclassified_length = len(unclassified_code) + 1
            else:
                unclassified_code.append(pred)
                unclassified += 1
        if not probabilistic_symbols.isdisjoint(deterministic_symbols):
            raise NeuroLangFrontendException(
                "An atom was defined as both deterministic and probabilistic"
            )
        if len(unclassified_code) > 0:
            raise NeuroLangFrontendException("There are unclassified atoms")
        return Union(deterministic_program), Union(probabilistic_program)

    def _union_of_idb(self):
        formulas = tuple()
        for union in self.solver.intensional_database().values():
            formulas += union.formulas
        return Union(formulas)


nl = ProbabilisticFrontend()
ns_study_id = nl.load_neurosynth_study_ids(name="ns_study_id")
ns_term_in_study = nl.load_neurosynth_term_study_associations(
    name="ns_term_in_study"
)
ns_activation = nl.load_neurosynth_reported_activations(name="ns_activation")
selected_study = nl.add_uniform_probabilistic_choice_over_set(
    ns_study_id.expression.value, name="selected_study"
)
with nl.scope as e:
    e.term_association[e.term] = (
        ns_term_in_study[e.study_id, e.term] & selected_study[e.study_id]
    )
    e.activation[e.voxel_id] = (
        ns_activation[e.study_id, e.voxel_id] & selected_study[e.study_id]
    )
    res = nl.solve_all()
