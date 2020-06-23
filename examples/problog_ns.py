from typing import Iterable
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
from neurolang.expressions import Symbol
from neurolang.frontend import QueryBuilderDatalog, RegionFrontendDatalogSolver
from neurolang.frontend.neurosynth_utils import NeuroSynthHandler
from neurolang.frontend.query_resolution_expressions import (
    Symbol as FrontEndSymbol,
)
from neurolang.logic import Implication, Union
from neurolang.probabilistic.cplogic import solve_succ_all
from neurolang.probabilistic.cplogic.program import (
    CPLogicMixin,
    CPLogicProgram,
)
from neurolang.region_solver import RegionSolver


class RegionFrontendCPLogicSolver(
    RegionSolver, CPLogicMixin, DatalogProgram, ExpressionBasicEvaluator,
):
    pass


class ProbabilisticFrontend(QueryBuilderDatalog):
    def __init__(self, probabilistic_solver="problog"):
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
            return solve_succ_all(cpl, solver_name=self.probabilistic_solver)
        return deterministic_solution

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
        self, deterministic_solution, probabilistic_idb,
    ):
        cpl = CPLogicProgram()
        for pred_symb, ra_set in deterministic_solution.items():
            cpl.add_extensional_predicate_from_tuples(pred_symb, ra_set.value)
        for pred_symb in self.solver.pfact_pred_symbs:
            cpl.add_probabilistic_facts_from_tuples(
                pred_symb, self.solver.symbol_table[pred_symb].value
            )
        for pred_symb in self.solver.pchoice_pred_symbs:
            cpl.add_probabilistic_choice_from_tuples(
                pred_symb, self.solver.symbol_table[pred_symb].value
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

# loading neurosynth data into the program
# nsh = NeuroSynthHandler()
# ns_term_in_study = nl.add_tuple_set(
# nsh.ns_load_term_study_associations(threshold=1e-3),
# name="ns_term_in_study",
# )
# ns_activation = nl.add_tuple_set(
# nsh.ns_load_reported_activations(), name="ns_activation"
# )
# selected_study = nl.add_uniform_probabilistic_choice_over_set(
# nsh.ns_load_all_study_ids(), name="selected_study"
# )
ns_term_in_study = nl.add_tuple_set(
    {
        ("insula", "59895"),
        ("brain", "34984"),
        ("auditory", "34984"),
        ("brain", "59895"),
    },
    name="ns_term_in_study",
)
ns_activation = nl.add_tuple_set(
    {("59895", 455), ("59895", 222), ("34984", 455),}, name="ns_activation"
)
selected_study = nl.add_uniform_probabilistic_choice_over_set(
    {("59895",), ("34984",)}, name="selected_study"
)
with nl.scope as e:
    e.term_association[e.term] = (
        ns_term_in_study[e.term, e.study_id] & selected_study[e.study_id]
    )
    e.activation[e.voxel_id] = (
        ns_activation[e.study_id, e.voxel_id] & selected_study[e.study_id]
    )
    res = nl.solve_all()
