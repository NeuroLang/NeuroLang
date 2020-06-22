from typing import AbstractSet

import problog

from ...expressions import Constant, FunctionApplication, Symbol
from ...logic import Implication
from ...relational_algebra import (
    NamedRelationalAlgebraFrozenSet,
    str2columnstr_constant,
)
from ...relational_algebra_provenance import ProvenanceAlgebraSet
from .gm_provenance_solver import build_always_true_provenance_relation
from .problog_solver import (
    cplogic_to_problog,
    nl_pred_to_pl_pred,
    pl_solution_to_nl_solution,
)


def solve_succ_all(cpl, solver_name="problog"):
    if solver_name == "problog":
        query_preds = (
            union.formulas[0].consequent
            for union in cpl.intensional_database().values()
        )
        pl = cplogic_to_problog(cpl)
        query_pl_term = problog.logic.Term("query")
        for nl_qpred in query_preds:
            pl += query_pl_term(nl_pred_to_pl_pred(nl_qpred))
        pl_solution = problog.core.ProbLog.convert(
            pl, problog.sdd_formula.SDD
        ).evaluate()
        nl_solution = pl_solution_to_nl_solution(pl_solution, query_preds)
        for pred_symb in cpl.extensional_database():
            nl_solution[pred_symb] = build_always_true_provenance_relation(
                Constant[AbstractSet](
                    NamedRelationalAlgebraFrozenSet(
                        iterable=cpl.symbol_table[pred_symb].value,
                        columns=tuple(
                            Symbol.fresh().name
                            for _ in range(
                                cpl.symbol_table[pred_symb].value.arity
                            )
                        ),
                    )
                )
            )
        for pred_symb in cpl.pfact_pred_symbs | cpl.pchoice_pred_symbs:
            nl_solution[pred_symb] = ProvenanceAlgebraSet(
                cpl.symbol_table[pred_symb].value,
                str2columnstr_constant(
                    cpl.symbol_table[pred_symb].value.columns[0],
                ),
            )
    else:
        raise NotImplementedError()
    return nl_solution
