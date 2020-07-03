import problog.core
import problog.logic
import problog.sdd_formula

from ....datalog.expressions import Fact
from ....expressions import Constant, Symbol
from ....logic import Implication
from ..problog_solver import cplogic_to_problog
from ..program import CPLogicProgram

P = Symbol("P")
a = Constant("a")


def test_convert_cpl_to_pl():
    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_facts_from_tuples(
        P, {(0.2, "a"), (1.0, "b")}
    )
    pl = cplogic_to_problog(cpl_program)
    query = problog.logic.Term("query")
    query_pred = problog.logic.Term("P", problog.logic.Var("v"))
    pl += query(query_pred)
    res = problog.core.ProbLog.convert(pl, problog.sdd_formula.SDD).evaluate()
    expected = {
        problog.logic.Term("P")(problog.logic.Constant("a")): 0.2,
        problog.logic.Term("P")(problog.logic.Constant("b")): 1.0,
    }
    assert res == expected


def test_zero_arity():
    cpl_program = CPLogicProgram()
    cpl_program.walk(Fact(P()))
    cplogic_to_problog(cpl_program)
    cpl_program = CPLogicProgram()
    cpl_program.walk(Fact(P(a)))
    cpl_program.walk(Implication(Symbol("yes")(), P(a)))
    pl = cplogic_to_problog(cpl_program)
    query = problog.logic.Term("query")
    query_pred = problog.logic.Term("yes")
    pl += query(query_pred)
    res = problog.core.ProbLog.convert(pl, problog.sdd_formula.SDD).evaluate()
    expected = {problog.logic.Term("yes"): 1.0}
    assert res == expected
