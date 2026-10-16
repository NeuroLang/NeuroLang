import problog.core
import problog.logic
import problog.sdd_formula
import pytest

from ....datalog.expressions import Fact
from ....expressions import Constant, Symbol
from ....logic import Conjunction, Implication, Union
from .. import gm_provenance_solver, problog_solver
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


def _build_program_with_unreachable_rules():
    Q, R, V, W, Y, Z = map(Symbol, "QRVWYZ")
    x = Symbol("x")
    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_facts_from_tuples(
        Q, {(0.9, "a"), (0.5, "b")}
    )
    cpl_program.add_probabilistic_facts_from_tuples(
        P, {(0.8, "a"), (0.2, "c")}
    )
    cpl_program.add_probabilistic_choice_from_tuples(
        R, {(0.3, "a"), (0.7, "b")}
    )
    cpl_program.add_extensional_predicate_from_tuples(V, {("a",), ("b",)})
    cpl_program.walk(Union((
        Implication(Z(x), Conjunction((Y(x), V(x)))),
        Implication(Y(x), Conjunction((Q(x), R(x)))),
        Implication(W(x), Conjunction((P(x), Q(x)))),
    )))
    return cpl_program


def _pl_clause_functors(pl):
    functors = set()
    for clause in pl:
        if isinstance(clause, problog.logic.AnnotatedDisjunction):
            functors |= {head.functor for head in clause.heads}
        elif isinstance(clause, problog.logic.Clause):
            functors.add(clause.head.functor)
        else:
            functors.add(clause.functor)
    return functors


def _prov_set_to_dict(prov_set):
    relation = prov_set.relation.value
    prob_col = prov_set.provenance_column.value
    columns = list(relation.columns)
    prob_idx = columns.index(prob_col)
    return {
        row[:prob_idx] + row[prob_idx + 1:]: row[prob_idx]
        for row in relation
    }


def test_convert_cpl_to_pl_restricted_to_query():
    cpl_program = _build_program_with_unreachable_rules()
    x = Symbol("x")
    assert _pl_clause_functors(cplogic_to_problog(cpl_program)) == set(
        "PQRVWYZ"
    )
    restricted = cplogic_to_problog(cpl_program, query_pred=Symbol("Z")(x))
    assert _pl_clause_functors(restricted) == set("QRVYZ")
    restricted = cplogic_to_problog(cpl_program, query_pred=P(x))
    assert _pl_clause_functors(restricted) == {"P"}


def test_restricted_succ_query_matches_full_program(monkeypatch):
    x = Symbol("x")
    queries = [Symbol(s)(x) for s in "ZYWPQRV"]
    expected = {}
    with monkeypatch.context() as m:
        m.setattr(
            problog_solver,
            "cplogic_to_problog",
            lambda cpl, query_pred=None: cplogic_to_problog(cpl),
        )
        for query in queries:
            result = problog_solver.solve_succ_query(
                query, _build_program_with_unreachable_rules()
            )
            expected[query] = _prov_set_to_dict(result)
    assert expected[Symbol("Z")(x)] == pytest.approx(
        {("a",): 0.27, ("b",): 0.35}
    )
    for query in queries:
        cpl_program = _build_program_with_unreachable_rules()
        result = _prov_set_to_dict(
            problog_solver.solve_succ_query(query, cpl_program)
        )
        assert result == pytest.approx(expected[query])
        result = _prov_set_to_dict(
            gm_provenance_solver.solve_succ_query(query, cpl_program)
        )
        assert result == pytest.approx(expected[query])
