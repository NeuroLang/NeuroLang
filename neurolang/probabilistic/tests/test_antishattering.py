from ...expressions import Constant, Symbol
from ...logic import Conjunction, Implication, Union
from .. import dalvi_suciu_lift
from ..cplogic import testing
from ..cplogic.program import CPLogicProgram

ans = Symbol("ans")
P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
x = Symbol("x")

a = Constant("a")
b = Constant("b")


def test_pchoice_with_constant():
    table = {
        (0.52, "a"),
        (0.48, "b"),
    }

    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_choice_from_tuples(P, table)

    query = Implication(ans(), Conjunction((P(a),)))
    cpl_program.walk(query)
    res = dalvi_suciu_lift.solve_succ_query(query, cpl_program)
    assert testing.eq_prov_relations(
        res, testing.make_prov_set({(0.52,)}, [res.provenance_column.value])
    )


def test_pchoice_with_constant_and_variable():
    table = {
        (0.52, "a"),
        (0.48, "b"),
    }

    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_choice_from_tuples(P, table)

    query = Implication(ans(), Conjunction((P(a), P(x))))
    cpl_program.walk(query)
    res = dalvi_suciu_lift.solve_succ_query(query, cpl_program)
    assert testing.eq_prov_relations(
        res, testing.make_prov_set({(0.52,)}, [res.provenance_column.value])
    )


def test_pchoice_empty_result():
    table = {
        (0.52, "a"),
        (0.48, "b"),
    }

    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_choice_from_tuples(P, table)

    query = Implication(ans(), Conjunction((P(a), P(b))))
    cpl_program.walk(query)
    res = dalvi_suciu_lift.solve_succ_query(query, cpl_program)
    assert testing.eq_prov_relations(
        res, testing.make_prov_set([], [res.provenance_column.value]),
    )


def test_pchoice_with_constant_and_projected_variable():
    table = {
        (0.52, "a"),
        (0.48, "b"),
    }

    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_choice_from_tuples(P, table)

    query = Implication(ans(x), Conjunction((P(a), P(x))))
    cpl_program.walk(query)
    res = dalvi_suciu_lift.solve_succ_query(query, cpl_program)
    assert testing.eq_prov_relations(
        res,
        testing.make_prov_set(
            {(0.52, "a")}, [res.provenance_column.value, "x"]
        ),
    )


def test_pchoice_with_constant_and_projected_variable_disjunction():
    table = {
        (0.52, "a"),
        (0.48, "b"),
    }

    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_choice_from_tuples(P, table)

    query = Implication(ans(), Q())
    program = Union((Implication(Q(), P(a)), Implication(Q(), P(b)), query))
    cpl_program.walk(program)
    res = dalvi_suciu_lift.solve_succ_query(query, cpl_program)
    assert testing.eq_prov_relations(
        res,
        testing.make_prov_set(
            [(0.48,), (0.52,)], [res.provenance_column.value]
        ),
    )


def test_pchoice_with_constant_and_projected_variable_disjunction_2():
    table = {
        (0.52, "a"),
        (0.48, "b"),
    }

    table2 = {(0.2, "a")}

    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_choice_from_tuples(P, table)
    cpl_program.add_probabilistic_facts_from_tuples(R, table2)

    query = Implication(ans(), Q())
    program = Union(
        (
            Implication(Q(), P(a)),
            Implication(Q(), P(b)),
            Implication(Q(), R(x)),
            query,
        )
    )
    cpl_program.walk(program)
    res = dalvi_suciu_lift.solve_succ_query(query, cpl_program)
    assert testing.eq_prov_relations(
        res,
        testing.make_prov_set(
            {(0.52, "a")}, [res.provenance_column.value, "x"]
        ),
    )
