from ...expressions import Constant, Symbol
from ...logic import Conjunction, Implication
from .. import dalvi_suciu_lift
from ..cplogic import testing
from ..cplogic.program import CPLogicProgram

ans = Symbol("ans")
P = Symbol("P")
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
        res, testing.make_prov_set({(0.52,)}, '_p_')
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
        res, testing.make_prov_set({(0.52,)}, ['_p_'])
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
    assert res.relation.value._container.empty


def test_pchoice_with_constant_and_proyected_variable():
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
        res, testing.make_prov_set({(0.52, 'a')}, ['_p_', 'x'])
    )