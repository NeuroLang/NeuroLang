from operator import eq

from sqlalchemy import false

from ..antishattering import (
    NestedExistentialChoiceSimplification,
    SelfjoinChoiceSimplification,
)

from ..probabilistic_ra_utils import (
    generate_probabilistic_symbol_table_for_query,
)

from ...expressions import Constant, Symbol
from ...logic import (
    Conjunction,
    ExistentialPredicate,
    Implication,
    Union,
)
from .. import dalvi_suciu_lift

from ..cplogic import testing
from ..cplogic.program import CPLogicProgram


ans = Symbol("ans")
P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
w = Symbol("w")
x = Symbol("x")
y = Symbol("y")
z = Symbol("z")

a = Constant("a")
b = Constant("b")
eq_ = Constant(eq)


def test_walkers_no_change():
    table = {
        (0.52, "a"),
        (0.48, "b"),
    }

    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_choice_from_tuples(P, table)

    formula = Conjunction((P(a),))
    symbol_table = generate_probabilistic_symbol_table_for_query(
        cpl_program, formula
    )

    new_formula = SelfjoinChoiceSimplification(symbol_table).walk(formula)
    new_formula = NestedExistentialChoiceSimplification(symbol_table).walk(
        new_formula
    )

    assert new_formula == P(a)


def test_walkers_conjuntion_selfjoin_constant():
    table = {
        (0.52, "a"),
        (0.48, "b"),
    }

    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_choice_from_tuples(P, table)

    formula = Conjunction((P(x), P(a)))
    symbol_table = generate_probabilistic_symbol_table_for_query(
        cpl_program, formula
    )

    new_formula = SelfjoinChoiceSimplification(symbol_table).walk(formula)
    new_formula = NestedExistentialChoiceSimplification(symbol_table).walk(
        new_formula
    )

    assert new_formula == Conjunction((P(a), eq_(x, a)))


def test_walkers_conjuntion_selfjoin_symbols():
    table = {
        (0.52, "a"),
        (0.48, "b"),
    }

    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_choice_from_tuples(P, table)

    formula = Conjunction((P(x), P(y)))
    symbol_table = generate_probabilistic_symbol_table_for_query(
        cpl_program, formula
    )

    new_formula = SelfjoinChoiceSimplification(symbol_table).walk(formula)
    new_formula = NestedExistentialChoiceSimplification(symbol_table).walk(
        new_formula
    )

    assert new_formula == Conjunction((P(y), eq_(x, y)))


def test_walkers_conjuntion_selfjoin_and_det():
    table = {
        (0.52, "a"),
        (0.48, "b"),
    }

    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_choice_from_tuples(P, table)

    formula = Conjunction((P(a), P(x), Q(x)))
    symbol_table = generate_probabilistic_symbol_table_for_query(
        cpl_program, formula
    )

    new_formula = SelfjoinChoiceSimplification(symbol_table).walk(formula)
    new_formula = NestedExistentialChoiceSimplification(symbol_table).walk(
        new_formula
    )

    for formula in new_formula.formulas:
        if formula not in (Q(a), P(a), eq_(x, a)):
            assert False
    assert True


def test_walkers_conjuntion_selfjoin_and_det2():
    table = {
        (0.52, "a"),
        (0.48, "b"),
    }

    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_choice_from_tuples(P, table)

    formula = Conjunction((P(a), P(x), Q(y)))
    symbol_table = generate_probabilistic_symbol_table_for_query(
        cpl_program, formula
    )

    new_formula = SelfjoinChoiceSimplification(symbol_table).walk(formula)
    new_formula = NestedExistentialChoiceSimplification(symbol_table).walk(
        new_formula
    )

    for formula in new_formula.formulas:
        if formula not in (Q(y), P(a), eq_(x, a)):
            assert False
    assert True


def test_walkers_conjuntion_existential():
    table = {
        (0.52, "a"),
        (0.48, "b"),
    }

    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_choice_from_tuples(P, table)

    formula = Conjunction(
        (
            ExistentialPredicate(
                x, ExistentialPredicate(y, Conjunction((P(x), P(y))))
            ),
        )
    )
    symbol_table = generate_probabilistic_symbol_table_for_query(
        cpl_program, formula
    )

    new_formula = SelfjoinChoiceSimplification(symbol_table).walk(formula)
    new_formula = NestedExistentialChoiceSimplification(symbol_table).walk(
        new_formula
    )

    assert new_formula == ExistentialPredicate(y, P(y))


def test_walkers_conjuntion_existential_constant():
    table = {
        (0.52, "a"),
        (0.48, "b"),
    }

    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_choice_from_tuples(P, table)

    formula = Conjunction(
        (ExistentialPredicate(x, Conjunction((P(x), P(a)))),)
    )
    symbol_table = generate_probabilistic_symbol_table_for_query(
        cpl_program, formula
    )

    new_formula = SelfjoinChoiceSimplification(symbol_table).walk(formula)
    new_formula = NestedExistentialChoiceSimplification(symbol_table).walk(
        new_formula
    )

    assert new_formula == P(a)


def test_walkers_choices_facts():
    table = {
        (0.52, "a"),
        (0.48, "b"),
    }

    table2 = {(0.2, "a")}

    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_choice_from_tuples(P, table)
    cpl_program.add_probabilistic_facts_from_tuples(R, table2)

    formula = Conjunction(
        (
            ExistentialPredicate(
                x, ExistentialPredicate(y, Conjunction((P(x), R(y))))
            ),
        )
    )
    symbol_table = generate_probabilistic_symbol_table_for_query(
        cpl_program, formula
    )

    new_formula = SelfjoinChoiceSimplification(symbol_table).walk(formula)
    new_formula = NestedExistentialChoiceSimplification(symbol_table).walk(
        new_formula
    )

    assert new_formula == Conjunction(
        (ExistentialPredicate(x, P(x)), ExistentialPredicate(y, R(y)),)
    )


def test_walkers_choices_facts_det():
    table = {
        (0.52, "a"),
        (0.48, "b"),
    }

    table2 = {(0.2, "a")}

    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_choice_from_tuples(P, table)
    cpl_program.add_probabilistic_facts_from_tuples(R, table2)

    formula = Conjunction(
        (
            ExistentialPredicate(
                x, ExistentialPredicate(y, Conjunction((P(x), R(y))))
            ),
            ExistentialPredicate(z, Conjunction((P(z), Q(z)))),
        )
    )
    symbol_table = generate_probabilistic_symbol_table_for_query(
        cpl_program, formula
    )

    new_formula = SelfjoinChoiceSimplification(symbol_table).walk(formula)
    new_formula = NestedExistentialChoiceSimplification(symbol_table).walk(
        new_formula
    )

    assert new_formula == Conjunction(
        (ExistentialPredicate(x, P(x)), ExistentialPredicate(y, R(y)),)
    )


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

    query = Implication(ans(), Conjunction((P(a), P(x), Q(x))))
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
        res, testing.make_prov_set([1.0], [res.provenance_column.value]),
    )
