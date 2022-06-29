from operator import eq

from ...expressions import Constant, FunctionApplication, Symbol
from ...logic import Conjunction, Disjunction, ExistentialPredicate, Implication, Union
from .. import dalvi_suciu_lift
from ..antishattering import (
    NestedExistentialChoiceSimplification,
    SelfjoinChoiceSimplification,
)
from ..cplogic import testing
from ..cplogic.program import CPLogicProgram
from ..probabilistic_ra_utils import (
    generate_probabilistic_symbol_table_for_query,
)

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

    assert new_formula == Conjunction((P(a),))


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

    # Conjunction((P(a), Fresh(x)))
    assert isinstance(new_formula, Conjunction)
    assert len(new_formula.formulas) == 2
    for formula in new_formula.formulas:
        assert isinstance(formula, FunctionApplication)
        if formula.functor == P:
            assert formula.args == (a,)
        if formula.functor.is_fresh:
            assert formula.args == (x,)


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

    assert isinstance(new_formula, Conjunction)
    assert len(new_formula.formulas) == 3
    for formula in new_formula.formulas:
        assert isinstance(formula, FunctionApplication)
        if formula.functor == P or formula.functor == Q:
            assert formula.args == (a,)
        if formula.functor.is_fresh:
            assert formula.args == (x,)


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

    assert isinstance(new_formula, Conjunction)
    assert len(new_formula.formulas) == 3
    for formula in new_formula.formulas:
        assert isinstance(formula, FunctionApplication)
        if formula.functor == P:
            assert formula.args == (a,)
        if formula.functor == Q:
            assert formula.args == (y,)
        if formula.functor.is_fresh:
            assert formula.args == (x,)


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

    assert new_formula == formula


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

    # assert new_formula == Conjunction(
    #    (
    #        ExistentialPredicate(z, Conjunction((Q(z), P(z)))),
    #        ExistentialPredicate(y, R(y)),
    #    )
    # )
    assert isinstance(new_formula, Conjunction)
    for formula in new_formula.formulas:
        assert isinstance(formula, ExistentialPredicate)
        if formula.head == z:
            assert isinstance(formula.body, Conjunction)
            assert len(formula.body.formulas) == 2
            for inner_formula in formula.body.formulas:
                assert inner_formula in [Q(z), P(z)]
        elif formula.head == y:
            assert formula.body == R(y)
        else:
            assert False


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

    cpl_program.add_extensional_predicate_from_tuples(Q, ["a", "b", "c"])

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
        res, testing.make_prov_set([0.0], [res.provenance_column.value]),
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
        res, testing.make_prov_set([(1.0,),], [res.provenance_column.value]),
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

def test_false_conjunction_inside_existential():
    table = {
        (0.52, "a", "1"),
        (0.28, "b", "2"),
        (0.20, "c", "1"),
    }

    two = Constant("2")
    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_choice_from_tuples(P, table)

    query = Implication(Q(), Conjunction((P(a, x), P(y, two), P(z, w))))

    cpl_program.walk(query)
    res = dalvi_suciu_lift.solve_succ_query(query, cpl_program)
    assert testing.eq_prov_relations(
        res, testing.make_prov_set([0.0], [res.provenance_column.value]),
    )

def test_disjunction_false():
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
            Implication(Q(), Conjunction((P(a), P(b)))), #(1) 0.2496
            Implication(Q(), P(b)), #(2) 0.48
            Implication(Q(), R(x)), #(3) 0.2
            query,
        )
    )
    # P(X) := P(1) + P(2) - P(1^2) = 0.2496 + 0.48 - 0.2496 = 0.48
    # P(X) + P(3) - P(X^3) = 0.48 + 0.2 - (0.48 * 0.2) = 0.584
    cpl_program.walk(program)
    res = dalvi_suciu_lift.solve_succ_query(query, cpl_program)
    assert testing.eq_prov_relations(
        res, testing.make_prov_set([0.584], [res.provenance_column.value]),
    )

def test_nested_conjunction_disjunction():
    table = {
        (0.52, "a"),
        (0.48, "b"),
    }

    table2 = {
        (0.2, "a"),
        (0.8, "b")
    }

    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_choice_from_tuples(P, table)
    cpl_program.add_probabilistic_choice_from_tuples(R, table2)
    query = Implication(ans(), Q())
    program = Union(
        (
            Implication(
                Q(), Conjunction((P(a), R(b)))
            ), #(1) 0.52 * 0.8 = 0.416
            Implication(
                Q(), Conjunction((P(x), R(b)))
            ), #(2) 0.384 + 0.416 = 0.8
            Implication(Q(), R(x)), #(3) 1
            query,
        )
    )
    # P(X) := P(1) + P(2) - P(1^2) = 0.416 + 0.8 - 0.416 = 0.8
    # P(X) + P(3) - P(X^3) = 0.8 + 1 - 0.416 = 1.384 ?!?
    cpl_program.walk(program)
    res = dalvi_suciu_lift.solve_succ_query(query, cpl_program)
    assert testing.eq_prov_relations(
        res, testing.make_prov_set([0.416], [res.provenance_column.value]),
    )

def test_false_conjunction_inside_existential():
    table = {
        (0.52, "a", "1"),
        (0.28, "b", "2"),
        (0.20, "c", "1"),
    }

    table2 = {
        (0.62, "a"),
        (0.38, "b"),
    }

    one = Constant("1")
    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_choice_from_tuples(P, table)
    cpl_program.add_probabilistic_choice_from_tuples(R, table2)

    query = Implication(Q(), Conjunction((P(a, x), P(y, one), R(y))))

    cpl_program.walk(query)
    res = dalvi_suciu_lift.solve_succ_query(query, cpl_program)
    assert testing.eq_prov_relations(
        res, testing.make_prov_set([0.3224], [res.provenance_column.value]),
    )