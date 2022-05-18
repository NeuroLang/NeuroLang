from operator import eq

from ...expressions import Constant, FunctionApplication, Symbol
from ...logic import (
    Conjunction,
    Disjunction,
    ExistentialPredicate,
    Implication,
    Union,
)
from .. import dalvi_suciu_lift
from ..antishattering import (
    EqualityVarsDetection,
    GetChoiceInConjunctionOrExistential,
    ReplaceFunctionApplicationInConjunctionWalker,
    ReplaceFunctionApplicationArgsWalker,
    pchoice_constants_as_head_variables,
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


def test_equiality_vars_detection():
    formula1 = Conjunction(
        (ExistentialPredicate(y, Conjunction((P(x, y), Q(a), eq_(x, a)))),)
    )

    eq_vars = EqualityVarsDetection().walk(formula1)
    assert len(eq_vars) == 1
    assert isinstance(eq_vars[0], list)
    assert x in eq_vars[0]
    assert a in eq_vars[0]

    formula2 = Conjunction(
        (ExistentialPredicate(y, Disjunction((P(x, y), Q(a), eq_(x, a)))),)
    )

    eq_vars = EqualityVarsDetection().walk(formula2)
    assert len(eq_vars) == 0


def test_get_choice_in_conjunction_or_existential():
    table = {
        (0.52, "a"),
        (0.48, "b"),
    }

    table2 = {(0.2, "a")}

    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_choice_from_tuples(P, table)
    cpl_program.add_probabilistic_facts_from_tuples(R, table2)

    formula = Conjunction((P(x), P(y), R(x), R(y)))

    symbol_table = generate_probabilistic_symbol_table_for_query(
        cpl_program, formula
    )

    gcce = GetChoiceInConjunctionOrExistential(symbol_table)
    choices = gcce.walk(formula)

    assert len(choices) == 2
    assert P(x) in choices
    assert P(y) in choices

    formula = Conjunction(
        (
            ExistentialPredicate(
                x, Conjunction((P(x), Disjunction((P(y), R(x)))))
            ),
            R(y),
        )
    )

    symbol_table = generate_probabilistic_symbol_table_for_query(
        cpl_program, formula
    )

    gcce = GetChoiceInConjunctionOrExistential(symbol_table)
    choices = gcce.walk(formula)

    assert len(choices) == 1
    assert P(x) in choices

    formula = Disjunction((P(x), R(y)))

    symbol_table = generate_probabilistic_symbol_table_for_query(
        cpl_program, formula
    )

    gcce = GetChoiceInConjunctionOrExistential(symbol_table)
    choices = gcce.walk(formula)

    assert len(choices) == 0


def test_nested_existential_choice_simplification():
    pass


def test_replace_nested_existential():
    pass


def test_replace_expression_in_conjunction():
    replacements = {
        P(x): Q(x),
        P(y): Q(y),
        R(y): Conjunction((Q(y), R(y))),
    }

    formula1 = Conjunction(
        (
            ExistentialPredicate(
                x, Conjunction((P(x), Disjunction((P(y), R(x)))))
            ),
            R(y),
        )
    )

    formula2 = Conjunction(
        (
            ExistentialPredicate(
                x, Disjunction((P(x), Conjunction((P(y), R(x)))))
            ),
            ExistentialPredicate(z, R(z)),
        )
    )

    rfac = ReplaceFunctionApplicationInConjunctionWalker(replacements)
    new_formula1 = rfac.walk(formula1)
    new_formula2 = rfac.walk(formula2)

    rfac2 = ReplaceFunctionApplicationArgsWalker({})
    new_formula3 = rfac2.walk(formula1)

    assert new_formula1 == Conjunction(
        (
            ExistentialPredicate(
                x, Conjunction((Q(x), Disjunction((P(y), R(x)))))
            ),
            Conjunction((Q(y), R(y))),
        )
    )

    assert new_formula2 == formula2
    assert new_formula3 == formula1


def test_replace_function_application_args():
    replacements = {
        x: a,
        y: z,
        w: b,
    }

    formula1 = P(x, w)
    formula2 = P(w, b)
    formula3 = Q(a)
    formula4 = P(x, x, y)

    new_formula1 = ReplaceFunctionApplicationArgsWalker(replacements).walk(
        formula1
    )
    new_formula2 = ReplaceFunctionApplicationArgsWalker(replacements).walk(
        formula2
    )
    new_formula3 = ReplaceFunctionApplicationArgsWalker(replacements).walk(
        formula3
    )
    new_formula4 = ReplaceFunctionApplicationArgsWalker(replacements).walk(
        formula4
    )

    assert new_formula1 == P(a, b)
    assert new_formula2 == P(b, b)
    assert new_formula3 == Q(a)
    assert new_formula4 == P(a, a, z)


def test_antishaterring_rewrite1():
    table = {
        (0.52, "a"),
        (0.48, "b"),
    }

    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_choice_from_tuples(P, table)

    a = Constant("a")
    query = Implication(ans(), Conjunction((P(a),)))
    new_query = pchoice_constants_as_head_variables(query, cpl_program)
    new_symbols = [
        fp
        for fp in new_query.antecedent.formulas
        if isinstance(fp, FunctionApplication) and fp.functor.is_fresh
    ]
    for arg in new_symbols[0].args:
        assert arg in new_query.antecedent.formulas[1].formulas[0].args


def test_antishaterring_rewrite2():
    table = {
        (0.52, "a", 1),
        (0.48, "b", 2),
    }

    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_choice_from_tuples(P, table)

    a = Constant("a")
    query = Implication(ans(), Conjunction((P(a, x),)))
    new_query = pchoice_constants_as_head_variables(query, cpl_program)
    new_symbols = [
        fp
        for fp in new_query.antecedent.formulas
        if isinstance(fp, FunctionApplication) and fp.functor.is_fresh
    ]
    for arg in new_symbols[0].args:
        assert arg in new_query.antecedent.formulas[1].formulas[0].args


def test_antishaterring_rewrite3():
    table = {
        (0.52, "a", 1),
        (0.48, "b", 2),
    }

    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_choice_from_tuples(P, table)

    a = Constant("a")
    one = Constant("1")
    query = Implication(ans(), Conjunction((P(a, one),)))
    new_query = pchoice_constants_as_head_variables(query, cpl_program)
    new_symbols = [
        fp
        for fp in new_query.antecedent.formulas
        if isinstance(fp, FunctionApplication) and fp.functor.is_fresh
    ]
    for arg in new_symbols[0].args:
        assert arg in new_query.antecedent.formulas[1].formulas[0].args


def test_antishaterring_rewrite4():
    table = {
        (0.52, "a", 1),
        (0.48, "b", 2),
    }

    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_choice_from_tuples(P, table)

    one = Constant("1")
    query = Implication(ans(), Conjunction((P(x, one),)))
    new_query = pchoice_constants_as_head_variables(query, cpl_program)
    new_symbols = [
        fp
        for fp in new_query.antecedent.formulas
        if isinstance(fp, FunctionApplication) and fp.functor.is_fresh
    ]
    for arg in new_symbols[0].args:
        assert arg in new_query.antecedent.formulas[1].formulas[0].args


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
        res, testing.make_prov_set([1.0], [res.provenance_column.value]),
    )
