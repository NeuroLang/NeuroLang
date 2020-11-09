from pytest import raises

from operator import eq
import typing

import numpy as np

from ...exceptions import SymbolNotFoundError
from ...expression_walker import ExpressionBasicEvaluator, ExpressionWalker
from ...expressions import Constant, ExpressionBlock, Symbol
from ...logic import (
    ExistentialPredicate, Implication,
    Negation, Conjunction, Union
)
from .. import DatalogProgram, Fact
from ..expression_processing import (
    TranslateToDatalogSemantics,
    is_conjunctive_expression,
    is_conjunctive_expression_with_nested_predicates,
    dependency_matrix,
    stratify, reachable_code, is_linear_rule,
    implication_has_existential_variable_in_antecedent,
    is_ground_predicate,
    extract_logic_free_variables,
    extract_logic_predicates,
    conjunct_if_needed,
    conjunct_formulas,
    program_has_loops,
    is_rule_with_builtin,
    HeadConstantToBodyEquality,
    HeadRepeatedVariableToBodyEquality,
    maybe_deconjunct_single_pred,
)

S_ = Symbol
C_ = Constant
Imp_ = Implication
B_ = ExpressionBlock
EP_ = ExistentialPredicate
T_ = Fact

EQ = Constant(eq)


DT = TranslateToDatalogSemantics()


def test_conjunctive_expression():
    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    x = S_('x')
    y = S_('y')

    assert is_conjunctive_expression(
        Q()
    )

    assert is_conjunctive_expression(
        Q(x)
    )

    assert is_conjunctive_expression(
        DT.walk(Q(x) & R(y, C_(1)))
    )

    assert not is_conjunctive_expression(
        DT.walk(R(x) | R(y))
    )

    assert not is_conjunctive_expression(
        DT.walk(R(x) & R(y) | R(x))
    )

    assert not is_conjunctive_expression(
        DT.walk(~R(x))
    )

    assert not is_conjunctive_expression(
        R(Q(x))
    )


def test_conjunctive_expression_with_nested():
    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    x = S_('x')
    y = S_('y')

    assert is_conjunctive_expression_with_nested_predicates(
        Q()
    )

    assert is_conjunctive_expression_with_nested_predicates(
        Q(x)
    )

    assert is_conjunctive_expression_with_nested_predicates(
        Q(x) & R(y, C_(1))
    )

    assert is_conjunctive_expression_with_nested_predicates(
        Q(x) & R(Q(y), C_(1))
    )

    assert not is_conjunctive_expression_with_nested_predicates(
        R(x) | R(y)
    )

    assert not is_conjunctive_expression_with_nested_predicates(
        R(x) & R(y) | R(x)
    )

    assert not is_conjunctive_expression_with_nested_predicates(
        ~R(x)
    )


def test_extract_free_variables():
    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    x = S_('x')
    y = S_('y')

    emptyset = set()
    assert extract_logic_free_variables(Q()) == emptyset
    assert extract_logic_free_variables(Q(C_(1))) == emptyset
    assert extract_logic_free_variables(x) == {x}
    assert extract_logic_free_variables(Q(x, y)) == {x, y}
    assert extract_logic_free_variables(Q(x, C_(1))) == {x}
    assert extract_logic_free_variables(Q(x) & R(y)) == {x, y}
    assert extract_logic_free_variables(EP_(x, Q(x, y))) == {y}
    assert extract_logic_free_variables(Imp_(R(x), Q(x, y))) == {y}
    assert extract_logic_free_variables(Imp_(R(x), Q(y) & Q(x))) == {y}
    assert extract_logic_free_variables(Q(R(y))) == {y}
    assert extract_logic_free_variables(Q(x) | R(y)) == {x, y}
    assert extract_logic_free_variables(~(R(y))) == {y}
    assert extract_logic_free_variables(B_([Q(x), R(y)])) == {x, y}


def test_extract_datalog_predicates():
    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    x = S_('x')
    y = S_('y')

    assert extract_logic_predicates(C_(1)) == set()
    assert extract_logic_predicates(Q) == set()

    expression = Q(x)
    assert extract_logic_predicates(expression) == {Q(x)}

    expression = DT.walk(Q(x) & R(y))
    assert extract_logic_predicates(expression) == {Q(x), R(y)}

    expression = DT.walk(B_([Q(x), Q(y) & R(y)]))
    assert extract_logic_predicates(expression) == {Q(x), Q(y), R(y)}

    expression = DT.walk(B_([Q(x), Q(y) & ~R(y)]))
    assert extract_logic_predicates(expression) == {
        Q(x), Q(y), Negation(R(y))
    }


def test_is_linear_rule():
    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    x = S_('x')
    y = S_('y')

    assert is_linear_rule(T_(Q(C_(1))))
    assert is_linear_rule(T_(Q(C_(x))))
    assert is_linear_rule(Imp_(Q(x), R(x, y)))
    assert is_linear_rule(Imp_(Q(x), Q(x)))
    assert is_linear_rule(DT.walk(Imp_(Q(x), R(x, y) & Q(x))))
    assert is_linear_rule(DT.walk(Imp_(Q(x), R(x, y) & ~Q(x))))
    assert not is_linear_rule(DT.walk(Imp_(Q(x), R(x, y) & Q(x) & Q(y))))
    assert not is_linear_rule(DT.walk(Imp_(Q(x), R(x, y) & Q(x) & ~Q(y))))


class Datalog(
    DatalogProgram,
    ExpressionBasicEvaluator
):
    pass


def test_stratification():
    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    S = S_('S')  # noqa: N806
    T = S_('T')  # noqa: N806
    x = S_('x')
    y = S_('y')

    code = DT.walk(B_([
        T_(Q(C_(1), C_(2))),
        Imp_(R(x, y), Q(x, y)),
        Imp_(R(x, y), Q(y, x)),
        Imp_(S(x), R(x, y) & S(y))
    ]))

    datalog = Datalog()
    datalog.walk(code)

    strata, stratifyiable = stratify(code, datalog)

    assert stratifyiable
    assert strata == [
        list(code.formulas[:1]),
        list(code.formulas[1: 3]),
        list(code.formulas[3:])
    ]

    code = DT.walk(B_([
        Imp_(Q(x, y), C_(eq)(x, y)),
        Imp_(R(x, y), Q(x, y)),
        Imp_(R(x, y), Q(y, x)),
        Imp_(S(x), R(x, y) & T(y)),
        Imp_(T(x), R(x, y) & S(x))
    ]))

    datalog = Datalog()
    datalog.walk(code)

    strata, stratifyiable = stratify(code, datalog)

    assert not stratifyiable
    assert strata == [
        list(code.formulas[:1]),
        list(code.formulas[1: 3]),
        list(code.formulas[3:])
    ]


def test_reachable():
    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    S = S_('S')  # noqa: N806
    T = S_('T')  # noqa: N806
    x = S_('x')
    y = S_('y')

    code = DT.walk(B_([
        Imp_(R(x, y), Q(x, y)),
        Imp_(R(x, y), Q(y, x)),
        Imp_(S(x), R(x, y) & S(y)),
        Imp_(T(x), Q(x, y)),
    ]))

    datalog = Datalog()
    datalog.walk(code)

    reached = reachable_code(code.formulas[-2], datalog)

    assert set(reached.formulas) == set(code.formulas[:-1])


def test_dependency_matrix():
    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    S = S_('S')  # noqa: N806
    T = S_('T')  # noqa: N806
    x = S_('x')
    y = S_('y')

    code = DT.walk(B_([
        Fact(Q(C_(0), C_(1))),
        Imp_(R(x, y), Q(x, y)),
        Imp_(R(x, y), T(y, x)),
        Imp_(S(x), R(x, y) & S(y) & C_(eq)(x, y)),
        Imp_(T(x), Q(x, y)),
    ]))

    datalog = Datalog()
    datalog.walk(code)

    idb_symbols, dep_matrix = dependency_matrix(datalog)

    assert idb_symbols == (R, S, T)
    assert np.array_equiv(dep_matrix, np.array(
        [
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 0]
        ]
    ))

    rules = Union(
        datalog.intensional_database()[R].formulas +
        datalog.intensional_database()[T].formulas
    )

    idb_symbols_2, dep_matrix_2 = dependency_matrix(
       datalog, rules=rules
    )

    assert idb_symbols_2 == (R, T)
    assert np.array_equiv(dep_matrix[(0, 2), :][:, (0, 2)], dep_matrix_2)

    code = DT.walk(B_([
        Fact(Q(C_(0), C_(1))),
        Imp_(R(x, y), Q(x, y)),
        Imp_(R(x, y), T(y, x)),
        Imp_(S(x), R(x, y) & S_('X')(y) & C_(eq)(x, y)),
        Imp_(T(x), Q(x, y)),
    ]))

    datalog = Datalog()
    datalog.walk(code)

    with raises(SymbolNotFoundError):
        dependency_matrix(datalog)


def test_program_has_loops():
    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    S = S_('S')  # noqa: N806
    T = S_('T')  # noqa: N806
    x = S_('x')
    y = S_('y')

    code = DT.walk(B_([
        Fact(Q(C_(0), C_(1))),
        Imp_(R(x, y), Q(x, y) & S(x)),
        Imp_(R(x, y), T(y, x)),
        Imp_(S(x), R(x, y) & C_(eq)(x, y)),
        Imp_(T(x, y), Q(x, y)),
    ]))

    datalog = Datalog()
    datalog.walk(code)

    assert program_has_loops(datalog)

    code = DT.walk(B_([
        Fact(Q(C_(0), C_(1))),
        Imp_(R(x, y), Q(x, y)),
        Imp_(R(x, x), T(x, x)),
        Imp_(S(x), R(x, y) & C_(eq)(x, y)),
        Imp_(T(x), Q(x, x))
    ]))

    datalog = Datalog()
    datalog.walk(code)

    assert not program_has_loops(datalog)


def test_implication_has_existential_variable_in_antecedent():
    Q = S_('Q')
    R = S_('R')
    x = S_('x')
    y = S_('y')
    assert implication_has_existential_variable_in_antecedent(
        Implication(Q(x), R(x, y)),
    )
    assert not implication_has_existential_variable_in_antecedent(
        Implication(Q(x), R(x)),
    )


def test_is_ground_predicate():
    P = S_("P")
    x = S_("x")
    a = C_("a")
    assert not is_ground_predicate(P(x))
    assert is_ground_predicate(P(a))


def test_conjunct_if_needed():
    P = S_("P")
    Q = S_("Q")
    x = S_("x")
    predicates = (P(x),)
    assert conjunct_if_needed(predicates) == P(x)
    predicates = (P(x), Q(x))
    assert conjunct_if_needed(predicates) == Conjunction(predicates)


def test_conjunct_formulas():
    P = S_("P")
    Q = S_("Q")
    x = S_("x")
    y = S_("y")
    f1 = Conjunction((P(x), Q(x, x)))
    f2 = Conjunction((Q(x, y), Q(y, y)))
    assert conjunct_formulas(f1, f2) == Conjunction(
        (P(x), Q(x, x), Q(x, y), Q(y, y))
    )
    assert conjunct_formulas(P(x), Q(x)) == Conjunction((P(x), Q(x)))



def test_is_rule_with_builtin():
    P = Symbol("P")
    Q = Symbol("Q")
    x = Symbol("x")
    y = Symbol("y")
    some_builtin = Constant(eq)
    rule = Implication(P(x), Conjunction((some_builtin(x, y), Q(x), Q(y))))
    assert is_rule_with_builtin(rule)
    rule = Implication(Q(x), P(x))
    assert not is_rule_with_builtin(rule)
    dl = Datalog()
    some_builtin = Symbol[typing.Callable[[int, int], str]]("some_builtin")
    my_builtin_fun = lambda u, v: str(u + v)
    dl.symbol_table[some_builtin] = Constant[typing.Callable[[int, int], str]](
        my_builtin_fun, auto_infer_type=False, verify_type=False,
    )
    rule = Implication(Q(x), some_builtin(x, y))
    assert is_rule_with_builtin(rule, known_builtins=dl.builtins())


class TestHeadConstantToBodyEquality(
    HeadConstantToBodyEquality,
    ExpressionWalker
):
    pass


class TestHeadRepeatedVariableToBodyEquality(
    HeadRepeatedVariableToBodyEquality,
    ExpressionWalker
):
    pass


def test_head_constant_to_body_equality():
    P = Symbol("P")
    Q = Symbol("Q")
    x = Symbol("x")
    a = Constant("a")
    walker = TestHeadConstantToBodyEquality()
    rule = Implication(P(x, a), Q(x))
    result = walker.walk(rule)
    assert result.consequent.args[1].is_fresh
    assert EQ(result.consequent.args[1], a) in result.antecedent.formulas


def test_head_repeated_variable_to_body_equality():
    P = Symbol("P")
    Q = Symbol("Q")
    x = Symbol("x")
    walker = TestHeadRepeatedVariableToBodyEquality()
    rule = Implication(P(x, x), Q(x))
    result = walker.walk(rule)
    assert result.consequent.args[1].is_fresh
    assert EQ(result.consequent.args[1], x) in result.antecedent.formulas



def test_maybe_deconjunct():
    P = Symbol("P")
    Q = Symbol("Q")
    x = Symbol("x")
    assert maybe_deconjunct_single_pred(Conjunction(tuple())) == Conjunction(
        tuple()
    )
    assert maybe_deconjunct_single_pred(
        Conjunction((P(x), Q(x)))
    ) == Conjunction((P(x), Q(x)))
    assert maybe_deconjunct_single_pred(Conjunction((P(x),))) == P(x)
