from operator import eq

import pytest

from ... import exceptions
from ... import expression_walker as ew
from ... import expressions
from ... import logic
from .. import Fact, Implication
from .. import negation as sdn
from ..expressions import TranslateToLogic
from ..chase import Chase as Chase_
from ..chase.negation import NegativeFactConstraints, DatalogChaseNegation


C_ = expressions.Constant
S_ = expressions.Symbol
F_ = expressions.FunctionApplication
L_ = expressions.Lambda
E_ = logic.ExistentialPredicate
U_ = logic.UniversalPredicate
Eb_ = expressions.ExpressionBlock


class Datalog(
    TranslateToLogic,
    sdn.DatalogProgramNegation,
    ew.ExpressionBasicEvaluator
):
    def function_gt(self, x: int, y: int) -> bool:
        return x > y


class Chase(NegativeFactConstraints, Chase_):
    pass


def test_non_recursive_negation():
    x = S_('x')
    G = S_('G')
    T = S_('T')
    V = S_('V')

    program = Eb_((
        Fact(V(C_(1))),
        Fact(V(C_(2))),
        Fact(V(C_(3))),
        Fact(T(C_(1))),
        Fact(T(C_(4))),
        Implication(G(x), V(x) & ~T(x))
    ))

    dl = Datalog()
    dl.walk(program)

    dc = Chase(dl)
    solution_instance = dc.build_chase_solution()

    assert solution_instance['V'].value == {(1,), (2,), (3,)}
    assert solution_instance['T'].value == {(1,), (4,)}
    assert solution_instance['G'].value == {(2,), (3,)}


def test_stratified_and_chase():
    x = S_('x')
    y = S_('y')
    z = S_('z')
    G = S_('G')
    T = S_('T')
    V = S_('V')
    ET = S_('ET')
    equals = S_('equals')

    program = Eb_((
        Fact(V(C_(1))),
        Fact(V(C_(2))),
        Fact(V(C_(3))),
        Fact(V(C_(4))),
        Fact(G(C_(1), C_(2))),
        Fact(G(C_(2), C_(3))),
        Implication(T(x, y),
                    V(x) & V(y) & G(x, y)),
        Implication(T(x, y),
                    G(x, z) & T(z, y)),
        Implication(ET(x, y),
                    V(x) & V(y) & ~(G(x, y)) & equals(x, y)),
    ))

    dl = Datalog()
    dl.walk(program)

    #dc = DatalogChaseNegation(dl)
    dc = Chase(dl)
    solution_instance = dc.build_chase_solution()

    final_instance = {
        G:
        C_({
            C_((C_(1), C_(2))),
            C_((C_(2), C_(3))),
        }),
        V:
        C_({
            C_((C_(2), )),
            C_((C_(3), )),
            C_((C_(1), )),
            C_((C_(4), )),
        }),
        ET:
        C_({
            C_((C_(1), C_(1))),
            C_((C_(3), C_(3))),
            C_((C_(4), C_(4))),
            C_((C_(2), C_(2))),
        }),
        T:
        C_({C_((C_(1), C_(2))),
            C_((C_(1), C_(3))),
            C_((C_(2), C_(3)))})
    }

    assert solution_instance == final_instance


def test_stratified_and_chase_builtin_equality():
    x = S_('x')
    y = S_('y')
    G = S_('G')
    V = S_('V')
    NT = S_('NT')
    equals = C_(eq)

    program = Eb_((
        Fact(V(C_(1))),
        Fact(V(C_(2))),
        Fact(V(C_(3))),
        Fact(V(C_(4))),
        Fact(G(C_(1), C_(2))),
        Fact(G(C_(2), C_(3))),
        Implication(
            NT(x, y),
            V(x) & V(y) & ~(G(x, y)) & ~(equals(x, y))
        ),
    ))

    dl = Datalog()
    dl.walk(program)

    dc = Chase(dl)
    solution_instance = dc.build_chase_solution()

    final_instance = {
        G:
        C_({
            C_((C_(1), C_(2))),
            C_((C_(2), C_(3))),
        }),
        V:
        C_({
            C_((C_(2), )),
            C_((C_(3), )),
            C_((C_(1), )),
            C_((C_(4), )),
        }),
        NT:
        C_({
            C_((C_(3), C_(2))),
            C_((C_(1), C_(3))),
            C_((C_(4), C_(1))),
            C_((C_(3), C_(1))),
            C_((C_(2), C_(1))),
            C_((C_(1), C_(4))),
            C_((C_(4), C_(3))),
            C_((C_(4), C_(2))),
            C_((C_(3), C_(4))),
            C_((C_(2), C_(4))),
        }),
    }

    assert solution_instance == final_instance


def test_negative_fact():
    V = S_('V')
    G = S_('G')
    T = S_('T')
    CT = S_('CT')
    x = S_('x')
    y = S_('y')
    z = S_('z')

    program = Eb_((
        Fact(V(C_(1))),
        Fact(V(C_(2))),
        Fact(V(C_(3))),
        Fact(V(C_(4))),
        Fact(G(C_(1), C_(2))),
        Fact(G(C_(2), C_(3))),
        sdn.NegativeFact(G(C_(1), C_(2))),
        Implication(T(x, y),
                    V(x) & V(y) & G(x, y)),
        Implication(T(x, y),
                    G(x, z) & T(z, y)),
        Implication(CT(x, y),
                    V(x) & V(y) & ~(G(x, y))),
    ))

    dl = Datalog()
    dl.walk(program)

    dc = Chase(dl)
    with pytest.raises(
        exceptions.NeuroLangException, match=r'There is a contradiction .*'
    ):
        dc.build_chase_solution()


def test_symbol_order_in_datalog_solver():
    n = S_("n")
    l = S_("l")
    R = S_("R")
    S = S_("S")
    Ans = S_("Ans")

    program = Eb_((Implication(Ans(l), R(l, n) & ~S(n, l)),))

    dl = Datalog()
    dl.walk(program)
    dl.add_extensional_predicate_from_tuples(
        R, {("a", 1), ("a", 2), ("a", 3), ("b", 3), ("b", 4)}
    )
    dl.add_extensional_predicate_from_tuples(
        S, {(1, "a"), (2, "a"), (3, "b"), (4, "b")}
    )

    dc = Chase(dl)
    solution_instance = dc.build_chase_solution()

    assert solution_instance["Ans"].value == {("a",)}


def test_negative_predicates_race_in_chase():
    x = S_("x")
    F = S_("F")
    G = S_("G")
    R = S_("R")
    S = S_("S")

    dl = Datalog()
    dl.add_extensional_predicate_from_tuples(
        F, {(1,), (2,), (3,)}
    )
    dl.add_extensional_predicate_from_tuples(
        G, {(1,), (2,), (3,), (4,), (5,)}
    )
    program = Eb_((
        Implication(R(x), F(x)),
        Implication(S(x), G(x) & ~R(x)),
    ))
    dl.walk(program)

    dc = DatalogChaseNegation(dl)
    solution_instance = dc.build_chase_solution()

    assert solution_instance["S"].value == {(4,), (5,)}
