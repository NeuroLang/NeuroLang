import numpy as np
from nl_expr.expressions import Constant, Symbol
from nl_expr.logic import Negation

from ..aggregation import AggregationApplication
from ..constant_restriction import restrict_to_constants_used
from ..expression_processing import EQ
from ..expressions import Conjunction, Implication

C_ = Constant
S_ = Symbol
p, q, r, e, f, ans = map(S_, ("p", "q", "r", "e", "f", "ans"))
x, y, t, n = map(S_, ("x", "y", "t", "n"))


def test_a_predicate_read_with_one_constant_is_computed_for_it():
    """p is read with 'a' at its second argument: its rule's body equates
    t with it."""
    rules = (
        Implication(p(x, t), Conjunction((e(x, t), f(t)))),
        Implication(q(x), Conjunction((p(x, C_("a")), Negation(p(y, C_("a")))))),
    )
    restricted = restrict_to_constants_used(rules, kept=(q,))
    assert set(map(str, restricted)) == set(map(str, (
        Implication(p(x, t), Conjunction((e(x, t), f(t), EQ(t, C_("a"))))),
        rules[1],
    )))


def test_a_predicate_read_with_constants_is_computed_for_them():
    """p is read with 'a' and, negated, 'b': its rule tests t is one of
    them."""
    rules = (
        Implication(p(x, t), Conjunction((e(x, t), f(t)))),
        Implication(q(x), Conjunction((p(x, C_("a")), Negation(p(x, C_("b")))))),
    )
    restricted, reader = restrict_to_constants_used(rules, kept=(q,))
    assert reader is rules[1]
    *body, one_of = restricted.antecedent.formulas
    assert tuple(body) == rules[0].antecedent.formulas
    assert one_of.args == (t,)
    assert one_of.functor.value.rows == {("a",), ("b",)}
    assert one_of.functor.value("a") and not one_of.functor.value("c")
    assert list(one_of.functor.value(np.array(["a", "c", "b"]))) == [
        True, False, True
    ]


def test_predicates_not_restricted():
    """Not restricted: a predicate also read with a variable there, one
    answering a query, one read by an unrestricted reader with a
    variable, an aggregated argument and one passed as an argument."""
    count = AggregationApplication(S_("count"), (y,))
    rules = (
        Implication(p(x, t), e(x, t)),
        Implication(q(x), p(x, C_("a"))),
        Implication(q(x), p(x, y)),                 # a variable
        Implication(ans(x), p(x, C_("a"))),
        Implication(r(x, count), e(x, y)),
        Implication(q(x), r(x, C_(3))),             # aggregated argument
    )
    assert restrict_to_constants_used(rules, kept=(q, ans)) == rules
    rules = (Implication(p(x, t), e(x, t)), Implication(q(x), p(x, C_("a"))))
    assert restrict_to_constants_used(rules, kept=(q, p)) == rules
    reader = Implication(ans(x), p(x, y))
    assert restrict_to_constants_used(rules, (reader,), kept=(q,)) == rules
    passed = Implication(ans(x), f(p, x))
    assert restrict_to_constants_used(rules, (passed,), kept=(q,)) == rules
