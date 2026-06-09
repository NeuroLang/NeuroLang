from operator import add, eq, lt, mul, pow, sub, truediv

import pytest


from neurolang.logic import ExistentialPredicate

from ....datalog import Conjunction, Fact, Implication, Negation, Union
from ....expressions import (
    Command,
    Constant,
    FunctionApplication,
    Lambda,
    Query,
    Statement,
    Symbol
)
from ....exceptions import UnexpectedTokenError
from ....probabilistic.expressions import (
    PROB,
    Condition,
    ProbabilisticChoice,
    ProbabilisticFact,
)
from ..standard_syntax import ExternalSymbol, parser


def test_facts():
    res = parser('A(3)')
    assert res == Union((Fact(Symbol('A')(Constant(3))),))

    res = parser('A("x")')
    assert res == Union((Fact(Symbol('A')(Constant('x'))),))

    res = parser("A('x', 3)")
    assert res == Union((Fact(Symbol('A')(Constant('x'), Constant(3))),))

    res = parser(
        'A("x", 3)\n'
        '`http://uri#test-fact`("x")\n'
        'ans():-A(x, y)'
    )
    assert res == Union((
        Fact(Symbol('A')(Constant('x'), Constant(3))),
        Fact(Symbol('http://uri#test-fact')(Constant('x'))),
        Query(
            Symbol('ans')(),
            Conjunction((
                Symbol('A')(Symbol('x'), Symbol('y')),
            ))
        )
    ))


def test_rules():
    A = Symbol('A')
    B = Symbol('B')
    C = Symbol('C')
    f = Symbol('f')
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    res = parser('A(x):-B(x, y), C(3, z)')
    assert res == Union((
        Implication(A(x), Conjunction((B(x, y), C(Constant(3), z)))),
    ))

    res = parser('A(x):-~B(x)')
    assert res == Union((
        Implication(A(x), Conjunction((Negation(B(x)),))),
    ))

    res = parser('A(x):-B(x, ...)')
    fresh_arg = res.formulas[0].antecedent.formulas[0].args[1]
    assert isinstance(fresh_arg, Symbol)
    assert res == Union((
        Implication(A(x), Conjunction((B(x, fresh_arg),))),
    ))

    res = parser('A(x):-B(x, y), C(3, z), (z == 4)')
    assert res == Union((
        Implication(
            A(x),
            Conjunction((
                B(x, y), C(Constant(3), z), Constant(eq)(z, Constant(4))
            ))
        ),
    ))

    res = parser('A(x):-B(x + 5 * 2, y), C(3, z), (z == 4)')
    assert res == Union((
        Implication(
            A(x),
            Conjunction((
                B(
                    Constant(add)(
                        x,
                        Constant(mul)(Constant(5), Constant(2))),
                    y
                ),
                C(Constant(3), z), Constant(eq)(z, Constant(4))
            ))
        ),
    ))

    res = parser('A(x):-B(x / 2, y), C(3, z), (z == 4)')
    assert res == Union((
        Implication(
            A(x),
            Conjunction((
                B(
                    Constant(truediv)(x, Constant(2)),
                    y
                ),
                C(Constant(3), z), Constant(eq)(z, Constant(4))
            ))
        ),
    ))

    res = parser('A(x):-B(f(x), y), C(3, z), (z == 4)')
    assert res == Union((
        Implication(
            A(x),
            Conjunction((
                B(f(x), y),
                C(Constant(3), z), Constant(eq)(z, Constant(4))
            ))
        ),
    ))

    res = parser('A(x):-B(x + (-5), "a")')
    assert res == Union((
        Implication(
            A(x),
            Conjunction((
                B(
                    Constant(add)(x, Constant(-5)),
                    Constant("a")
                ),
            ))
        ),
    ))

    res = parser('A(x):-B(x - 5 * 2, @y ** -2)')
    assert res == Union((
        Implication(
            A(x),
            Conjunction((
                B(
                    Constant(sub)(
                        x,
                        Constant(mul)(Constant(5), Constant(2))
                    ),
                    Constant(pow)(ExternalSymbol('y'), Constant(-2))
                ),
            ))
        ),
    ))


def test_aggregation():
    A = Symbol('A')
    B = Symbol('B')
    f = Symbol('f')
    x = Symbol('x')
    y = Symbol('y')
    res = parser('A(x, f(y)):-B(x, y)')
    expected_result = Union((
        Implication(
            A(x, FunctionApplication(f, (y,))),
            Conjunction((B(x, y),))
        ),
    ))
    assert res == expected_result


def test_uri():
    from rdflib import RDFS

    label = Symbol(name=str(RDFS.label))
    regional_part = Symbol(
        name='http://sig.biostr.washington.edu/fma3.0#regional_part_of'
    )
    x = Symbol('x')
    y = Symbol('y')

    res = parser(f'`{str(label.name)}`(x):-`{str(regional_part.name)}`(x, y)')
    expected_result = Union((
        Implication(
            label(x),
            Conjunction((regional_part(x, y),))
        ),
    ))

    assert res == expected_result


def test_probabilistic_fact():
    A = Symbol('A')
    p = Symbol('p')
    res = parser('p::A(3)')
    assert res == Union((
        Implication(
            ProbabilisticFact(p, A(Constant(3.))),
            Constant(True)
        ),
    ))

    res = parser('0.8::A("a b", 3)')
    assert res == Union((
        Implication(
            ProbabilisticFact(
                Constant(0.8),
                A(Constant("a b"), Constant(3.))
            ),
            Constant(True)
        ),
    ))

    exp = Symbol("exp")
    d = Symbol("d")
    x = Symbol("x")
    B = Symbol("B")
    res = parser("B(x) :: exp(-d / 5.0) :- A(x, d) , (d < 0.8)")
    expected = Union(
        (
            Implication(
                ProbabilisticFact(
                    FunctionApplication(
                        exp,
                        (
                            Constant(truediv)(
                                Constant(mul)(Constant(-1), d), Constant(5.0)
                            ),
                        ),
                    ),
                    B(x),
                ),
                Conjunction((A(x, d), Constant(lt)(d, Constant(0.8)))),
            ),
        )
    )
    assert res == expected


# ── Probabilistic choice syntax (^ and :~:) ──────────────────────────────────


def test_probabilistic_choice_fact_caret():
    A = Symbol('A')
    p = Symbol('p')
    res = parser('p ^ A(3)')
    assert res == Union((
        Implication(
            ProbabilisticChoice(p, A(Constant(3.))),
            Constant(True),
        ),
    ))


def test_probabilistic_choice_fact_tilde():
    A = Symbol('A')
    res = parser('0.8 :~: A("a b", 3)')
    assert res == Union((
        Implication(
            ProbabilisticChoice(
                Constant(0.8),
                A(Constant("a b"), Constant(3.)),
            ),
            Constant(True),
        ),
    ))


def test_probabilistic_choice_rule_caret():
    B = Symbol("B")
    A = Symbol("A")
    x = Symbol("x")
    d = Symbol("d")
    exp = Symbol("exp")
    res = parser("B(x) ^ exp(-d / 5.0) :- A(x, d)")
    expected = Union((
        Implication(
            ProbabilisticChoice(
                FunctionApplication(
                    exp,
                    (Constant(truediv)(
                        Constant(mul)(Constant(-1), d), Constant(5.0)
                    ),),
                ),
                B(x),
            ),
            Conjunction((A(x, d),)),
        ),
    ))
    assert res == expected


def test_probabilistic_choice_rule_tilde():
    B = Symbol("B")
    x = Symbol("x")
    res = parser("B(x) :~: 0.3 :- A(x)")
    expected = Union((
        Implication(
            ProbabilisticChoice(Constant(0.3), B(x)),
            Conjunction((Symbol("A")(x),)),
        ),
    ))
    assert res == expected


def test_condition():
    A = Symbol('A')
    B = Symbol('B')
    C = Symbol('C')
    x = Symbol('x')
    res = parser('C(x) :- A(x) // B(x)')

    expected = Union((
        Implication(
            C(x),
            Condition(A(x), B(x))
        ),
    ))

    assert res == expected

    res = parser('C(x) :- (A(x), B(x)) // B(x)')

    expected = Union((
        Implication(
            C(x),
            Condition(Conjunction((A(x), B(x))), B(x))
        ),
    ))

    assert res == expected

    res = parser('C(x) :- A(x) // (A(x), B(x))')

    expected = Union((
        Implication(
            C(x),
            Condition(A(x), Conjunction((A(x), B(x))))
        ),
    ))

    assert res == expected


def test_existential():
    A = Symbol("A")
    B = Symbol("B")
    C = Symbol("C")
    x = Symbol("x")
    s1 = Symbol("s1")
    s2 = Symbol("s2")

    res = parser("C(x) :- B(x), exists(s1; A(s1))")
    expected = Union(
        (
            Implication(
                C(x), Conjunction((B(x), ExistentialPredicate(s1, A(s1))))
            ),
        )
    )
    assert res == expected

    res = parser("C(x) :- B(x), ∃(s1 st A(s1))")
    assert res == expected

    res = parser("C(x) :- B(x), exists(s1, s2; A(s1), A(s2))")

    expected = Union(
        (
            Implication(
                C(x),
                Conjunction(
                    (
                        B(x),
                        ExistentialPredicate(
                            s2,
                            ExistentialPredicate(
                                s1, Conjunction((A(s1), A(s2)))
                            ),
                        ),
                    )
                ),
            ),
        )
    )

    assert res == expected

    # try :
    #     res = parser("C(x) :- B(x), exists(s1; )")
    # except UnexpectedToken as e:
    #     raise UnexpectedTokenError from e
    with pytest.raises(UnexpectedTokenError):
        res = parser("C(x) :- B(x), exists(s1; )")


def test_query():
    ans = Symbol("ans")
    B = Symbol("B")
    C = Symbol("C")
    x = Symbol("x")
    y = Symbol("y")
    res = parser("ans(x) :- B(x, y), C(3, y)")
    assert res == Union(
        (Query(ans(x), Conjunction((B(x, y), C(Constant(3), y)))),)
    )


def test_prob_implicit():
    B = Symbol("B")
    C = Symbol("C")
    x = Symbol("x")
    y = Symbol("y")
    res = parser("B(x, PROB, y) :- C(x, y)")
    assert res == Union(
        (Implication(B(x, PROB(x, y), y), Conjunction((C(x, y),))),)
    )


def test_prob_explicit():
    B = Symbol("B")
    C = Symbol("C")
    x = Symbol("x")
    y = Symbol("y")
    res = parser("B(x, PROB(x, y), y) :- C(x, y)")
    assert res == Union(
        (Implication(B(x, PROB(x, y), y), Conjunction((C(x, y),))),)
    )


def test_lambda_definition():
    c = Symbol("c")
    x = Symbol("x")

    res = parser("c := lambda x: x + 1")
    expression = Lambda((x,), FunctionApplication(
        Constant(add), (x, Constant[int](1))
    ))
    expected = Union((
        Statement(c, expression),
    ))
    assert expected == res


def test_lambda_definition_statement():
    c = Symbol("c")
    x = Symbol("x")
    y = Symbol("y")

    res = parser("c(x, y) := x + y")
    expression = Lambda((x, y), FunctionApplication(
        Constant(add), (x, y)
    ))
    expected = Union((
        Statement(c, expression),
    ))
    assert expected == res


def test_lambda_application():
    c = Symbol("c")
    x = Symbol("x")

    res = parser("c := (lambda x: x + 1)(2)")
    expression = FunctionApplication(
        Lambda((x,), FunctionApplication(
            Constant(add), (x, Constant[int](1))
        )),
        (Constant[int](2),)
    )
    expected = Union((
        Statement(c, expression),
    ))
    assert expected == res


def test_command_syntax():
    res = parser('.load_csv(A, "http://myweb/file.csv", B)')
    expected = Union(
        (
            Command(
                Symbol("load_csv"),
                (Symbol("A"), Constant("http://myweb/file.csv"), Symbol("B")),
                (),
            ),
        )
    )
    assert res == expected

    res = parser('.load_csv("http://myweb/file.csv")')
    expected = Union(
        (
            Command(
                Symbol("load_csv"), (Constant("http://myweb/file.csv"),), ()
            ),
        )
    )
    assert res == expected

    res = parser(".load_csv()")
    expected = Union((Command(Symbol("load_csv"), (), ()),))
    assert res == expected

    res = parser('.load_csv(sep=",")')
    expected = Union(
        (Command("load_csv", (), ((Symbol("sep"), Constant(",")),)),)
    )
    assert res == expected

    res = parser('.load_csv(sep=",", header=None, index_col=0)')
    expected = Union(
        (
            Command(
                "load_csv",
                (),
                (
                    (Symbol("sep"), Constant(",")),
                    (Symbol("header"), Symbol("None")),
                    (Symbol("index_col"), Constant(0)),
                ),
            ),
        )
    )
    assert res == expected

    res = parser('.load_csv(A, "http://myweb/file.csv", sep=",", header=None)')
    expected = Union(
        (
            Command(
                "load_csv",
                (Symbol("A"), "http://myweb/file.csv"),
                (
                    (Symbol("sep"), Constant(",")),
                    (Symbol("header"), Symbol("None")),
                ),
            ),
        )
    )
    assert res == expected


# ── New syntax: Aggregation in rule head ─────────────────────────────────────

from ....datalog.expressions import AggregationApplication  # noqa: E402


def test_head_aggregation_in_query():
    """ans(x, count(y)):-body(x, y) → Query with AggregationApplication."""
    res = parser("ans(x, count(y)):-body(x, y)")
    ans = Symbol("ans")
    body = Symbol("body")
    x = Symbol("x")
    y = Symbol("y")

    expected = Union((
        Query(
            ans(x, AggregationApplication(Symbol("count"), (y,))),
            Conjunction((body(x, y),)),
        ),
    ))
    assert res == expected


def test_head_aggregation_in_rule():
    """Non-ans head with aggregation → Implication with AggregationApplication."""
    res = parser("result(x, count(y)):-body(x, y)")
    result = Symbol("result")
    body = Symbol("body")
    x = Symbol("x")
    y = Symbol("y")

    expected = Union((
        Implication(
            result(x, AggregationApplication(Symbol("count"), (y,))),
            Conjunction((body(x, y),)),
        ),
    ))
    assert res == expected


def test_head_aggregation_multiple():
    """Multiple aggregation functions in head args."""
    res = parser("ans(x, count(y), sum(z)):-body(x, y, z)")
    ans = Symbol("ans")
    body = Symbol("body")
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")

    expected = Union((
        Query(
            ans(
                x,
                AggregationApplication(Symbol("count"), (y,)),
                AggregationApplication(Symbol("sum"), (z,)),
            ),
            Conjunction((body(x, y, z),)),
        ),
    ))
    assert res == expected


def test_head_aggregation_mean_std():
    """Mean and std aggregation functions are recognised."""
    res = parser("ans(x, mean(y), std(z)):-body(x, y, z)")

    q = res.formulas[0]
    assert isinstance(q, Query)
    args = q.head.args
    assert isinstance(args[1], AggregationApplication)
    assert args[1].functor == Symbol("mean")
    assert isinstance(args[2], AggregationApplication)
    assert args[2].functor == Symbol("std")


# ── New syntax: PROB head rule ───────────────────────────────────────────────

def test_prob_head_rule_simple():
    """PROB[pred(x)]:-body(x) → Implication with pred as head."""
    res = parser("PROB[p(x)]:-body(x)")
    p = Symbol("p")
    body = Symbol("body")
    x = Symbol("x")

    expected = Union((
        Implication(
            p(x, FunctionApplication(PROB, (x,))),
            Conjunction((body(x),)),
        ),
    ))
    assert res == expected


def test_prob_head_rule_conditional():
    """PROB[pred(x)]:-body(x)//cond(x) → Implication with Condition."""
    res = parser("PROB[p(x)]:-a(x) // b(x)")
    p = Symbol("p")
    a = Symbol("a")
    b = Symbol("b")
    x = Symbol("x")

    expected = Union((
        Implication(
            p(x, FunctionApplication(PROB, (x,))),
            Condition(a(x), b(x)),
        ),
    ))
    assert res == expected


# ── New syntax: PROB body predicate ──────────────────────────────────────────

def test_prob_body_simple():
    """ans(x, p):-PROB[pred(x)]=p desugars to fresh(x, PROB(x)):-pred(x), ans(x,p):-fresh(x,p)."""
    ans = Symbol("ans")
    pred = Symbol("pred")
    x = Symbol("x")
    p = Symbol("p")
    res = parser("ans(x, p):-PROB[pred(x)]=p")

    fresh = res.formulas[0].consequent.functor
    assert fresh.is_fresh

    expected = Union((
        Implication(
            fresh(x, FunctionApplication(PROB, (x,))),
            Conjunction((pred(x),)),
        ),
        Query(
            ans(x, p),
            Conjunction((fresh(x, p),)),
        ),
    ))
    assert res == expected


def test_prob_body_with_filter():
    """ans(x, p):-filter(x) & PROB[pred(x)]=p → fresh rule + query with filter."""
    ans = Symbol("ans")
    filter_ = Symbol("filter")
    pred = Symbol("pred")
    x = Symbol("x")
    p = Symbol("p")
    res = parser("ans(x, p):-filter(x) & PROB[pred(x)]=p")

    fresh = res.formulas[0].consequent.functor
    assert fresh.is_fresh

    expected = Union((
        Implication(
            fresh(x, FunctionApplication(PROB, (x,))),
            Conjunction((pred(x),)),
        ),
        Query(
            ans(x, p),
            Conjunction((filter_(x), fresh(x, p))),
        ),
    ))
    assert res == expected


def test_prob_body_conditional():
    """ans(x, p):-PROB[pred(x) // cond(x)]=p → cond_body added to fresh rule body."""
    ans = Symbol("ans")
    pred = Symbol("pred")
    cond = Symbol("cond")
    x = Symbol("x")
    p = Symbol("p")
    res = parser("ans(x, p):-PROB[pred(x) // cond(x)]=p")

    fresh = res.formulas[0].consequent.functor
    assert fresh.is_fresh

    expected = Union((
        Implication(
            fresh(x, FunctionApplication(PROB, (x,))),
            Condition(pred(x), Conjunction((cond(x),))),
        ),
        Query(
            ans(x, p),
            Conjunction((fresh(x, p),)),
        ),
    ))
    assert res == expected


def test_prob_body_conditional_with_filter():
    """ans(x, p):-filter(x) & PROB[pred(x) // cond(x)]=p → filter + conditional PROB."""
    ans = Symbol("ans")
    filter_ = Symbol("filter")
    pred = Symbol("pred")
    cond = Symbol("cond")
    x = Symbol("x")
    p = Symbol("p")
    res = parser("ans(x, p):-filter(x) & PROB[pred(x) // cond(x)]=p")

    fresh = res.formulas[0].consequent.functor
    assert fresh.is_fresh

    expected = Union((
        Implication(
            fresh(x, FunctionApplication(PROB, (x,))),
            Condition(pred(x), Conjunction((cond(x),))),
        ),
        Query(
            ans(x, p),
            Conjunction((filter_(x), fresh(x, p))),
        ),
    ))
    assert res == expected


# ── New syntax: MARG body predicate ──────────────────────────────────────────

def test_marg_body_simple():
    """ans(x, p):-MARG[pred(x)]=p → fresh rule + query with conjunction-wrapped pred."""
    ans = Symbol("ans")
    pred = Symbol("pred")
    x = Symbol("x")
    p = Symbol("p")
    res = parser("ans(x, p):-MARG[pred(x)]=p")

    fresh = res.formulas[0].consequent.functor
    assert fresh.is_fresh

    expected = Union((
        Implication(
            fresh(FunctionApplication(PROB, (Conjunction((pred(x),)),))),
            Conjunction((pred(x),)),
        ),
        Query(
            ans(x, p),
            Conjunction((fresh(p),)),
        ),
    ))
    assert res == expected


def test_marg_body_conjunction():
    """ans(x, y, p):-MARG[pred1(x) & pred2(y)]=p → fresh extracts union of vars."""
    ans = Symbol("ans")
    pred1 = Symbol("pred1")
    pred2 = Symbol("pred2")
    x = Symbol("x")
    y = Symbol("y")
    p = Symbol("p")
    res = parser("ans(x, y, p):-MARG[pred1(x) & pred2(y)]=p")

    fresh = res.formulas[0].consequent.functor
    assert fresh.is_fresh

    expected = Union((
        Implication(
            fresh(x, y, FunctionApplication(PROB, (x, y))),
            Conjunction((pred1(x), pred2(y))),
        ),
        Query(
            ans(x, y, p),
            Conjunction((fresh(x, y, p),)),
        ),
    ))
    assert res == expected


def test_marg_body_with_filter():
    """ans(x, p):-filter(x) & MARG[pred(x)]=p → mixed query with fresh rule."""
    ans = Symbol("ans")
    filter_ = Symbol("filter")
    pred = Symbol("pred")
    x = Symbol("x")
    p = Symbol("p")
    res = parser("ans(x, p):-filter(x) & MARG[pred(x)]=p")

    fresh = res.formulas[0].consequent.functor
    assert fresh.is_fresh

    expected = Union((
        Implication(
            fresh(x, FunctionApplication(PROB, (x,))),
            Conjunction((pred(x),)),
        ),
        Query(
            ans(x, p),
            Conjunction((filter_(x), fresh(x, p))),
        ),
    ))
    assert res == expected


def test_marg_body_conditional():
    """ans(x, p):-MARG[pred(x) // cond(x)]=p → cond_body added to fresh rule body."""
    ans = Symbol("ans")
    pred = Symbol("pred")
    cond = Symbol("cond")
    x = Symbol("x")
    p = Symbol("p")
    res = parser("ans(x, p):-MARG[pred(x) // cond(x)]=p")

    fresh = res.formulas[0].consequent.functor
    assert fresh.is_fresh

    expected = Union((
        Implication(
            fresh(x, FunctionApplication(PROB, (x,))),
            Condition(pred(x), Conjunction((cond(x),))),
        ),
        Query(
            ans(x, p),
            Conjunction((fresh(x, p),)),
        ),
    ))
    assert res == expected


def test_marg_body_conditional_with_filter():
    """ans(x, p):-filter(x) & MARG[pred(x) // cond(x)]=p → filter + conditional MARG."""
    ans = Symbol("ans")
    filter_ = Symbol("filter")
    pred = Symbol("pred")
    cond = Symbol("cond")
    x = Symbol("x")
    p = Symbol("p")
    res = parser("ans(x, p):-filter(x) & MARG[pred(x) // cond(x)]=p")

    fresh = res.formulas[0].consequent.functor
    assert fresh.is_fresh

    expected = Union((
        Implication(
            fresh(x, FunctionApplication(PROB, (x,))),
            Condition(pred(x), Conjunction((cond(x),))),
        ),
        Query(
            ans(x, p),
            Conjunction((filter_(x), fresh(x, p))),
        ),
    ))
    assert res == expected


# ── | separator (alias for //) ────────────────────────────────────────────────

def test_prob_body_conditional_pipe():
    """PROB[pred | cond] parses to same IR as PROB[pred // cond] (modulo fresh names)."""
    # Both must parse, both produce Condition antecedents
    res_pipe = parser("ans(x, p):-PROB[pred(x) | cond(x)]=p")
    res_slash = parser("ans(x, p):-PROB[pred(x) // cond(x)]=p")

    for res in (res_pipe, res_slash):
        fml = res.formulas[0]
        assert isinstance(fml, Implication)
        assert isinstance(fml.antecedent, Condition)
        assert isinstance(fml.antecedent.conditioned, FunctionApplication)
        assert isinstance(fml.antecedent.conditioning, Conjunction)


def test_marg_body_conditional_pipe():
    """MARG[pred | cond] parses to same IR as MARG[pred // cond] (modulo fresh names)."""
    res_pipe = parser("ans(x, p):-MARG[pred(x) | cond(x)]=p")
    res_slash = parser("ans(x, p):-MARG[pred(x) // cond(x)]=p")

    for res in (res_pipe, res_slash):
        fml = res.formulas[0]
        assert isinstance(fml, Implication)
        assert isinstance(fml.antecedent, Condition)
        assert isinstance(fml.antecedent.conditioned, FunctionApplication)
        assert isinstance(fml.antecedent.conditioning, Conjunction)


# ── MARG head rule ────────────────────────────────────────────────────────────

def test_marg_head_rule_simple():
    """MARG[pred(x)] :- body(x) — MARG wraps head in Conjunction."""
    pred = Symbol("pred")
    body = Symbol("body")
    x = Symbol("x")
    res = parser("MARG[pred(x)] :- body(x)")
    expected = Union((
        Implication(
            Conjunction((pred(x, FunctionApplication(PROB, (x,))),)),
            Conjunction((body(x),)),
        ),
    ))
    assert res == expected


def test_marg_head_rule_conditional():
    """MARG[pred(x) // cond(x)] :- body(x) — conditional marginal rule."""
    pred = Symbol("pred")
    cond = Symbol("cond")
    x = Symbol("x")
    res = parser("MARG[pred(x) // cond(x)] :- body(x)")
    expected = Union((
        Implication(
            pred(x, FunctionApplication(PROB, (x,))),
            Condition(
                Conjunction((pred(x),)),
                Conjunction((cond(x),)),
            ),
        ),
    ))
    assert res == expected


# ── SUCC head rule (no-op) ────────────────────────────────────────────────────

def test_succ_head_rule():
    """SUCC[...] :- body is a no-op — returns Constant(True)."""
    res = parser("SUCC[anything] :- body(x)")
    expected = Union((
        Constant(True),
    ))
    assert res == expected


# ── Negation and Existential inside PROB body ─────────────────────────────────

def test_prob_body_negation():
    """ans(x, p):-PROB[~pred(x)]=p — Negation inside PROB body."""
    ans = Symbol("ans")
    pred = Symbol("pred")
    x = Symbol("x")
    p = Symbol("p")
    res = parser("ans(x, p):-PROB[~pred(x)]=p")

    fresh = res.formulas[0].consequent.functor
    assert fresh.is_fresh

    expected = Union((
        Implication(
            fresh(x, FunctionApplication(PROB, (x,))),
            Conjunction((Negation(pred(x)),)),
        ),
        Query(
            ans(x, p),
            Conjunction((fresh(x, p),)),
        ),
    ))
    assert res == expected


def test_prob_body_negation_conditional():
    """ans(x, p):-PROB[~pred(x) // cond(x)]=p — Negation with conditioning."""
    ans = Symbol("ans")
    pred = Symbol("pred")
    cond = Symbol("cond")
    x = Symbol("x")
    p = Symbol("p")
    res = parser("ans(x, p):-PROB[~pred(x) // cond(x)]=p")

    fresh = res.formulas[0].consequent.functor
    assert fresh.is_fresh

    expected = Union((
        Implication(
            fresh(x, FunctionApplication(PROB, (x,))),
            Condition(Negation(pred(x)), Conjunction((cond(x),))),
        ),
        Query(
            ans(x, p),
            Conjunction((fresh(x, p),)),
        ),
    ))
    assert res == expected


def test_prob_body_existential():
    """ans(x, p):-PROB[exists(Y; pred(x, Y))]=p — Existential inside PROB body."""
    ans = Symbol("ans")
    pred = Symbol("pred")
    x = Symbol("x")
    p = Symbol("p")
    res = parser("ans(x, p):-PROB[exists(Y; pred(x, Y))]=p")

    fresh = res.formulas[0].consequent.functor
    assert fresh.is_fresh

    # ExistentialPredicate(head=Y, body=pred(x, Y))
    expected = Union((
        Implication(
            fresh(x, FunctionApplication(PROB, (x,))),
            Conjunction((
                ExistentialPredicate(Symbol("Y"), pred(x, Symbol("Y"))),
            )),
        ),
        Query(
            ans(x, p),
            Conjunction((fresh(x, p),)),
        ),
    ))
    assert res == expected


# ── New syntax: SUCC body predicate (no-op) ──────────────────────────────────

def test_succ_body_noop():
    """SUCC[...]=p is treated as a no-op (removed from body)."""
    ans = Symbol("ans")
    x = Symbol("x")
    res = parser("ans(x):-SUCC[anything]=p")
    expected = Union((
        Query(ans(x), Constant(True)),
    ))
    assert res == expected


def test_succ_body_with_filter():
    """SUCC with other atoms: SUCC is removed, rest stays."""
    ans = Symbol("ans")
    filter_ = Symbol("filter")
    x = Symbol("x")
    p = Symbol("p")
    res = parser("ans(x, p):-filter(x) & SUCC[anything]=p")
    expected = Union((
        Query(
            ans(x, p),
            Conjunction((filter_(x),)),
        ),
    ))
    assert res == expected


# ── Preprocessing: % comments ────────────────────────────────────────────────

def test_percent_comment_stripping():
    """% comments are stripped before parsing."""
    ans = Symbol("ans")
    body = Symbol("body")
    x = Symbol("x")
    res = parser("ans(x):-body(x) % this is a comment")
    expected = Union((
        Query(
            ans(x),
            Conjunction((body(x),)),
        ),
    ))
    assert res == expected


def test_percent_comment_full_line():
    """Full-line % comments are stripped."""
    ans = Symbol("ans")
    body = Symbol("body")
    x = Symbol("x")
    res = parser("% full line comment\nans(x):-body(x)")
    expected = Union((
        Query(
            ans(x),
            Conjunction((body(x),)),
        ),
    ))
    assert res == expected


# ── Preprocessing: trailing dots ─────────────────────────────────────────────

def test_trailing_dot_stripping():
    """Prolog-style trailing dots are stripped before parsing."""
    ans = Symbol("ans")
    body = Symbol("body")
    x = Symbol("x")
    res = parser("ans(x):-body(x).")
    expected = Union((
        Query(
            ans(x),
            Conjunction((body(x),)),
        ),
    ))
    assert res == expected


# ── Wildcard: _ ──────────────────────────────────────────────────────────────

def test_underscore_wildcard():
    """_ is treated as an anonymous wildcard (fresh Symbol like ...)."""
    res = parser("A(x):-B(x, _)")
    fresh_arg = res.formulas[0].antecedent.formulas[0].args[1]
    assert isinstance(fresh_arg, Symbol)
    assert res == Union((
        Implication(
            Symbol("A")(Symbol("x")),
            Conjunction((Symbol("B")(Symbol("x"), fresh_arg),)),
        ),
    ))


# ── AGGREGATE body syntax ──────────────────────────────────────────────────────

def test_agg_body_simple():
    """
    AGGREGATE[group](body @ count(var)) = result
    → Implication with AggregationApplication in head args.
    """
    res = parser(
        "study_count(r, c) :-"
        " AGGREGATE[r](reported_activation(s, x, y, z)"
        " & voxel_in_region(x, y, z, r) @ count(s)) = c"
    )
    fml = res.formulas[0]
    assert isinstance(fml, Implication)
    head_args = fml.consequent.args
    assert len(head_args) == 2
    assert head_args[0] == Symbol("r")
    assert isinstance(head_args[1], AggregationApplication)
    assert head_args[1].functor == Symbol("count")
    assert head_args[1].args == (Symbol("s"),)
    assert isinstance(fml.antecedent, Conjunction)
    assert len(fml.antecedent.formulas) == 2


def test_agg_body_empty_group():
    """AGGREGATE[()] — empty group desugars into fresh predicate + main rule."""
    res = parser("avg_w(m) :- AGGREGATE[()](weights(w) @ mean(w)) = m")
    assert len(res.formulas) == 2
    fresh_fml = res.formulas[0]
    main_fml = res.formulas[1]
    assert isinstance(fresh_fml, Implication)
    assert isinstance(main_fml, Implication)
    fresh_args = fresh_fml.consequent.args
    assert len(fresh_args) == 1
    assert fresh_args[0] == Symbol("mean")(Symbol("w"))


def test_agg_body_single_predicate():
    """AGGREGATE with a single body atom (no conjunction), empty group."""
    res = parser("max_v(m) :- AGGREGATE[()](value(v) @ max(v)) = m")
    assert len(res.formulas) == 2
    fresh_fml = res.formulas[0]
    assert isinstance(fresh_fml, Implication)
    fresh_args = fresh_fml.consequent.args
    assert len(fresh_args) == 1
    assert fresh_args[0] == Symbol("max")(Symbol("v"))
    assert isinstance(fresh_fml.antecedent, Conjunction)
    assert len(fresh_fml.antecedent.formulas) == 1


def test_agg_body_mixed_disjunction_refused():
    """AGGREGATE body does NOT support disjunction (;) —parser-level restriction."""
    with pytest.raises(Exception):
        parser(
            "bad(r, c) :-"
            " AGGREGATE[r](p(r) ; q(r) @ count(s)) = c"
        )


# ── CHOICE statement ────────────────────────────────────────────────────────────
# CHOICE syntax ({ }) is not yet implemented — tests removed pending
# the grammar+lexer changes required to support curly-brace terminals.
