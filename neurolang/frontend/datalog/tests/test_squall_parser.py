from inspect import signature
from itertools import product
from operator import add, eq, mul, pow, sub, truediv
import logging

import pytest

from .... import config
from ....datalog import Conjunction, Fact, Implication, Negation, Union
from ....datalog.aggregation import AggregationApplication
from ....expression_pattern_matching import add_match
from ....expression_walker import ExpressionWalker, ReplaceExpressionWalker
from ....expressions import Constant, Query, Symbol
from ....logic import (
    ExistentialPredicate,
    LogicOperator,
    NaryLogicOperator,
    UniversalPredicate
)
from ....probabilistic.expressions import ProbabilisticPredicate
from ..squall import InvertedFunctionApplication, LogicSimplifier
from ..squall_syntax_lark import parser
from ..standard_syntax import ExternalSymbol
from ...probabilistic_frontend import RegionFrontendCPLogicSolver, Chase


LOGGER = logging.getLogger()


config.disable_expression_type_printing()


EQ = Constant(eq)


class LogicWeakEquivalence(ExpressionWalker):
    @add_match(EQ(UniversalPredicate, UniversalPredicate))
    def eq_universal_predicates(self, expression):
        left, right = expression.args
        if left.head != right.head:
            new_head = Symbol.fresh()
            rew = ReplaceExpressionWalker(
                {left.head: new_head, right.head: new_head}
            )
            left = rew.walk(left)
            right = rew.walk(right)

        return self.walk(EQ(left.body, right.body))

    @add_match(EQ(ExistentialPredicate, ExistentialPredicate))
    def eq_existential_predicates(self, expression):
        return self.eq_universal_predicates(expression)

    @add_match(
        EQ(NaryLogicOperator, NaryLogicOperator),
        lambda exp: (
            len(exp.args[0].formulas) == len(exp.args[1].formulas) and
            type(exp.args[0]) == type(exp.args[1])
        )
    )
    def n_ary_sort(self, expression):
        left, right = expression.args

        return all(
            self.walk(EQ(a1, a2))
            for a1, a2 in zip(
                sorted(left.formulas, key=repr),
                sorted(right.formulas, key=repr)
            )
        )

    @add_match(EQ(LogicOperator, LogicOperator))
    def eq_logic_operator(self, expression):
        left, right = expression.args
        for l, r in zip(left.unapply(), right.unapply()):
            if isinstance(l, tuple) and isinstance(r, tuple):
                return (
                    (len(l) == len(r)) and
                    all(
                        self.walk(EQ(ll, rr))
                        for ll, rr in zip(l, r)
                    )
                )
            else:
                return self.walk(EQ(l, r))

    @add_match(EQ(..., ...))
    def eq_expression(self, expression):
        return expression.args[0] == expression.args[1]


def weak_logic_eq(left, right):
    left = LogicSimplifier().walk(left)
    right = LogicSimplifier().walk(right)
    return LogicWeakEquivalence().walk(EQ(left, right))


@pytest.fixture(scope="module")
def nouns():
    return [
        ("'marseillaise'", lambda x: x(Constant('marseillaise'))),
        ("?x", lambda x:x(Symbol('x')))
    ]


@pytest.fixture(scope="module")
def verb1():
    return [
        ("plays", lambda x: Symbol("plays")(x)),
    ]


@pytest.fixture(scope="module")
def verb2():
    return [
        ("~sings", lambda x, y: InvertedFunctionApplication(Symbol("sings"), (x, y)))
    ]


@pytest.fixture(scope="module")
def verbs(verb1, verb2):
    return verb1 + verb2


def op_application(np):
    return lambda d: np(lambda y: d(y))


@pytest.fixture(scope="module")
def op(noun_phrases):
    return [
        (np[0], op_application(np[1]))
        for np in noun_phrases
    ]


def vp_op_application(op, v2):
    return lambda x: op(lambda y: v2(x, y))


@pytest.fixture(scope="module")
def verb_phrases_do_op(verbs, op):
    return [
        (f"{v2[0]} {op[0]}", vp_op_application(op[1], v2[1]))
        for v2, op in product(verbs, op)
        if is_transitive(v2[1])
    ]


@pytest.fixture(scope="module")
def nouns_1():
    return [
        ("person", Symbol("person")),
        ("country", Symbol("country"))
    ]


@pytest.fixture(scope="module")
def nouns_2():
    return [
        ("~author", lambda x, y: InvertedFunctionApplication(Symbol("author"), (x, y))),
        ("~publication_year", lambda x, y: InvertedFunctionApplication(Symbol("publication_year"), (x, y)))
    ]


def lambda_simple(arg):
    return lambda x: arg(x)


def lambda_conjunction(*args):
    return lambda x: Conjunction(tuple(a(x) for a in args))


def lambda_conjunction_2(arg2, *args):
    return lambda x: lambda y: Conjunction(
        tuple(a(x) for a in args) + (arg2(x, y),)
    )


@pytest.fixture(scope="module")
def noun_groups_1(nouns_1):
    return [
        (f"{n1[0]}", lambda_conjunction(n1[1]))
        for n1 in nouns_1
    ]


@pytest.fixture(scope="module")
def noun_groups_2(nouns_2):
    return [
        (f"{n2[0]}", lambda_conjunction_2(n2[1]))
        for n2 in nouns_2
    ]



@pytest.fixture(scope="module")
def det():
    res = []

    fresh = lambda: Symbol.fresh()
    res.append((
        "every",
        lambda d1: lambda d2: (
            (
                lambda x: UniversalPredicate(x, Implication(d2(x), d1(x))))
                (fresh()
            )
        )
    ))

    for det in ("a", "an", "some"):
        res.append((
            det,
            lambda d1: lambda d2: (
                (lambda x: ExistentialPredicate(x, Conjunction((d1(x), d2(x)))))(fresh())
            )
        ))

    res.append((
        "no",
        lambda d1: lambda d2: (
            (lambda x: Negation(ExistentialPredicate(x, Conjunction((d1(x), d2(x))))))(fresh())
        )
    ))
    return res


@pytest.fixture(scope="module")
def noun_phrase_quantified_1(det, noun_groups_1):
    res = []
    for det_, ng1 in product(det, noun_groups_1):
        exp = det_[1](ng1[1])
        res.append((
            f"{det_[0]} {ng1[0]}", lambda_simple(exp)
        ))
    return res


def np_np2_composition(np, np2):
    return lambda d: np(lambda x: (np2(x)(d)))


@pytest.fixture(scope="module")
def noun_phrase_quantified_2(noun_phrase_2, nouns):
    res = []
    for np2, np in product(noun_phrase_2, nouns):
        exp = np_np2_composition(np[1], np2[1])
        res.append((f"{np2[0]} of {np[0]}", exp))
    return res


def det_ng2_composition(det, ng2):
    return lambda x: lambda d: (
        det(lambda y: ng2(x)(y))(d)
    )


@pytest.fixture(scope="module")
def noun_phrase_2(det, noun_groups_2):
    res = []
    for det_, ng2 in product(det, noun_groups_2):
        exp = det_ng2_composition(det_[1], ng2[1])
        res.append((
            f"{det_[0]} {ng2[0]}", exp
        ))
    return res


@pytest.fixture(scope="module")
def verb_phrases(verbs, verb_phrases_do_op):
    return [
        v for v in verbs if not is_transitive(v[1])
    ] + verb_phrases_do_op


@pytest.fixture(scope="module")
def noun_phrases(nouns, noun_phrase_quantified_1, noun_phrase_quantified_2):
    return nouns + noun_phrase_quantified_1 + noun_phrase_quantified_2 


@pytest.fixture(scope="module")
def s_np_vp(noun_phrases, verb_phrases):
    return [
        (f"{np[0]} {vp[0]}", np[1](vp[1]))
        for np, vp in product(noun_phrases, verb_phrases)
    ]


@pytest.fixture(scope="module")
def s_for(s_np_vp, noun_phrases):
    res = list(s_np_vp)
    for _ in range(1):
        res += [
            (f"for {np[0]}, {s[0]}", np[1](lambda x: s[1]))
            for np, s in product(noun_phrases, res)
        ]
    res = res[len(s_np_vp):]
    return res


@pytest.fixture(scope="module")
def s(s_np_vp, s_for):
    return s_np_vp + s_for


@pytest.mark.slow
def test_squall_s(s):
    for query, expected in s:
        LOGGER.log(logging.INFO + 1, "Query to test %s", query)
        res = parser(f"squall {query}")
        assert weak_logic_eq(res, expected)


def is_transitive(verb):
    return len(signature(verb).parameters) > 1


def test_squall_simple_np_nv(noun_phrases, verb_phrases):
    query_result_pairs = []
    for np, vp in product(noun_phrases, verb_phrases):
        query = f"{np[0]} {vp[0]}"
        result = np[1](vp[1])
        query_result_pairs.append((query, result))

    for query, expected in query_result_pairs:
        res = parser(f"squall {query}")
        assert weak_logic_eq(res, expected)


def test_squall_quantified_np_nv(noun_phrase_quantified_1, verb_phrases):
    query_result_pairs = []
    for np, vp in product(noun_phrase_quantified_1, verb_phrases):
        query_result_pairs.append(
            (f"{np[0]} {vp[0]}", np[1](vp[1]))
        )

    for query, expected in query_result_pairs:
        res = parser(f"squall {query}")
        assert weak_logic_eq(res, expected)


def test_squall_quantified_np2_nv(noun_phrase_quantified_2, verb_phrases):
    query_result_pairs = []
    for np, vp in product(noun_phrase_quantified_2, verb_phrases):
        query_result_pairs.append(
            (f"{np[0]} {vp[0]}", np[1](vp[1]))
        )

    for query, expected in query_result_pairs:
        res = parser(f"squall {query}")
        assert weak_logic_eq(res, expected)


def test_squall_quantified_np_nv2(
    noun_phrase_quantified_1, verb2, noun_phrases
):
    query_result_pairs = []
    for npq, vp, np in product(
        noun_phrase_quantified_1, verb2, noun_phrases
    ):
        vp_ = lambda x: np[1](lambda y: vp[1](x, y))
        query_result_pairs.append(
            (f"{npq[0]} {vp[0]} {np[0]}", npq[1](vp_))
        )

    for query, expected in query_result_pairs:
        res = parser(f"squall {query}")
        assert weak_logic_eq(res, expected)


def test_squall_voxel_activation():
    query = "every voxel (?x; ?y; ?z) that a study ?s ~reports activates"
    res = parser(f"squall {query}")
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    s = Symbol("s")
    voxel = Symbol("voxel")
    study = Symbol("study")
    reports = Symbol("reports")
    activates = Symbol("activates")
    expected = UniversalPredicate(
        z,
        UniversalPredicate(
            y,
            UniversalPredicate(
                x,
                Implication(
                    activates(x, y, z),
                    Conjunction((
                        voxel(x, y, z),
                        ExistentialPredicate(
                            s,
                            Conjunction((
                                reports(s, x, y, z),
                                study(s)
                            ))
                        )

                    ))
                )
            )
        )
    )

    assert weak_logic_eq(res, expected)


def test_squall():
    # res = parser("squall ?t1 runs")
    # res = parser("squall every woman sees a man")
    # res = parser("squall a woman that eats sees a man that sleeps")
    res = parser("squall ?s reports")
    parser("squall every voxel that a study ~reports is an activation")
    parser("squall every voxel that 'neuro study' ~reports activates")
    parser("squall every voxel that a study that ~mentions a word that is 'auditory' ~reports, activates")
    parser("squall every voxel that a study that ~reports no region ~reports activates")
    # parser("squall every voxel that a study --that ~reports no region-- ~reports activates")

    assert res


@pytest.fixture
def datalog_simple():
    datalog = RegionFrontendCPLogicSolver()

    datalog.add_extensional_predicate_from_tuples(
        Symbol("item"),
        [('a',), ('b',), ('c',), ('d',)]
    )

    datalog.add_extensional_predicate_from_tuples(
        Symbol("item_count"),
        [('a', 0), ('a', 1), ('b', 2), ('c', 3)]
    )

    datalog.add_extensional_predicate_from_tuples(
        Symbol("quantity"),
        [(i,) for i in range(5)]
    )
    return datalog


def test_lark_semantics_item_selection(datalog_simple):
    code = (
        "define as Large every Item "
        "that has an item_count greater equal than 2."
    )
    logic_code = parser(code)
    datalog_simple.walk(logic_code)
    chase = Chase(datalog_simple)
    solution = chase.build_chase_solution()['large'].value
    expected = set([('b',), ('c',)])

    assert solution == expected


def test_lark_semantics_join(datalog_simple):
    code = (
        "define as merge for every Item ?i ;"
        " with every Quantity that ?i item_count"
    )
    logic_code = parser(code)
    datalog_simple.walk(logic_code)
    chase = Chase(datalog_simple)
    solution = chase.build_chase_solution()['merge'].value
    expected = set([('a', 0), ('a', 1), ('b', 2), ('c', 3)])

    assert solution == expected


def test_squall_obtain_produces_squall_program():
    """'obtain' clause produces a SquallProgram with a Query expression."""
    from neurolang.frontend.datalog.squall_syntax_lark import SquallProgram
    from neurolang.expressions import Query

    prog = parser("obtain every item that plays.")
    assert isinstance(prog, SquallProgram)
    assert len(prog.queries) == 1
    assert isinstance(prog.queries[0], Query), (
        f"Expected Query, got {type(prog.queries[0]).__name__}: {repr(prog.queries[0])}"
    )
    q = prog.queries[0]
    assert "item" in repr(q)
    assert "plays" in repr(q)


def test_squall_obtain_executes_against_engine(datalog_simple):
    """'obtain every Item that has item_count' returns correct results."""
    from neurolang.frontend.datalog.squall_syntax_lark import SquallProgram
    from neurolang.expressions import Query
    from neurolang.datalog import Implication

    prog = parser("obtain every Item that has an item_count.")
    assert isinstance(prog, SquallProgram)
    assert len(prog.queries) == 1
    q = prog.queries[0]
    assert isinstance(q, Query)

    # Execute by converting Query(x, body) → head(x) :- body
    head_sym = Symbol.fresh()
    datalog_simple.walk(Implication(head_sym(q.head), q.body))
    chase = Chase(datalog_simple)
    sol = chase.build_chase_solution()
    # Items with an item_count: a, b, c (not d)
    result = sol[head_sym].value
    assert result == {('a',), ('b',), ('c',)}


    """'whose NG2 VP' must include the binary noun predicate in the body.

    'define as published every person whose writer plays' should produce:
    published(x) :- person(x), ∃y. writer(x, y) ∧ plays(y)
    i.e. the 'writer' relation must appear in the output.
    """
    res = parser("define as published every person whose writer plays.")
    assert "writer" in repr(res), (
        f"rel_ng2: binary noun 'writer' missing from output: {repr(res)}"
    )
    assert "plays" in repr(res)
    assert "person" in repr(res)


def test_squall_rel_ng2_whose_with_object():
    """'whose NG2 VP' with transitive VP."""
    res = parser("define as notable every person whose writer ~creates a book.")
    assert "writer" in repr(res)
    assert "creates" in repr(res)


def test_squall_aggregation_ir():
    """'every Max of the Quantity where ?i item_count per ?i' produces
    Implication with AggregationApplication(max, (q,)) in the consequent
    and item_count(i, q) in the antecedent."""
    from neurolang.datalog.aggregation import AggregationApplication as AA
    code = (
        "define as max_items for every Item ?i ; "
        "where every Max of the Quantity where ?i item_count per ?i."
    )
    rule = parser(code)
    assert isinstance(rule, Implication), f"Expected Implication, got {type(rule)}"

    # The consequent must include an AggregationApplication argument
    consequent = rule.consequent
    agg_args = [
        a for a in consequent.args if isinstance(a, AA)
    ]
    assert agg_args, (
        f"Expected AggregationApplication in consequent args, got: {consequent.args}"
    )
    agg = agg_args[0]
    # The functor should be max
    assert agg.functor == Constant(max), (
        f"Expected Constant(max) functor, got {agg.functor}"
    )

    # The antecedent must contain item_count somewhere
    antecedent_repr = repr(rule.antecedent)
    assert "item_count" in antecedent_repr, (
        f"Expected item_count in antecedent, got: {antecedent_repr}"
    )
    assert "item" in antecedent_repr


def test_lark_semantics_aggregation(datalog_simple):
    code = """
        define as max_items for every Item ?i ;
            where every Max of the Quantity where ?i item_count per ?i.
    """
    logic_code = parser(code)

    datalog_simple.walk(logic_code)
    chase = Chase(datalog_simple)
    solution = chase.build_chase_solution()["max_items"].value
    expected = set([('a', 1), ('b', 2), ('c', 3)])
    assert solution == expected


def test_inverted_function_application_ir_node():
    """InvertedFunctionApplication reverses args when walked through mixin."""
    from neurolang.frontend.datalog.squall import (
        InvertedFunctionApplication,
        ResolveInvertedFunctionApplicationMixin,
    )
    from neurolang.expression_walker import ExpressionWalker

    class _Resolver(ResolveInvertedFunctionApplicationMixin, ExpressionWalker):
        pass

    f = Symbol("reports")
    s = Symbol("s")
    x = Symbol("x")
    # Surface order: (s, x) — after resolution should become (x, s)
    inv = InvertedFunctionApplication(f, (s, x))
    result = _Resolver().walk(inv)
    assert result == f(x, s), f"Expected reports(x, s), got {result}"

    # Three-arg full reversal: (a, b, c) → (c, b, a)
    a = Symbol("a")
    b = Symbol("b")
    c = Symbol("c")
    inv3 = InvertedFunctionApplication(f, (a, b, c))
    result3 = _Resolver().walk(inv3)
    assert result3 == f(c, b, a), f"Expected reports(c, b, a), got {result3}"


def test_squall_transitive_inv_argument_order():
    """~verb in a relative clause produces InvertedFunctionApplication in the IR."""
    from neurolang.frontend.datalog.squall import InvertedFunctionApplication

    # "define as authored every Paper ?p that a Person ~author ?p."
    # The ~author relative clause: a Person ~author [the paper]
    # transitive_inv fires → _InverseVerbSymbol(author)
    # Surface call: author(person_var, paper_var) → InvertedFunctionApplication
    result = parser(
        "define as authored every Paper ?p that a Person ~author ?p."
    )
    assert isinstance(result, Implication), f"Expected Implication, got {type(result)}"

    # Recursive search for InvertedFunctionApplication nodes (with cycle guard)
    import neurolang.frontend.datalog.squall as sq

    found = []
    seen = set()

    def _search(obj):
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        if isinstance(obj, sq.InvertedFunctionApplication):
            found.append(obj)
        if hasattr(obj, '__dict__'):
            for v in obj.__dict__.values():
                if hasattr(v, '__iter__') and not isinstance(v, (str, bytes)):
                    try:
                        for item in v:
                            _search(item)
                    except TypeError:
                        pass
                elif hasattr(v, '__dict__'):
                    _search(v)

    _search(result)
    assert found, (
        f"Expected InvertedFunctionApplication in IR, but none found.\n"
        f"Full IR: {repr(result)}"
    )
    assert any(n.functor == Symbol("author") for n in found), (
        f"Expected InvertedFunctionApplication with functor 'author', got: {found}"
    )
