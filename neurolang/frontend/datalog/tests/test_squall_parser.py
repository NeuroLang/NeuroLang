from inspect import signature
from itertools import product
from operator import eq
import logging

import pytest

from .... import config
from ....datalog import Conjunction, Implication, Negation
from ....expression_pattern_matching import add_match
from ....expression_walker import ExpressionWalker, ReplaceExpressionWalker
from ....expressions import Constant, FunctionApplication, Symbol
from ....logic import (
    ExistentialPredicate,
    LogicOperator,
    NaryLogicOperator,
    UniversalPredicate
)
from ..squall import LogicSimplifier
from ..squall_syntax_lark import parser, SquallProgram
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
        ("'marseillaise'", lambda x: x(Constant('marseillaise'))),  # pylint: disable=W0108
        ("?x", lambda x:x(Symbol('x')))  # pylint: disable=W0108
    ]


@pytest.fixture(scope="module")
def verb1():
    return [
        ("plays", lambda x: Symbol("plays")(x)),  # pylint: disable=W0108
    ]


@pytest.fixture(scope="module")
def verb2():
    # Parser now resolves InvertedFunctionApplication at parse time.
    # _apply_ops pre-swaps inverse-verb args before the resolver reverses
    # them back, so the final IR has normal subject-first order.
    return [
        ("~sings", lambda x, y: Symbol("sings")(x, y))  # pylint: disable=W0108
    ]


@pytest.fixture(scope="module")
def verbs(verb1, verb2):
    return verb1 + verb2


def op_application(np):
    return lambda d: np(lambda y: d(y))  # pylint: disable=W0108


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
    # Parser now resolves InvertedFunctionApplication at parse time, so
    # expected IR uses plain FunctionApplication with reversed arguments.
    return [
        ("~author", lambda x, y: Symbol("author")(y, x)),
        ("~publication_year", lambda x, y: Symbol("publication_year")(y, x))
    ]


def lambda_simple(arg):
    return lambda x: arg(x)  # pylint: disable=W0108


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

    fresh = Symbol.fresh
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
    return lambda x: lambda d: (  # pylint: disable=W0108
        det(lambda y: ng2(x)(y))(d)  # pylint: disable=W0108
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
    from neurolang.frontend.datalog.squall import InvertedFunctionApplication

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
                                InvertedFunctionApplication(reports, (s, x, y, z)),
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


    # 'whose NG2 VP' must include the binary noun predicate in the body.
    #
    # 'define as published every person whose writer plays' should produce:
    # published(x) :- person(x), ∃y. writer(x, y) ∧ plays(y)
    # i.e. the 'writer' relation must appear in the output.
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
    """Test aggregation IR structure for 'every Max of the Quantity where ?i item_count per ?i'."""
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
    """Test: InvertedFunctionApplication reverses args when walked through mixin."""
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
    """Tilde-verb in a relative clause resolves to FunctionApplication with reversed argument order."""
    result = parser(
        "define as authored every Paper ?p that a Person ~author ?p."
    )
    assert isinstance(result, Implication), f"Expected Implication, got {type(result)}"

    # Walk the rule body and locate the author(...) atom.
    # Because the parser simplifier resolves InvertedFunctionApplication,
    # the IR should contain author(?p, fresh) — paper first, person second.
    body_atoms = []
    stack = [result.antecedent]
    while stack:
        expr = stack.pop()
        if isinstance(expr, FunctionApplication):
            body_atoms.append(expr)
        elif hasattr(expr, 'formulas'):
            stack.extend(expr.formulas)
        elif hasattr(expr, 'body'):
            stack.append(expr.body)

    author_atoms = [a for a in body_atoms if a.functor == Symbol("author")]
    assert author_atoms, (
        f"Expected author(...) in rule body, but none found.\n"
        f"Full IR: {repr(result)}"
    )
    atom = author_atoms[0]
    # author(?p, fresh) — paper (subject) first, person (object) second
    assert atom.args[0] == Symbol("p"), (
        f"Expected first arg to be ?p, got {atom.args[0]}"
    )
    assert isinstance(atom.args[1], Symbol), (
        f"Expected second arg to be a fresh Symbol, got {atom.args[1]}"
    )


def test_squall_conditioned_prior_produces_condition_node():
    """rule_body1_cond_prior returns an Implication with Condition antecedent."""
    from neurolang.probabilistic.expressions import Condition

    # Grammar: define as probably verb1 rule_body1_cond_prior
    # rule_body1_cond_prior: det ng1 CONDITIONED TO s
    # verb1 is intransitive (upper_identifier); s must be a full sentence (np vp)
    result = parser(
        "define as probably Published every Voxel conditioned to every Study activates."
    )
    assert isinstance(result, Implication), (
        f"Expected Implication, got {type(result).__name__}"
    )
    assert isinstance(result.antecedent, Condition), (
        f"Expected Condition in antecedent, got {type(result.antecedent).__name__}: "
        f"{repr(result.antecedent)}"
    )
    cond = result.antecedent
    # In the prior form (det ng1 CONDITIONED TO s):
    #   conditioned = the NP predicate (Voxel)
    #   conditioning = the sentence (Study activates)
    cond_syms = {s.name.lower() for s in cond.conditioned._symbols}
    ing_syms = {s.name.lower() for s in cond.conditioning._symbols}
    assert "voxel" in cond_syms, (
        f"Expected 'voxel' in conditioned (first arg), got: {repr(cond.conditioned)}"
    )
    assert "study" in ing_syms or "activates" in ing_syms, (
        f"Expected 'study' or 'activates' in conditioning (second arg), got: {repr(cond.conditioning)}"
    )


def test_squall_conditioned_posterior_produces_condition_node():
    """rule_body1_cond_posterior returns an Implication with Condition antecedent."""
    from neurolang.probabilistic.expressions import Condition

    # Grammar: define as probably verb1 rule_body1_cond_posterior
    # rule_body1_cond_posterior: s CONDITIONED TO det ng1
    # verb1 is intransitive (upper_identifier); s must be a full sentence (np vp)
    result = parser(
        "define as probably Published every Study activates conditioned to every Voxel."
    )
    assert isinstance(result, Implication), (
        f"Expected Implication, got {type(result).__name__}"
    )
    assert isinstance(result.antecedent, Condition), (
        f"Expected Condition in antecedent, got {type(result.antecedent).__name__}: "
        f"{repr(result.antecedent)}"
    )
    cond = result.antecedent
    # In the posterior form (s CONDITIONED TO det ng1):
    #   conditioned = the sentence (Study activates)
    #   conditioning = the NP predicate (Voxel)
    cond_syms = {s.name.lower() for s in cond.conditioned._symbols}
    ing_syms = {s.name.lower() for s in cond.conditioning._symbols}
    assert "study" in cond_syms or "activates" in cond_syms, (
        f"Expected 'study' or 'activates' in conditioned (first arg), got: {repr(cond.conditioned)}"
    )
    assert "voxel" in ing_syms, (
        f"Expected 'voxel' in conditioning (second arg), got: {repr(cond.conditioning)}"
    )


def test_rule_body1_cond_prior_tuple_var_info():
    """rule_body1_cond_prior unpacks tuple _var_info into separate head args."""
    from neurolang.probabilistic.expressions import Condition

    result = parser(
        "define as probably Published every Voxel (?x; ?y; ?z) "
        "conditioned to every Study activates."
    )
    assert isinstance(result, Implication), f"Expected Implication, got {type(result)}"
    assert isinstance(result.antecedent, Condition), (
        f"Expected Condition in antecedent, got {type(result.antecedent)}"
    )
    # head should have 3 args (x, y, z), not 1 arg that is a tuple
    head_args = result.consequent.args
    assert len(head_args) == 3, (
        f"Expected 3 head args for (?x;?y;?z), got {len(head_args)}: {head_args}"
    )
    from neurolang.expressions import Symbol
    assert all(isinstance(a, Symbol) for a in head_args), (
        f"All head args should be Symbols, got: {head_args}"
    )


def test_rule_op_marg_produces_prob_query_in_head():
    """'with probability … conditioned to …' produces ProbabilisticQuery(PROB,...) in head."""
    from neurolang.probabilistic.expressions import Condition, ProbabilisticQuery, PROB

    result = parser(
        "define as Published with probability every Voxel "
        "conditioned to every Study activates."
    )
    assert isinstance(result, Implication), f"Expected Implication, got {type(result)}"
    assert isinstance(result.antecedent, Condition), (
        f"Expected Condition in antecedent, got {type(result.antecedent)}"
    )
    # Head must contain a ProbabilisticQuery(PROB, ...) argument
    head_args = result.consequent.args
    prob_args = [a for a in head_args if isinstance(a, ProbabilisticQuery)
                 and a.functor == PROB]
    assert prob_args, (
        f"Expected ProbabilisticQuery(PROB, ...) in head args, got: {head_args}"
    )



def test_ng1_agg_npc_arbitrary_functor():
    """'every Custom_func of the Relation' uses Symbol as aggregation functor."""
    from neurolang.expressions import Symbol, FunctionApplication
    from neurolang.datalog.expressions import AggregationApplication

    # Non-builtin aggregation function: create_overlay is not in _AGG_FUNC_MAP
    result = parser(
        "define as Result every Create_overlay of the Prob_map."
    )
    assert isinstance(result, Implication), f"Expected Implication, got {type(result)}"
    head_args = result.consequent.args
    assert len(head_args) >= 1, f"Expected at least 1 head arg, got: {head_args}"
    # The arg should be a FunctionApplication with functor Symbol("create_overlay")
    agg_arg = head_args[0]
    assert isinstance(agg_arg, (FunctionApplication, AggregationApplication)), (
        f"Expected FunctionApplication/AggregationApplication, got {type(agg_arg)}"
    )
    assert agg_arg.functor == Symbol("create_overlay"), (
        f"Expected functor Symbol('create_overlay'), got {agg_arg.functor}"
    )


def test_det_every_agg_free_var_fallback():
    """Global aggregation: agg args are all free vars in npc body, not just one var."""
    from neurolang.expressions import FunctionApplication
    from neurolang.datalog.expressions import AggregationApplication

    # Prob_map is a binary relation (x, p) — both should be agg args
    result = parser(
        "define as Result every Create_overlay of the Prob_map."
    )
    assert isinstance(result, Implication), (
        f"Expected Implication, got {type(result)}"
    )
    head_args = result.consequent.args
    assert len(head_args) >= 1, f"Expected head args, got: {head_args}"
    agg_arg = head_args[0]
    assert isinstance(agg_arg, (FunctionApplication, AggregationApplication)), (
        f"Expected FunctionApplication/AggregationApplication, got {type(agg_arg)}"
    )
    # When Prob_map introduces a single variable, agg_arg.args should have >= 1 arg
    # (the free variable from the npc body)
    assert len(agg_arg.args) >= 1, (
        f"Expected at least 1 agg arg (free var from npc body), got: {agg_arg.args}"
    )


def test_rule_body2_cond_parses():
    """Test: define as … with probability every A conditioned to every B should parse."""
    from ....probabilistic.expressions import Condition, ProbabilisticQuery

    result = parser(
        "define as spread with probability every virus conditioned to every study."
    )
    assert isinstance(result, Implication), (
        f"Expected Implication, got {type(result)}"
    )
    body = result.antecedent
    assert isinstance(body, Condition), f"Expected Condition, got {type(body)}"
    head_args = result.consequent.args
    prob_args = [a for a in head_args if isinstance(a, ProbabilisticQuery)]
    assert prob_args, f"Expected ProbabilisticQuery in head args, got: {head_args}"


def test_rel_body_function_call_parses():
    """Euclidean body atom includes the subject variable alongside the two labels."""
    from ..squall_syntax_lark import parser
    from ....datalog import Implication
    from ....logic.expression_processing import extract_logic_free_variables

    result = parser(
        "squall define as Close every Person that euclidean(?x, ?y) holds."
    )
    rules = result if isinstance(result, list) else [result]
    implications = [r for r in rules if isinstance(r, Implication)]
    assert len(implications) == 1
    # The body should contain a FunctionApplication for euclidean
    body_str = str(implications[0].antecedent)
    assert "euclidean" in body_str.lower(), f"euclidean not in body: {body_str}"
    # subject var + ?x + ?y → at least 3 free variables in the antecedent
    free_vars = extract_logic_free_variables(implications[0].antecedent)
    assert len(free_vars) >= 3, (
        f"Expected >=3 free vars (subject + 2 labels), got {free_vars}"
    )


def test_rel_comp_variable_rhs_parses():
    """'that has an item_weight ?w greater than ?threshold' — variable RHS comparison."""
    from ..squall_syntax_lark import parser

    result = parser(
        "define as Heavy every Item "
        "that has an item_weight ?w greater than ?threshold."
    )
    rules = result if isinstance(result, list) else [result]
    implications = [r for r in rules if isinstance(r, Implication)]
    assert len(implications) == 1
    body_str = str(implications[0].antecedent)
    # Both ?w and ?threshold should appear as free variables in the body
    assert "threshold" in body_str, f"threshold not in body: {body_str}"


def test_anonymous_wildcard_in_nary_predicate():
    """'every Study that _ activates' — wildcard fills first arg of activates."""
    result = parser("define as HasActivation every Study that _ activates.")
    rules = result if isinstance(result, list) else [result]
    implications = [r for r in rules if isinstance(r, Implication)]
    assert len(implications) == 1


def test_two_anonymous_wildcards_get_distinct_symbols():
    """Two '_' labels in the same rule produce two distinct fresh symbols."""
    result = parser(
        "define as TwoCols every Study ?s that ~activates _ and ~reports _ ."
    )
    rules = result if isinstance(result, list) else [result]
    implications = [r for r in rules if isinstance(r, Implication)]
    assert len(implications) == 1
    body = implications[0].antecedent
    # Collect fresh Symbol args (generated from '_') across all body formulas.
    fresh_syms = []
    for formula in body.formulas:
        if hasattr(formula, "args"):
            for arg in formula.args:
                if isinstance(arg, Symbol) and arg.name.startswith("fresh_"):
                    fresh_syms.append(arg)
    # Each '_' should produce a distinct fresh symbol
    assert len(fresh_syms) >= 2, (
        f"Expected >=2 fresh symbols, got {fresh_syms}"
    )
    assert fresh_syms[0] is not fresh_syms[1], (
        "The two '_' wildcards must produce distinct symbols"
    )


def test_vpdo_explicit_prob_v1_accepts_variable():
    """'activates with probability ?p' should parse and produce ProbabilisticFact(Symbol('p'), ...)."""
    from neurolang.probabilistic.expressions import ProbabilisticFact

    result = parser(
        "define as Probable every Study that activates with probability ?p."
    )
    rules = result if isinstance(result, list) else [result]
    implications = [r for r in rules if isinstance(r, Implication)]
    assert len(implications) == 1
    # The ProbabilisticFact appears in the rule body (antecedent)
    body = implications[0].antecedent
    # Collect all subexpressions to find the ProbabilisticFact
    def find_pf(expr):
        if isinstance(expr, ProbabilisticFact):
            return expr
        if hasattr(expr, 'formulas'):
            for f in expr.formulas:
                pf = find_pf(f)
                if pf is not None:
                    return pf
        return None
    pf = find_pf(body)
    assert pf is not None, f"No ProbabilisticFact found in body: {body}"
    # Probability should be a Symbol (the label ?p)
    assert isinstance(pf.probability, Symbol), f"Expected Symbol, got {type(pf.probability)}"
    assert pf.probability.name == "p"


def test_rel_fun_call_tuple_subject_no_prepend():
    """
    rel_fun_call with a tuple-subject noun must NOT prepend the tuple as first arg.

    'every Focus_reported (?i2;?j2;?k2;?s) that is_near(?i1,?j1,?k1,?i2,?j2,?k2) holds'
    should emit is_near(i1,j1,k1,i2,j2,k2) — the six explicit labels, nothing extra.
    Specifically, both i1 and i2 must appear as free variables in the rule body.
    """
    from ....datalog import Implication
    from ....logic.expression_processing import extract_logic_free_variables

    result = parser(
        "squall define as Near every Focus_reported (?i2; ?j2; ?k2; ?s) "
        "that is_near(?i1, ?j1, ?k1, ?i2, ?j2, ?k2) holds."
    )
    rules = result if isinstance(result, list) else [result]
    implications = [r for r in rules if isinstance(r, Implication)]
    assert len(implications) == 1
    body_str = str(implications[0].antecedent)
    assert "is_near" in body_str.lower(), f"is_near not in body: {body_str}"
    free_vars = extract_logic_free_variables(implications[0].antecedent)
    var_names = {v.name for v in free_vars}
    assert "i1" in var_names, f"i1 missing from free vars {var_names}"
    assert "i2" in var_names, f"i2 missing from free vars {var_names}"
    # Confirm no extra args were prepended (tuple must not appear as first arg):
    # 6 explicit labels in is_near + ?s from the tuple subject → exactly 7 free variables expected
    assert len(free_vars) == 7, (
        f"Expected exactly 7 free vars (i1,j1,k1,i2,j2,k2,s), got {free_vars}"
    )


def test_anaphora_the_noun_resolves_to_quantifier_var():
    result = parser(
        "define as Test for every Region where a Study activates the Region."
    )
    assert isinstance(result, Implication)
    # Find the region(...) and activates(...) atoms in the body
    body_atoms = []
    stack = [result.antecedent]
    while stack:
        expr = stack.pop()
        if isinstance(expr, FunctionApplication):
            body_atoms.append(expr)
        elif hasattr(expr, 'formulas'):
            stack.extend(expr.formulas)
        elif hasattr(expr, 'body'):
            stack.append(expr.body)

    region_atom = next(a for a in body_atoms if a.functor == Symbol("region"))
    activates_atom = next(a for a in body_atoms if a.functor == Symbol("activates"))
    # The variable in region(r) should be the same as the second arg of activates(s, r)
    region_var = region_atom.args[0]
    activates_region_var = activates_atom.args[1]
    assert region_var == activates_region_var, (
        f"Expected anaphora resolution: region var {region_var} != activates arg {activates_region_var}"
    )


def test_anaphora_the_noun_with_nd_annotation_resolves():
    result = parser(
        "define as ActiveVoxel for every Voxel in 3D where a Study reports the Voxel."
    )
    assert isinstance(result, Implication)
    body_atoms = []
    stack = [result.antecedent]
    while stack:
        expr = stack.pop()
        if isinstance(expr, FunctionApplication):
            body_atoms.append(expr)
        elif hasattr(expr, 'formulas'):
            stack.extend(expr.formulas)
        elif hasattr(expr, 'body'):
            stack.append(expr.body)

    voxel_atom = next(a for a in body_atoms if a.functor == Symbol("voxel"))
    reports_atom = next(a for a in body_atoms if a.functor == Symbol("reports"))
    voxel_var = voxel_atom.args[0]
    reports_voxel_var = reports_atom.args[1]
    assert voxel_var == reports_voxel_var, (
        f"Expected ND anaphora resolution: voxel var {voxel_var} != reports arg {reports_voxel_var}"
    )


def test_anaphora_unbound_noun_creates_existential():
    result = parser(
        "obtain every Region that activates the Term."
    )
    assert isinstance(result, SquallProgram)
    q = result.queries[0]
    # Body should contain an existential for Term (not in scope)
    body_atoms = []
    stack = [q.body]
    while stack:
        expr = stack.pop()
        if isinstance(expr, FunctionApplication):
            body_atoms.append(expr)
        elif hasattr(expr, 'formulas'):
            stack.extend(expr.formulas)
        elif hasattr(expr, 'body'):
            stack.append(expr.body)

    term_atoms = [a for a in body_atoms if a.functor == Symbol("term")]
    assert len(term_atoms) >= 1, "Expected at least one term(...) atom from existential fallback"


def test_compound_quantifier_explicit_vars():
    result = parser(
        "define as Cooccurrence for every Region ?r and for every Term ?t "
        "where a Selected_study ?s activates ?r and mentions ?t."
    )
    assert isinstance(result, Implication)
    # Head should be cooccurrence(r, t)
    assert result.consequent.functor == Symbol("cooccurrence")
    assert len(result.consequent.args) == 2
    assert result.consequent.args[0] == Symbol("r")
    assert result.consequent.args[1] == Symbol("t")
    # Body should contain region(r), term(t), selected_study(s), activates(s, r), mentions(s, t)
    body_atoms = []
    stack = [result.antecedent]
    while stack:
        expr = stack.pop()
        if isinstance(expr, FunctionApplication):
            body_atoms.append(expr)
        elif hasattr(expr, 'formulas'):
            stack.extend(expr.formulas)
        elif hasattr(expr, 'body'):
            stack.append(expr.body)
    functors = {a.functor.name for a in body_atoms}
    assert "region" in functors
    assert "term" in functors
    assert "selected_study" in functors
    assert "activates" in functors
    assert "mentions" in functors


def test_compound_quantifier_marg():
    result = parser(
        "define as Joint_probability with inferred probability "
        "for every Region ?r and for every Term ?t "
        "where a Selected_study ?s activates ?r and mentions ?t."
    )
    assert isinstance(result, Implication)
    assert result.consequent.functor == Symbol("joint_probability")
    # Head should have 3 args: r, t, PROB(r, t)
    assert len(result.consequent.args) == 3
    from neurolang.probabilistic.expressions import ProbabilisticQuery, PROB
    prob_term = result.consequent.args[2]
    assert isinstance(prob_term, ProbabilisticQuery)
    assert prob_term.functor == PROB


def test_compound_quantifier_explicit_prob():
    from neurolang.probabilistic.expressions import ProbabilisticFact
    result = parser(
        "define as Weighted with probability 0.5 "
        "for every Region ?r and for every Term ?t "
        "where a Selected_study ?s activates ?r and mentions ?t."
    )
    assert isinstance(result, Implication)
    assert isinstance(result.consequent, ProbabilisticFact)
    assert result.consequent.probability == Constant(0.5)
    assert result.consequent.body.functor == Symbol("weighted")
    assert len(result.consequent.body.args) == 2


def test_obtain_unnamed_simple_predicate():
    from neurolang.frontend.datalog.squall_syntax_lark import SquallProgram
    from neurolang.expressions import Query, Symbol

    result = parser(
        "define as Active every Person that plays. "
        "obtain every Active."
    )
    assert isinstance(result, SquallProgram)
    assert len(result.queries) == 1
    q = result.queries[0]
    assert isinstance(q, Query)
    assert isinstance(q.head, Symbol), f"Expected Symbol head, got {type(q.head)}"
    assert "active" in repr(q.body).lower()


def test_obtain_unnamed_with_tuple_label_parses():
    from neurolang.frontend.datalog.squall_syntax_lark import SquallProgram
    from neurolang.expressions import Query

    result = parser(
        "define as term_prob with inferred probability "
        "for every Term that a Selected_study study_term. "
        "obtain every term_prob (?t; ?p)."
    )
    assert isinstance(result, SquallProgram)
    assert len(result.queries) == 1
    q = result.queries[0]
    assert isinstance(q, Query)
    assert isinstance(q.head, tuple), f"Expected tuple head, got {type(q.head)}"
    assert len(q.head) == 2, f"Expected 2 head vars, got {len(q.head)}"
    head_names = [h.name for h in q.head]
    assert "t" in head_names and "p" in head_names, f"Expected t and p in head, got {head_names}"


def test_probably_prefix_in_wlq_head_is_preserved():
    """
    Regression: probably_X in a rule head must not be stripped to X.

    Before the fix, terminal handlers unconditionally stripped the
    ``probably_`` prefix from all identifiers, including rule-head verbs.
    This caused ``define as probably_mentions ...`` to produce a rule for
    ``mentions`` instead of ``probably_mentions``, colliding with a
    deterministic ``mentions`` rule and causing the chase to hang.
    """
    from neurolang.frontend.datalog.squall_syntax_lark import SquallProgram
    from neurolang.probabilistic.expressions import ProbabilisticQuery

    program_text = (
        "define as mentions every Term_in_study (?s; ?t). "
        "define as probably_mentions with inferred probability "
        "for every Term that a Study mentions. "
        "obtain every probably_mentions."
    )
    result = parser(program_text)

    assert isinstance(result, SquallProgram), (
        f"Expected SquallProgram, got {type(result)}"
    )
    rules = result.rules
    # First rule: deterministic mentions
    det_rule = rules[0]
    assert det_rule.consequent.functor.name == "mentions", (
        f"Expected deterministic rule for 'mentions', "
        f"got '{det_rule.consequent.functor.name}'"
    )
    # Second rule: WLQ probably_mentions — head functor must be probably_mentions,
    # NOT 'mentions' (which would cause a collision with the deterministic rule).
    wlq_rule = rules[1]
    wlq_functor_name = wlq_rule.consequent.functor.name
    assert wlq_functor_name == "probably_mentions", (
        f"WLQ head functor was stripped to '{wlq_functor_name}'; "
        "expected 'probably_mentions'. This indicates the probably_ prefix is "
        "being incorrectly stripped from rule heads."
    )
    # The WLQ head must also carry a ProbabilisticQuery PROB argument.
    assert any(
        isinstance(arg, ProbabilisticQuery)
        for arg in wlq_rule.consequent.args
    ), "Expected a ProbabilisticQuery PROB arg in the WLQ head"


def test_full_squall_program_ir_structure():
    """
    The full 6-rule SQUALL program from the Datalog→SQUALL translation
    produces the correct IR structure for every rule type:

    1. rule_op_prob_agg (CBMA-style) — AggregationApplication + ProbabilisticFact
    2. probably + non-cond rule_body1 — ProbabilisticFact with fresh Symbol
    3. deterministic rule with where filter
    4. deterministic projection rule
    5. MARG with conditioned-to
    6. aggregation in head with AggregationApplication
    """
    from neurolang.probabilistic.expressions import (
        Condition, ProbabilisticFact, ProbabilisticQuery,
    )
    from neurolang.logic import Implication, Conjunction
    from neurolang.datalog.aggregation import AggregationApplication as AggApp

    program_text = """
define as Voxel_reported with a probability of
    the Proximity_indicator of the Peak_reported (?x2; ?y2; ?z2; ?s)
        for each ?x, ?y, ?z and for each ?s
        where (?x; ?y; ?z) is a Voxel
        and where EUCLIDEAN(?x, ?y, ?z, ?x2, ?y2, ?z2) is lower than 4.

define as probably Term_in_study every Term_in_study_tfidf (?term; _; ?study).

define as Term_association every Term_in_study (?term; ?study)
    where ?study is a Selected_study.

define as Activation every Voxel_reported (?x; ?y; ?z; ?s)
    where ?s is a Selected_study.

define as Activation_given_term with probability
    every Activation (?x; ?y; ?z)
    conditioned to every Term_association ?t that is 'emotion'.

define as Activation_given_term_image
    every Agg_create_region_overlay of the Activation_given_term (?x; ?y; ?z; ?p).

obtain every Activation_given_term_image (?result).
"""
    result = parser(program_text)
    assert isinstance(result, SquallProgram), (
        f"Expected SquallProgram, got {type(result)}"
    )
    rules = result.rules
    assert len(rules) == 6, f"Expected 6 rules, got {len(rules)}"
    assert len(result.queries) == 1, (
        f"Expected 1 obtain query, got {len(result.queries)}"
    )

    r1 = rules[0]
    assert isinstance(r1, Implication), f"Rule 1 expected Implication, got {type(r1)}"
    assert isinstance(r1.consequent, ProbabilisticFact)
    assert r1.consequent.functor.name == "voxel_reported"
    assert len(r1.consequent.args) == 4
    agg_prob = r1.consequent.probability
    assert isinstance(agg_prob, AggApp)
    assert agg_prob.functor.name == "proximity_indicator"
    body1 = r1.antecedent
    assert isinstance(body1, Conjunction)
    body1_str = str(body1)
    assert "voxel" in body1_str
    assert "peak_reported" in body1_str
    assert "EUCLIDEAN" in body1_str
    assert "lt" in body1_str

    r2 = rules[1]
    assert isinstance(r2, Implication)
    assert isinstance(r2.consequent, ProbabilisticFact)
    assert r2.consequent.probability.is_fresh
    assert r2.consequent.functor.name == "term_in_study"
    assert len(r2.consequent.args) == 2

    r3 = rules[2]
    assert isinstance(r3, Implication)
    assert not isinstance(r3.consequent, ProbabilisticFact)
    assert r3.consequent.functor.name == "term_association"
    assert "selected_study" in str(r3.antecedent)

    r4 = rules[3]
    assert isinstance(r4, Implication)
    assert not isinstance(r4.consequent, ProbabilisticFact)
    assert r4.consequent.functor.name == "activation"

    r5 = rules[4]
    assert isinstance(r5, Implication)
    assert isinstance(r5.antecedent, Condition)
    assert any(isinstance(a, ProbabilisticQuery) for a in r5.consequent.args)
    assert r5.consequent.functor.name == "activation_given_term"

    r6 = rules[5]
    assert isinstance(r6, Implication)
    has_agg_app = any(isinstance(a, AggApp) for a in r6.consequent.args)
    assert has_agg_app
    agg_app = next(a for a in r6.consequent.args if isinstance(a, AggApp))
    assert agg_app.functor.name == "agg_create_region_overlay"
    assert r6.consequent.functor.name == "activation_given_term_image"

    q = result.queries[0]
    assert q.body.functor.name == "activation_given_term_image"


def test_rel_fun_call_with_string_literal():
    """
    rel_fun_call with a string literal argument parses correctly.

    ``every Atlas_label ?label that startswith('L ') holds`` should emit
    a body atom ``startswith(label, 'L ')`` with the constant ``'L '``
    as the second argument.
    """
    from ....datalog import Implication
    from ....expressions import Constant, FunctionApplication

    result = parser(
        "define as Left_label every Atlas_label ?label "
        "that startswith('L ') holds."
    )
    assert isinstance(result, Implication)
    body = result.antecedent
    # Walk the body to find the startswith atom
    startswith_atom = None
    stack = [body]
    while stack:
        expr = stack.pop()
        if isinstance(expr, FunctionApplication) and expr.functor.name == "startswith":
            startswith_atom = expr
            break
        if hasattr(expr, 'formulas'):
            stack.extend(expr.formulas)
        elif hasattr(expr, 'body'):
            stack.append(expr.body)
    assert startswith_atom is not None, (
        f"No startswith atom found in body: {body}"
    )
    # Second arg should be a Constant with value 'L '
    second_arg = startswith_atom.args[1]
    assert isinstance(second_arg, Constant), (
        f"Expected Constant for string literal, got {type(second_arg)}: {second_arg}"
    )
    assert second_arg.value == 'L ', (
        f"Expected 'L ', got {second_arg.value!r}"
    )


def test_rel_fun_call_with_numeric_literal():
    """rel_fun_call with a numeric literal argument parses correctly."""
    from ....datalog import Implication
    from ....expressions import Constant, FunctionApplication

    result = parser(
        "define as Large every Item ?i that size_greater(42) holds."
    )
    assert isinstance(result, Implication)
    body = result.antecedent
    size_atom = None
    stack = [body]
    while stack:
        expr = stack.pop()
        if isinstance(expr, FunctionApplication) and expr.functor.name == "size_greater":
            size_atom = expr
            break
        if hasattr(expr, 'formulas'):
            stack.extend(expr.formulas)
        elif hasattr(expr, 'body'):
            stack.append(expr.body)
    assert size_atom is not None, f"No size_greater atom found in body: {body}"
    second_arg = size_atom.args[1]
    assert isinstance(second_arg, Constant), (
        f"Expected Constant for numeric literal, got {type(second_arg)}"
    )
    assert second_arg.value == 42


def test_command_parses_and_preserves_in_squall_program():
    """#set_backend('pandas') parses and is preserved in SquallProgram.commands."""
    from ..squall_syntax_lark import SquallProgram
    from ....expressions import FunctionApplication

    result = parser(
        "#set_backend('pandas').\n"
        "define as Active every person that plays.\n"
        "obtain every Active."
    )
    assert isinstance(result, SquallProgram), f"Expected SquallProgram, got {type(result)}"
    assert len(result.commands) == 1, f"Expected 1 command, got {len(result.commands)}"
    cmd = result.commands[0]
    assert isinstance(cmd, FunctionApplication), f"Expected FunctionApplication, got {type(cmd)}"
    assert cmd.functor.name == "set_backend", f"Expected 'set_backend', got {cmd.functor.name}"


def test_squall_predicate_call():
    """Bare predicate call (``Predicate (?x, ?y, ?z)``) in a ``define … where`` body."""
    from ....datalog import Implication
    from ....expressions import Constant, FunctionApplication, Symbol

    result = parser(
        "define as Bayes_factor (?x; ?y; ?bf)\n"
        "    where Joint_probability (?x, ?y, ?p_rt)\n"
        "    and Region_probability (?x, ?p_r)."
    )
    assert isinstance(result, Implication), f"Expected Implication, got {type(result)}"
    head = result.consequent
    assert head.functor == Symbol("bayes_factor"), f"Expected 'bayes_factor', got {head.functor}"
    assert len(head.args) == 3, f"Expected 3 head args, got {len(head.args)}: {head.args}"
    body = result.antecedent
    assert isinstance(body, Conjunction), f"Expected Conjunction body, got {type(body)}"
    fa_list = [f for f in body.formulas if isinstance(f, FunctionApplication)]
    assert len(fa_list) == 2, f"Expected 2 function applications in body, got {len(fa_list)}"
    pred_names = {f.functor.name for f in fa_list}
    assert pred_names == {"joint_probability", "region_probability"}, (
        f"Expected joint_probability and region_probability, got {pred_names}"
    )


def test_squall_predicate_call_with_literal():
    """Bare predicate call with a string literal argument."""
    from ....expressions import Constant, FunctionApplication, Symbol

    result = parser(
        "define as Filtered (?x)\n"
        "    where Region_probability (?x, ?p_r)\n"
        "    and ?x is 'target_region'."
    )
    assert isinstance(result, Implication)
    body = result.antecedent
    assert isinstance(body, Conjunction)
    atoms = [f for f in body.formulas if isinstance(f, FunctionApplication)]
    eq_atoms = [a for a in atoms if a.functor == EQ]
    assert eq_atoms, f"Expected equality atom in body, got {atoms}"
    eq_atom = eq_atoms[0]
    assert eq_atom.args[1] == Constant("target_region"), (
        f"Expected 'target_region' constant, got {eq_atom.args[1]}"
    )


def test_squall_arithmetic_assign():
    """Arithmetic expression (``?bf = expression``) in a rule body."""
    from ....datalog import Implication
    from ....expressions import Constant, FunctionApplication, Symbol
    from operator import truediv

    code = """
    define as Bayes_factor (?x; ?y; ?bf)
        where Joint_probability (?x, ?y, ?p_rt)
        and Region_probability (?x, ?p_r)
        and Term_probability (?y, ?p_t)
        and ?bf is (?p_rt / ?p_r) / ((?p_t - ?p_rt) / (1.0 - ?p_r)).
    """
    result = parser(code)
    assert isinstance(result, Implication), f"Expected Implication, got {type(result)}"
    head = result.consequent
    assert head.functor == Symbol("bayes_factor"), f"Expected 'bayes_factor', got {head.functor}"
    assert len(head.args) == 3, f"Expected 3 head args, got {len(head.args)}"
    # Body must contain an equality atom with truediv/sub arithmetic.
    body = result.antecedent
    assert isinstance(body, Conjunction), f"Expected Conjunction body, got {type(body)}"
    eq_atoms = [f for f in body.formulas
                if isinstance(f, FunctionApplication) and f.functor == EQ]
    assert eq_atoms, f"Expected at least one equality atom (assignment), got {body.formulas}"
    eq_atom = eq_atoms[0]
    rhs = eq_atom.args[1]
    assert isinstance(rhs, FunctionApplication), (
        f"Expected FunctionApplication RHS, got {type(rhs)}: {rhs}"
    )
    assert rhs.functor == Constant(truediv), (
        f"Expected truediv at outermost level of RHS, got {rhs.functor}"
    )
    assert isinstance(eq_atom.args[0], Symbol), f"Expected Symbol LHS, got {type(eq_atom.args[0])}"
    assert eq_atom.args[0].name.startswith("bf"), f"Expected 'bf' variable, got {eq_atom.args[0].name}"
    inner_ops = set()
    stack = [rhs]
    while stack:
        expr = stack.pop()
        if isinstance(expr, FunctionApplication):
            inner_ops.add(expr.functor)
            stack.extend(expr.args)
    assert Constant(truediv) in inner_ops, f"Expected truediv operator in {inner_ops}"


def test_squall_predicate_and_arithmetic_roundtrip():
    """End-to-end: define a rule with predicate calls + arithmetic, then execute against engine."""
    from ....datalog import Implication
    from ....expressions import Constant, FunctionApplication, Symbol
    from ....datalog import Conjunction
    from ....datalog.expressions import Fact
    from ....frontend import NeurolangPDL

    nl = NeurolangPDL()
    nl.add_tuple_set([(1, 10.0)], name="joint_probability")
    nl.add_tuple_set([(1, 5.0)], name="region_probability")
    nl.add_tuple_set([(1, 2.0)], name="term_probability")

    squall_code = """
    define as Bayes_factor (?x; ?y; ?bf)
        where Joint_probability (?x, ?y, ?p_rt)
        and Region_probability (?x, ?p_r)
        and Term_probability (?y, ?p_t)
        and ?bf is (?p_rt / ?p_r) / ((?p_t - ?p_rt) / (1.0 - ?p_r)).
    obtain every Bayes_factor (?x; ?y; ?bf) as BF.
    """
    # Parse and simplify
    parsed = parser(squall_code)
    assert isinstance(parsed, SquallProgram), f"Expected SquallProgram, got {type(parsed)}"
    # The define rule (bayes_factor) plus the obtain-as projection rule (bf)
    define_rules = [
        r for r in parsed.rules
        if r.consequent.functor.name == "bayes_factor"
    ]
    assert len(define_rules) == 1, (
        f"Expected 1 bayes_factor define rule, got {len(define_rules)}"
    )
    rule = define_rules[0]
    assert isinstance(rule, Implication), f"Expected Implication, got {type(rule)}"
    head = rule.consequent
    assert len(head.args) == 3, f"Expected 3 head args, got {len(head.args)}"
    body = rule.antecedent
    assert isinstance(body, Conjunction), f"Expected Conjunction, got {type(body)}"
    assert len(body.formulas) >= 4, (
        f"Expected at least 4 body formulas (3 pred + 1 eq), got {len(body.formulas)}"
    )


def test_squall_where_label_is_expr_does_not_conflict_with_where_s():
    """Existing ``?x is 'string'`` pattern (via s_np_vp) should not break."""
    from ....expressions import Constant, FunctionApplication

    result = parser(
        "obtain every Item (?r; ?x) where ?r is 'target' as Items."
    )
    assert isinstance(result, SquallProgram)
    q = result.queries[0]
    body = q.body
    eq_atoms = []
    stack = [body]
    while stack:
        expr = stack.pop()
        if isinstance(expr, FunctionApplication) and expr.functor == EQ:
            eq_atoms.append(expr)
        elif hasattr(expr, 'formulas'):
            stack.extend(expr.formulas)
        elif hasattr(expr, 'body'):
            stack.append(expr.body)
    assert eq_atoms, f"Expected equality atoms in body, got {body}"
    assert any(a.args[1] == Constant("target") for a in eq_atoms), (
        f"Expected 'target' constant in equality, got {[str(a) for a in eq_atoms]}"
    )
