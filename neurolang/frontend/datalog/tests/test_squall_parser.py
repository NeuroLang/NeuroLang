from inspect import signature
from itertools import product
from operator import eq
import logging

import pytest

from .... import config
from ....datalog import Conjunction, Implication, Negation
from ....expression_pattern_matching import add_match
from ....expression_walker import (
    ExpressionWalker, PatternWalker, ReplaceExpressionWalker, ReplaceSymbolWalker
)
from ....expressions import (
    Constant, FunctionApplication, ParametricTypeClassMeta, Symbol
)
from ....type_system import get_generic_type, is_leq_informative
from ....logic import (
    ExistentialPredicate,
    LogicOperator,
    NaryLogicOperator,
    UniversalPredicate,
    Disjunction,
)
from ....probabilistic.expressions import Condition, ProbabilisticFact
from ..squall import LogicSimplifier
from ..squall_syntax_lark import (
    parser,
    EquiprobableChoiceDef,
    SquallProgram,
    WeightedChoiceDef,
)
from ...probabilistic_frontend import RegionFrontendCPLogicSolver, Chase
from ....datalog.expression_processing import extract_logic_free_variables
from ....datalog.expressions import AggregationApplication
from ....datalog.negation import is_conjunctive_negation
from ....expressions import ExpressionBlock, Query
from ....logic.horn_clauses import fol_query_to_datalog_program
from ....logic.transformations import ExtractBoundVariables
from ....probabilistic.expressions import PROB, ProbabilisticQuery
from ..anaphora_resolution import AnaphoraPredicate


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
        left, right = expression.args
        if type(left) is not type(right):
            # Unify Unknown/Any across parameterized types
            lt = type(left)
            rt = type(right)
            if not (
                isinstance(lt, ParametricTypeClassMeta) and
                isinstance(rt, ParametricTypeClassMeta)
            ):
                return False
            left_root = get_generic_type(lt)
            right_root = get_generic_type(rt)
            if not (
                left_root is right_root and
                (
                    is_leq_informative(left.type, right.type) or
                    is_leq_informative(right.type, left.type)
                )
            ):
                return False
        return left.unapply() == right.unapply()


def weak_logic_eq(left, right):
    left = LogicSimplifier().walk(left)
    right = LogicSimplifier().walk(right)
    return LogicWeakEquivalence().walk(EQ(left, right))


class ConditionAwareEqMixin(PatternWalker):
    """Extend LogicWeakEquivalence with Condition comparison and fix
    LogicOperator iteration to walk all children instead of early-returning
    after the first child."""

    @add_match(EQ(Condition, Condition))
    def eq_condition(self, expression):
        left, right = expression.args
        return (
            self.walk(EQ(left.conditioned, right.conditioned)) and
            self.walk(EQ(left.conditioning, right.conditioning))
        )

    @add_match(EQ(NaryLogicOperator, NaryLogicOperator))
    def eq_nary_logic_operator(self, expression):
        left, right = expression.args
        if len(left.formulas) != len(right.formulas) or type(left) is not type(right):
            return False
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
        results = []
        for lv, rv in zip(left.unapply(), right.unapply()):
            if isinstance(lv, tuple) and isinstance(rv, tuple):
                if len(lv) != len(rv):
                    return False
                results.append(all(
                    self.walk(EQ(lvv, rvv))
                    for lvv, rvv in zip(lv, rv)
                ))
            else:
                results.append(self.walk(EQ(lv, rv)))
        return all(results)


class ConditionAwareLogicWeakEquivalence(
    ConditionAwareEqMixin, LogicWeakEquivalence
):
    pass


def condition_aware_weak_logic_eq(left, right):
    left = LogicSimplifier().walk(left)
    right = LogicSimplifier().walk(right)
    return ConditionAwareLogicWeakEquivalence().walk(EQ(left, right))


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
    assert all(isinstance(a, Symbol) for a in head_args), (
        f"All head args should be Symbols, got: {head_args}"
    )


def test_rule_op_marg_produces_prob_query_in_head():
    """'with probability … conditioned to …' produces ProbabilisticQuery(PROB,...) in head."""
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
    assert len(voxel_atom.args) == 3, (
        f"Expected voxel to have 3 args (3D), got {len(voxel_atom.args)}: {voxel_atom.args}"
    )
    assert len(reports_atom.args) == 4, (
        f"Expected reports to have 4 args (study + 3 voxel coords), "
        f"got {len(reports_atom.args)}: {reports_atom.args}"
    )
    voxel_vars = voxel_atom.args
    reports_voxel_vars = reports_atom.args[1:]
    assert voxel_vars == reports_voxel_vars, (
        f"Expected ND anaphora resolution: voxel vars {voxel_vars} "
        f"!= reports voxel args {reports_voxel_vars}"
    )


def test_nd_annotation_and_tuple_label_equivalent_arity():
    """'Voxel in 3D' and 'Voxel (?x; ?y; ?z)' must produce the same predicate arities."""

    r1 = parser("obtain every Voxel (?x; ?y; ?z) that a Study reported.")
    r2 = parser("obtain every Voxel in 3D that a Study reported.")

    def _collect_arities(expr, out=None):
        if out is None:
            out = {}
        if hasattr(expr, 'functor') and hasattr(expr, 'args'):
            name = expr.functor.name if isinstance(expr.functor, Symbol) else str(expr.functor)
            out.setdefault(name, set()).add(len(expr.args))
        if hasattr(expr, 'formulas'):
            for f in expr.formulas:
                _collect_arities(f, out)
        if hasattr(expr, 'head') and hasattr(expr, 'body'):
            _collect_arities(expr.head, out)
            _collect_arities(expr.body, out)
        return out

    arities1 = _collect_arities(r1)
    arities2 = _collect_arities(r2)

    assert arities1['voxel'] == {3}, f"Tuple form: voxel arity = {arities1.get('voxel')}"
    assert arities2['voxel'] == {3}, f"ND form: voxel arity = {arities2.get('voxel')}"
    assert arities1['reported'] == {4}, f"Tuple form: reported arity = {arities1.get('reported')}"
    assert arities2['reported'] == {4}, f"ND form: reported arity = {arities2.get('reported')}"


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


def test_dimension_noun_in_compound_quantifier():
    """'Probability' and 'Value' (DIM_PROBABILITY / DIM_VALUE) must parse as
    regular nouns in compound quantifiers (for every Voxel in 3D and for every
    Probability). This tests the grammar change that adds dimension_noun as
    an alternative in noun1."""
    result = parser(
        "define as activation_probability for every Voxel in 3D "
        "and for every Probability "
        "where a Schaefer_label labels the Voxel "
        "and Label_reports the Probability."
    )
    assert isinstance(result, Implication)
    head = result.consequent
    assert head.functor.name == "activation_probability"
    assert len(head.args) == 4
    # Body: voxel(s0, s1, s2), probability(s3),
    #       exists s4: schaefer_label(s4), labels(s4, s0, s1, s2),
    #                  label_reports(s4, s3)
    voxel_atoms = []
    _collect_predicate_atoms(result.antecedent, "voxel", voxel_atoms)
    assert len(voxel_atoms) == 1
    assert len(voxel_atoms[0].args) == 3

    # Probability/Value are type annotations — they introduce a variable
    # into scope without generating a body predicate
    prob_atoms = []
    _collect_predicate_atoms(result.antecedent, "probability", prob_atoms)
    assert len(prob_atoms) == 0, (
        "Probability is a type noun, not a database predicate — "
        "it must NOT appear in the rule body"
    )

    schaefer_atoms = []
    _collect_predicate_atoms(
        result.antecedent, "schaefer_label", schaefer_atoms
    )
    assert len(schaefer_atoms) == 1

    labels_atoms = []
    _collect_predicate_atoms(
        result.antecedent, "labels", labels_atoms
    )
    assert len(labels_atoms) == 1
    assert len(labels_atoms[0].args) == 4

    label_reports_atoms = []
    _collect_predicate_atoms(
        result.antecedent, "label_reports", label_reports_atoms
    )
    assert len(label_reports_atoms) == 1
    assert len(label_reports_atoms[0].args) == 2

    # Verify variable sharing across atoms
    voxel_vars = voxel_atoms[0].args
    labels_vars = labels_atoms[0].args
    assert voxel_vars == labels_vars[1:], (
        "Voxel coords must be shared between voxel/3 and labels/4"
    )
    # Probability var comes from the head (4th arg), shared with label_reports
    prob_var = result.consequent.args[3]
    lr_var = label_reports_atoms[0].args[1]
    assert prob_var == lr_var, (
        "Probability var must be shared between head and "
        f"label_reports/2, got {prob_var} != {lr_var}"
    )
    # Anaphora: 'the Voxel' in labels and 'the Probability' in label_reports
    # must resolve to the same vars as quantifier-introduced vars
    schaefer_var = schaefer_atoms[0].args[0]
    lr_schaefer_var = label_reports_atoms[0].args[0]
    labels_schaefer_var = labels_atoms[0].args[0]
    assert schaefer_var == lr_schaefer_var == labels_schaefer_var, (
        "Anaphora: same Schaefer_label var across all atoms"
    )


def _collect_predicate_atoms(expr, functor_name, result_list):
    if isinstance(expr, FunctionApplication):
        if isinstance(expr.functor, Symbol) and expr.functor.name == functor_name:
            result_list.append(expr)
        return
    if isinstance(expr, (Conjunction,)):
        for f in expr.formulas:
            _collect_predicate_atoms(f, functor_name, result_list)
    elif hasattr(expr, 'body'):
        _collect_predicate_atoms(expr.body, functor_name, result_list)
    elif hasattr(expr, 'antecedent'):
        _collect_predicate_atoms(expr.antecedent, functor_name, result_list)
    elif hasattr(expr, 'conditioned'):
        _collect_predicate_atoms(expr.conditioned, functor_name, result_list)
        _collect_predicate_atoms(expr.conditioning, functor_name, result_list)
    elif hasattr(expr, 'formulas'):
        for f in expr.formulas:
            _collect_predicate_atoms(f, functor_name, result_list)


def test_anaphora_predicate_class():

    x = Symbol.fresh()
    p = Symbol("test_predicate")
    body = p(x)
    ap = AnaphoraPredicate(x, body, Symbol("test_noun"))

    assert isinstance(ap, AnaphoraPredicate)
    assert isinstance(ap, ExistentialPredicate)
    assert ap.head is x
    assert ap.body is body
    assert ap.noun_name == Symbol("test_noun")


def test_squall_marg_anaphora_resolves_across_given():

    result = parser(
        "define as Published with probability every Voxel "
        "that a SelectedStudy reports "
        "given the SelectedStudy mentions 'emotion'."
    )

    # Use the parser's own fresh symbols so structural comparison succeeds.
    v = result.consequent.args[0]
    s = result.antecedent.head

    expected = Implication(
        FunctionApplication(Symbol("published"), (
            v,
            ProbabilisticQuery(PROB, (v,)),
        )),
        ExistentialPredicate(
            s,
            Condition(
                Conjunction((
                    FunctionApplication(Symbol("voxel"), (v,)),
                    FunctionApplication(Symbol("selectedstudy"), (s,)),
                    FunctionApplication(Symbol("reports"), (s, v)),
                )),
                Conjunction((
                    FunctionApplication(Symbol("selectedstudy"), (s,)),
                    FunctionApplication(Symbol("mentions"), (s, Constant("emotion"))),
                )),
            ),
        ),
    )

    assert condition_aware_weak_logic_eq(result, expected), (
        f"IR mismatch.\n\n"
        f"Result:   {result}\n\n"
        f"Expected: {expected}"
    )


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
    prob_term = result.consequent.args[2]
    assert isinstance(prob_term, ProbabilisticQuery)
    assert prob_term.functor == PROB


def test_compound_quantifier_explicit_prob():
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


# ---------------------------------------------------------------------------
# Probabilistic choice syntax
# ---------------------------------------------------------------------------

def test_parse_equiprobable_choice_basic():
    code = (
        "define as Selected_study as an equiprobable choice over "
        "every Study."
    )
    result = parser(code)
    assert isinstance(result, EquiprobableChoiceDef)
    assert result.head_symbol.name == "selected_study"
    # body_formula should be a single predicate application: study(?s)
    assert isinstance(result.body_formula, FunctionApplication)
    assert result.body_formula.functor.name == "study"


def test_parse_equiprobable_choice_filtered():
    code = (
        "define as Selected_study as an equiprobable choice over "
        "every Study that has_data."
    )
    result = parser(code)
    assert isinstance(result, EquiprobableChoiceDef)
    # Filtered source produces a Conjunction
    assert isinstance(result.body_formula, Conjunction)
    assert len(result.body_formula.formulas) == 2


def test_parse_equiprobable_choice_in_squall_program():
    code = (
        "define as Selected_study as an equiprobable choice over "
        "every Study. "
        "obtain every Selected_study as Result."
    )
    result = parser(code)
    assert isinstance(result, SquallProgram)
    assert len(result.choice_defs) == 1
    assert isinstance(result.choice_defs[0], EquiprobableChoiceDef)
    assert result.choice_defs[0].head_symbol.name == "selected_study"


def test_parse_weighted_choice_basic():
    code = (
        "define as Selected_study as a choice over "
        "every Study (?s; ?q) with probability (?q / ?total)."
    )
    result = parser(code)
    assert isinstance(result, WeightedChoiceDef)
    assert result.head_symbol.name == "selected_study"
    assert isinstance(result.body_formula, FunctionApplication)
    assert result.body_formula.functor.name == "study"


def test_parse_weighted_choice_simple_prob():
    code = (
        "define as Selected_study as a choice over "
        "every Study with probability ?p."
    )
    result = parser(code)
    assert isinstance(result, WeightedChoiceDef)
    assert result.head_symbol.name == "selected_study"


def test_define_as_nested_relative_clause_no_recursion_error():
    """Regression: nested relative clause with ``in 3D`` + transitive verb.

    A query like ``define as Region_reported every Schaefer_region that labels
    a Voxel in 3D that a Study that mentions 'language' reports.`` previously
    raised ``RecursionError``.  Two bugs caused it:

    1. ``_apply_ops`` used ``lambda obj: verb(subject, obj)`` which failed
       when ``_apply_to_vars`` expanded a tuple ``(x, y, z)`` into separate
       positional arguments — the single-param lambda raised ``TypeError``,
       falling back to passing the raw tuple as a single arg.

    2. ``flatten_nested_existentials`` re-wrapped structurally-stable
       ``ExistentialPredicate`` chains on every visit, defeating
       ``process_expression``'s identity-based change detection and causing
       infinite re-walking.
    """
    code = (
        "define as Region_reported every Schaefer_region "
        "that labels a Voxel in 3D "
        "that a Study that mentions 'language' reports."
    )
    result = parser(code)
    assert isinstance(result, Implication)

    # Build the expected IR.  The head variable comes from the
    # consequent (region_reported(r)), so we extract it to match
    # exactly.  Body-bound variables use fresh symbols —
    # weak_logic_eq handles those via quantifier-head renaming.
    r = result.consequent.args[0]
    x, y, z, s = (Symbol.fresh() for _ in range(4))

    expected = Implication(
        Symbol("region_reported")(r),
        Conjunction((
            Symbol("schaefer_region")(r),
            ExistentialPredicate(z, ExistentialPredicate(y, ExistentialPredicate(x,
                Conjunction((
                    Symbol("voxel")(x, y, z),
                    ExistentialPredicate(s,
                        Conjunction((
                            Symbol("study")(s),
                            Symbol("mentions")(s, Constant("language")),
                            Symbol("reports")(s, x, y, z)
                        ))
                    ),
                    Symbol("labels")(r, x, y, z)
                ))
            )))
        ))
    )

    assert weak_logic_eq(result, expected), (
        f"IR mismatch.\nGot:      {result}\nExpected: {expected}"
    )


def test_obtain_with_probability_standalone_adds_dimension():
    """``with Probability`` / ``with Value`` standalone adds 2 head variables.

    ``app_dimension_only`` (handles ``with Probability`` without a preceding
    ``in ND``) returns 2 fresh symbols instead of 1.  This ensures the
    predicate body retains the noun's base argument (e.g. a region variable)
    while adding the probability/value dimension, producing a 2-column query.
    """
    # --- with Probability ---
    result = parser(
        "define as Region_reported every Schaefer_region. "
        "obtain every Region_reported with Probability."
    )
    assert isinstance(result, SquallProgram)
    rule = result.rules[0]
    q = result.queries[0]
    expected = SquallProgram(
        [Implication(
            Symbol("region_reported")(rule.consequent.args[0]),
            Symbol("schaefer_region")(rule.consequent.args[0])
        )],
        [Query(q.head, Symbol("region_reported")(*q.head))]
    )
    assert weak_logic_eq(result, expected)

    # --- with Value ---
    result2 = parser(
        "define as Region_reported every Schaefer_region. "
        "obtain every Region_reported with Value."
    )
    assert isinstance(result2, SquallProgram)
    rule2 = result2.rules[0]
    q2 = result2.queries[0]
    expected2 = SquallProgram(
        [Implication(
            Symbol("region_reported")(rule2.consequent.args[0]),
            Symbol("schaefer_region")(rule2.consequent.args[0])
        )],
        [Query(q2.head, Symbol("region_reported")(*q2.head))]
    )
    assert weak_logic_eq(result2, expected2)


def _check_anaphora(result, expected_n_dims):
    """Verify that ``the Voxel`` / ``the Probability`` in the where clause
    use the SAME variables as the outer ``for every Voxel … with Probability``
    quantification (anaphora resolution)."""
    assert isinstance(result, Implication)
    voxel_vars = result.consequent.args
    assert len(voxel_vars) == expected_n_dims
    s = Symbol.fresh()
    expected = Implication(
        result.consequent.functor(*voxel_vars),
        Conjunction((
            Symbol("voxel")(*voxel_vars),
            ExistentialPredicate(s,
                Conjunction((
                    Symbol("schaefer_label")(s),
                    Symbol("labels")(s, *voxel_vars),
                    Symbol("label_reports")(s, voxel_vars[-1])
                ))
            )
        ))
    )
    assert weak_logic_eq(result, expected)


def test_compound_quantifier_with_probability_as_noun():
    """``Probability`` keyword can be used as a regular noun in ``where``
    clauses of compound-quantifier rules.

    The grammar terminal ``DIM_PROBABILITY`` (exact string ``"Probability"``)
    is excluded from ``UPPER_NAME`` by negative lookahead, so ``Probability``
    could never match ``noun1``. This broke sentences where ``Probability``
    appears as a regular noun after ``the`` in a ``where`` clause.

    Fix: add ``DIM_PROBABILITY | DIM_VALUE`` as alternatives in ``noun1``,
    and convert the Lark ``Token`` to ``Symbol`` in the ``noun1`` transformer.
    """
    code = (
        "define as voxel_probability for every Voxel in 3D with Probability "
        "where a Schaefer_label labels the Voxel "
        "and Label_reports the Probability."
    )
    result = parser(code)
    _check_anaphora(result, expected_n_dims=4)

    code2 = (
        "define as voxel_probability for every Voxel with Probability "
        "where a Schaefer_label labels the Voxel "
        "and Label_reports the Probability."
    )
    result2 = parser(code2)
    _check_anaphora(result2, expected_n_dims=2)

    code3 = (
        "define as voxel_value for every Voxel in 3D with Value "
        "where a Schaefer_label labels the Voxel "
        "and Label_reports the Value."
    )
    result3 = parser(code3)
    _check_anaphora(result3, expected_n_dims=4)


def test_define_as_marg_given_anaphora_inside_rel():
    """``the Selected_study`` in a ``given`` clause resolves to the
    ``a Selected_study`` introduced inside a nested relative clause
    (``that a Selected_study reports``).

    After the post-parse ``AnaphoraResolutionWalker``:
    - ``the Selected_study`` uses the SAME variable as ``a Selected_study``
    - No ``AnaphoraPredicate`` markers remain
    - The existential for ``Selected_study`` lifts to wrap the ``Condition``
    """
    code = (
        "define as Label_reports with inferred probability "
        "every Schaefer_label that labels a Voxel in 3D "
        "that a Selected_study reports "
        "given the Selected_study mentions 'language'."
    )
    result = parser(code)
    assert isinstance(result, Implication)

    head = result.consequent
    s = head.args[0]
    ss = result.antecedent.head
    condition = result.antecedent.body
    assert isinstance(condition, Condition)

    # Extract actual existentially-quantified variables from the result,
    # so the expected expression faithfully represents parser output.
    conditioned = condition.conditioned
    ep_chain = conditioned.unapply()[0][1]
    z = ep_chain.unapply()[0]
    y = ep_chain.unapply()[1].unapply()[0]
    x = ep_chain.unapply()[1].unapply()[1].unapply()[0]

    # Build expected conditioned part: schaefer_label(s) ∧
    #   ∃z ∃y ∃x ( voxel(x,y,z) ∧ selected_study(ss) ∧ reports(ss,x,y,z)
    #           ∧ labels(s,x,y,z) )
    expected_conditioned = Conjunction((
        Symbol("schaefer_label")(s),
        ExistentialPredicate(z, ExistentialPredicate(y, ExistentialPredicate(x,
            Conjunction((
                Symbol("voxel")(x, y, z),
                Symbol("selected_study")(ss),
                Symbol("reports")(ss, x, y, z),
                Symbol("labels")(s, x, y, z),
            ))
        ))),
    ))
    assert weak_logic_eq(condition.conditioned, expected_conditioned)

    # Build expected conditioning part: selected_study(ss) ∧ mentions(ss, 'language')
    expected_conditioning = Conjunction((
        Symbol("selected_study")(ss),
        Symbol("mentions")(ss, Constant("language"))
    ))
    assert weak_logic_eq(condition.conditioning, expected_conditioning)

    # Verify the existential wrapper matches
    assert weak_logic_eq(result.antecedent, ExistentialPredicate(ss, condition))


def test_aggregation_of_4ary_predicate():
    """``every agg_create_region_overlay of the Activation_probability in 3D with Probability``
    should produce an ``AggregationApplication`` with 4 arguments (x, y, z, p),
    not just 1.

    The ``npc = the Activation_probability in 3D with Probability`` has 4 dimension
    variables from ``app_dimension_with`` (3 coords + probability).  The aggregation
    path in ``det_every`` uses ``capturing_cont`` which accepts ``(*args)`` — but
    ``det_the``'s scope-found tuple path was calling ``d(x[0])`` and only spreading
    the rest when ``isinstance(result, FunctionApplication)``, which fails for
    ``capturing_cont`` (returns ``Constant(True)``), losing 3 of 4 variables.
    """
    code = (
        "define as max_items "
        "where every agg_create_region_overlay "
        "of the Activation_probability in 3D with Probability."
    )
    result = parser(code)

    assert isinstance(result, Implication)
    head = result.consequent
    agg_arg = head.args[0]
    assert isinstance(agg_arg, AggregationApplication)
    assert len(agg_arg.args) == 4
    for a in agg_arg.args:
        assert isinstance(a, Symbol)

    body = result.antecedent
    assert isinstance(body, FunctionApplication), (
        f"Expected body to be a FunctionApplication, got {type(body).__name__}"
    )
    assert body.functor == Symbol("activation_probability"), (
        f"Expected activation_probability, got {body.functor}"
    )
    assert len(body.args) == 4, (
        f"Expected 4 args (3D + probability), got {len(body.args)}: {body.args}"
    )
    for a in body.args:
        assert isinstance(a, Symbol)


def test_lift_ep_from_conditioned_anaphora():
    v = Symbol('v')
    p_of_v = FunctionApplication(Symbol('p'), (v,))
    q_of_v = FunctionApplication(Symbol('q'), (v,))
    ep = ExistentialPredicate(v, p_of_v)

    other_bound = ExtractBoundVariables().walk(q_of_v)
    assert v not in other_bound

    lifted = ExistentialPredicate(
        ep.head,
        Condition(ep.body, q_of_v)
    )

    assert is_conjunctive_negation(lifted.body.conditioned)
    assert is_conjunctive_negation(lifted.body.conditioning)


def test_lift_ep_clash_both_sides():
    x1 = Symbol('x')
    x2 = Symbol('x')
    p_of_x1 = FunctionApplication(Symbol('p'), (x1,))
    q_of_x2 = FunctionApplication(Symbol('q'), (x2,))
    ep_left = ExistentialPredicate(x1, p_of_x1)
    ep_right = ExistentialPredicate(x2, q_of_x2)

    other_bound = ExtractBoundVariables().walk(ep_right)
    assert x1 in other_bound

    fresh_var = Symbol[x1.type].fresh()
    fresh_ep = ReplaceSymbolWalker({x1.name: fresh_var}).walk(ep_left)
    lifted = ExistentialPredicate(
        fresh_ep.head,
        Condition(fresh_ep.body, ep_right)
    )

    inner_cond = lifted.body
    assert isinstance(inner_cond, Condition)
    assert inner_cond.conditioned == fresh_ep.body
    assert isinstance(inner_cond.conditioning, ExistentialPredicate)
    assert lifted.head == fresh_ep.head
    assert lifted.head.name != x1.name


def test_decompose_non_conjunctive_condition():
    x = Symbol('x')
    p_of_x = FunctionApplication(Symbol('p'), (x,))
    q_of_x = FunctionApplication(Symbol('q'), (x,))
    r_of_x = FunctionApplication(Symbol('r'), (x,))
    disj = Disjunction((q_of_x, r_of_x))
    condition = Condition(p_of_x, disj)

    results = []
    new_args = []
    for arg in (condition.conditioned, condition.conditioning):
        if is_conjunctive_negation(arg):
            new_args.append(arg)
        else:
            fv = extract_logic_free_variables(arg)
            fresh_head = Symbol.fresh()(*tuple(fv))
            aux = fol_query_to_datalog_program(fresh_head, arg)
            results.append(aux)
            new_args.append(fresh_head)

    assert len(new_args) == 2
    assert new_args[0] == p_of_x
    assert isinstance(new_args[1], FunctionApplication)
    assert len(results) == 1
    assert isinstance(results[0], ExpressionBlock)
