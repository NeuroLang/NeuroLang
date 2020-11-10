import pytest

from ...exceptions import ForbiddenDisjunctionError
from ...expression_walker import IdentityWalker
from ...expressions import MATMUL, Constant, FunctionApplication, Symbol
from ...logic import Implication
from ..cplogic.program import (
    CPLogicProgram,
    TranslateQueryBasedProbabilisticFactMixin,
)
from ..expressions import ProbabilisticPredicate
from ..query_resolution import QueryBasedProbFactToDetRule

P = Symbol("P")
Q = Symbol("Q")
x = Symbol("x")
y = Symbol("y")
p = Symbol("p")


class QueryBasedProbFactToDetRuleProgramTest(
    QueryBasedProbFactToDetRule,
    CPLogicProgram,
):
    pass


def test_query_based_pfact():
    """
    Translate rule
        P(x) : (p / 2) :- Q(x)
    to the two rules
        _f1_(_f2_, x) :- Q(x), _f2_ = (p / 2)
        P(x) : _f2_ :- _f1_(_f2_, x)

    """
    pfact = Implication(
        ProbabilisticPredicate(p / Constant(2), P(x)),
        Q(x, p),
    )
    program = QueryBasedProbFactToDetRuleProgramTest()
    program.walk(pfact)
    program_rules = list()
    for union in program.intensional_database().values():
        program_rules += union.formulas
    det_rule = next(
        formula
        for formula in program_rules
        if (
            isinstance(formula.consequent, FunctionApplication)
            # check _f1_
            and formula.consequent.functor.is_fresh
            # check _f2_ first arg in consequent
            and formula.consequent.args[0].is_fresh
            and formula.consequent.args[1:] == (x,)
        )
    )
    assert any(
        isinstance(formula.consequent, ProbabilisticPredicate)
        and formula.consequent.probability == det_rule.consequent.args[0]
        and formula.antecedent == det_rule.consequent
        and formula.consequent.body == P(x)
        for formula in program_rules
    )


def test_prevent_combination_of_query_based_and_set_based():
    pfact = Implication(
        ProbabilisticPredicate(p / Constant(2), P(x)),
        Q(x, p),
    )
    cpl = QueryBasedProbFactToDetRuleProgramTest()
    cpl.add_probabilistic_facts_from_tuples(P, [(0.2, "a")])
    with pytest.raises(ForbiddenDisjunctionError):
        cpl.walk(pfact)


class TestTranslateQueryBasedProbabilisticFact(
    TranslateQueryBasedProbabilisticFactMixin,
    IdentityWalker,
):
    pass


def test_translation_sugar_syntax():
    pfact = Implication(MATMUL(P, (p / Constant(2)))(x), Q(x, p))
    translator = TestTranslateQueryBasedProbabilisticFact()
    result = translator.walk(pfact)
    expected = Implication(
        ProbabilisticPredicate(p / Constant(2), P(x)), Q(x, p)
    )
    assert result == expected
