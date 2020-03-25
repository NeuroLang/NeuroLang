import pytest

from ....exceptions import NeuroLangException
from ....expressions import Symbol, Constant, ExpressionBlock
from ....logic import Implication
from ....datalog import Fact
from ...expressions import DeltaTerm
from ...expression_processing import is_probabilistic_fact
from ..from_ppdl import PPDLToCPLogicTranslator
from ..program import CPLogicProgram


P = Symbol("P")
Q = Symbol("P")
Z = Symbol("P")
x = Symbol("x")
bernoulli = Symbol("bernoulli")
a = Constant("a")
b = Constant("b")


@pytest.mark.skip("translation should produce probabilistic choices")
def test_simple_program():
    probabilistic_rule = Implication(
        Q(x, DeltaTerm(bernoulli, (Constant(0.2),))), P(x)
    )
    deterministic_rule = Implication(Z(x), P(x))
    program = ExpressionBlock(
        (probabilistic_rule, deterministic_rule, Fact(P(a)), Fact(P(b)))
    )
    translator = PPDLToCPLogicTranslator()
    translated = translator.walk(program)
    matching_exps = [
        exp for exp in translated.expressions if is_probabilistic_fact(exp)
    ]
    assert len(matching_exps) == 1
    probfact = matching_exps[0]
    assert x in probfact.consequent.body.args
    assert probfact.consequent.probability == Constant(0.2)
    program = CPLogicProgram()
    program.walk(translated)
    assert deterministic_rule in program.intensional_database()[Z].formulas
    with pytest.raises(NeuroLangException, match=r".*bernoulli.*"):
        bad_rule = Implication(
            Q(x, DeltaTerm(Symbol("unknown_distrib"), tuple())), P(x)
        )
        translator = PPDLToCPLogicTranslator()
        translator.walk(bad_rule)
