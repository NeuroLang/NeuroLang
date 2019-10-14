import pytest
from typing import AbstractSet, Mapping

from ...exceptions import NeuroLangException
from ...expressions import Symbol, Constant, ExpressionBlock
from ...datalog.expressions import Implication
from ..probdatalog import ProbFact
from ..probdatalog_bn import (
    TranslatorGroundedProbDatalogToBN, pfact_cpd_factory
)
from ..distributions import TableDistribution

shop = Symbol('shop')
bought = Symbol('bought')
john = Constant[str]('john')
mary = Constant[str]('mary')
spaghetti = Constant[str]('spaghetti')
fish = Constant[str]('fish')

shopping_program_code = ExpressionBlock([
    ProbFact(Constant[float](0.2), shop(john)),
    ProbFact(Constant[float](0.6), shop(mary)),
    Implication(bought(spaghetti), shop(john)),
    Implication(bought(fish), shop(mary)),
])


def test_probdatalog_bn_translation():
    translator = TranslatorGroundedProbDatalogToBN()
    bn = translator.walk(shopping_program_code)
    expected_rv_symbols = Constant(
        frozenset({
            Symbol('shop(john)'),
            Symbol('shop(mary)'),
            Symbol('bought(spaghetti)'),
            Symbol('bought(fish)'),
            Symbol('c_1'),
            Symbol('c_2'),
        })
    )
    assert bn.random_variables == expected_rv_symbols
    # TODO re-enable type inference when #197 is fixed
    expected_edges = Constant[Mapping](
        {
            Symbol('shop(john)'): frozenset({Symbol('c_1')}),
            Symbol('shop(mary)'): frozenset({Symbol('c_2')}),
            Symbol('bought(spaghetti)'): frozenset({Symbol('shop(john)')}),
            Symbol('bought(fish)'): frozenset({Symbol('shop(mary)')}),
        },
        auto_infer_type=False,
    )
    assert bn.edges == expected_edges
    assert bn.rv_to_cpd_factory.value[Symbol('shop(john)')](
        Constant({Symbol('c_1'): Constant(1)}, auto_infer_type=False)
    ) == Constant(TableDistribution({
        0: 0.0,
        1: 1.0,
    }))
    assert bn.rv_to_cpd_factory.value[Symbol('shop(john)')](
        Constant({Symbol('c_1'): Constant(0)}, auto_infer_type=False)
    ) == Constant(TableDistribution({
        0: 1.0,
        1: 0.0,
    }))
    assert bn.rv_to_cpd_factory.value[Symbol('shop(mary)')](
        Constant({Symbol('c_2'): Constant(1)}, auto_infer_type=False)
    ) == Constant(TableDistribution({
        0: 0.0,
        1: 1.0,
    }))
    assert bn.rv_to_cpd_factory.value[Symbol('c_1')](Constant({})) == Constant(
        TableDistribution({
            0: 0.8,
            1: 0.2
        })
    )


def test_bn_translation_unique_rv_symbols():
    translator = TranslatorGroundedProbDatalogToBN()
    translator.walk(ExpressionBlock([]))
    translator._add_choice_variable(
        Symbol('c'), TableDistribution({
            0: 0,
            1: 1
        })
    )
    with pytest.raises(NeuroLangException):
        translator._add_choice_variable(
            Symbol('c'), TableDistribution({
                0: 0,
                1: 1
            })
        )


def test_forbid_parents_for_choice_vars():
    with pytest.raises(NeuroLangException):
        factory = pfact_cpd_factory(ProbFact(Constant(0.2), shop(john)))
        factory(Constant({'x': 4}, auto_infer_type=False))
