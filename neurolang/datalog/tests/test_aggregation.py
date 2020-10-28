from neurolang.exceptions import ForbiddenUnstratifiedAggregation
from typing import AbstractSet

import pytest

from ...expression_walker import ExpressionBasicEvaluator
from ...expressions import (Constant, ExpressionBlock, NeuroLangException,
                            Symbol)
from ...type_system import Unknown
from .. import DatalogProgram, Fact, Implication
from ..aggregation import (AggregationApplication, Chase,
                           DatalogWithAggregationMixin,
                           TranslateToLogicWithAggregation)
from ..expressions import Union

S_ = Symbol
C_ = Constant
Imp_ = Implication
Fa_ = AggregationApplication
Eb_ = ExpressionBlock
U_ = Union
F_ = Fact


class Datalog(
    TranslateToLogicWithAggregation,
    DatalogWithAggregationMixin, DatalogProgram,
    ExpressionBasicEvaluator
):
    def function_sum(self, x: AbstractSet) -> Unknown:
        return sum(v for v in x)

    def function_sum2(self, x: AbstractSet, y: AbstractSet) -> Unknown:
        return sum(v + w for v, w in zip(x, y))

    def function_set_create(self, x: AbstractSet) -> Unknown:
        return frozenset(x)


def test_aggregation_parsing():
    dl = Datalog()

    P = S_('P')  # noqa: N806
    Q = S_('Q')  # noqa: N806
    x = S_('x')
    y = S_('y')

    edb = [
        F_(P(C_(i), C_(j)))
        for i in range(3)
        for j in range(3)
    ]

    code = U_(edb + [
        Imp_(Q(x, S_('sum')(y,)), P(x, y)),
    ])

    dl.walk(code)

    assert Q in dl.intensional_database()
    assert P in dl.extensional_database()

    with pytest.raises(NeuroLangException):
        dl.walk(U_([
            Imp_(Q(x, Fa_(C_(sum), (y,)), Fa_(C_(sum), (y,))), P(x, y)),
        ]))


def test_aggregation_non_stratified():
    P = S_('P')  # noqa: N806
    Q = S_('Q')  # noqa: N806
    x = S_('x')

    edb = [
        F_(P(C_(i)))
        for i in range(3)
    ]

    code = Eb_(edb + [
        Imp_(Q(Fa_(S_('sum'), (x,))), Q(x)),
    ])

    dl = Datalog()
    dl.walk(code)

    chase = Chase(dl)

    with pytest.raises(ForbiddenUnstratifiedAggregation):
        chase.build_chase_solution()


def test_aggregation_chase_no_grouping():

    P = S_('P')  # noqa: N806
    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    x = S_('x')

    edb = [
        F_(P(C_(i)))
        for i in range(3)
    ]

    code = Eb_(edb + [
        Imp_(Q(Fa_(S_('sum'), (x,))), P(x)),
    ])

    dl = Datalog()
    dl.walk(code)

    chase = Chase(dl)

    solution = chase.build_chase_solution()

    dl.add_extensional_predicate_from_tuples(R, {(3,)})

    res = dl.extensional_database()['R']

    assert solution[Q] == res


def test_aggregation_chase_no_grouping_2args():

    P = S_('P')  # noqa: N806
    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    x = S_('x')
    y = S_('y')

    edb = [
        F_(P(C_(i), C_(2 * i)))
        for i in range(3)
    ]

    code = U_(edb + [
        Imp_(Q(Fa_(S_('sum2'), (x, y))), P(x, y)),
    ])

    dl = Datalog()
    dl.walk(code)

    chase = Chase(dl)

    solution = chase.build_chase_solution()

    dl.add_extensional_predicate_from_tuples(R, {(9,)})

    res = dl.extensional_database()['R']

    assert solution[Q] == res


def test_aggregation_chase_single_grouping():
    dl = Datalog()

    P = S_('P')  # noqa: N806
    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    x = S_('x')
    y = S_('y')

    edb = [
        F_(P(C_(i), C_(i * j)))
        for i in range(3)
        for j in range(3)
    ]

    code = Eb_(edb + [
        Imp_(Q(x, Fa_(S_('sum'), (y,))), P(x, y)),
    ])

    dl.walk(code)

    chase = Chase(dl)

    solution = chase.build_chase_solution()

    dl.add_extensional_predicate_from_tuples(
        R,
        {
            (0, 0),
            (1, 3),
            (2, 6)
        }
    )
    res = dl.extensional_database()['R']

    assert solution[Q] == res


def test_aggregation_chase_single_grouping_muliple_columns():
    dl = Datalog()

    P = S_('P')  # noqa: N806
    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    x = S_('x')
    y = S_('y')
    z = S_('z')

    edb = [
        F_(P(C_(i), C_(i * j), C_(0)))
        for i in range(3)
        for j in range(3)
    ]

    code = Eb_(edb + [
        Imp_(Q(x, Fa_(S_('sum2'), (y, z))), P(x, y, z)),
    ])

    dl.walk(code)

    chase = Chase(dl)

    solution = chase.build_chase_solution()

    dl.add_extensional_predicate_from_tuples(
        R,
        {
            (0, 0),
            (1, 3),
            (2, 6)
        }
    )
    res = dl.extensional_database()['R']

    assert solution[Q] == res


def test_aggregation_set_creation():
    dl = Datalog()
    P = S_('P')
    Q = S_('Q')
    x = S_('x')
    y = S_('y')

    edb = tuple(
        F_(P(C_(0), C_(i)))
        for i in range(3)
    )

    code = Eb_(edb + (
        Imp_(
            Q(x, Fa_(S_('set_create'), (y,))),
            P(x, y)
        ),
    ))

    dl.walk(code)
    chase = Chase(dl)
    solution = chase.build_chase_solution()

    assert Q in solution
    assert set(solution[Q].value.unwrapped_iter()) == {
        (0, frozenset(i for i in range(3)))
    }


def test_aggregation_emptyset():
    dl = Datalog()

    P = S_('P')  # noqa: N806
    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    x = S_('x')
    y = S_('y')

    edb = [
        F_(P(C_(i), C_(i * j)))
        for i in range(3)
        for j in range(3)
    ] + [F_(R(C_(10)))]

    code = Eb_(edb + [
        Imp_(Q(x, Fa_(S_('sum'), (y,))), R(x) & P(x, y)),
    ])

    dl.walk(code)

    chase = Chase(dl)

    solution = chase.build_chase_solution()

    assert Q not in solution or solution[Q] == set()
