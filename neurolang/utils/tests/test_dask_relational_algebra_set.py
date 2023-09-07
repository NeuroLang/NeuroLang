import typing
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import pytest
from neurolang.type_system import Unknown
from neurolang.utils.relational_algebra_set.dask_helpers import (
    DaskContextManager,
    try_to_infer_type_of_operation,
)
from neurolang.utils.relational_algebra_set.dask_sql import (
    NamedRelationalAlgebraFrozenSet,
    RelationalAlgebraFrozenSet,
    RelationalAlgebraStringExpression,
)


def test_dask_context_manager_is_singleton():
    with pytest.raises(
        TypeError,
        match=r"Can't instantiate abstract class DaskContextManager"
        " with abstract methods _do_not_instantiate_singleton_class",
    ):
        DaskContextManager()
    ctx1 = DaskContextManager.get_context()
    from neurolang.utils.relational_algebra_set.dask_helpers import (
        DaskContextManager as DCM,
    )

    ctx2 = DCM.get_context()
    assert ctx1 is ctx2
    ctx3 = DaskContextManager.get_context(new=True)
    assert ctx3 is not ctx1


def test_set_init():
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    ras_a = RelationalAlgebraFrozenSet(a)
    ras_b = RelationalAlgebraFrozenSet(b)
    assert not ras_a.is_empty()
    assert not ras_b.is_empty()


def test_named_set_init():
    assert not NamedRelationalAlgebraFrozenSet.dee().is_empty()
    assert NamedRelationalAlgebraFrozenSet.dum().is_empty()
    assert NamedRelationalAlgebraFrozenSet(iterable=[]).is_empty()
    assert NamedRelationalAlgebraFrozenSet(
        columns=("x",), iterable=[]
    ).is_empty()
    a = [(i, i * 2) for i in range(5)]
    ras_a = NamedRelationalAlgebraFrozenSet(("x", "y"), a)
    assert not ras_a.is_empty()
    assert ras_a.columns == ("x", "y")


def test_set_length():
    assert len(RelationalAlgebraFrozenSet.dee()) == 1
    assert len(RelationalAlgebraFrozenSet.dum()) == 0
    assert len(RelationalAlgebraFrozenSet([])) == 0
    ras_a = RelationalAlgebraFrozenSet([(i, i * 2) for i in range(5)])
    assert len(ras_a) == 5
    assert len(ras_a - ras_a) == 0


def test_fetch_one():
    assert RelationalAlgebraFrozenSet.dee().fetch_one() == tuple()
    assert RelationalAlgebraFrozenSet.dum().fetch_one() is None
    assert RelationalAlgebraFrozenSet([]).fetch_one() is None

    a = [(i, i * 2) for i in range(5)]
    ras_a = RelationalAlgebraFrozenSet(a)
    assert ras_a.fetch_one() in a
    assert (ras_a - ras_a).fetch_one() is None


def test_named_fetch_one():
    assert NamedRelationalAlgebraFrozenSet.dee().fetch_one() == tuple()
    assert NamedRelationalAlgebraFrozenSet.dum().fetch_one() is None
    assert NamedRelationalAlgebraFrozenSet(("x",), []).fetch_one() is None

    a = [(i, i * 2) for i in range(5)]
    ras_a = NamedRelationalAlgebraFrozenSet(("x", "y"), a)
    assert ras_a.fetch_one() in a
    assert (ras_a - ras_a).fetch_one() is None


def test_is_empty():
    assert not RelationalAlgebraFrozenSet.dee().is_empty()
    assert RelationalAlgebraFrozenSet.dum().is_empty()
    assert RelationalAlgebraFrozenSet([]).is_empty()
    ras_a = RelationalAlgebraFrozenSet([(i, i * 2) for i in range(5)])
    assert not ras_a.is_empty()
    assert (ras_a - ras_a).is_empty()


def test_iter():
    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]
    a += a[3:]
    ras_a = RelationalAlgebraFrozenSet(a)
    res = list(iter(ras_a))
    assert res == a[:6]


def test_named_iter():
    a = [(i, i * j) for i in (1, 2) for j in (2, 3, 4)]
    a += a[3:]
    ras_a = NamedRelationalAlgebraFrozenSet(("y", "x"), a)
    res = list(iter(ras_a))
    assert res == a[:6]


def test_set_dtypes():
    data = [
        (5, "dog", frozenset({(1, 2), (5, 6)}), np.nan, 45.34, False),
        (10, "cat", frozenset({(5, 6), (8, 9)}), np.nan, np.nan, True),
        (3, "cow", np.nan, np.nan, np.nan, True),
    ]
    ras_a = NamedRelationalAlgebraFrozenSet(
        ("a", "b", "c", "d", "e", "f"), data
    )
    expected_row_types = [
        int,
        str,
        typing.AbstractSet[typing.Tuple[int, int]],
        float,
        float,
        bool,
    ]
    expected_dtypes = [
        np.dtype(int),
        pd.StringDtype(),
        np.object_,
        np.float64,
        np.float64,
        np.bool_,
    ]
    assert all(ras_a.row_types == expected_row_types)
    assert all(ras_a.dtypes == expected_dtypes)
    ras_b = NamedRelationalAlgebraFrozenSet(
        ("aa", "bb", "cc", "dd", "ee", "ff"), ras_a
    )
    assert all(ras_b.row_types == expected_row_types)
    assert all(ras_b.dtypes == expected_dtypes)
    assert all(
        ras_b.row_types.index.values == ("aa", "bb", "cc", "dd", "ee", "ff")
    )


def test_infer_types():
    row_types = pd.Series(
        [
            int,
            str,
            typing.AbstractSet[typing.Tuple[int, int]],
            float,
            float,
            bool,
        ],
        index=("a", "b", "c", "d", "e", "f"),
    )

    # lambda expression cannot be inferred, should return default type
    assert (
        try_to_infer_type_of_operation(lambda x: x + 1, row_types) is Unknown
    )
    func: Callable[[int], int] = lambda x: x ** 2
    func.__annotations__["return"] = int
    assert try_to_infer_type_of_operation(func, row_types) == int
    assert (
        try_to_infer_type_of_operation("count", row_types) == pd.Int32Dtype()
    )
    assert try_to_infer_type_of_operation("sum", row_types) == np.dtype(object)
    assert (
        try_to_infer_type_of_operation(
            RelationalAlgebraStringExpression("e + 1"), row_types
        )
        == row_types["e"]
    )
    assert (
        try_to_infer_type_of_operation(
            RelationalAlgebraStringExpression("a * 12"), row_types
        )
        == row_types["a"]
    )
    assert try_to_infer_type_of_operation("0", row_types) == int
    assert (
        try_to_infer_type_of_operation(1.0, row_types, np.dtype(object))
        == float
    )
    assert try_to_infer_type_of_operation("hello", row_types) == str
    assert try_to_infer_type_of_operation("hello world", row_types) == str


def test_row_type():
    data = [
        (5, "dog", frozenset({(1, 2), (5, 6)}), np.nan, 45.34, False),
        (10, "cat", frozenset({(5, 6), (8, 9)}), np.nan, np.nan, True),
        (3, "cow", np.nan, np.nan, np.nan, True),
    ]
    ras_a = NamedRelationalAlgebraFrozenSet(
        ("a", "b", "c", "d", "e", "f"), data
    )
    expected_row_types = [
        int,
        str,
        typing.AbstractSet[typing.Tuple[int, int]],
        float,
        float,
        bool,
    ]
    assert ras_a.set_row_type == Tuple[tuple(expected_row_types)]
    assert NamedRelationalAlgebraFrozenSet.dee().set_row_type == Tuple
    assert (
        NamedRelationalAlgebraFrozenSet(("x", "y"), []).set_row_type
        == Tuple[tuple([Unknown, Unknown])]
    )


def test_aggregate():
    initial_set = NamedRelationalAlgebraFrozenSet(
        ("x", "y", "z"), [(7, 8, 1), (7, 8, 9)]
    )
    expected_lambda = NamedRelationalAlgebraFrozenSet(
        ("x", "y", "z"), [(7, 8, 8)]
    )

    new_set = initial_set.aggregate(["x", "y"], {"z": lambda x: max(x) - 1})
    assert expected_lambda == new_set


def test_create_view_from():
    a = [(i, i * 2) for i in range(5)]
    ras_a = RelationalAlgebraFrozenSet(a)
    ras_a = ras_a.selection({0: 1})

    ras_b = RelationalAlgebraFrozenSet.create_view_from(ras_a)
    assert ras_b._container is None
    ras_a.fetch_one()
    assert ras_a._container is not None
    assert ras_b._container is None
    assert ras_b == ras_a


def test_ra_string_expression():
    ras = RelationalAlgebraStringExpression("x == 1 and y == 2")
    assert ras == "x == 1 and y == 2"

    ras = RelationalAlgebraStringExpression("((x != 3))")
    assert ras == "((x <> 3))"

    ras = RelationalAlgebraStringExpression("d ** 3")
    assert ras == "POWER(d ,  3)"

    ras = RelationalAlgebraStringExpression("(d ** 3)")
    assert ras == "POWER(d ,  3)"
