from neurolang.utils.relational_algebra_set.pandas import RelationalAlgebraColumn, RelationalAlgebraColumnInt, RelationalAlgebraStringExpression
from neurolang.utils.relational_algebra_set.sql import (
    NamedRelationalAlgebraFrozenSet,
    RelationalAlgebraFrozenSet,
    SQLAEngineFactory
)
from sqlalchemy import create_engine, Table, MetaData
from unittest.mock import patch
import pytest


@pytest.fixture(scope="session", autouse=True)
def mock_sql_engine():
    """yields a SQLAlchemy engine which is suppressed after the test session"""
    engine_ = create_engine("sqlite:///test.db", echo=False)

    with patch.object(SQLAEngineFactory, "_create_engine") as _fixture:
        _fixture.return_value = engine_
        yield _fixture

    # meta = MetaData()
    # meta.reflect(bind=engine_, views=True)
    # meta.drop_all(engine_)
    engine_.dispose()


def get_table_from_engine(table_name):
    return Table(
        table_name,
        MetaData(),
        autoload=True,
        autoload_with=SQLAEngineFactory.get_engine(),
    )


def test_set_init():
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    ras_a = RelationalAlgebraFrozenSet(a)
    ras_b = RelationalAlgebraFrozenSet(b)
    assert ras_a.is_empty() == False
    assert ras_b.is_empty() == False


def test_named_set_init():
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    ras_a = NamedRelationalAlgebraFrozenSet(("z", "y"), a)
    ras_b = NamedRelationalAlgebraFrozenSet(("y", "x"), b)

    assert ras_a.is_empty() == False
    assert (4, 8) in ras_a
    assert ras_b.is_empty() == False
    assert (4, 6) in ras_b


def test_try_to_create_index():
    """
    try_to_create_index should not create duplicate indexes.
    """
    a = [(i, i * 2) for i in range(5)]
    ras_a = NamedRelationalAlgebraFrozenSet(("z", "y"), a)

    ras_a._try_to_create_index(["y", "z"])
    table = get_table_from_engine(ras_a._table_name)
    assert len(table.indexes) == 1
    assert set(next(iter(table.indexes)).columns.keys()) == {"y", "z"}

    ras_a._try_to_create_index(["z", "y"])
    table = get_table_from_engine(ras_a._table_name)
    assert len(table.indexes) == 1

    ras_a._try_to_create_index(["y"])
    table = get_table_from_engine(ras_a._table_name)
    assert len(table.indexes) == 2


def test_try_to_create_index_on_views():
    """
    try_to_create_index should create indexes on parent tables when
    called on views.
    """
    a = [(i, i * 2) for i in range(5)]
    ras_a = NamedRelationalAlgebraFrozenSet(("z", "y"), a)
    ras_b = NamedRelationalAlgebraFrozenSet(("v", "u"), a)
    ab = ras_a.cross_product(ras_b)
    aba = ab.naturaljoin(ras_a)

    table = get_table_from_engine(ras_a._table_name)
    assert len(table.indexes) == 1
    assert set(next(iter(table.indexes)).columns.keys()) == {"y", "z"}

    table = get_table_from_engine(ras_b._table_name)
    assert len(table.indexes) == 0

    abab = aba.naturaljoin(ras_b)
    table = get_table_from_engine(ras_a._table_name)
    assert len(table.indexes) == 1
    table = get_table_from_engine(ras_b._table_name)
    assert len(table.indexes) == 1
    assert set(next(iter(table.indexes)).columns.keys()) == {"u", "v"}


def test_natural_join_creates_view():
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    ras_a = NamedRelationalAlgebraFrozenSet(("z", "y"), a)
    ras_b = NamedRelationalAlgebraFrozenSet(("y", "x"), b)

    res = ras_a.naturaljoin(ras_b)
    assert res._is_view == True

def test_extended_projection():
    initial_set = NamedRelationalAlgebraFrozenSet(
        ("x", "y"), [(7, 8), (9, 2)]
    )
    expected_sum = NamedRelationalAlgebraFrozenSet(
        ("z",), [(15,), (11,)]
    )
    expected_lambda = NamedRelationalAlgebraFrozenSet(
        ("z",), [(14,), (10,)]
    )
    expected_lambda2 = NamedRelationalAlgebraFrozenSet(
        ("z", "x"), [(14, 8), (10, 10)]
    )
    expected_new_colum_str = NamedRelationalAlgebraFrozenSet(
        ("x", "z",), [(7, "a",), (9, "a",)]
    )
    expected_new_colum_int = NamedRelationalAlgebraFrozenSet(
        ("z",), [(1,), (1,)]
    )
    new_set = initial_set.extended_projection({"z": sum})
    assert expected_sum == new_set
    new_set = initial_set.extended_projection(
        {"z": RelationalAlgebraStringExpression("x+y")}
    )
    assert expected_sum == new_set
    new_set = initial_set.extended_projection({"z": lambda r: r.x + r.y - 1})
    assert expected_lambda == new_set
    new_set = initial_set.extended_projection(
        {
            "z": lambda r: (r.x + r.y - 1),
            "x": RelationalAlgebraStringExpression("x+1"),
        }
    )
    assert expected_lambda2 == new_set
    new_set = initial_set.extended_projection(
        {"z": "a", "x": RelationalAlgebraStringExpression("x")}
    )
    assert expected_new_colum_str == new_set
    new_set = initial_set.extended_projection({"z": 1})
    assert expected_new_colum_int == new_set

    new_set = initial_set.extended_projection(
        {"x": RelationalAlgebraColumn("x")}
    )
    assert initial_set.projection("x") == new_set

    base_set = NamedRelationalAlgebraFrozenSet(
        (1, 2), [(7, 8), (9, 2)]
    )

    new_set = base_set.extended_projection({
        "x": RelationalAlgebraColumnInt(1),
        "y": RelationalAlgebraColumnInt(2)
    })

    assert initial_set == new_set
