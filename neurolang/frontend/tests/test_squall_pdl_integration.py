"""
Tests for NeurolangPDL.execute_squall_program.

These tests exercise the high-level API that mirrors execute_datalog_program
but accepts SQUALL controlled-English programs.
"""
import pytest

from ..probabilistic_frontend import NeurolangPDL


@pytest.fixture
def nl():
    """Fresh NeurolangPDL instance with basic extensional facts."""
    engine = NeurolangPDL()
    engine.add_tuple_set(
        [("alice",), ("bob",), ("carol",)], name="person"
    )
    engine.add_tuple_set(
        [("alice",), ("carol",)], name="plays"
    )
    return engine


@pytest.fixture
def nl_counts():
    """NeurolangPDL instance with item / item_count / quantity facts."""
    engine = NeurolangPDL()
    engine.add_tuple_set(
        [("a",), ("b",), ("c",), ("d",)], name="item"
    )
    engine.add_tuple_set(
        [("a", 0), ("a", 1), ("b", 2), ("c", 3)], name="item_count"
    )
    engine.add_tuple_set(
        [(i,) for i in range(5)], name="quantity"
    )
    return engine


# ---------------------------------------------------------------------------
# Rules-only programs return None
# ---------------------------------------------------------------------------

def test_execute_squall_rules_only_returns_none(nl):
    """define-as programs with no obtain clause return None."""
    result = nl.execute_squall_program(
        "define as Active every person that plays."
    )
    assert result is None


def test_execute_squall_rules_fire_into_program(nl):
    """Rules walked in via execute_squall_program are visible in solve_all."""
    nl.execute_squall_program(
        "define as Active every person that plays."
    )
    solution = nl.solve_all()
    assert "active" in solution
    assert (
        set(solution["active"].as_pandas_dataframe().iloc[:, 0].tolist())
        == {"alice", "carol"}
    )


# ---------------------------------------------------------------------------
# Single obtain clause returns NamedRelationalAlgebraFrozenSet
# ---------------------------------------------------------------------------

def test_execute_squall_single_obtain_returns_relation(nl):
    """A single obtain clause returns a NamedRelationalAlgebraFrozenSet."""
    from neurolang.utils.relational_algebra_set.pandas import (
        NamedRelationalAlgebraFrozenSet,
    )
    result = nl.execute_squall_program("obtain every Person that plays.")
    assert isinstance(result, NamedRelationalAlgebraFrozenSet)


def test_execute_squall_single_obtain_correct_rows(nl):
    """obtain every Person that plays returns alice and carol."""
    result = nl.execute_squall_program("obtain every Person that plays.")
    rows = set(result.as_pandas_dataframe().iloc[:, 0].tolist())
    assert rows == {"alice", "carol"}


# ---------------------------------------------------------------------------
# Multiple obtain clauses return dict keyed obtain_0, obtain_1, …
# ---------------------------------------------------------------------------

def test_execute_squall_multiple_obtains_returns_dict(nl):
    """Multiple obtain clauses produce a dict with obtain_0 / obtain_1 keys."""
    result = nl.execute_squall_program(
        "obtain every Person that plays. "
        "obtain every Person."
    )
    assert isinstance(result, dict)
    assert set(result.keys()) == {"obtain_0", "obtain_1"}


def test_execute_squall_multiple_obtains_correct_rows(nl):
    """Each entry in the multi-obtain dict holds the right result set."""
    result = nl.execute_squall_program(
        "obtain every Person that plays. "
        "obtain every Person."
    )
    plays_rows = set(result["obtain_0"].as_pandas_dataframe().iloc[:, 0].tolist())
    all_rows = set(result["obtain_1"].as_pandas_dataframe().iloc[:, 0].tolist())
    assert plays_rows == {"alice", "carol"}
    assert all_rows == {"alice", "bob", "carol"}


# ---------------------------------------------------------------------------
# Combined rules + obtain in one program
# ---------------------------------------------------------------------------

def test_execute_squall_rules_then_obtain(nl):
    """Rules defined before obtain are visible to the query."""
    result = nl.execute_squall_program(
        "define as Active every person that plays. "
        "obtain every Active."
    )
    rows = set(result.as_pandas_dataframe().iloc[:, 0].tolist())
    assert rows == {"alice", "carol"}


# ---------------------------------------------------------------------------
# Comparison filtering
# ---------------------------------------------------------------------------

def test_execute_squall_comparison_filter(nl_counts):
    """Filtering with 'greater equal than' produces correct rows."""
    nl_counts.execute_squall_program(
        "define as Large every Item "
        "that has an item_count greater equal than 2."
    )
    solution = nl_counts.solve_all()
    rows = set(solution["large"].as_pandas_dataframe().iloc[:, 0].tolist())
    assert rows == {"b", "c"}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_execute_squall_aggregation(nl_counts):
    """Aggregation rule produces correct per-item max values."""
    nl_counts.execute_squall_program(
        "define as max_items for every Item ?i ;"
        " where every Max of the Quantity where ?i item_count per ?i."
    )
    solution = nl_counts.solve_all()
    df = solution["max_items"].as_pandas_dataframe()
    # Build set of (item, max_count) pairs from whatever columns are present
    rows = set(map(tuple, df.values.tolist()))
    assert rows == {("a", 1), ("b", 2), ("c", 3)}


@pytest.fixture
def nl_author():
    """NeurolangPDL with paper/author facts stored as author(paper, person)."""
    engine = NeurolangPDL()
    engine.add_tuple_set(
        [("p1",), ("p2",), ("p3",)], name="paper"
    )
    engine.add_tuple_set(
        [("alice",), ("bob",)], name="person"
    )
    engine.add_tuple_set(
        [("p1", "alice"), ("p2", "alice"), ("p3", "bob")], name="author"
    )
    return engine


def test_execute_squall_tilde_inversion_end_to_end(nl_author):
    """~author reverses argument order so author(paper, person) is matched correctly.

    'obtain every Paper that a Person ~author.' means:
      - for each paper p, there exists a person x such that author(p, x)
      - ~author means the SQUALL subject (paper) is arg[0] of the stored relation
    Expected: all three papers are returned.
    """
    result = nl_author.execute_squall_program(
        "obtain every Paper that a Person ~author."
    )
    rows = set(result.as_pandas_dataframe().iloc[:, 0].tolist())
    assert rows == {"p1", "p2", "p3"}
