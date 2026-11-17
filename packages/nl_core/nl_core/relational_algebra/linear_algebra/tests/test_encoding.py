import gc
from typing import AbstractSet

import numpy as np
import pandas as pd
import pytest
from nl_expr.expressions import Constant, Symbol

from ....datalog.expressions import Conjunction, Implication, Union
from ....utils.relational_algebra_set import (
    NamedRelationalAlgebraFrozenSet, RelationalAlgebraFrozenSet,
    RelationalAlgebraSet
)
from .. import encoding
from ..encoding import (
    Axis, EntityIndex, SparseView, ViewDeclaration, build_sparse_views,
    sparse_view, stored_relation
)


def _sorted(frame):
    frame = frame.reset_index(drop=True)
    return frame.sort_values(list(frame.columns)).reset_index(drop=True)


def test_entity_index_round_trip_composite():
    voxels = pd.DataFrame({"i": [3, 0, 3, 1], "j": [1, 2, 1, 0],
                           "k": [-5, 7, -5, 70_000_000]})
    index = EntityIndex([voxels])
    assert len(index) == 3 and index.width == 3
    codes = index.encode(voxels)
    assert codes.dtype == np.int32
    assert codes[0] == codes[2]
    # Numbered in the lexicographic order of the values.
    assert list(codes) == [2, 0, 2, 1]
    back = index.decode(codes, ["i", "j", "k"])
    pd.testing.assert_frame_equal(back, voxels, check_dtype=False)


def test_entity_index_strings_and_unknowns():
    studies = pd.DataFrame({"s": pd.array(["b", "a", "c", "a"], "str")})
    index = EntityIndex([studies, pd.DataFrame({"s": ["d"]})])
    assert len(index) == 4
    assert list(index.encode(studies)) == [1, 0, 2, 0]
    assert list(index.encode(pd.DataFrame({"s": ["z", "d", None]}))) == [
        -1, 3, -1
    ]
    assert list(index.decode([3, 0])[0]) == ["d", "a"]


def test_entity_index_composite_unknown_combination():
    index = EntityIndex([pd.DataFrame({"a": [0, 1], "b": ["x", "y"]})])
    # Each value is in a dictionary, but the pair is no entity.
    codes = index.encode(pd.DataFrame({"a": [0, 0, 1], "b": ["y", "x", "y"]}))
    assert list(codes) == [-1, 0, 1]


def test_numbered_frames_are_encoded_alike():
    frames = [
        pd.DataFrame({"a": [2, 0, 2], "b": pd.array(["x", "y", "x"], "str")}),
        pd.DataFrame({"a": [5, 0], "b": pd.array(["z", "y"], "str")}),
    ]
    index, numbers = EntityIndex.numbered(frames)
    assert len(index) == 3
    for frame, number in zip(frames, numbers):
        assert number.dtype == np.int32
        np.testing.assert_array_equal(number, index.encode(frame))


def test_integers_outside_the_dictionary_are_unknown():
    # Rows outnumber the span of the values: looked up in a table.
    index = EntityIndex([pd.DataFrame({"i": [3, 4, 6, 3, 4, 6]})])
    codes = index.encode(pd.DataFrame({"i": [-1, 2, 3, 5, 6, 7, 99, 4]}))
    assert list(codes) == [-1, -1, 0, -1, 2, -1, -1, 1]


def test_entity_index_empty():
    index = EntityIndex([pd.DataFrame(columns=["i", "j"])])
    assert len(index) == 0
    assert len(index.encode(pd.DataFrame(columns=["i", "j"]))) == 0
    assert list(index.encode(pd.DataFrame({"i": [1], "j": [2]}))) == [-1]
    assert index.decode([]).shape == (0, 2)


def _activated():
    return RelationalAlgebraFrozenSet([
        (0, 0, 0, "s1"), (0, 0, 0, "s2"), (1, 2, 3, "s2"), (5, 5, 5, "s3"),
    ])


def test_sparse_view_round_trip():
    relation = _activated()
    frame = relation.as_pandas_dataframe()
    voxels = EntityIndex([frame[[0, 1, 2]]])
    studies = EntityIndex([frame[[3]]])
    view = SparseView(relation, (0, 1, 2), (3,), voxels, studies)
    matrix = view.matrix
    assert matrix.format == "csr" and matrix.shape == (3, 3)
    assert matrix.indices.dtype == np.int32
    assert matrix.indptr.dtype == np.int32
    assert matrix.dtype == bool and matrix.nnz == 4
    assert matrix.T.format == "csc"
    np.testing.assert_array_equal(
        matrix.toarray(),
        [[True, True, False], [False, True, False], [False, False, True]],
    )
    pd.testing.assert_frame_equal(
        _sorted(view.to_frame()[[0, 1, 2, 3]]), _sorted(frame),
        check_dtype=False,
    )


def test_sparse_view_drops_duplicates():
    relation = RelationalAlgebraFrozenSet(
        [(0, "a"), (0, "a"), (1, "b"), (0, "a")]
    )
    assert relation.might_have_duplicates
    frame = relation.as_pandas_dataframe()
    view = SparseView(
        relation, (0,), (1,),
        EntityIndex([frame[[0]]]), EntityIndex([frame[[1]]]),
    )
    assert view.matrix.nnz == 2


def test_sparse_view_value_column():
    relation = NamedRelationalAlgebraFrozenSet(
        ("s", "t", "w"),
        [("a", "x", 0.5), ("a", "y", 0.0), ("b", "x", 2.0), ("a", "x", 0.5)],
    )
    frame = relation.as_pandas_dataframe()
    view = SparseView(
        relation, ("s",), ("t",),
        EntityIndex([frame[["s"]]]), EntityIndex([frame[["t"]]]), value="w",
    )
    # The explicit zero is a tuple of the relation.
    assert view.matrix.nnz == 3
    np.testing.assert_array_equal(view.matrix.toarray(), [[.5, 0], [2, 0]])
    pd.testing.assert_frame_equal(
        _sorted(view.to_frame()), _sorted(frame), check_dtype=False
    )


def test_sparse_view_value_not_a_function():
    relation = NamedRelationalAlgebraFrozenSet(
        ("s", "t", "w"), [("a", "x", 0.5), ("a", "x", 0.25)]
    )
    frame = relation.as_pandas_dataframe()
    with pytest.raises(ValueError, match="not a function"):
        SparseView(
            relation, ("s",), ("t",),
            EntityIndex([frame[["s"]]]), EntityIndex([frame[["t"]]]),
            value="w",
        )


def test_sparse_view_spans_every_column():
    relation = _activated()
    frame = relation.as_pandas_dataframe()
    with pytest.raises(ValueError, match="spans each column"):
        SparseView(
            relation, (0, 1), (3,),
            EntityIndex([frame[[0, 1]]]), EntityIndex([frame[[3]]]),
        )


def test_sparse_view_entity_not_in_index():
    relation = _activated()
    frame = relation.as_pandas_dataframe()
    with pytest.raises(ValueError, match="not in its index"):
        SparseView(
            relation, (0, 1, 2), (3,), EntityIndex([frame[[0, 1, 2]]]),
            EntityIndex([pd.DataFrame({"s": ["s1"]})]),
        )


def test_sparse_view_empty_relation():
    for relation in (
        RelationalAlgebraFrozenSet(),
        NamedRelationalAlgebraFrozenSet(("v", "s")),
    ):
        columns = tuple(relation.columns) or (0, 1)
        empty = pd.DataFrame(columns=["c"])
        view = SparseView(
            relation, columns[:1], columns[1:],
            EntityIndex([empty]), EntityIndex([empty]),
        )
        assert view.matrix.shape == (0, 0) and view.matrix.nnz == 0
        assert len(view.to_frame()) == 0


def test_sparse_view_cached_for_the_relation_lifetime():
    relation = _activated()
    frame = relation.as_pandas_dataframe()
    voxels, studies = EntityIndex([frame[[0, 1, 2]]]), EntityIndex(
        [frame[[3]]]
    )
    view = sparse_view(relation, (0, 1, 2), (3,), voxels, studies)
    assert sparse_view(relation, (0, 1, 2), (3,), voxels, studies) is view
    # A set made from the relation without changing its rows shares them.
    same = RelationalAlgebraFrozenSet.create_view_from(relation)
    assert sparse_view(same, (0, 1, 2), (3,), voxels, studies) is view
    # Another numbering of the entities is another view.
    other = EntityIndex([frame[[3]], pd.DataFrame({3: ["s0"]})])
    assert sparse_view(relation, (0, 1, 2), (3,), voxels, other) is not view
    key = id(frame)
    assert key in encoding._VIEWS
    del relation, same, frame, view
    gc.collect()
    assert key not in encoding._VIEWS


def test_sparse_view_rebuilt_when_rows_removed_in_place():
    relation = RelationalAlgebraSet([(0, "a"), (1, "b")])
    frame = relation.as_pandas_dataframe()
    rows, columns = EntityIndex([frame[[0]]]), EntityIndex([frame[[1]]])
    view = sparse_view(relation, (0,), (1,), rows, columns)
    relation.discard((1, "b"))
    again = sparse_view(relation, (0,), (1,), rows, columns)
    assert again is not view and again.matrix.nnz == 1


def _constant(relation):
    return Constant[AbstractSet](
        relation, auto_infer_type=False, verify_type=False
    )


def _symbol_table(relation, rule_args, body_args):
    fresh, predicate = Symbol("fresh"), Symbol("activated")
    return {
        fresh: _constant(relation),
        predicate: Union((Implication(
            predicate(*map(Symbol, rule_args)),
            Conjunction((fresh(*map(Symbol, body_args)),)),
        ),)),
    }


def test_stored_relation_follows_a_renaming():
    relation = _activated()
    table = _symbol_table(relation, "ijks", "sijk")
    stored, positions = stored_relation(table, Symbol("activated"))
    assert stored is relation and positions == (1, 2, 3, 0)
    stored, positions = stored_relation(table, Symbol("fresh"))
    assert stored is relation and positions == (0, 1, 2, 3)
    for body in ("iiks", "ijkt"):
        with pytest.raises(ValueError, match="not a stored relation"):
            stored_relation(
                _symbol_table(relation, "ijks", body), Symbol("activated")
            )


def test_build_sparse_views_share_the_index_of_a_sort():
    activated = _activated()
    tfidf = NamedRelationalAlgebraFrozenSet(
        ("study", "term", "w"), [("s4", "pain", 0.5), ("s1", "pain", 0.25)]
    )
    table = _symbol_table(activated, "ijks", "ijks")
    table[Symbol("tfidf")] = _constant(tfidf)
    views = build_sparse_views(table, {
        "activated": ViewDeclaration(
            Axis("voxel", (0, 1, 2)), Axis("study", (3,))
        ),
        "tfidf": ViewDeclaration(
            Axis("study", (0,)), Axis("term", (1,)), value=2
        ),
    })
    assert views["activated"].column_index is views["tfidf"].row_index
    studies = views["tfidf"].row_index
    assert list(studies.decode(range(len(studies)))[0]) == [
        "s1", "s2", "s3", "s4"
    ]
    # The studies of both relations are numbered alike: the product of the
    # views is the join on the study, summed over it.
    product = views["activated"].matrix @ views["tfidf"].matrix
    np.testing.assert_array_equal(product.toarray(), [[.25], [0], [0]])


def test_weighted_view_counts_in_products():
    # Region x study incidence: the studies two regions share are the
    # entries of R @ R.T, which the boolean matrix would not count.
    rng = np.random.default_rng(0)
    pairs = {(int(r), int(s)) for r, s in rng.integers(0, [7, 40], (120, 2))}
    relation = RelationalAlgebraFrozenSet(sorted(pairs))
    frame = relation.as_pandas_dataframe()
    view = SparseView(
        relation, (0,), (1,),
        EntityIndex([frame[[0]]]), EntityIndex([frame[[1]]]),
    )
    weighted = view.weighted()
    assert weighted.dtype == np.float64
    assert np.shares_memory(weighted.indices, view.matrix.indices)
    assert np.shares_memory(weighted.indptr, view.matrix.indptr)
    shared = (weighted @ weighted.T).toarray()
    regions = view.row_index.decode(range(len(view.row_index)))[0]
    for a, first in enumerate(regions):
        for b, second in enumerate(regions):
            studies = {s for r, s in pairs if r == first} & {
                s for r, s in pairs if r == second
            }
            assert shared[a, b] == len(studies)
    assert (view.matrix @ view.matrix.T).toarray().max() == 1


def test_weighted_view_of_values():
    relation = NamedRelationalAlgebraFrozenSet(
        ("s", "t", "w"), [("a", "x", 0.5), ("b", "x", 2.0)]
    )
    frame = relation.as_pandas_dataframe()
    view = SparseView(
        relation, ("s",), ("t",),
        EntityIndex([frame[["s"]]]), EntityIndex([frame[["t"]]]), value="w",
    )
    np.testing.assert_array_equal(
        view.weighted(np.float32).toarray(), [[0.5], [2.0]]
    )
