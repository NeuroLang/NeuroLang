"""A directional query run end to end, from a tractogram and an atlas,
over the relations of a store given to an engine."""

import numpy as np
import pandas as pd
import pytest

from neurolang.atlases import Atlas
from nl_tracts.indexing import IndexingParams, RelationStore
from nl_tracts.tractogram import Tractogram

from .conftest import AFFINES, brute_force_extent, synthetic_volume

TABLE = pd.DataFrame(
    {"id": [0, 1, 2, 3, 4],
     "name": ["bg", "Left-A", "Right-B", "C", "D"]}
)


def lines(seed=0, n=40):
    rng = np.random.default_rng(seed)
    return [
        # around label 1 (x -7..-1, y -18.5..-9.5, z -32..-20 mm), so that
        # some streamlines are on each side of its box
        (rng.normal(size=(rng.integers(2, 20), 3)) * 6
         + [-4, -14, -26]).astype(np.float32)
        for _ in range(n)
    ]


def make_store():
    atlas = Atlas("fs", synthetic_volume(), AFFINES["aligned"], TABLE, "x")
    tractogram = Tractogram.from_streamlines("t1", lines(), "x")
    store = RelationStore()
    store.add_atlas(atlas)
    store.add_tractogram(tractogram, IndexingParams())
    return store, atlas, tractogram


#: The relations of a store, with the names of their columns (DESIGN.md, 3).
RELATION_NAMES = {
    "tractogram": ("t",),
    "streamline": ("x", "t", "s"),
    "atlas": ("a",),
    "label": ("a", "l", "name", "hemi"),
    "label_term": ("a", "l", "term"),
    "streamline_label": ("a", "x", "l"),
    "endpoint_label": ("a", "x", "e", "l"),
    "streamline_extent": ("x", "axis", "lo", "hi"),
    "endpoint_coord": ("x", "e", "axis", "v"),
    "endpoint_point": ("x", "e", "px", "py", "pz"),
    "label_extent": ("a", "l", "axis", "lo", "hi"),
    "streamline_length": ("x", "mm"),
}


def test_the_relations_have_the_columns_of_the_schema():
    store, _, _ = make_store()
    for name, frame in store.relations.items():
        assert tuple(frame.columns) == RELATION_NAMES[name], name


def test_the_relations_a_directional_query_reads_are_registered():
    store, _, _ = make_store()
    assert {"tractogram", "label_extent", "streamline_extent",
            "endpoint_coord"} <= set(store.relations)


@pytest.mark.parametrize(
    "axis,face,op,edge,cmp",
    [
        (1, "hi", np.greater, "hi", ">"),  # anterior of label 1
        (1, "lo", np.less, "lo", "<"),  # posterior of label 1
    ],
)
def test_a_directional_query_end_to_end(axis, face, op, edge, cmp):
    """From a tractogram and an atlas to the streamlines, on NeurolangDL:
    a streamline, or one of its ends, beyond a face of the box of label 1,
    in Datalog over the relations, against numpy over the raw points and
    the voxels of the label (loops, no relation of the package)."""
    from neurolang.frontend import NeurolangDL

    store, atlas, tractogram = make_store()
    box = f'label_extent("fs", 1, {axis}, lo_1, hi_1)'
    nl = NeurolangDL()
    for name, frame in store.relations.items():
        nl.add_tuple_set(frame, name=name)
    nl.execute_datalog_program(
        f"tracts(x) :- streamline_extent(x, {axis}, lo, hi), {box}, "
        f"({edge} {cmp} {face}_1)\n"
        + "".join(
            f"end{e}(x) :- endpoint_coord(x, {e}, {axis}, v), {box}, "
            f"(v {cmp} {face}_1)\n"
            for e in (0, 1)
        )
    )
    solution = nl.solve_all()

    low, high = brute_force_extent(synthetic_volume(), AFFINES["aligned"], 1)
    box = (high if face == "hi" else low)[axis]
    pts = lines()
    stat = np.array([(p[:, axis].max() if edge == "hi" else p[:, axis].min())
                     for p in pts])
    expected_tracts = set(np.flatnonzero(op(stat, box)).tolist())
    expected_end0 = {s for s, p in enumerate(pts) if op(p[0, axis], box)}
    expected_end1 = {s for s, p in enumerate(pts) if op(p[-1, axis], box)}
    assert expected_tracts and expected_tracts != set(range(len(pts)))

    def got(pred):
        # one tractogram, the first: x is s
        frame = solution[pred].as_pandas_dataframe()
        assert frame.shape[1] == 1
        return set(frame.iloc[:, 0])

    assert got("tracts") == expected_tracts
    assert got("end0") == expected_end0
    assert got("end1") == expected_end1
