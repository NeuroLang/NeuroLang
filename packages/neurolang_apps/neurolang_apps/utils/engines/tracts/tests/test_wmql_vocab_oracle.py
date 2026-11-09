"""The translation of WMQL to the vocabulary against tract_querier's own
evaluator, where the natural semantics of the vocabulary and tract_querier's
coincide (DESIGN.md, 16.5 and 17.3).

Both run on the same synthetic data: the labels each streamline traverses,
the label at each end, the boxes of the labels and of the streamlines and
the positions of the ends. The reference reads them as dictionaries and
arrays; the translated rules, with the vocabulary, as the relations of
the engine. Where the two coincide here: each name of WMQL is the labels of
one term of one hemisphere (``p.left = 1``, the term ``p`` of label 1, of
hemisphere ``left``), and ``u = 9`` of hemisphere ``none``, so that a term
is a name and ``HEMISPHERE.left``, the union of the ``.left`` names, is the
labels of hemisphere ``left``; every label has a box, so that the box of a
term is that of its name; the background is never traversed, as with
``.index_tracts``' default, so that ``only`` is the same. The vertical
directions are anatomical in the vocabulary, inverted in tract_querier
(DESIGN.md, 15.20): checked against numpy instead.
"""

import sys

import numpy as np
import pandas as pd
import pytest

from ..wmql_vocab import translate

from .reference_tq import (
    Indexing, evaluate, load_indexing, load_reference,
)

TERMS = {  # label: (term, hemisphere)
    1: ("p", "left"), 2: ("p", "right"), 3: ("q", "left"),
    4: ("q", "right"), 5: ("r", "left"), 6: ("r", "right"),
    7: ("s", "left"), 8: ("s", "right"), 9: ("u", "none"),
}
LABELS = "".join(
    f"{term}.{hemi} |= {label}\n" if hemi != "none"
    else f"{term} |= {label}\n"
    for label, (term, hemi) in TERMS.items()
) + "HEMISPHERE.left |= '*.left'\nHEMISPHERE.right |= '*.right'\n"

QUERIES = """
x1.side = p.side or q.side
x2.side = p.side and q.side
x3.side = p.side not in q.side not in r.side
x4.side = (p.side or q.side) and r.side not in s.side
x5.side = p.side not in hemisphere.opposite
x6.side = hemisphere.side and p.opposite
x7.side = endpoints_in(p.side)
x8.side = endpoints_in(p.side or q.side) and endpoints_in(r.side)
x9.side = endpoints_in(p.side and anterior_of(q.side))
x10.side = endpoints_in(p.side not in posterior_of(q.side)) and r.side
x11.side = both_endpoints_in(p.side or q.side)
x12.side = only(p.side)
x13.side = only(p.side or q.side)
x14.side = only(p.side and q.side)
x15.side = anterior_of(p.side) and q.side
x16.side = posterior_of(p.side) not in r.side
x17.side = medial_of(p.side) and q.side
x18.side = lateral_of(q.side) not in medial_of(p.opposite)
x19.side = q.side and u
x20.side = q.side not in u
x21 = p.left and p.right
x22 = (p.left or q.right) not in r.left
x23.side = x1.side not in x2.side
x24.side = x5.side or x7.side
x25.side = q.side not in endpoints_in(p.side and anterior_of(r.side))
x26.side = only(q.side and u)
x27 = endpoints_in(p.left) and endpoints_in(p.right)
x28.side = endpoints_in(p.side and anterior_of(q.side)) or r.side
uf.side = ((p.side or q.side) and endpoints_in(r.side and anterior_of(s.side))
           and u) not in posterior_of(q.side) not in hemisphere.opposite
"""


def synthetic(seed, n=80, background=False):
    """crossing, endings, label boxes, streamline boxes, endpoints. With
    ``background``, about half the streamlines traverse label 0, which has
    no term (``.index_tracts(..., emit_background=1)``)."""
    rng = np.random.default_rng(seed)
    crossing = {
        s: set(rng.choice(list(TERMS), size=rng.integers(0, 5),
                          replace=False).tolist())
        for s in range(n)
    }
    if background:
        for s in range(n):
            if rng.random() < 0.5:
                crossing[s].add(0)
    endings = tuple(  # 0: an end in the background, of no label
        {s: int(rng.integers(0, len(TERMS) + 1)) for s in range(n)}
        for _ in range(2)
    )
    boxes = {}
    for label in TERMS:
        low = rng.integers(-20, 20, 3) * 0.5
        boxes[label] = (low, low + rng.integers(1, 16, 3) * 0.5)
    low = rng.integers(-40, 40, (n, 3)) * 0.25
    size = rng.integers(0, 40, (n, 3))
    tract_boxes = np.stack([low, low + size * 0.25], axis=1)
    endpoints = np.stack([
        low + rng.integers(0, size + 1) * 0.25 for _ in range(2)], axis=1)
    return crossing, endings, boxes, tract_boxes, endpoints


def relations(crossing, endings, boxes, tract_boxes, endpoints):
    """The relations of the engine of the data; one tractogram, ``x = s``."""
    n = len(crossing)
    f32 = np.float32
    return {
        "label": pd.DataFrame(
            [("fs", lab, f"{t}_{h}", h) for lab, (t, h) in TERMS.items()],
            columns=["a", "l", "name", "hemi"]),
        "label_term": pd.DataFrame(
            [("fs", lab, t) for lab, (t, _) in TERMS.items()],
            columns=["a", "l", "term"]),
        "label_extent": pd.DataFrame(
            [("fs", lab, axis, f32(lo[axis]), f32(hi[axis]))
             for lab, (lo, hi) in boxes.items() for axis in range(3)],
            columns=["a", "l", "axis", "lo", "hi"]),
        "streamline_label": pd.DataFrame(
            [("fs", s, lab) for s, labels in crossing.items()
             for lab in sorted(labels)], columns=["a", "x", "l"]),
        "endpoint_label": pd.DataFrame(
            [("fs", s, e, lab) for e in (0, 1)
             for s, lab in endings[e].items() if lab != 0],
            columns=["a", "x", "e", "l"]),
        "streamline_extent": pd.DataFrame(
            [(s, axis, f32(tract_boxes[s, 0, axis]),
              f32(tract_boxes[s, 1, axis]))
             for s in range(n) for axis in range(3)],
            columns=["x", "axis", "lo", "hi"]),
        "endpoint_coord": pd.DataFrame(
            [(s, e, axis, f32(endpoints[s, e, axis]))
             for s in range(n) for e in (0, 1) for axis in range(3)],
            columns=["x", "e", "axis", "v"]),
        "opposite_hemisphere": pd.DataFrame(
            [("left", "right"), ("right", "left"), ("none", "none")],
            columns=["h1", "h2"]),
    }


def solve(program, data, label_extent=None):
    from ..tests.engine import tracts_engine

    nl = tracts_engine()  # with the vocabulary
    frames = relations(*data)
    if label_extent is not None:
        frames["label_extent"] = label_extent
    for name, frame in frames.items():
        nl.add_tuple_set(frame, name=name)
    nl.execute_datalog_program(program)
    return {
        name: set(sol.as_pandas_dataframe().itertuples(index=False, name=None))
        for name, sol in nl.solve_all().items()
    }


def streamlines(solution, atom):
    """The streamlines of a query, ``uf(x, "left")`` or ``x21(x)``."""
    pred, _, side = atom[:-1].partition("(x")
    side = side.strip(', "')
    return {row[0] for row in solution.get(pred, ())
            if not side or row[1] == side}


@pytest.mark.parametrize("seed", [0, 1])
def test_the_translation_is_tract_queriers_evaluation(seed):
    reference = load_reference()
    source = LABELS + QUERIES
    translation = translate(source)
    assert translation.skipped == {}
    data = synthetic(seed)
    crossing, endings, boxes, tract_boxes, endpoints = data
    expected = evaluate(reference, source, Indexing(
        crossing, endings, boxes, tract_boxes, endpoints))
    solution = solve(translation.text, data)
    assert len(translation.queries) == 2 * 26 + 3
    nonempty = 0
    for name, atom in translation.queries.items():
        got = streamlines(solution, atom)
        assert got == expected[name], (name, atom, seed)
        nonempty += bool(got)
    assert nonempty > 30  # the data tells the queries apart


def test_the_vertical_directions_are_anatomical():
    """``superior_of(p.left)``: the streamlines whose box reaches above that
    of ``p.left``; ``inferior_of`` below (tract_querier has the reverse)."""
    source = LABELS + (
        "up.side = superior_of(p.side)\ndown.side = inferior_of(p.side)\n")
    data = synthetic(2)
    _, _, boxes, tract_boxes, _ = data
    solution = solve(translate(source).text, data)
    for side, label in (("left", 1), ("right", 2)):
        lo, hi = boxes[label]
        assert streamlines(solution, f'up(x, "{side}")') == set(
            np.flatnonzero(tract_boxes[:, 1, 2] > hi[2]).tolist())
        assert streamlines(solution, f'down(x, "{side}")') == set(
            np.flatnonzero(tract_boxes[:, 0, 2] < lo[2]).tolist())


@pytest.mark.parametrize("seed", [1, 2])
def test_the_background_among_the_labels_is_as_in_tract_querier(seed):
    """With ``emit_background=1`` (rows of label 0 in ``streamline_label``,
    as the reference's ``crossing_tracts_labels``), every query, ``only``
    included, is the reference's: the background is a label outside the
    term, so that a streamline in it is not ``only`` in a term."""
    reference = load_reference()
    source = LABELS + QUERIES
    translation = translate(source)
    data = synthetic(seed, background=True)
    expected = evaluate(reference, source, Indexing(*data))
    solution = solve(translation.text, data)
    for name, atom in translation.queries.items():
        assert streamlines(solution, atom) == expected[name], (name, seed)
    # without the rows of the background (the default of .index_tracts) the
    # `only` queries are supersets of the reference's
    plain = solve(translation.text, (
        {s: labels - {0} for s, labels in data[0].items()}, *data[1:]))
    more = [
        name for name, atom in translation.queries.items()
        if name.startswith(("x12.", "x13.", "x14.", "x26."))
        and streamlines(plain, atom) > expected[name]
    ]
    assert more


@pytest.mark.parametrize("affine_name", ["aligned", "flipped", "permuted"])
def test_the_directions_with_tract_queriers_boxes(affine_name, monkeypatch):
    """``extent="centre"``: ``label_extent`` of a label volume is the
    reference's ``compute_label_bounding_boxes`` where scipy is not
    installed (its voxel centres; blocked here, for it is installed in the
    tests), and the directional queries, with streamline boxes on the faces
    of the labels' (half a voxel from the edge's), are the reference's."""
    from neurolang.atlases import Atlas
    from ..atlas import label_extent_relation

    from .conftest import AFFINES

    affine = AFFINES[affine_name]
    rng = np.random.default_rng(3)
    volume = np.zeros((12, 10, 8), dtype=np.int32)
    for label in TERMS:
        lo = rng.integers(0, [10, 8, 6])
        volume[tuple(slice(a, a + 1 + rng.integers(0, 3)) for a in lo)] = label
    table = pd.DataFrame({
        "id": [0, *TERMS], "name": ["bg", *[f"{t}_{h}" for t, h in
                                            TERMS.values()]]})
    atlas = Atlas("fs", volume, affine, table, "s")
    extent = label_extent_relation(atlas, "centre")
    monkeypatch.setitem(sys.modules, "scipy", None)  # no scipy: ImportError
    reference_boxes = load_indexing().compute_label_bounding_boxes(
        volume, affine)
    monkeypatch.undo()
    boxes = {label: (np.asarray(box[:3]), np.asarray(box[3:]))
             for label, box in reference_boxes.items()}
    assert set(boxes) == set(extent["l"]) == set(TERMS)
    for label, (lo, hi) in boxes.items():
        got = extent[extent.l == label].sort_values("axis")
        np.testing.assert_allclose(got["lo"], lo, atol=1e-4)
        np.testing.assert_allclose(got["hi"], hi, atol=1e-4)

    crossing, endings, _, tract_boxes, endpoints = synthetic(4)
    # streamline boxes on a 1 mm grid, which the faces of the boxes are on
    low = np.round(rng.uniform(-30, 30, tract_boxes.shape[0:1] + (3,)))
    tract_boxes = np.stack(
        [low, low + rng.integers(0, 40, low.shape)], axis=1)
    endpoints = tract_boxes[:, [0, 1]]
    data = (crossing, endings, boxes, tract_boxes, endpoints)
    source = LABELS + """
a.side = anterior_of(p.side) and q.side
b.side = posterior_of(p.side) not in r.side
c.side = anterior_of(s.side) or posterior_of(q.side)
d.side = medial_of(p.side) and q.side
e.side = lateral_of(q.side) not in medial_of(p.opposite)
f.side = endpoints_in(p.side and anterior_of(q.side))
"""
    translation = translate(source)
    expected = evaluate(
        load_reference(), source, Indexing(*data))
    solution = solve(translation.text, data, label_extent=extent)
    nonempty = 0
    for name, atom in translation.queries.items():
        got = streamlines(solution, atom)
        assert got == expected[name], (name, atom, affine_name)
        nonempty += bool(got)
    assert nonempty >= 6
