"""The translation of WMQL to the vocabulary against tract_querier's own
evaluator, where the natural semantics of the vocabulary and tract_querier's
coincide (DESIGN.md, 16.5 and 17.3).

Both run on the same synthetic data: the labels each streamline traverses,
the label at each end, the boxes of the labels and of the streamlines and
the positions of the ends. The reference reads them as dictionaries and
arrays; the translated rules, with the vocabulary, as the relations of
nl_tracts. Where the two coincide here: each name of WMQL is the labels of
one term of one hemisphere (``p.left = 1``, the term ``p`` of label 1, of
hemisphere ``left``), and ``u = 9`` of hemisphere ``none``, so that a term
is a name and ``HEMISPHERE.left``, the union of the ``.left`` names, is the
labels of hemisphere ``left``; every label has a box, so that the box of a
term is that of its name; the background is never traversed, as with
``.index_tracts``' default, so that ``only`` is the same. The vertical
directions are anatomical in the vocabulary, inverted in tract_querier
(DESIGN.md, 15.20): checked against numpy instead.
"""

import numpy as np
import pandas as pd
import pytest

from nl_tracts.wmql_vocab import translate

from .reference_tq import Indexing, evaluate, load_reference

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


def synthetic(seed, n=80):
    """crossing, endings, label boxes, streamline boxes, endpoints."""
    rng = np.random.default_rng(seed)
    crossing = {
        s: set(rng.choice(list(TERMS), size=rng.integers(0, 5),
                          replace=False).tolist())
        for s in range(n)
    }
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
    """The relations of nl_tracts of the data; one tractogram, ``x = s``."""
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


def solve(program, data):
    from nl_tracts.frontend import tracts_engine

    nl = tracts_engine()  # with the vocabulary
    for name, frame in relations(*data).items():
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
