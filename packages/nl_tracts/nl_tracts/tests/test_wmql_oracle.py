"""The WMQL transpiler against tract_querier's own evaluator.

Both run on the same synthetic data: ``crossing`` (the labels each streamline
traverses), ``endings`` (the label at each end), and for the directional
terms the boxes of the labels and of the streamlines and the positions of
their endpoints. The reference reads them as dictionaries and arrays, one
tractography at a time; the Datalog program reads them as the relations
``streamline``, ``streamline_label``, ``endpoint_label``, ``label_extent``,
``streamline_extent`` and ``endpoint_coord``, for all the tractograms at once,
each streamline an id ``x = (index << 32) | s`` (the index of its tractogram
in ``TRACTOGRAMS``). Every query *defined* by a program, not only the saved
ones, is compared, for each tractogram, on its four components: the
streamlines (``x`` decoded to ``t`` and ``s``), the label set, and the
streamlines whose first / last point is in the labels.

Data conventions (DESIGN.md, 6.3 and 11): the background, label 0, is among
the traversed labels (``emit_background=True``) and is never the label of an
endpoint, because ``endpoint_label`` has no background rows; the labels 1 to 7
have boxes, label 0 has none (the reference fails on a label without a box, so
the directional arguments never have label 0); the coordinates are multiples of
0.25, which a float32 holds exactly.
"""

import numpy as np
import pandas as pd
import pytest

from nl_tracts.wmql import transpile

from .reference_tq import Indexing, evaluate, load_reference

pytestmark = pytest.mark.slow  # 3 minutes: each test starts an engine

ATLAS = "fs"
TRACTOGRAMS = ("t1", "t2")
LABELS = 7


@pytest.fixture(scope="module")
def reference():
    return load_reference()


def label_boxes(seed):
    """{label: (low, high)} of the atlas, shared by the tractograms."""
    rng = np.random.default_rng(1000 + seed)
    boxes = {}
    for label in range(1, LABELS + 1):
        low = rng.integers(-20, 20, 3) * 0.5
        boxes[label] = (low, low + rng.integers(1, 16, 3) * 0.5)
    return boxes


def synthetic(seed, n=60):
    """What one tractogram has: crossing, endings, boxes, endpoints."""
    rng = np.random.default_rng(seed)
    crossing = {}
    for s in range(n):
        k = rng.integers(0, 5)
        chosen = set(rng.choice(LABELS + 1, size=k, replace=False).tolist())
        if rng.random() < 0.5:
            chosen.add(0)  # the background, as in emit_background=True
        crossing[s] = chosen
    # every label is traversed by someone in every tractogram: the labels
    # the reference gets from the data (``not``, the labels of a directional
    # result) are those of one tractogram
    for label in range(LABELS + 1):
        crossing[label].add(label)
    endings = tuple(
        {s: int(rng.integers(1, LABELS + 1)) for s in range(n)}
        for _ in range(2)
    )
    low = rng.integers(-40, 40, (n, 3)) * 0.25
    size = rng.integers(0, 40, (n, 3))
    high = low + size * 0.25
    endpoints = np.empty((n, 2, 3))
    for e in range(2):
        endpoints[:, e, :] = low + rng.integers(0, size + 1) * 0.25
    return crossing, endings, np.stack([low, high], axis=1), endpoints


def streamline_ids(data):
    """{x: (t, s)} of {t: synthetic(...)}: ``x = (index << 32) | s``, with
    the index of ``t`` in the order of ``data``, as the store has it."""
    return {
        (index << 32) | s: (t, s)
        for index, (t, (crossing, *_)) in enumerate(data.items())
        for s in crossing
    }


def relations(data, boxes):
    """The relations of {t: synthetic(...)} and of the label boxes."""
    label, endpoint, extent, coord = [], [], [], []
    ids = {ts: x for x, ts in streamline_ids(data).items()}
    for t, (crossing, endings, tract_boxes, endpoints) in data.items():
        for s, labels in crossing.items():
            label += [(ATLAS, ids[t, s], lab) for lab in sorted(labels)]
        for e in (0, 1):
            endpoint += [
                (ATLAS, ids[t, s], e, lab) for s, lab in endings[e].items()
                if lab != 0
            ]
            for s in range(len(crossing)):
                for axis in range(3):
                    coord.append((ids[t, s], e, axis, endpoints[s, e, axis]))
        for s in range(len(crossing)):
            for axis in range(3):
                extent.append(
                    (ids[t, s], axis, tract_boxes[s, 0, axis],
                     tract_boxes[s, 1, axis])
                )
    boxes = [
        (ATLAS, lab, axis, lo[axis], hi[axis])
        for lab, (lo, hi) in boxes.items() for axis in range(3)
    ]
    return {
        "streamline": pd.DataFrame(
            [(x, t, s) for x, (t, s) in streamline_ids(data).items()],
            columns=["x", "t", "s"],
        ).astype({"x": np.int64, "s": np.int32}),
        "streamline_label": pd.DataFrame(
            label, columns=["a", "x", "l"]
        ).astype({"x": np.int64, "l": np.int32}),
        "endpoint_label": pd.DataFrame(
            endpoint, columns=["a", "x", "e", "l"]
        ).astype({"x": np.int64, "e": np.int8, "l": np.int32}),
        "tractogram": pd.DataFrame({"t": list(data)}),
        "label_extent": pd.DataFrame(
            boxes, columns=["a", "l", "axis", "lo", "hi"]
        ).astype({"l": np.int32, "axis": np.int8, "lo": np.float32,
                  "hi": np.float32}),
        "streamline_extent": pd.DataFrame(
            extent, columns=["x", "axis", "lo", "hi"]
        ).astype({"x": np.int64, "axis": np.int8, "lo": np.float32,
                  "hi": np.float32}),
        "endpoint_coord": pd.DataFrame(
            coord, columns=["x", "e", "axis", "v"]
        ).astype({"x": np.int64, "e": np.int8, "axis": np.int8,
                  "v": np.float32}),
    }


def run_datalog(program, data, boxes):
    from neurolang.frontend import NeurolangDL

    nl = NeurolangDL()
    for name, frame in relations(data, boxes).items():
        nl.add_tuple_set(frame, name=name)
    nl.execute_datalog_program(program)
    return {
        name: {tuple(row) for row in sol.as_pandas_dataframe().itertuples(
            index=False, name=None)}
        for name, sol in nl.solve_all().items()
    }


def labels_of(rows, t):
    """The labels of tractogram ``t`` in rows ``(t, l)``."""
    return {r[1] for r in rows if r[0] == t}


def streamlines_of(rows, t, ids):
    """The streamlines ``s`` of tractogram ``t`` in rows ``(x,)``, by the
    decoding ``ids`` ({x: (t, s)})."""
    assert all(len(r) == 1 for r in rows)
    return {ids[r[0]][1] for r in rows if ids[r[0]][0] == t}


def check_program(reference, source, seeds=(0, 1, 2), boxes=True,
                  **transpile_kwargs):
    program = transpile(source, atlas=ATLAS, **transpile_kwargs)
    for seed in seeds:
        atlas_boxes = label_boxes(seed) if boxes else {}
        data = {
            t: synthetic(seed * 10 + i) for i, t in enumerate(TRACTOGRAMS)
        }
        solution = run_datalog(program.text, data, atlas_boxes)
        ids = streamline_ids(data)
        for t, (crossing, endings, tract_boxes, endpoints) in data.items():
            expected = evaluate(
                reference, source,
                Indexing(crossing, endings, atlas_boxes, tract_boxes,
                         endpoints),
            )
            assert set(expected) == set(program.defined), (
                set(expected) ^ set(program.defined)
            )
            for name, q in program.defined.items():
                got = tuple(
                    set() if pred is None
                    else labels_of(solution.get(pred, set()), t)
                    if kind == "labels"
                    else streamlines_of(solution.get(pred, set()), t, ids)
                    for kind, pred in vars(q).items()
                )
                assert got == expected[name], (name, seed, t)


BASE = "a = 1\nb = 2\nc = 3\nd = 4\ne = 5\n"

EXPRESSIONS = [
    "x = a",
    "x = a or b",
    "x = a + b",
    "x = a and b",
    "x = a * b",
    "x = a - b",
    "x = a not in b",
    "x = a and b not in c",
    "x = a or b not in c and d",
    "x = a not in b not in c",
    "x = (a or b) and (c or d)",
    "x = not a",
    "x = -a",
    "x = +a",
    "x = ~a",
    "x = ~(a or b)",
    "x = not (a and b)",
    "x = only(a)",
    "x = only(a or b)",
    "x = only(a and b)",
    "x = only(a or b or c)",
    "x = endpoints_in(a)",
    "x = both_endpoints_in(a)",
    "x = endpoints_in(a or b)",
    "x = both_endpoints_in(a or b)",
    "x = endpoints_in(a) and endpoints_in(b)",
    "x = endpoints_in(a) not in endpoints_in(b)",
    "x = only(endpoints_in(a))",
    "x = only(both_endpoints_in(a or b))",
    "x = endpoints_in(only(a or b))",
    "x = (a and b) not in only(c)",
    "x = not only(a)",
    "x = only(not a)",
    "x = not not a",
    "x = a and not b",
    "x = a - (b - c)",
    "x = (a - b) - c",
    "x = 1",
    "x = 0",
    "x = 7 or 3",
    "x = a and 99",
]


@pytest.mark.parametrize("expression", EXPRESSIONS)
def test_expressions_against_the_reference(reference, expression):
    check_program(reference, BASE + expression + "\n", boxes=False)


# the directional terms: each with a name, medial and lateral with both
# sides, inside the other constructs, on a union of labels, with .side, and
# in the shape of the uncinate fasciculus of freesurfer_queries.qry
SIDED = "p.left = 6\np.right = 7\nq.left = 1\nq.right = 2\n"
DIRECTIONAL = [
    "x = anterior_of(a)",
    "x = posterior_of(a)",
    "x = superior_of(a)",
    "x = inferior_of(a)",
    "x = medial_of(p.left)",
    "x = medial_of(p.right)",
    "x = lateral_of(p.left)",
    "x = lateral_of(p.right)",
    "u = a or b\nx = anterior_of(u)",
    "u = a and b and c\nx = posterior_of(u)",
    "x = endpoints_in(anterior_of(a))",
    "x = both_endpoints_in(posterior_of(b))",
    "x = only(anterior_of(a))",
    "x = anterior_of(a) not in posterior_of(b)",
    "x = anterior_of(a) and posterior_of(b) and c",
    "x = not anterior_of(a)",
    "x = anterior_of(a) or medial_of(p.left)",
    "x = only(superior_of(a) or inferior_of(b))",
    "x = anterior_of(Q.left)",
    "y.side = medial_of(q.side) and not p.opposite\n"
    "z.side = lateral_of(q.opposite) not in q.side",
    "uf.side = (a and endpoints_in(b and anterior_of(c)))"
    " not in posterior_of(d)",
    "uf.side = ((p.side or q.side) and endpoints_in(c and anterior_of(a)))"
    " not in posterior_of(b) not in q.opposite",
]


@pytest.mark.parametrize("expression", DIRECTIONAL)
def test_directional_terms_against_the_reference(reference, expression):
    check_program(reference, BASE + SIDED + expression + "\n")


def test_the_box_of_a_label_set_that_differs_by_tractogram(reference):
    """The box of a directional term is per tractogram, and a streamline
    meets the box of its own: ``y`` has, on seed 1, label sets that differ
    between the two tractograms (and are not empty, where the reference
    fails), so the boxes do. So do the complement and ``only`` of it, whose
    streamlines and ends meet the labels of their own tractogram: the
    other tests have the same label sets in both tractograms, where a rule
    that forgets ``streamline(x, t, s)`` gives the same answer."""
    source = (
        BASE + "y = (1 or 2 or 3 or 4 or 5 or 6 or 7) - superior_of(c)\n"
        "x = posterior_of(y) or anterior_of(y)\n"
        "z = endpoints_in(inferior_of(y))\n"
        "w = not y\n"
        "v = only(y)\n"
    )
    labels = []
    for i in range(len(TRACTOGRAMS)):  # the data check_program makes
        crossing, endings, tract_boxes, endpoints = synthetic(10 + i)
        labels.append(evaluate(
            reference, source,
            Indexing(crossing, endings, label_boxes(1), tract_boxes,
                     endpoints),
        )["y"][1])
    assert all(labels) and labels[0] != labels[1], labels
    check_program(reference, source, seeds=(1,))


def test_directional_terms_with_no_label_boxes_select_nothing(reference):
    """``label_bounding_boxes`` empty: the reference returns an empty
    query, and so do the rules, over an empty ``label_extent``."""
    check_program(
        reference, BASE + "x = anterior_of(a)\ny = endpoints_in(x) or a\n",
        boxes=False,
    )


def test_the_vertical_terms_are_the_references_by_default(reference):
    """``superior_of`` and ``inferior_of`` select, in the reference, the
    streamlines whose top is *below*, and whose bottom is *above*, those of
    the box: the opposite of the other terms (DESIGN.md, 11.2). That is
    what is reproduced, unless ``anatomical_vertical`` says otherwise."""
    source = BASE + "x = superior_of(a)\ny = inferior_of(a)\n"
    check_program(reference, source)
    ref = transpile(source, atlas=ATLAS)
    fixed = transpile(source, atlas=ATLAS, anatomical_vertical=True)
    assert ref.text != fixed.text


def test_sequential_definitions_and_the_overwriting_of_names(reference):
    source = (
        BASE
        + "x = a or b\n"
        + "y = x and c\n"
        + "x |= d\n"  # overwrites: y keeps the first x
        + "z = x or y\n"
    )
    check_program(reference, source, boxes=False)


def test_a_bare_name_marks_a_query_as_saved(reference):
    program = transpile(BASE + "x = a or b\ny |= c\ny\n", atlas=ATLAS)
    assert program.saved == ("a", "b", "c", "d", "e", "x", "y")


def test_dotted_names_and_wildcards(reference):
    source = (
        "alpha.left = 1\nalpha.right = 2\nbeta.left = 3\nbeta.right = 4\n"
        "hemisphere.left |= '*.left'\nhemisphere.right |= '*.right'\n"
        "all_alpha = 'alpha*'\n"
        "none = 'nomatch*'\n"
        "mixed = hemisphere.left and not hemisphere.right\n"
    )
    check_program(reference, source, boxes=False)


def test_wildcards_match_only_the_names_defined_so_far(reference):
    source = "p_1 = 1\nearly = 'p_*'\np_2 = 2\nlate = 'p_*'\n"
    check_program(reference, source, boxes=False)


def test_names_are_case_insensitive(reference):
    source = "Alpha = 1\nBETA |= 2\nx = ALPHA or beta\ny = Only(Alpha)\n"
    check_program(reference, source, boxes=False)


def test_side_expands_to_left_and_right(reference):
    source = (
        "ctx.left = 1\nctx.right = 2\nput.left = 3\nput.right = 4\n"
        "ctx_put.side = ctx.side and put.side not in ctx.opposite\n"
        "both.side |= ctx.side or put.side\n"
        "other.side = (ctx.side and put.opposite) not in put.side\n"
    )
    check_program(reference, source, boxes=False)


def test_for_loops_over_a_list_and_a_pattern(reference):
    """Only the *names* equal to the loop variable are replaced, so a loop
    redefines the variable's values or the same query over and over."""
    source = (
        "r1 = 1\nr2 = 2\nr3 = 3\n"
        "for i in [r1, r2]:\n    i |= i or r3\n    g = only(i)\n"
        "for j in 'r*':\n    h = endpoints_in(j)\n    k = h or j\n"
    )
    check_program(reference, source, boxes=False)


def test_a_redefinition_is_a_new_definition(reference):
    check_program(
        reference, BASE + "x = a\ny = x or b\nx = c\nz = x or y\n",
        boxes=False,
    )


def test_a_query_that_is_statically_empty(reference):
    source = BASE + "x = not a\ny = x and endpoints_in(b)\nz = y or c\n"
    check_program(reference, source, boxes=False)
