"""The vocabulary of tractography (``vocabulary.dl``), each predicate on
relations written by hand, with the answers written out.

One atlas ``fs``; the left amygdala is two labels (1 and 6), so that its box
is the union of their boxes, label 6 being traversed by no streamline:

====  ===============  =====  ===================  ====================
l     name             hemi   terms                box x / y / z (mm)
====  ===============  =====  ===================  ====================
0     Unknown          none   -                    -
1     Left-Amygdala    left   amygdala             -30 -20 / -10 0 / -20 -10
6     Left-Amygdala-2  left   amygdala             -35 -25 / -5 5 / -25 -15
2     Right-Amygdala   right  amygdala             20 30 / -10 0 / -20 -10
3     ctx-lh-a         left   temporal             -60 -40 / -30 10 / -30 0
4     ctx-lh-b         left   temporal, insula     -40 -30 / 0 20 / -10 10
5     Brain-Stem       none   brainstem            -5 5 / -30 -20 / -40 -10
====  ===============  =====  ===================  ====================

The box of the left amygdala is x -35 -20, y -10 5, z -25 -10.
"""

import pytest

from nl_tracts.frontend import tracts_engine
from nl_tracts.vocabulary import predicates, prelude

LABELS = [
    (0, "Unknown", "none"), (1, "Left-Amygdala", "left"),
    (2, "Right-Amygdala", "right"), (3, "ctx-lh-a", "left"),
    (4, "ctx-lh-b", "left"), (5, "Brain-Stem", "none"),
    (6, "Left-Amygdala-2", "left"),
]
TERMS = [
    (1, "amygdala"), (2, "amygdala"), (6, "amygdala"), (3, "temporal"),
    (4, "temporal"), (4, "insula"), (5, "brainstem"),
]
BOXES = {
    1: [(-30, -20), (-10, 0), (-20, -10)],
    6: [(-35, -25), (-5, 5), (-25, -15)],
    2: [(20, 30), (-10, 0), (-20, -10)],
    3: [(-60, -40), (-30, 10), (-30, 0)],
    4: [(-40, -30), (0, 20), (-10, 10)],
    5: [(-5, 5), (-30, -20), (-40, -10)],
}
# the labels each streamline goes through (0, the background, as with
# emit_background=1), and the labels of its ends (2 has no label at end 1)
TRAVERSED = {0: {1, 3}, 1: {3, 4}, 2: {2}, 3: {1, 2, 5}, 4: {4}, 5: {0, 3}}
ENDS = {(0, 0): 3, (0, 1): 1, (1, 0): 3, (2, 0): 2, (3, 0): 1, (3, 1): 2,
        (4, 0): 4, (4, 1): 4, (5, 0): 3}
INSIDE = [(-30, -25), (-5, 0), (-20, -15)]  # in the amygdala's box
EXTENTS = {
    0: [(-40, -22), (-12, 6), (-30, -12)],  # left, posterior, anterior, inf.
    1: [(-30, -18), (-5, 4), (-20, -5)],    # right, superior; 4 < 5
    2: [(-30, -25), (-5, 5), (-20, -15)],   # 5 is the face: not beyond it
}
CENTRE = (-30, 0, -20)
COORDS = {
    (0, 0): (-50, 8, -20),    # lateral, anterior
    (0, 1): (-25, -12, -30),  # posterior, inferior
    (1, 0): (-50, 2, -20),    # lateral
    (1, 1): (-10, 12, 0),     # medial, anterior, superior
}


def frames():
    return {
        "atlas": [("fs",)],
        "label": [("fs", i, n, h) for i, n, h in LABELS],
        "label_term": [("fs", i, t) for i, t in TERMS],
        "label_extent": [
            ("fs", i, axis, float(lo), float(hi))
            for i, box in BOXES.items() for axis, (lo, hi) in enumerate(box)
        ],
        "tractogram": [("t1",)],
        "streamline": [(x, "t1", x) for x in TRAVERSED],
        "streamline_label": [
            ("fs", x, i) for x, labels in TRAVERSED.items() for i in labels
        ],
        "endpoint_label": [("fs", x, e, i) for (x, e), i in ENDS.items()],
        "streamline_extent": [
            (x, axis, float(lo), float(hi))
            for x in TRAVERSED
            for axis, (lo, hi) in enumerate(EXTENTS.get(x, INSIDE))
        ],
        "endpoint_coord": [
            (x, e, axis, float(v))
            for x in TRAVERSED for e in (0, 1)
            for axis, v in enumerate(COORDS.get((x, e), CENTRE))
        ],
        "opposite_hemisphere": [
            ("left", "right"), ("right", "left"), ("none", "none")
        ],
    }


@pytest.fixture(scope="module")
def nl():
    engine = tracts_engine()
    for name, rows in frames().items():
        engine.add_tuple_set(rows, name=name)
    return engine


def ask(nl, query):
    with nl.scope:
        result = nl.execute_datalog_program(query)
    return {tuple(r) for r in result.as_pandas_dataframe().itertuples(
        index=False, name=None)} if result is not None else set()


def test_the_prelude_is_datalog_with_percent_comments():
    text = prelude()
    assert "#" not in text
    assert {"traverses", "endpoint_in", "ends_in", "in_hemisphere",
            "only_in", "term_box_lo", "term_box_hi", "anterior_of",
            "medial_of", "endpoint_lateral_of"} <= set(predicates())


SORTS = {
    "x": "streamline", "e": "end", "term": "term", "h": "hemisphere",
    "axis": "axis", "a": "atlas", "l": "label_id", "lo": "coordinate",
    "hi": "coordinate",
}


def test_each_predicate_is_documented_with_the_sorts_of_its_arguments():
    documentation = tracts_engine().predicate_documentation()
    assert set(documentation) == set(predicates())
    for name, doc in documentation.items():
        assert doc.description, name
        assert doc.arguments, name
        assert all(SORTS[a] == sort for a, sort in doc.arguments), name


def test_traverses(nl):
    assert ask(nl, "ans(x, term, h) :- traverses(x, term, h)") == {
        (0, "amygdala", "left"), (0, "temporal", "left"),
        (1, "temporal", "left"), (1, "insula", "left"),
        (2, "amygdala", "right"),
        (3, "amygdala", "left"), (3, "amygdala", "right"),
        (3, "brainstem", "none"),
        (4, "temporal", "left"), (4, "insula", "left"),
        (5, "temporal", "left"),
    }


def test_endpoint_in_and_ends_in(nl):
    assert ask(nl, "ans(x, e, term, h) :- endpoint_in(x, e, term, h)") == {
        (0, 0, "temporal", "left"), (0, 1, "amygdala", "left"),
        (1, 0, "temporal", "left"), (2, 0, "amygdala", "right"),
        (3, 0, "amygdala", "left"), (3, 1, "amygdala", "right"),
        (4, 0, "temporal", "left"), (4, 0, "insula", "left"),
        (4, 1, "temporal", "left"), (4, 1, "insula", "left"),
        (5, 0, "temporal", "left"),
    }
    assert ask(nl, 'ans(x) :- ends_in(x, "amygdala", "left")') == {(0,), (3,)}


def test_in_hemisphere(nl):
    assert ask(nl, "ans(x, h) :- in_hemisphere(x, h)") == {
        (0, "left"), (1, "left"), (2, "right"), (3, "left"), (3, "right"),
        (3, "none"), (4, "left"), (5, "none"), (5, "left"),
    }


def test_only_in(nl):
    # 5 goes through the background too, which is of no term
    assert ask(nl, "ans(x, term, h) :- only_in(x, term, h)") == {
        (1, "temporal", "left"), (2, "amygdala", "right"),
        (4, "temporal", "left"), (4, "insula", "left"),
    }


def test_the_box_of_a_term_is_that_of_all_its_labels(nl):
    # label 6, which no streamline goes through, is in the box
    assert ask(nl, 'ans(axis, lo) :- term_box_lo("amygdala", "left", '
                   'axis, lo)') == {(0, -35.0), (1, -10.0), (2, -25.0)}
    assert ask(nl, 'ans(axis, hi) :- term_box_hi("amygdala", "left", '
                   'axis, hi)') == {(0, -20.0), (1, 5.0), (2, -10.0)}


@pytest.mark.parametrize("predicate, expected", [
    # strictly beyond a face: 2 reaches the anterior face, 5, and is not
    # anterior; 1 (4) would be, if the box were label 1's alone (0)
    ("anterior_of", {0}), ("posterior_of", {0}),
    ("superior_of", {1}), ("inferior_of", {0}),
    ("left_of", {0}), ("right_of", {1}),
    # relative to the hemisphere of the term: medial of left is right
    ("medial_of", {1}), ("lateral_of", {0}),
])
def test_the_directions_are_anatomical(nl, predicate, expected):
    assert ask(nl, f'ans(x) :- {predicate}(x, "amygdala", "left")') == {
        (x,) for x in expected
    }


def test_medial_and_lateral_of_the_right_hemisphere(nl):
    # every streamline is to the left of the right amygdala (x 20 30)
    assert ask(nl, 'ans(x) :- medial_of(x, "amygdala", "right")') == {
        (x,) for x in TRAVERSED
    }
    assert ask(nl, 'ans(x) :- lateral_of(x, "amygdala", "right")') == set()
    assert ask(nl, 'ans(x) :- medial_of(x, "brainstem", "none")') == set()


@pytest.mark.parametrize("predicate, expected", [
    ("anterior_of", {(0, 0), (1, 1)}), ("posterior_of", {(0, 1)}),
    ("superior_of", {(1, 1)}), ("inferior_of", {(0, 1)}),
    ("left_of", {(0, 0), (1, 0)}), ("right_of", {(1, 1)}),
    ("medial_of", {(1, 1)}), ("lateral_of", {(0, 0), (1, 0)}),
])
def test_the_directions_of_an_end(nl, predicate, expected):
    assert ask(
        nl, f'ans(x, e) :- endpoint_{predicate}(x, e, "amygdala", "left")'
    ) == expected


def test_one_end_in_a_region_and_beyond_a_face(nl):
    """``e`` names an end: the same end is in the temporal lobe and anterior
    of the amygdala for 0; for 1, one end is in it and the other anterior."""
    assert ask(
        nl, 'ans(x) :- endpoint_in(x, e, "temporal", "left"), '
            'endpoint_anterior_of(x, e, "amygdala", "left")'
    ) == {(0,)}
    assert ask(
        nl, 'ans(x) :- endpoint_in(x, e, "temporal", "left"), '
            'endpoint_anterior_of(x, f, "amygdala", "left")'
    ) == {(0,), (1,)}


def test_the_stored_relations_and_functions_are_kept(nl):
    assert ask(nl, 'ans(h) :- opposite_hemisphere("left", h)') == {
        ("right",)}
    assert ask(nl, 'ans(x, t, s) :- streamline(x, t, s), (x == 2)') == {
        (2, "t1", 2)}


def test_the_region_relations_of_the_same_names_are_not_there(nl):
    """``NeurolangPDL`` has ``anterior_of(r1, r2)`` and the like over
    ``Region`` values: the engine of the vocabulary has its rules instead,
    and the other functions."""
    functions = set(nl.program_ir.included_functions)
    assert not functions & set(predicates())
    assert "anatomical_anterior_of" in functions
