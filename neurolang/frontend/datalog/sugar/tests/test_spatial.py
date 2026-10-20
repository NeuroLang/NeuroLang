import itertools
import math
import operator
import re
import warnings

import numpy
import pytest

from .....datalog.basic_representation import DatalogProgram
from .....exceptions import UnsupportedProgramError
from .....expressions import Constant, Symbol
from .....logic import Conjunction, Implication
from .....frontend import NeurolangDL, NeurolangPDL
from .....regions import ExplicitVBR
from ..spatial import (
    DetectEuclideanDistanceBoundMatrix,
    TranslateEuclideanDistanceBoundMatrixMixin,
)

EQ = Constant(operator.eq)
LT = Constant(operator.lt)
LE = Constant(operator.le)
GT = Constant(operator.gt)
GE = Constant(operator.ge)
P = Symbol("P")
EUCLIDEAN = Symbol("EUCLIDEAN")
euclidean = Symbol("euclidean")
i1, j1, k1 = Symbol("i1"), Symbol("j1"), Symbol("k1")
i2, j2, k2 = Symbol("i2"), Symbol("j2"), Symbol("k2")
FirstRange = Symbol("FirstRange")
SecondRange = Symbol("SecondRange")
d = Symbol("d")


def test_euclidean_spatial_bound_detection():
    max_dist = Constant(2)
    detector = DetectEuclideanDistanceBoundMatrix()
    expression = Implication(
        P(i2, j2, k2),
        Conjunction(
            (
                FirstRange(i1, j1, k1),
                SecondRange(i2, j2, k2),
                EQ(d, EUCLIDEAN(i1, j1, k1, i2, j2, k2)),
                LT(d, max_dist),
            )
        ),
    )
    assert detector.walk(expression)
    expression = Implication(
        P(i2, j2, k2),
        Conjunction(
            (
                FirstRange(i1, j1, k1),
                SecondRange(i2, j2, k2),
                EQ(d, EUCLIDEAN(i1, j1, k1, i2, j2, k2)),
                LE(d, max_dist),
            )
        ),
    )
    assert detector.walk(expression)
    expression = Implication(
        P(i2, j2, k2),
        Conjunction(
            (
                FirstRange(i1, j1, k1),
                SecondRange(i2, j2, k2),
                EQ(d, EUCLIDEAN(i1, j1, k1, i2, j2, k2)),
                GE(max_dist, d),
            )
        ),
    )
    assert detector.walk(expression)


def test_reversed_equality():
    max_dist = Constant(10)
    detector = DetectEuclideanDistanceBoundMatrix()
    expression = Implication(
        P(i2, j2, k2),
        Conjunction(
            (
                FirstRange(i1, j1, k1),
                SecondRange(i2, j2, k2),
                EQ(EUCLIDEAN(i1, j1, k1, i2, j2, k2), d),
                LT(d, max_dist),
            )
        ),
    )
    assert detector.walk(expression)
    expression = Implication(
        P(i2, j2, k2),
        Conjunction(
            (
                FirstRange(i1, j1, k1),
                SecondRange(i2, j2, k2),
                EQ(EUCLIDEAN(i1, j1, k1, i2, j2, k2), d),
                GT(max_dist, d),
            )
        ),
    )
    assert detector.walk(expression)


def test_unsupported_unbounded_distance():
    detector = DetectEuclideanDistanceBoundMatrix()
    expression = Implication(
        P(i2, j2, k2),
        Conjunction(
            (
                FirstRange(i1, j1, k1),
                SecondRange(i2, j2, k2),
                EQ(EUCLIDEAN(i1, j1, k1, i2, j2, k2), d),
            )
        ),
    )
    with pytest.raises(UnsupportedProgramError):
        detector.walk(expression)


class ProgramTest(TranslateEuclideanDistanceBoundMatrixMixin, DatalogProgram):
    pass


def test_non_extensional_safe_range_pred_falls_back():
    max_dist = Constant(2)
    translator = ProgramTest()
    expression = Implication(
        P(i2, j2, k2),
        Conjunction(
            (
                FirstRange(i1, j1, k1),
                SecondRange(i2, j2, k2),
                EQ(EUCLIDEAN(i1, j1, k1, i2, j2, k2), d),
                GT(max_dist, d),
            )
        ),
    )
    with pytest.warns(UserWarning, match="euclidean"):
        translator.walk(expression)
    assert P in translator.intensional_database()


def test_lower_bound_is_not_a_distance_bound():
    detector = DetectEuclideanDistanceBoundMatrix()
    expression = Implication(
        P(i2, j2, k2),
        Conjunction(
            (
                FirstRange(i1, j1, k1),
                SecondRange(i2, j2, k2),
                EQ(d, EUCLIDEAN(i1, j1, k1, i2, j2, k2)),
                GT(d, Constant(2)),
            )
        ),
    )
    with pytest.raises(UnsupportedProgramError, match="EUCLIDEAN"):
        detector.walk(expression)


@pytest.mark.parametrize("functor", [EUCLIDEAN, euclidean])
@pytest.mark.parametrize(
    "bound",
    [
        lambda app: LT(app, Constant(2)),
        lambda app: LE(app, Constant(2.5)),
        lambda app: GT(Constant(2), app),
        lambda app: GE(Constant(2), app),
    ],
)
def test_direct_bound_detection(functor, bound):
    detector = DetectEuclideanDistanceBoundMatrix()
    expression = Implication(
        P(i2, j2, k2),
        Conjunction(
            (
                FirstRange(i1, j1, k1),
                SecondRange(i2, j2, k2),
                bound(functor(i1, j1, k1, i2, j2, k2)),
            )
        ),
    )
    assert detector.walk(expression)


def test_lowercase_without_bound_is_a_plain_function():
    detector = DetectEuclideanDistanceBoundMatrix()
    expression = Implication(
        P(i2, j2, k2, d),
        Conjunction(
            (
                FirstRange(i1, j1, k1),
                SecondRange(i2, j2, k2),
                EQ(d, euclidean(i1, j1, k1, i2, j2, k2)),
            )
        ),
    )
    assert not detector.walk(expression)


VOXELS = list(itertools.product(range(6), repeat=3))
PEAKS = [
    (2, 2, 2, "s1"),
    (2, 2, 2, "s2"),
    (4, 3, 2, "s1"),
    (0, 0, 5, "s3"),
    (9, 9, 9, "s4"),
]


def _distance(p, q):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))


def _expected(op, radius, with_distance=False):
    result = set()
    for voxel, peak in itertools.product(VOXELS, PEAKS):
        dist = _distance(voxel, peak[:3])
        if op(dist, radius):
            row = voxel + (peak[3],)
            result.add(row + (dist,) if with_distance else row)
    return result


def _engine(frontend=NeurolangPDL):
    nl = frontend()
    nl.add_tuple_set(VOXELS, name="voxel")
    nl.add_tuple_set(PEAKS, name="peak_reported")
    return nl


def _run(nl, rule, head="close(i, j, k, s)"):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = nl.execute_datalog_program(
            f"{rule}\nans({head[6:-1]}) :- {head}"
        )
    return set(result.to_unnamed())


def _rule_body(nl, name="close"):
    idb = nl.program_ir.intensional_database()
    union = next(v for k, v in idb.items() if k.name == name)
    return union.formulas[0].antecedent


CLOSE = "close(i, j, k, s) :- voxel(i, j, k) & peak_reported(i1, j1, k1, s)"


@pytest.mark.parametrize(
    "bound,op,radius",
    [
        ("d < 2", operator.lt, 2),
        ("d <= 2", operator.le, 2),
        ("2 > d", operator.lt, 2),
        ("2 >= d", operator.le, 2),
        ("d < 1", operator.lt, 1),
        ("d <= 1", operator.le, 1),
        ("d < 2.5", operator.lt, 2.5),
    ],
)
def test_reserved_euclidean_strictness(bound, op, radius):
    nl = _engine()
    rule = (
        f"{CLOSE} & (d == EUCLIDEAN(i, j, k, i1, j1, k1)) & ({bound})"
    )
    assert _run(nl, rule) == _expected(op, radius)
    assert "EUCLIDEAN" not in str(_rule_body(nl))


@pytest.mark.parametrize("frontend", [NeurolangPDL, NeurolangDL])
@pytest.mark.parametrize(
    "bound,op,radius",
    [
        ("euclidean(i, j, k, i1, j1, k1) < 2", operator.lt, 2),
        ("euclidean(i, j, k, i1, j1, k1) <= 2", operator.le, 2),
        ("2. > euclidean(i, j, k, i1, j1, k1)", operator.lt, 2),
        ("3 >= euclidean(i1, j1, k1, i, j, k)", operator.le, 3),
    ],
)
def test_lowercase_euclidean_bound(frontend, bound, op, radius):
    nl = _engine(frontend)
    assert _run(nl, f"{CLOSE} & ({bound})") == _expected(op, radius)
    body = str(_rule_body(nl))
    assert "euclidean" not in body
    # Both range atoms are absorbed into the precomputed relation
    assert "voxel" not in body and "peak_reported" not in body


def test_probabilistic_range_predicate_is_kept():
    nl = _engine()
    nl.add_probabilistic_facts_from_tuples(
        [(0.5, 2, 2, 3), (0.25, 5, 5, 5)], name="active"
    )
    with pytest.warns(UserWarning, match="active"):
        nl.execute_datalog_program(
            "near(s) :- active(i, j, k) & peak_reported(i1, j1, k1, s) "
            "& (euclidean(i, j, k, i1, j1, k1) < 2)"
        )
    assert "active" in str(_rule_body(nl, "near"))
    with nl.environment as e:
        e.pnear[e.s, e.PROB[e.s]] = e.near[e.s]
        result = nl.query((e.s, e.p), e.pnear[e.s, e.p])
    assert dict(result.to_unnamed()) == pytest.approx({"s1": 0.5, "s2": 0.5})


def test_distance_variable_available_in_head():
    nl = _engine()
    rule = (
        "close(i, j, k, s, d) :- voxel(i, j, k) & "
        "peak_reported(i1, j1, k1, s) & "
        "(d == euclidean(i, j, k, i1, j1, k1)) & (d <= 2)"
    )
    result = _run(nl, rule, head="close(i, j, k, s, d)")
    expected = _expected(operator.le, 2, with_distance=True)
    assert {r[:4] for r in result} == {r[:4] for r in expected}
    for row in result:
        assert row[4] == pytest.approx(_distance(row[:3], next(
            p for p in PEAKS
            if p[3] == row[3] and _distance(row[:3], p[:3]) == row[4]
        )[:3]))


def test_extra_distance_conditions_are_kept():
    nl = _engine()
    rule = (
        f"{CLOSE} & (d == euclidean(i, j, k, i1, j1, k1)) & (d < 2) "
        "& (d > 1)"
    )
    expected = {
        voxel + (peak[3],)
        for voxel, peak in itertools.product(VOXELS, PEAKS)
        if 1 < _distance(voxel, peak[:3]) < 2
    }
    assert _run(nl, rule) == expected


def test_builtin_euclidean_as_plain_function():
    nl = NeurolangPDL()
    nl.add_tuple_set([(0, 0, 0), (1, 2, 2)], name="a")
    result = nl.execute_datalog_program(
        "dist(d) :- a(0, 0, 0) & a(i, j, k) & "
        "(d == euclidean(0, 0, 0, i, j, k))\n"
        "ans(d) :- dist(d)"
    )
    assert set(result.to_unnamed()) == {(0.0,), (3.0,)}


def test_user_defined_euclidean_is_respected():
    nl = _engine()

    @nl.add_symbol
    def euclidean(
        i1: int, j1: int, k1: int, i2: int, j2: int, k2: int
    ) -> float:
        return numpy.abs(i1 - i2) + numpy.abs(j1 - j2) + numpy.abs(k1 - k2)

    result = _run(nl, f"{CLOSE} & (euclidean(i, j, k, i1, j1, k1) < 2)")
    expected = {
        voxel + (peak[3],)
        for voxel, peak in itertools.product(VOXELS, PEAKS)
        if sum(abs(a - b) for a, b in zip(voxel, peak[:3])) < 2
    }
    assert result == expected


def test_derived_range_predicate_falls_back():
    nl = _engine()
    nl.execute_datalog_program("near_origin(i, j, k) :- voxel(i, j, k)")
    rule = (
        "close(i, j, k, s) :- near_origin(i, j, k) & "
        "peak_reported(i1, j1, k1, s) & "
        "(euclidean(i, j, k, i1, j1, k1) < 2)"
    )
    with pytest.warns(UserWarning, match="near_origin"):
        nl.execute_datalog_program(rule)
    result = nl.execute_datalog_program("ans(i, j, k, s) :- close(i, j, k, s)")
    assert set(result.to_unnamed()) == _expected(operator.lt, 2)


def test_two_dimensional_coordinates():
    nl = NeurolangPDL()
    pixels = list(itertools.product(range(5), repeat=2))
    nl.add_tuple_set(pixels, name="pixel")
    nl.add_tuple_set([(1, 1, "a"), (4, 0, "b")], name="spot")
    result = _run(
        nl,
        "close(i, j, s) :- pixel(i, j) & spot(i1, j1, s) "
        "& (euclidean(i, j, i1, j1) < 1.5)",
        head="close(i, j, s)",
    )
    expected = {
        (i, j, s)
        for (i, j), (a, b, s) in itertools.product(
            pixels, [(1, 1, "a"), (4, 0, "b")]
        )
        if math.hypot(i - a, j - b) < 1.5
    }
    assert result == expected


@pytest.mark.parametrize(
    "op_text,op", [("<", operator.lt), ("<=", operator.le)]
)
def test_float_coordinates(op_text, op):
    nl = NeurolangPDL()
    first = [(0.0, 0.0, 0.0), (0.5, 0.25, 0.0), (3.0, 0.0, 0.0)]
    second = [(2.0, 0.0, 0.0, "x"), (0.5, 0.25, 0.0, "y")]
    nl.add_tuple_set(first, name="pa")
    nl.add_tuple_set(second, name="pb")
    result = _run(
        nl,
        "close(x, y, z, s) :- pa(x, y, z) & pb(x1, y1, z1, s) "
        f"& (euclidean(x, y, z, x1, y1, z1) {op_text} 2)",
        head="close(x, y, z, s)",
    )
    expected = {
        p + (q[3],)
        for p, q in itertools.product(first, second)
        if op(_distance(p, q[:3]), 2)
    }
    assert result == expected


def test_unbounded_reserved_error_mentions_forms():
    nl = _engine()
    with pytest.raises(UnsupportedProgramError, match="euclidean"):
        nl.execute_datalog_program(
            f"{CLOSE} & (d == EUCLIDEAN(i, j, k, i1, j1, k1))"
        )


SQUALL_CLOSE = (
    "define as Close (?i; ?j; ?k; ?s) where Voxel (?i, ?j, ?k) "
    "and Peak_reported (?i1, ?j1, ?k1, ?s) and "
)


@pytest.mark.parametrize(
    "condition,op",
    [
        (
            "euclidean(?i, ?j, ?k, ?i1, ?j1, ?k1) is lower than 2.",
            operator.lt,
        ),
        (
            "EUCLIDEAN(?i, ?j, ?k, ?i1, ?j1, ?k1) is lower than 2.",
            operator.lt,
        ),
        (
            "?d is euclidean(?i, ?j, ?k, ?i1, ?j1, ?k1) "
            "and ?d is lower equal than 2.",
            operator.le,
        ),
    ],
)
def test_squall_euclidean_bound(condition, op):
    nl = _engine()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        nl.execute_squall_program(SQUALL_CLOSE + condition)
    assert "euclidean" not in str(_rule_body(nl)).lower()
    result = nl.execute_datalog_program("ans(i, j, k, s) :- close(i, j, k, s)")
    assert set(result.to_unnamed()) == _expected(op, 2)


BOUND_TEMPLATES = {
    "<": ("{app} < 2", "d < 2", operator.lt),
    "<=": ("{app} <= 2", "d <= 2", operator.le),
    ">": ("2 > {app}", "2 > d", operator.lt),
    ">=": ("2 >= {app}", "2 >= d", operator.le),
}


def _plan(nl, capsys, program):
    capsys.readouterr()
    nl.execute_datalog_program(
        program + "\nans(i, j, k, s) :- close(i, j, k, s)", show_ra=True
    )
    text = capsys.readouterr().out
    return re.sub(r"fresh_\w+", "FRESH", text)


@pytest.mark.parametrize("functor", ["EUCLIDEAN", "euclidean"])
@pytest.mark.parametrize("op_text", list(BOUND_TEMPLATES))
def test_direct_and_distance_variable_forms_are_equivalent(
    functor, op_text, capsys
):
    direct, with_variable, op = BOUND_TEMPLATES[op_text]
    app = f"{functor}(i, j, k, i1, j1, k1)"
    rules = {
        "direct": f"{CLOSE} & ({direct.format(app=app)})",
        "variable": f"{CLOSE} & (d == {app}) & ({with_variable})",
    }
    rows, plans = {}, {}
    for form, rule in rules.items():
        nl = _engine()
        rows[form] = _run(nl, rule)
        body = str(_rule_body(nl))
        assert functor not in body and "voxel" not in body
        plans[form] = _plan(_engine(), capsys, rule)
    assert rows["direct"] == rows["variable"] == _expected(op, 2)
    assert plans["direct"] == plans["variable"]
    assert "FRESH" in plans["direct"]


@pytest.mark.parametrize("functor", ["EUCLIDEAN", "euclidean"])
def test_forms_agree_when_range_predicate_is_missing(functor):
    app = f"{functor}(i, j, k, i1, j1, k1)"
    rules = [
        f"close(i, j, k, s) :- peak_reported(i1, j1, k1, s) & ({app} < 2)",
        f"close(i, j, k, s) :- peak_reported(i1, j1, k1, s) "
        f"& (d == {app}) & (d < 2)",
    ]
    outcomes = []
    for rule in rules:
        nl = _engine()
        with pytest.warns(UserWarning) as record:
            try:
                nl.execute_datalog_program(rule)
                nl.execute_datalog_program(
                    "ans(i, j, k, s) :- close(i, j, k, s)"
                )
                outcomes.append(None)
            except Exception as exc:
                outcomes.append(type(exc))
        assert any("i, j, k" in str(w.message) for w in record)
    assert outcomes[0] is not None and outcomes[0] == outcomes[1]


LABELLED = [(i, j, k, f"l{(i + j) % 3}") for i, j, k in VOXELS]


@pytest.mark.parametrize(
    "head,body,expected",
    [
        (
            "close(l, s)",
            "labelled(i, j, k, l) & peak_reported(i1, j1, k1, s)",
            lambda v, p, d: (v[3], p[3]),
        ),
        (
            "close(l)",
            "labelled(i, j, k, l) & peak_reported(i1, j1, k1, s)",
            lambda v, p, d: (v[3],),
        ),
        (
            "close(i, j, k, i1, j1, k1, s, l)",
            "labelled(i, j, k, l) & peak_reported(i1, j1, k1, s)",
            lambda v, p, d: v[:3] + p + (v[3],),
        ),
        (
            "close(i, j, k, s)",
            "labelled(i, j, k, s) & peak_reported(i1, j1, k1, s)",
            lambda v, p, d: v[:3] + (p[3],) if v[3] == p[3] else None,
        ),
        (
            "close(i, j, k, s)",
            "labelled(i, j, k, 'l1') & peak_reported(i1, j1, k1, s)",
            lambda v, p, d: v[:3] + (p[3],) if v[3] == "l1" else None,
        ),
        (
            "close(s, d)",
            "labelled(i, j, k, l) & peak_reported(i1, j1, k1, s)",
            lambda v, p, d: (p[3], d),
        ),
    ],
)
def test_range_atoms_with_other_columns(head, body, expected):
    nl = NeurolangPDL()
    nl.add_tuple_set(LABELLED, name="labelled")
    nl.add_tuple_set(PEAKS + [(2, 2, 2, "l1")], name="peak_reported")
    rule = (
        f"{head} :- {body} & (d == euclidean(i, j, k, i1, j1, k1)) & (d < 2)"
    )
    result = _run(nl, rule, head=head)
    rows = (
        expected(v, p, _distance(v[:3], p[:3]))
        for v, p in itertools.product(LABELLED, PEAKS + [(2, 2, 2, "l1")])
        if _distance(v[:3], p[:3]) < 2
    )
    assert result == {row for row in rows if row is not None}


def _region_relation_rows(engine, name):
    """
    Return the type and the rows of the stored relation *name*.

    Parameters
    ----------
    engine : QueryBuilderDatalog
        Engine storing the relation.
    name : str
        Name of the relation.

    Returns
    -------
    tuple
        The type of the relation and its rows.
    """
    stored = engine.symbol_table[name]
    value = stored.value
    rows = set(value.unwrap() if hasattr(value, "unwrap") else value)
    return stored.type, rows


@pytest.mark.parametrize("frontend", [NeurolangDL, NeurolangPDL])
def test_region_destroy_keeps_region_relation(frontend):
    engine = frontend()
    first = ExplicitVBR(numpy.array([[1, 0, 0], [1, 1, 0]]), numpy.eye(4))
    second = ExplicitVBR(numpy.array([[2, 0, 0]]), numpy.eye(4))
    engine.add_tuple_set([("A", first), ("B", second)], name="region")
    before = _region_relation_rows(engine, "region")

    engine.execute_datalog_program(
        "voxels(i, j, k, l) :- region(l, r), contains(r, (i, j, k))."
    )

    relation_type, rows = _region_relation_rows(engine, "region")
    assert relation_type == before[0]
    assert rows == before[1]
    assert all(isinstance(region, ExplicitVBR) for _, region in rows)

    voxels = engine.execute_datalog_program(
        "ans(i, j, k, l) :- voxels(i, j, k, l)"
    )
    assert set(voxels.to_unnamed()) == {
        (1, 0, 0, "A"),
        (1, 1, 0, "A"),
        (2, 0, 0, "B"),
    }

    regions = engine.execute_datalog_program("ans(l, r) :- region(l, r)")
    assert set(regions.to_unnamed()) == {("A", first), ("B", second)}


def _expected_near(point, op, radius):
    """The studies of PEAKS within *radius* of the constant *point*."""
    return {
        (peak[3],) for peak in PEAKS if op(_distance(peak[:3], point), radius)
    }


@pytest.mark.parametrize("frontend", [NeurolangPDL, NeurolangDL])
@pytest.mark.parametrize(
    "bound,op,radius",
    [
        ("euclidean(i1, j1, k1, 2, 2, 3) < 1.5", operator.lt, 1.5),
        ("euclidean(2, 2, 3, i1, j1, k1) <= 1", operator.le, 1),
        ("2.5 > euclidean(i1, j1, k1, 2.0, 2.0, 3.0)", operator.lt, 2.5),
        ("EUCLIDEAN(i1, j1, k1, 2, 2, 3) <= 2.5", operator.le, 2.5),
    ],
)
def test_constant_point_bound_is_precomputed(frontend, bound, op, radius):
    """A point of numbers is ranged by a relation of its own, so the pairs
    within the bound are precomputed rather than the distance computed
    for every row, with a warning."""
    nl = _engine(frontend)
    result = _run(
        nl, f"close(s) :- peak_reported(i1, j1, k1, s) & ({bound})",
        head="close(s)",
    )
    assert result == _expected_near((2, 2, 3), op, radius)
    body = str(_rule_body(nl))
    assert "euclidean" not in body.lower() and "peak_reported" not in body


def test_constant_point_with_a_distance_variable():
    nl = _engine()
    result = _run(
        nl,
        "close(s, d) :- peak_reported(i1, j1, k1, s) & "
        "(d == euclidean(i1, j1, k1, 4, 3, 2)) & (d <= 2)",
        head="close(s, d)",
    )
    assert result == {
        (peak[3], _distance(peak[:3], (4, 3, 2))) for peak in PEAKS
        if _distance(peak[:3], (4, 3, 2)) <= 2
    }


def test_two_constant_points():
    nl = _engine()
    result = _run(
        nl,
        "close(s) :- peak_reported(i1, j1, k1, s) & "
        "(euclidean(i1, j1, k1, 2, 2, 2) < 1) & "
        "(euclidean(i1, j1, k1, 4, 3, 2) < 3)",
        head="close(s)",
    )
    assert result == {("s1",), ("s2",)}


def test_constant_point_without_a_bound_is_a_plain_function():
    nl = NeurolangPDL()
    nl.add_tuple_set([(0, 0, 0), (1, 2, 2)], name="a")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = nl.execute_datalog_program(
            "dist(d) :- a(i, j, k) & (d == euclidean(i, j, k, 0, 0, 0))\n"
            "ans(d) :- dist(d)"
        )
    assert set(result.to_unnamed()) == {(0.0,), (3.0,)}


def test_constant_point_of_a_user_defined_euclidean():
    nl = _engine()

    @nl.add_symbol
    def euclidean(
        i1: int, j1: int, k1: int, i2: int, j2: int, k2: int
    ) -> float:
        return numpy.abs(i1 - i2) + numpy.abs(j1 - j2) + numpy.abs(k1 - k2)

    result = _run(
        nl,
        "close(s) :- peak_reported(i1, j1, k1, s) & "
        "(euclidean(i1, j1, k1, 2, 2, 3) < 2)",
        head="close(s)",
    )
    assert result == {
        (peak[3],) for peak in PEAKS
        if sum(abs(a - b) for a, b in zip(peak[:3], (2, 2, 3))) < 2
    }
