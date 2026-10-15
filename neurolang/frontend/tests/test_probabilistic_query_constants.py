"""
Queries on probabilistic programs whose query atom carries constants.

The constants of a query such as ``ans(i, p) :- r(i, "pain", p)`` must
reach the rules defining ``r`` through magic sets, so that only the
``"pain"`` part of ``r`` is computed, and the result must be exactly the
one of the unrestricted query, or of the query with the constant written
into the rule by hand.
"""
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from ..probabilistic_frontend import NeurolangPDL

TERMS = ("pain", "memory", "vision")
VARIABLES = ("i", "j", "k", "s", "t", "p")

PROGRAM = """
reports_close(i, j, k, s) :- voxel(i, j, k) & peak_reported(i, j, k, s)
activation(i, j, k, s) :- reports_close(i, j, k, s), selected_study(s)
term_association(s, t) :- mentions(s, t), selected_study(s)
r(i, j, k, t, PROB(i, j, k, t)) :- activation(i, j, k, s) // term_association(s, t)
w(i, j, k, t, PROB(i, j, k, t)) :- activation(i, j, k, s) & term_association(s, t)
r_pain(i, j, k, PROB(i, j, k)) :- activation(i, j, k, s) // term_association(s, "pain")
w_pain(i, j, k, PROB(i, j, k)) :- activation(i, j, k, s) & term_association(s, "pain")
"""


def _engine(studies="choice"):
    nl = NeurolangPDL()
    rng = np.random.default_rng(0)
    n_studies = 6
    nl.add_tuple_set(
        [(i, j, 0) for i in range(5) for j in range(5)], name="voxel"
    )
    nl.add_tuple_set(
        [
            (int(rng.integers(5)), int(rng.integers(5)), 0, f"s{s}")
            for s in range(n_studies)
            for _ in range(4)
        ],
        name="peak_reported",
    )
    nl.add_tuple_set(
        [
            (f"s{s}", term)
            for s in range(n_studies)
            for t, term in enumerate(TERMS)
            if (s + t) % 2 == 0 or s == t
        ],
        name="mentions",
    )
    study_tuples = [(f"s{s}",) for s in range(n_studies)]
    if studies == "choice":
        nl.add_uniform_probabilistic_choice_over_set(
            study_tuples, name="selected_study"
        )
    else:
        nl.add_probabilistic_facts_from_tuples(
            [(0.3 + 0.1 * n, s) for n, (s,) in enumerate(study_tuples)],
            name="selected_study",
        )
    nl.execute_datalog_program(PROGRAM)
    return nl


@pytest.fixture(params=["choice", "facts"])
def nl(request):
    return _engine(request.param)


def _rows(nl, head, atom):
    with nl.scope as e:
        res = nl.query(
            tuple(getattr(e, v) for v in head),
            getattr(e, atom[0])(
                *(getattr(e, a) if a in VARIABLES else a for a in atom[1:])
            ),
        )
    return _as_dict(res, head)


def _as_dict(res, head):
    df = res.as_pandas_dataframe()
    df = df[list(head)]
    prob = "p" if "p" in head else None
    keys = [c for c in head if c != prob]
    out = {}
    for row in df.itertuples(index=False):
        row = dict(zip(head, row))
        out[tuple(row[k] for k in keys)] = row.get(prob, None)
    return out


def _assert_same(found, expected):
    assert found.keys() == expected.keys()
    for key, value in expected.items():
        if value is None:
            assert found[key] is None
        else:
            assert np.isclose(found[key], value), key


def _full(nl, predicate):
    return _rows(
        nl, ("i", "j", "k", "t", "p"), (predicate, "i", "j", "k", "t", "p")
    )


def _restrict(full, position, value, drop=True):
    out = {}
    for key, p in full.items():
        if key[position] == value:
            new_key = key[:position] + key[position + 1:] if drop else key
            out[new_key] = p
    return out


@pytest.mark.parametrize("predicate", ["r", "w"])
def test_constant_on_term_equals_hand_pushed(nl, predicate):
    found = _rows(
        nl, ("i", "j", "k", "p"), (predicate, "i", "j", "k", "pain", "p")
    )
    expected = _rows(
        nl, ("i", "j", "k", "p"), (f"{predicate}_pain", "i", "j", "k", "p")
    )
    assert len(expected) > 0
    _assert_same(found, expected)
    _assert_same(found, _restrict(_full(nl, predicate), 3, "pain"))


@pytest.mark.parametrize("predicate", ["r", "w"])
def test_constant_on_voxel_coordinate(nl, predicate):
    full = _full(nl, predicate)
    i_value = next(iter(full))[0]
    with nl.scope as e:
        res = nl.query(
            (e.j, e.k, e.t, e.p),
            getattr(e, predicate)(i_value, e.j, e.k, e.t, e.p),
        )
    found = _as_dict(res, ("j", "k", "t", "p"))
    expected = _restrict(full, 0, i_value)
    assert len(expected) > 0
    _assert_same(found, expected)


@pytest.mark.parametrize("predicate", ["r", "w"])
def test_constants_on_term_and_coordinate(nl, predicate):
    full = _full(nl, predicate)
    i_value = next(key for key in full if key[3] == "pain")[0]
    with nl.scope as e:
        res = nl.query(
            (e.j, e.k, e.p),
            getattr(e, predicate)(i_value, e.j, e.k, "pain", e.p),
        )
    found = _as_dict(res, ("j", "k", "p"))
    expected = _restrict(_restrict(full, 3, "pain"), 0, i_value)
    assert len(expected) > 0
    _assert_same(found, expected)


@pytest.mark.parametrize("predicate", ["r", "w"])
def test_constant_matching_nothing_is_empty(nl, predicate):
    found = _rows(
        nl, ("i", "j", "k", "p"), (predicate, "i", "j", "k", "nothing", "p")
    )
    assert found == {}


@pytest.mark.parametrize("predicate", ["r", "w"])
def test_repeated_variable_in_query(nl, predicate):
    full = _full(nl, predicate)
    with nl.scope as e:
        res = nl.query(
            (e.i, e.k, e.t, e.p),
            getattr(e, predicate)(e.i, e.i, e.k, e.t, e.p),
        )
    found = _as_dict(res, ("i", "k", "t", "p"))
    expected = {
        (i, k, t): p for (i, j, k, t), p in full.items() if i == j
    }
    assert len(expected) > 0
    _assert_same(found, expected)


def test_no_constant_is_unchanged(nl, capsys):
    full = _full(nl, "r")
    assert len(full) > 0
    with nl.scope as e:
        nl.query(
            (e.i, e.j, e.k, e.t, e.p), e.r(e.i, e.j, e.k, e.t, e.p),
            show_rewritten=True, dry_run=True,
        )
    assert "magic_" not in capsys.readouterr().out


def test_deterministic_idb_query_with_constant(nl):
    full = _rows(
        nl, ("i", "j", "k", "s"), ("reports_close", "i", "j", "k", "s")
    )
    study = next(iter(full))[3]
    with nl.scope as e:
        res = nl.query(
            (e.i, e.j, e.k), e.reports_close(e.i, e.j, e.k, study),
        )
    found = _as_dict(res, ("i", "j", "k"))
    expected = {key[:3]: None for key in full if key[3] == study}
    assert found == expected


def _rewritten_rules(text):
    """Split a printed rewritten program into (head, body) strings."""
    rules = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("──"):
            continue
        if line[0].isspace():
            rules[-1][1].append(line.strip())
        else:
            rules.append((line.strip(), []))
    return [(head, " ".join(body)) for head, body in rules]


def test_plan_propagates_constant_to_term_association(nl, capsys):
    with nl.scope as e:
        nl.query(
            (e.i, e.j, e.k, e.p), e.r(e.i, e.j, e.k, "pain", e.p),
            show_rewritten=True, dry_run=True,
        )
    rules = _rewritten_rules(capsys.readouterr().out)
    heads = [head for head, _ in rules]
    assert any(
        head.startswith("magic_") and "'pain'" in head for head in heads
    ), heads
    term_association_rules = [
        body for head, body in rules
        if head.startswith("term_association")
    ]
    assert term_association_rules
    assert all("magic_" in body for body in term_association_rules), rules
    numerator_rules = [
        body for head, body in rules if "cond_num" in head.split("(")[0]
        and not head.startswith("magic_")
    ]
    assert numerator_rules
    assert all("magic_" in body for body in numerator_rules), rules


SIZE_SCRIPT = textwrap.dedent(
    """
    from neurolang.frontend import NeurolangPDL

    n_voxels, n_studies, n_terms = 2000, 400, 2000
    nl = NeurolangPDL()
    nl.add_tuple_set([(v,) for v in range(n_voxels)], name="voxel")
    nl.add_tuple_set(
        [
            (v, s) for s in range(n_studies) for v in range(n_voxels)
            if (v * 7 + s) % 5 == 0
        ],
        name="peak_reported",
    )
    nl.add_tuple_set(
        [
            (s, f"t{t}") for s in range(n_studies) for t in range(n_terms)
            if (s * 3 + t) % 4 == 0
        ],
        name="mentions",
    )
    nl.add_uniform_probabilistic_choice_over_set(
        [(s,) for s in range(n_studies)], name="selected_study"
    )
    nl.execute_datalog_program('''
    activation(v, s) :- voxel(v), peak_reported(v, s), selected_study(s)
    term_association(s, t) :- mentions(s, t), selected_study(s)
    r(v, t, PROB(v, t)) :- activation(v, s) // term_association(s, t)
    ''')
    with nl.scope as e:
        res = nl.query((e.v, e.p), e.r(e.v, "t0", e.p))
    print(len(res))
    """
)


def test_query_constant_keeps_large_program_feasible():
    """
    Solving ``r`` for all 2000 terms takes minutes; for one, a second.
    """
    try:
        out = subprocess.run(
            [sys.executable, "-c", SIZE_SCRIPT],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("query constant not propagated: query timed out")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "2000"
