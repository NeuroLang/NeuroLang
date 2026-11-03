"""The WMQL transpiler against tract_querier's own evaluator.

Both run on the same synthetic data: ``crossing`` (the labels each streamline
traverses) and ``endings`` (the label at each end). The reference reads them
as four dictionaries; the Datalog program reads them as the relations
``traverses`` and ``endpoint``. Every query *defined* by a program, not only
the saved ones, is compared on its four components: the streamlines, the
label set, and the streamlines whose first / last point is in the labels.

Data conventions (DESIGN.md, 6.3 and 11): the background, label 0, is among
the traversed labels (``emit_background=True``) and is never the label of an
endpoint, because ``endpoint`` has no background rows.
"""

import numpy as np
import pandas as pd
import pytest

from nl_tracts.wmql import transpile

from .reference_tq import Indexing, evaluate, load_reference

pytestmark = pytest.mark.slow  # 3 minutes: each test starts an engine

ATLAS = "fs"
TRACTOGRAMS = ("t1", "t2")


@pytest.fixture(scope="module")
def reference():
    return load_reference()


def synthetic(seed, n=60, labels=7):
    """crossing and endings of one tractogram."""
    rng = np.random.default_rng(seed)
    crossing = {}
    for s in range(n):
        k = rng.integers(0, 5)
        chosen = set(rng.choice(labels + 1, size=k, replace=False).tolist())
        if rng.random() < 0.5:
            chosen.add(0)  # the background, as in emit_background=True
        crossing[s] = chosen
    endings = tuple(
        {s: int(rng.integers(1, labels + 1)) for s in range(n)}
        for _ in range(2)
    )
    return crossing, endings


def relations(data):
    """``traverses`` and ``endpoint`` of {t: (crossing, endings)}."""
    traverses = [
        (ATLAS, t, s, label)
        for t, (crossing, _) in data.items()
        for s, labels in crossing.items()
        for label in sorted(labels)
    ]
    endpoint = [
        (ATLAS, t, s, e, label)
        for t, (_, endings) in data.items()
        for e in (0, 1)
        for s, label in endings[e].items()
        if label != 0
    ]
    return (
        pd.DataFrame(traverses, columns=list("atsl")),
        pd.DataFrame(endpoint, columns=list("atsel")),
    )


def run_datalog(program, data):
    from neurolang.frontend import NeurolangDL

    nl = NeurolangDL()
    traverses, endpoint = relations(data)
    nl.add_tuple_set(traverses, name="traverses")
    nl.add_tuple_set(endpoint, name="endpoint")
    nl.execute_datalog_program(program)
    return {
        name: {tuple(row) for row in sol.as_pandas_dataframe().itertuples(
            index=False, name=None)}
        for name, sol in nl.solve_all().items()
    }


def split_by_t(rows):
    out = {t: set() for t in TRACTOGRAMS}
    for t, s in rows:
        out[t].add(s)
    return out


def check_program(reference, source, seeds=(0, 1, 2)):
    program = transpile(source, atlas=ATLAS)
    for seed in seeds:
        data = {
            t: synthetic(seed * 10 + i) for i, t in enumerate(TRACTOGRAMS)
        }
        solution = run_datalog(program.text, data)
        for t, (crossing, endings) in data.items():
            expected = evaluate(reference, source, Indexing(crossing, endings))
            assert set(expected) == set(program.defined), (
                set(expected) ^ set(program.defined)
            )
            for name, q in program.defined.items():
                got = (
                    split_by_t(solution.get(q.tracts, set()))[t]
                    if q.tracts else set(),
                    {x for (x,) in solution.get(q.labels, set())}
                    if q.labels else set(),
                    split_by_t(solution.get(q.end0, set()))[t]
                    if q.end0 else set(),
                    split_by_t(solution.get(q.end1, set()))[t]
                    if q.end1 else set(),
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
    check_program(reference, BASE + expression + "\n")


def test_sequential_definitions_and_the_overwriting_of_names(reference):
    source = (
        BASE
        + "x = a or b\n"
        + "y = x and c\n"
        + "x |= d\n"  # overwrites: y keeps the first x
        + "z = x or y\n"
    )
    check_program(reference, source)


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
    check_program(reference, source)


def test_wildcards_match_only_the_names_defined_so_far(reference):
    source = "p_1 = 1\nearly = 'p_*'\np_2 = 2\nlate = 'p_*'\n"
    check_program(reference, source)


def test_names_are_case_insensitive(reference):
    source = "Alpha = 1\nBETA |= 2\nx = ALPHA or beta\ny = Only(Alpha)\n"
    check_program(reference, source)


def test_side_expands_to_left_and_right(reference):
    source = (
        "ctx.left = 1\nctx.right = 2\nput.left = 3\nput.right = 4\n"
        "ctx_put.side = ctx.side and put.side not in ctx.opposite\n"
        "both.side |= ctx.side or put.side\n"
        "other.side = (ctx.side and put.opposite) not in put.side\n"
    )
    check_program(reference, source)


def test_for_loops_over_a_list_and_a_pattern(reference):
    """Only the *names* equal to the loop variable are replaced, so a loop
    redefines the variable's values or the same query over and over."""
    source = (
        "r1 = 1\nr2 = 2\nr3 = 3\n"
        "for i in [r1, r2]:\n    i |= i or r3\n    g = only(i)\n"
        "for j in 'r*':\n    h = endpoints_in(j)\n    k = h or j\n"
    )
    check_program(reference, source)


def test_a_redefinition_is_a_new_definition(reference):
    check_program(reference, BASE + "x = a\ny = x or b\nx = c\nz = x or y\n")


def test_a_query_that_is_statically_empty(reference):
    source = BASE + "x = not a\ny = x and endpoints_in(b)\nz = y or c\n"
    check_program(reference, source)
