"""
The probabilities computed as tensor contractions (the provenance
solver's step ``SolveAsContractions``) are those of relational algebra
and those of the possible worlds: random programs of a choice,
probabilistic facts and deterministic relations with duplicate rows,
probabilities of 0 and 1, constants and repeated variables in heads and
atoms, self-joins, conditionals and resamplings.
"""
import itertools

import numpy as np
import pytest

from neurolang.frontend import NeurolangPDL
from nl_core.probabilistic import sums_of_products
from nl_core.relational_algebra.linear_algebra.rewrites import NotApplicable
from nl_expr.config import config


@pytest.fixture(autouse=True)
def no_failed_contractions(caplog):
    """A plan whose contraction fails, rather than declines, is computed
    by relational algebra, with a warning: a test fails on it."""
    yield
    assert not [
        r for r in caplog.records if r.name == sums_of_products.LOG.name
    ]


@pytest.fixture
def contractions(monkeypatch):
    """The plans computed as contractions, by solve: those whose
    evaluation gave an answer. An error of an evaluation (which leaves
    the plan to relational algebra, with its answers) fails the test;
    a plan the backend does not compute (NotApplicable) does not."""
    computed, errors = [], []
    evaluate = sums_of_products.evaluate

    def recording(plan, *args):
        try:
            answer = evaluate(plan, *args)
        except NotApplicable:
            raise
        except Exception as error:
            errors.append(error)
            raise
        computed.append(plan)
        return answer

    monkeypatch.setattr(sums_of_products, "evaluate", recording)
    yield computed
    assert not errors, f"contractions failed: {errors!r}"


def _solve(nl, program, contract=True):
    previous = config.get("PROBABILISTIC_SOLVER", "contractions")
    config.set("PROBABILISTIC_SOLVER", "contractions", str(contract))
    try:
        with nl.scope:
            result = nl.execute_datalog_program(program)
    finally:
        config.set("PROBABILISTIC_SOLVER", "contractions", previous)
    return {
        tuple(row[:-1]): row[-1]
        for row in result.as_pandas_dataframe().itertuples(
            index=False, name=None
        )
    }


class Const:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return repr(self.value)


def _text(atoms):
    return " & ".join(
        f"{name}({', '.join(map(str, args))})" for name, args in atoms
    )


# Queries: the head's terms and the body's atoms, or the conditioned and
# conditioning atoms of a conditional; variables are strings.
QUERIES = {
    "marginal": (("v",), [("act", ("v", "s")), ("sel", ("s",))]),
    "constant_in_body": (("v",), [
        ("act", ("v", "s")), ("sel", ("s",)), ("men", ("s", Const("t0"))),
    ]),
    "conditional": (("v",), (
        [("act", ("v", "s")), ("sel", ("s",))],
        [("men", ("s", Const("t0"))), ("sel", ("s",))],
    )),
    "pairs": (("v1", "v2"), [
        ("act", ("v1", "s")), ("act", ("v2", "s")), ("sel", ("s",)),
    ]),
    "pair_conditional": (("v1", "v2", "t"), (
        [("act", ("v1", "s")), ("act", ("v2", "s")), ("sel", ("s",))],
        [("men", ("s", "t")), ("sel", ("s",))],
    )),
    "repeated_head_variable": (("v", "v"), [
        ("act", ("v", "s")), ("sel", ("s",)),
    ]),
    "constant_in_head": ((Const("c"), "v"), [
        ("act", ("v", "s")), ("sel", ("s",)),
    ]),
    "two_free_variables": (("t", "v"), [
        ("act", ("v", "s")), ("men", ("s", "t")), ("sel", ("s",)),
    ]),
    "facts": (("v",), [
        ("pf", ("v",)), ("act", ("v", "s")), ("sel", ("s",)),
    ]),
    "repeated_variable_in_atom": (("s",), [
        ("sym", ("v", "v", "s")), ("sel", ("s",)), ("pf", ("v",)),
    ]),
    "scalar": ((), [("act", (Const(1), "s")), ("sel", ("s",))]),
    "noisy_or": (("s",), [("pf", ("v",)), ("act", ("v", "s"))]),
}
# Those of the first solver, computed as contractions.
# Every shape is computed as contractions, noisy-or (an independent
# projection over a fact variable out of any choice) included.
CONTRACTED = set(QUERIES)


def _random_program(seed):
    """A choice sel of a study s (weights of 0 included), facts pf of a
    voxel v (probabilities of 0 and 1 included), deterministic act(v, s),
    men(s, t) and sym(v, w, s) with duplicate rows."""
    rng = np.random.default_rng(seed)
    studies = range(int(rng.integers(1, 5)))
    weights = rng.random(len(studies)) * (rng.random(len(studies)) < .8)
    weights = weights + (weights.sum() == 0)
    weights /= weights.sum()
    relations = {
        "sel": {(s,): float(w) for s, w in zip(studies, weights)},
        "pf": {
            (v,): float(rng.choice([0., 1., rng.random()]))
            for v in range(3)
        },
    }
    voxels = range(3)
    relations["act"] = [
        (v, s) for v, s in itertools.product(voxels, studies)
        if rng.random() < .5
    ]
    relations["men"] = [
        (s, t) for s, t in itertools.product(studies, ("t0", "t1"))
        if rng.random() < .5
    ]
    relations["sym"] = [
        (v, w, s) for v, w, s in itertools.product(voxels, voxels, studies)
        if rng.random() < .3
    ]
    nl = NeurolangPDL()
    for name in ("act", "men", "sym"):
        rows = relations[name]
        nl.add_tuple_set(rows + rows[:2] or [(-1,) * (3 if name == "sym"
                                                      else 2)], name=name)
    nl.add_probabilistic_choice_from_tuples(
        [(w, s) for (s,), w in relations["sel"].items()], name="sel"
    )
    nl.add_probabilistic_facts_from_tuples(
        [(p, v) for (v,), p in relations["pf"].items()], name="pf"
    )
    return nl, relations


def _answers(atoms, world, head):
    """The answers of the conjunction *atoms* in *world*."""
    bindings = [{}]
    for name, args in atoms:
        bindings = [
            {**binding, **dict(zip(args, row))}
            for binding in bindings for row in world[name]
            if all(
                (arg.value == value) if isinstance(arg, Const)
                else binding.get(arg, value) == value
                for arg, value in zip(args, row)
            ) and all(
                a == b or not (a == b) is False
                for a, b in [(row[i], row[j])
                             for i, x in enumerate(args)
                             for j, y in enumerate(args)
                             if x == y and not isinstance(x, Const)]
            )
        ]
    return {
        tuple(t.value if isinstance(t, Const) else b[t] for t in head)
        for b in bindings
    }


def _worlds(relations):
    """The possible worlds and their probabilities."""
    facts = list(relations["pf"].items())
    for (study,), weight in relations["sel"].items():
        for chosen in itertools.product((False, True), repeat=len(facts)):
            probability = weight * np.prod([
                p if c else 1 - p for (_, p), c in zip(facts, chosen)
            ])
            world = {
                "sel": [(study,)],
                "pf": [f for (f, _), c in zip(facts, chosen) if c],
                "act": set(relations["act"]),
                "men": set(relations["men"]),
                "sym": set(relations["sym"]),
            }
            yield world, probability


def _brute_force(relations, head, body):
    """The probabilities of the answers, by summing over the worlds."""
    conditioned, conditioning = (
        body if isinstance(body, tuple) else (body, None)
    )
    joint, evidence = {}, {}
    for world, probability in _worlds(relations):
        atoms = conditioned + (conditioning or [])
        for answer in _answers(atoms, world, head):
            joint[answer] = joint.get(answer, 0) + probability
        if conditioning is not None:
            given = [a for a in head if a in {
                x for _, args in conditioning for x in args
            }]
            for answer in _answers(conditioning, world, given):
                evidence[answer] = evidence.get(answer, 0) + probability
    if conditioning is not None:
        given = [i for i, a in enumerate(head) if a in {
            x for _, args in conditioning for x in args
        }]
        joint = {
            answer: p / evidence[tuple(answer[i] for i in given)]
            for answer, p in joint.items()
        }
    return {answer: p for answer, p in joint.items() if p > 0}


def _program(head, body):
    if isinstance(body, tuple):
        formula = f"{_text(body[0])} // {_text(body[1])}"
    else:
        formula = _text(body)
    terms = ", ".join(map(str, head + ("p",)))
    variables = ", ".join(f"x{i}" for i in range(len(head)))
    return (
        f"q({terms}) :- PROB[{formula}] = p\n"
        f"ans({variables}{', ' if head else ''}p) :- q({variables}"
        f"{', ' if head else ''}p)"
    )


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("name", sorted(QUERIES))
def test_random_programs(seed, name, contractions):
    """The same probabilities as relational algebra, as summed over the
    possible worlds."""
    head, body = QUERIES[name]
    nl, relations = _random_program(seed)
    program = _program(head, body)
    contracted = _solve(nl, program)
    relational = _solve(nl, program, contract=False)
    expected = _brute_force(relations, head, body)
    for result in (contracted, relational):
        assert set(result) == set(expected)
        for key, value in expected.items():
            assert result[key] == pytest.approx(value, rel=1e-12, abs=0)
    assert bool(contractions) == (name in CONTRACTED), contractions


class _Unordered:
    """A value with equality and a hash but no order, as a region."""

    def __init__(self, n):
        self.n = n

    def __eq__(self, other):
        return isinstance(other, _Unordered) and other.n == self.n

    def __hash__(self):
        return hash(self.n)


_A, _B = _Unordered(0), _Unordered(1)
_NAN = float("nan")


@pytest.mark.parametrize("dense_cells", [0, 1000000])
@pytest.mark.parametrize("act, sel, pf, body, head, expected", [
    # values without an order: no index of them to number
    ([(0, _A), (1, _B), (2, _B)], [(.4, _A), (.6, _B)], [(.3, _A)],
     "act(v, s) & sel(s)", "v", {(0,): .4, (1,): .6, (2,): .6}),
    ([(0, _A), (1, _B), (2, _B)], [(.4, _A), (.6, _B)], [(.3, _A)],
     "act(v, s) & pf(s)", "v", {(0,): .3}),
    ([(0, 0), (1, "a")], [(.5, 0), (.5, "a")], [(.3, 0)],
     "act(v, s) & sel(s)", "v", {(0,): .5, (1,): .5}),
    # a tuple of two probabilities: two facts, two outcomes of the choice
    ([(0, 0), (0, 1), (1, 1), (2, 0)], [(.5, 0), (.5, 1)],
     [(.3, 0), (.5, 0), (.6, 1)],
     "pf(v) & act(v, s)", "s", {(0,): 1 - .7 * .5, (1,): 1 - .7 * .5 * .4}),
    ([(0, 0), (1, 1), (2, 0)], [(.2, 0), (.3, 0), (.5, 1)], [(.3, 0)],
     "act(v, s) & sel(s)", "v", {(0,): .5, (1,): .5, (2,): .5}),
    # missing values, which relational algebra joins with each other
    ([(0, 0.), (1, _NAN), (2, 1.)], [(.5, 0.), (.5, _NAN)], [(.3, 0.)],
     "act(v, s) & sel(s)", "v", {(0,): .5, (1,): .5}),
    ([(0, "a"), (1, None)], [(.5, "a"), (.5, None)], [(.3, "a")],
     "act(v, s) & sel(s)", "v", {(0,): .5, (1,): .5}),
])
def test_relations_the_translator_declines(
    act, sel, pf, body, head, expected, dense_cells
):
    """Values without an order, tuples of two probabilities and missing
    values in a joined column are left to relational algebra, whose
    answers they keep."""
    previous = config.get("PROBABILISTIC_SOLVER", "dense_cells")
    config.set("PROBABILISTIC_SOLVER", "dense_cells", str(dense_cells))
    nl = NeurolangPDL()
    nl.add_tuple_set(act, name="act")
    nl.add_probabilistic_choice_from_tuples(sel, name="sel")
    nl.add_probabilistic_facts_from_tuples(pf, name="pf")
    try:
        result = _solve(
            nl, f"q({head}, p) :- PROB[{body}] = p\nans(x, p) :- q(x, p)"
        )
    finally:
        config.set("PROBABILISTIC_SOLVER", "dense_cells", previous)
    assert set(result) == set(expected)
    for key, value in expected.items():
        assert result[key] == pytest.approx(value, rel=1e-12)


@pytest.mark.parametrize("uniform", [True, False])
def test_a_view_by_one_number_for_every_entity_is_its_sums(
    uniform, contractions, monkeypatch
):
    """
    A view times the same number for every entity of a sort, summed
    over it (the probability of a voxel's activation over a uniform
    choice of all the studies), is that number times the sums of the
    view kept with it: the view is not read (57 s cold for the 1 billion
    entries of neurostore's activated). By another choice, the view is
    multiplied by its vector. The answers are those of relational
    algebra.
    """
    from scipy import sparse

    from nl_core.relational_algebra.linear_algebra import (
        Axis,
        ViewDeclaration,
        build_sparse_views,
    )

    rng = np.random.default_rng(0)
    voxels = list(itertools.product(range(3), range(2), range(2)))
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [v + (s,) for v in voxels for s in range(5) if rng.random() < .5],
        name="activated",
    )
    weights = [.2] * 5 if uniform else [.1, .2, .3, .15, .25]
    nl.add_probabilistic_choice_from_tuples(
        list(zip(weights, range(5))), name="sel"
    )
    nl.sparse_views = build_sparse_views(
        nl.program_ir.symbol_table,
        {"activated": ViewDeclaration(
            Axis("voxel", (0, 1, 2)), Axis("study", (3,))
        )},
    )
    read = []
    matmul = sparse._compressed._cs_matrix._matmul_vector
    monkeypatch.setattr(
        sparse._compressed._cs_matrix, "_matmul_vector",
        lambda self, other: read.append(self.nnz) or matmul(self, other),
    )
    program = (
        "q(i, j, k, p) :- PROB[activated(i, j, k, s) & sel(s)] = p\n"
        "ans(i, j, k, p) :- q(i, j, k, p)"
    )
    contracted = _solve(nl, program)
    assert contractions
    view_nnz = nl.sparse_views["activated"].matrix.nnz
    assert (view_nnz in read) != uniform
    relational = _solve(nl, program, contract=False)
    assert set(contracted) == set(relational)
    for key, value in relational.items():
        assert contracted[key] == pytest.approx(value, rel=1e-12)


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("few_rows", [True, False])
@pytest.mark.parametrize("body, head", [
    ("activated(i, j, k, s) & sel(s)", "i, j, k"),
    ("activated(i, j, k, s) & sel(s) // term(s, t) & sel(s)", "i, j, k, t"),
])
def test_views_are_read_as_they_are(
    body, head, few_rows, indexed, contractions, monkeypatch
):
    """A relation of an engine's sparse view is read as its matrix,
    neither copied nor converted to another layout (a transposed view is
    a CSC matrix: the other operand of a product is converted to it; of
    a product of two views, one is), with the answers of relational
    algebra. By a few rows (the terms), it is a product by a dense
    matrix, not a sparse product; a sparse product takes operands of
    the view's type of indices, so it converts neither. Indexed by its
    columns (the studies), the view is read through its index by the
    studies of a term, restricted to them, and by a vector of studies."""
    from scipy import sparse

    from nl_core.relational_algebra.linear_algebra import (
        Axis,
        ViewDeclaration,
        build_sparse_views,
        scipy_backend,
    )

    rng = np.random.default_rng(0)
    voxels = list(itertools.product(range(3), range(2), range(2)))
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [v + (s,) for v in voxels for s in range(5) if rng.random() < .5],
        name="activated",
    )
    nl.add_tuple_set(
        [(0, "pain"), (1, "pain"), (2, "fear"), (3, "fear"), (4, "pain")],
        name="term",
    )
    nl.add_tuple_set([(0, 0, 0), (1, 1, 1), (2, 0, 1)], name="region")
    nl.add_probabilistic_choice_from_tuples(
        [(.1, 0), (.2, 1), (.3, 2), (.15, 3), (.25, 4)], name="sel"
    )
    nl.sparse_views = build_sparse_views(
        nl.program_ir.symbol_table,
        {"activated": ViewDeclaration(
            Axis("voxel", (0, 1, 2)), Axis("study", (3,))
        )},
        indexed=["activated"] if indexed else (),
    )
    view_nnz = nl.sparse_views["activated"].matrix.nnz
    read_by_columns = []
    if indexed:
        # every study is few enough, in so small a view
        monkeypatch.setitem(
            config["PROBABILISTIC_SOLVER"], "index_share", "1"
        )
        index = nl.sparse_views["activated"].by_columns
        matrix = index.matrix
        monkeypatch.setattr(
            index, "matrix",
            lambda columns: read_by_columns.append(len(columns))
            or matrix(columns),
        )
    converted = []
    for kind, method in (
        (sparse.csc_array, "tocsr"), (sparse.csr_array, "tocsc")
    ):
        convert = getattr(kind, method)
        monkeypatch.setattr(
            kind, method, lambda self, *a, convert=convert, **k: (
                converted.append(self.nnz) or convert(self, *a, **k)
            ),
        )
    if not few_rows:
        monkeypatch.setitem(
            config["PROBABILISTIC_SOLVER"], "few_cells", "0"
        )
    multiplied = []
    matmul = sparse._compressed._cs_matrix._matmul_sparse
    monkeypatch.setattr(
        sparse._compressed._cs_matrix, "_matmul_sparse",
        lambda self, other: multiplied.append((
            self.nnz, other.nnz, self.indices.dtype, other.indices.dtype
        )) or matmul(self, other),
    )
    previous = config.get("PROBABILISTIC_SOLVER", "dense_cells")
    config.set("PROBABILISTIC_SOLVER", "dense_cells", "0")
    try:
        program = (
            f"q({head}, p) :- PROB[{body}] = p\n"
            f"ans({head}, p) :- q({head}, p)"
        )
        contracted = _solve(nl, program)
        assert contractions and view_nnz not in converted
        assert all(a == b for _, _, a, b in multiplied)
        # with every study few enough, the index is read for the studies
        # of a term and for the vector of the choice over the studies
        assert bool(read_by_columns) == indexed
        assert not few_rows or read_by_columns or not any(
            view_nnz in (n, m) for n, m, _, _ in multiplied
        )
        relational = _solve(nl, program, contract=False)
    finally:
        config.set("PROBABILISTIC_SOLVER", "dense_cells", previous)
    assert set(contracted) == set(relational)
    for key, value in relational.items():
        assert contracted[key] == pytest.approx(value, rel=1e-12)


def test_a_view_restricted_by_magic_sets_is_read_as_its_matrix(
    contractions, monkeypatch
):
    """A view bound by magic sets by a relation (the voxels of an image)
    is restricted in the plan by the join with its magic predicate, which
    the product applies to the view's matrix: no restricted copy, the
    answers of relational algebra."""
    from nl_core.relational_algebra.linear_algebra import (
        Axis,
        ViewDeclaration,
        build_sparse_views,
        scipy_backend,
    )

    rng = np.random.default_rng(0)
    voxels = list(itertools.product(range(4), range(2), range(2)))
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [v + (s,) for v in voxels for s in range(6) if rng.random() < .5],
        name="stored_activated",
    )
    nl.add_tuple_set([(0, 0, 0), (1, 1, 1), (2, 0, 1)], name="mask")
    nl.add_tuple_set(
        [(s, t) for s in range(6) for t in ("pain", "fear")
         if rng.random() < .5],
        name="term_in_study",
    )
    nl.add_uniform_probabilistic_choice_over_set(
        [(s,) for s in range(6)], name="selected_study"
    )
    nl.add_uniform_probabilistic_choice_over_set(
        voxels, name="selected_voxel"
    )
    nl.execute_datalog_program(
        "activated(i, j, k, s) :- stored_activated(i, j, k, s)"
    )
    nl.sparse_views = build_sparse_views(
        nl.program_ir.symbol_table,
        {"activated": ViewDeclaration(
            Axis("voxel", (0, 1, 2)), Axis("study", (3,))
        )},
    )
    viewed = []
    view_of = scipy_backend._view_of
    monkeypatch.setattr(
        scipy_backend, "_view_of",
        lambda factor: (lambda found: viewed.append(found) or found)(
            view_of(factor)
        ),
    )
    program = """
image_active(i, j, k) :- mask(i, j, k)
image_hit(s) :- selected_voxel(i, j, k), image_active(i, j, k), \
activated(i, j, k, s)
q(t, PROB) :- (image_hit(s) & selected_study(s)) // \
(term_in_study(s, t) & selected_study(s))
ans(t, p) :- q(t, p)
"""
    contracted = _solve(nl, program)
    assert contractions and any(found is not None for found in viewed)
    relational = _solve(nl, program, contract=False)
    assert set(contracted) == set(relational)
    for key, value in relational.items():
        assert contracted[key] == pytest.approx(value, rel=1e-12)


def test_a_contraction_failing_falls_back(caplog, monkeypatch):
    """A contraction that fails, its answer's set included (a backend's
    set not taking the frame), is logged, and the plan computed by
    relational algebra: the answers do not change."""
    nl = NeurolangPDL()
    nl.add_tuple_set([(0, 0), (0, 1), (1, 1)], name="act")
    nl.add_probabilistic_choice_from_tuples(
        [(.4, 0), (.6, 1)], name="sel"
    )
    program = "q(v, p) :- PROB[act(v, s) & sel(s)] = p\nans(v, p) :- q(v, p)"
    expected = _solve(nl, program, contract=False)

    def failing(*args):
        raise RuntimeError("a set not taking the frame")

    monkeypatch.setattr(
        sums_of_products, "NamedRelationalAlgebraFrozenSet", failing
    )
    assert _solve(nl, program) == pytest.approx(expected)
    assert any(
        r.name == sums_of_products.LOG.name for r in caplog.records
    )
    caplog.clear()


@pytest.mark.parametrize("body, head", [
    # facts on a few voxels: the rows of the view (an image's voxels)
    ("activated(i, j, k, s) & pv(i, j, k) & sel(s)", "s"),
    # facts on a few studies: its columns
    ("activated(i, j, k, s) & ps(s)", "i, j, k"),
])
def test_a_view_weighted_by_a_few_facts(body, head, contractions, caplog):
    """A view weighted by probabilistic facts on a few of its rows (the
    voxels of an image) or columns (studies) is scaled by the diagonal of
    their probabilities, with the answers of relational algebra."""
    from nl_core.relational_algebra.linear_algebra import (
        Axis,
        ViewDeclaration,
        build_sparse_views,
    )

    rng = np.random.default_rng(1)
    voxels = list(itertools.product(range(3), range(2), range(2)))
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [v + (s,) for v in voxels for s in range(5) if rng.random() < .6],
        name="activated",
    )
    nl.add_probabilistic_facts_from_tuples(
        [(.9, 0, 0, 0), (.4, 1, 1, 1), (.7, 2, 0, 1)], name="pv"
    )
    nl.add_probabilistic_facts_from_tuples([(.3, 1), (.8, 3)], name="ps")
    nl.add_probabilistic_choice_from_tuples(
        [(.1, 0), (.2, 1), (.3, 2), (.15, 3), (.25, 4)], name="sel"
    )
    nl.sparse_views = build_sparse_views(
        nl.program_ir.symbol_table,
        {"activated": ViewDeclaration(
            Axis("voxel", (0, 1, 2)), Axis("study", (3,))
        )},
    )
    previous = config.get("PROBABILISTIC_SOLVER", "dense_cells")
    config.set("PROBABILISTIC_SOLVER", "dense_cells", "0")
    try:
        program = (
            f"q({head}, p) :- PROB[{body}] = p\n"
            f"ans({head}, p) :- q({head}, p)"
        )
        contracted = _solve(nl, program)
        assert contractions
        # an error of the contraction would leave it to relational algebra
        assert "not computed as contractions" not in caplog.text
        relational = _solve(nl, program, contract=False)
    finally:
        config.set("PROBABILISTIC_SOLVER", "dense_cells", previous)
    assert set(contracted) == set(relational) and relational
    for key, value in relational.items():
        assert contracted[key] == pytest.approx(value, rel=1e-12)


@pytest.mark.parametrize(
    "index_share, rows_alone", [("1", True), ("0", False)]
)
def test_a_choice_over_a_few_rows_of_a_view(
    index_share, rows_alone, contractions, caplog, monkeypatch
):
    """A choice over a few voxels (those of an image) by the voxel by
    study view, summed over the voxels: the product reads those rows of the
    view alone when their entries are at most ``index_share`` of the
    view's, the whole view otherwise; the answers of relational algebra."""
    from nl_core.relational_algebra.linear_algebra import (
        Axis,
        ViewDeclaration,
        build_sparse_views,
        scipy_backend,
    )

    rng = np.random.default_rng(2)
    voxels = list(itertools.product(range(3), range(2), range(2)))
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [v + (s,) for v in voxels for s in range(5) if rng.random() < .6],
        name="activated",
    )
    nl.add_probabilistic_choice_from_tuples(
        [(.5, 0, 0, 0), (.3, 1, 1, 1), (.2, 2, 0, 1)], name="cv"
    )
    nl.sparse_views = build_sparse_views(
        nl.program_ir.symbol_table,
        {"activated": ViewDeclaration(
            Axis("voxel", (0, 1, 2)), Axis("study", (3,))
        )},
    )
    monkeypatch.setitem(
        config["PROBABILISTIC_SOLVER"], "index_share", index_share
    )
    few = []
    record = scipy_backend._few
    monkeypatch.setattr(
        scipy_backend, "_few",
        lambda entries, nnz: few.append(record(entries, nnz)) or few[-1],
    )
    previous = config.get("PROBABILISTIC_SOLVER", "dense_cells")
    config.set("PROBABILISTIC_SOLVER", "dense_cells", "0")
    try:
        program = (
            "q(s, p) :- PROB[cv(i, j, k) & activated(i, j, k, s)] = p\n"
            "ans(s, p) :- q(s, p)"
        )
        contracted = _solve(nl, program)
        assert contractions
        assert "not computed as contractions" not in caplog.text
        relational = _solve(nl, program, contract=False)
    finally:
        config.set("PROBABILISTIC_SOLVER", "dense_cells", previous)
    assert few and any(few) is rows_alone
    assert set(contracted) == set(relational) and relational
    for key, value in relational.items():
        assert contracted[key] == pytest.approx(value, rel=1e-12)
