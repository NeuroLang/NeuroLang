"""Tests of how much studies cover the regions of an atlas."""

import nibabel as nib
import numpy as np
import pytest

from neurolang_apps.utils.engine_registry import (
    _register_atlas_coverage, atlas_coverage, cached_atlas_coverage,
)

SIGMA = 10 / np.sqrt(8 * np.log(2))
NAMES = {1: "a", 2: "b", 3: "c"}
AFFINE = np.array([[-2.0, 0, 0, 20.0], [0, 2.0, 0, -12.0], [0, 0, 2.0, 4.0],
                   [0, 0, 0, 1.0]])


def toy(seed=0, n_peaks=40, n_studies=9):
    """A 14 x 12 x 10 atlas of three regions and random peaks around it."""
    rng = np.random.default_rng(seed)
    data = np.zeros((14, 12, 10), dtype=np.int16)
    data[1:5, 2:6, 2:5] = 1
    data[8:13, 1:4, 5:9] = 2
    data[6, 9, 3] = 3
    img = nib.Nifti1Image(np.asfortranarray(data), AFFINE)  # as nibabel's
    low = nib.affines.apply_affine(AFFINE, [-8, -8, -8])
    high = nib.affines.apply_affine(AFFINE, [21, 19, 17])
    peaks = rng.uniform(np.minimum(low, high), np.maximum(low, high),
                        (n_peaks, 3))
    studies = rng.integers(0, n_studies, n_peaks) * 10
    return img, peaks, studies


def brute_force(img, peaks, studies):
    """The mean over each region of the max over the study's peaks of the
    Gaussian kernel truncated at 3 sigma, voxel by voxel."""
    data = np.asanyarray(img.dataobj)
    voxels = np.argwhere(data > 0)
    centres = nib.affines.apply_affine(img.affine, voxels)
    coverage = {}
    for study in np.unique(studies):
        distances = np.linalg.norm(
            centres[:, None] - peaks[studies == study][None], axis=-1
        )
        kernel = np.where(
            distances <= 3 * SIGMA, np.exp(-distances ** 2 / (2 * SIGMA ** 2)),
            0,
        ).max(axis=1)
        for value, name in NAMES.items():
            mean = kernel[data[tuple(voxels.T)] == value].mean()
            if mean >= 1e-3:
                coverage[name, int(study)] = mean
    return coverage


def as_dict(rows):
    return {
        (label, int(study)): p for p, label, study in
        rows[["probability", "label", "study"]].itertuples(index=False)
    }


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("chunk", [1, 3, 4096])
def test_the_coverage_is_the_brute_force_one(seed, chunk):
    img, peaks, studies = toy(seed)
    rows = atlas_coverage(img, NAMES, peaks, studies, chunk=chunk)
    found, expected = as_dict(rows), brute_force(img, peaks, studies)
    assert found.keys() == expected.keys() and len(rows) == len(found)
    assert all(np.isclose(found[k], expected[k]) for k in expected)


def test_a_peak_on_a_one_voxel_region_covers_it_surely():
    img, _, _ = toy()
    peak = nib.affines.apply_affine(AFFINE, [6, 9, 3])[None]
    rows = as_dict(atlas_coverage(img, NAMES, peak, np.array([7])))
    assert np.isclose(rows["c", 7], 1.0)


def test_the_rows_are_kept_and_read(tmp_path):
    img, peaks, studies = toy()
    found = cached_atlas_coverage(tmp_path, img, NAMES, peaks, studies)
    assert len(list(tmp_path.glob("atlas_coverage-*.npz"))) == 1
    read = cached_atlas_coverage(tmp_path, img, NAMES, peaks, studies)
    assert as_dict(read).keys() == as_dict(found).keys()
    assert all(
        np.isclose(as_dict(read)[k], p, atol=1e-6)
        for k, p in as_dict(found).items()
    )
    cached_atlas_coverage(tmp_path, img, NAMES, peaks, studies, fwhm_mm=8)
    assert len(list(tmp_path.glob("atlas_coverage-*.npz"))) == 2


def lifted_engine():
    """An engine that fails rather than falls back to weighted model
    counting: its probabilistic queries are answered by lifted plans."""
    from neurolang.frontend import NeurolangPDL
    from neurolang.probabilistic import (
        dalvi_suciu_lift, small_dichotomy_theorem_based_solver,
    )

    return NeurolangPDL(
        probabilistic_solvers=(
            small_dichotomy_theorem_based_solver.solve_succ_query,
            dalvi_suciu_lift.solve_succ_query,
        ),
        probabilistic_marg_solvers=(
            small_dichotomy_theorem_based_solver.solve_marg_query,
            dalvi_suciu_lift.solve_marg_query,
        ),
    )


QUERIES = {
    "forward": """
        c(l, s) :- toy_covered(l, s), selected_study(s)
        t(s) :- mentions(s, 'x'), selected_study(s)
        q(l, p) :- PROB[c(l, s) // t(s)] = p""",
    "reverse": """
        c(l, s) :- toy_covered(l, s), selected_study(s)
        t(s) :- mentions(s, 'x'), selected_study(s)
        q(l, p) :- PROB[t(s) // c(l, s)] = p""",
    "co-coverage": """
        co(l1, l2, s) :- toy_covered(l1, s), toy_covered(l2, s),
                         selected_study(s)
        q(l1, l2, p) :- (l1 < l2), PROB[co(l1, l2, s)] = p""",
    "not covered": """
        n(l, s) :- label(l), selected_study(s), ~toy_covered(l, s)
        q(l, p) :- PROB[n(l, s)] = p""",
    "union": """
        u(s) :- toy_covered('a', s), selected_study(s)
        u(s) :- toy_covered('b', s), selected_study(s)
        t(s) :- mentions(s, 'x'), selected_study(s)
        q(p) :- PROB[u(s) // t(s)] = p""",
}


def expected(name, cover, studies, term):
    """Brute force: a study is chosen uniformly, then each of its facts
    holds independently with its probability."""
    def p(label, s):
        return cover.get((label, s), 0.0)

    def given(f, cond):
        return (sum(f(s) * cond(s) for s in studies)
                / sum(cond(s) for s in studies))

    labels = sorted(NAMES.values())
    if name == "forward":
        return {(l,): given(lambda s: p(l, s), lambda s: s in term)
                for l in labels}
    if name == "reverse":
        return {(l,): given(lambda s: s in term, lambda s: p(l, s))
                for l in labels if any(p(l, s) for s in studies)}
    if name == "co-coverage":
        return {(a, b): given(lambda s: p(a, s) * p(b, s), lambda s: 1)
                for a in labels for b in labels if a < b}
    if name == "not covered":
        return {(l,): given(lambda s: 1 - p(l, s), lambda s: 1)
                for l in labels}
    return {(): given(lambda s: 1 - (1 - p("a", s)) * (1 - p("b", s)),
                      lambda s: s in term)}


@pytest.mark.parametrize("name", QUERIES)
def test_lifted_queries_over_the_coverage_are_the_brute_force_ones(name):
    img, peaks, studies = toy(1)
    all_studies = sorted(set(int(s) for s in studies)) + [1000]
    term = set(all_studies[::2])
    nl = lifted_engine()
    nl.add_tuple_set(
        [(*peak, int(s)) for peak, s in zip(peaks, studies)], name="peak_mni"
    )
    _register_atlas_coverage(nl, NAMES, img, {}, "toy")
    nl.add_uniform_probabilistic_choice_over_set(
        [(s,) for s in all_studies], name="selected_study"
    )
    nl.add_tuple_set([(s, "x") for s in term], name="mentions")
    nl.add_tuple_set([(l,) for l in NAMES.values()], name="label")
    cover = as_dict(atlas_coverage(img, NAMES, peaks, studies))
    with nl.scope:
        nl.execute_datalog_program(QUERIES[name])
        frame = nl.solve_all()["q"].as_pandas_dataframe()
    found = {tuple(row[:-1]): row[-1] for row in frame.itertuples(index=False)}
    want = {  # rows of probability 0 are not in a result
        k: v for k, v in expected(name, cover, all_studies, term).items()
        if v > 0
    }
    assert found.keys() == want.keys()
    assert all(np.isclose(found[k], want[k]) for k in want)
