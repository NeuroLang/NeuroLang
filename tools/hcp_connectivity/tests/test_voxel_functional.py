"""The partial correlations of cortical voxels with the parcels of an atlas,
on small synthetic data: against a brute-force reference, the worker and
the group table."""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hcp_connectivity import combine, voxel_functional  # noqa: E402
from hcp_connectivity.voxel_functional import (  # noqa: E402
    GroupSums, centres, connectivity, family_weights, ledoit_wolf_alpha,
    load_phenotypes, partial_correlations)

N_PARCELS = 4
PER_PARCEL = 6
KEY = "schaefer400_17_voxel_partial"


def synthetic(T=200, seed=0):
    """T time points of N_PARCELS * PER_PARCEL voxels: a signal of the
    parcel (the parcels correlated), the voxel's own and noise, the
    voxels' variances different; the parcel of each voxel, from 1."""
    rng = np.random.default_rng(seed)
    parcels = np.repeat(np.arange(1, N_PARCELS + 1), PER_PARCEL)
    mixing = np.eye(N_PARCELS) + 0.4 * rng.standard_normal(
        (N_PARCELS, N_PARCELS))
    signals = rng.standard_normal((T, N_PARCELS)) @ mixing
    series = signals[:, parcels - 1] + rng.standard_normal(
        (T, len(parcels)))
    series = series * rng.uniform(0.5, 3.0, len(parcels)) + 100
    return series, parcels


def reference(series, parcels, n, alpha):
    """For each voxel, the matrix of the correlations of the voxel with the
    parcels' means, its own parcel's without it, shrunk by *alpha* and
    inverted: the partial correlations."""
    T, V = series.shape
    out = np.zeros((V, n))
    for v in range(V):
        means = np.column_stack([
            series[:, [u for u in range(V)
                       if parcels[u] == j + 1 and u != v]].mean(axis=1)
            for j in range(n)])
        correlation = np.corrcoef(np.column_stack([series[:, v], means]).T)
        shrunk = (1 - alpha) * correlation + alpha * np.eye(n + 1)
        precision = np.linalg.inv(shrunk)
        out[v] = -precision[0, 1:] / np.sqrt(
            precision[0, 0] * np.diag(precision)[1:])
    return out


@pytest.mark.parametrize("alpha", [0.0, 0.05, 0.4])
def test_partial_correlations_are_those_of_the_reference(alpha):
    series, parcels = synthetic()
    expected = reference(series, parcels, N_PARCELS, alpha)
    found = partial_correlations(series, parcels, N_PARCELS, alpha)
    assert found.shape == expected.shape
    np.testing.assert_allclose(found, expected, atol=1e-8)


def test_chunks_of_voxels_do_not_change_the_result():
    series, parcels = synthetic(seed=1)
    whole = partial_correlations(series, parcels, N_PARCELS, 0.1)
    chunked = partial_correlations(series, parcels, N_PARCELS, 0.1, chunk=5)
    np.testing.assert_allclose(chunked, whole, atol=1e-10)


def test_without_shrinkage_they_are_those_of_regression_residuals():
    """The correlation of the residuals of the voxel and of the parcel,
    each regressed on the other parcels."""
    series, parcels = synthetic(seed=2)
    n = N_PARCELS
    found = partial_correlations(series, parcels, n, 0.0)
    for v in (0, 7, 13, 23):
        means = np.column_stack([
            series[:, [u for u in range(len(parcels))
                       if parcels[u] == j + 1 and u != v]].mean(axis=1)
            for j in range(n)])
        for j in range(n):
            others = np.delete(means, j, axis=1)
            others = np.column_stack([np.ones(len(others)), others])

            def residual(y):
                return y - others @ np.linalg.lstsq(others, y, rcond=None)[0]

            r = np.corrcoef(residual(series[:, v]), residual(means[:, j]))
            assert found[v, j] == pytest.approx(r[0, 1], abs=1e-8)


def test_a_voxel_is_not_part_of_its_own_parcel_signal():
    """A voxel that is its parcel's only signal-carrier would be perfectly
    correlated with a mean that contains it; left out, its partial
    correlation with its parcel is far from 1."""
    series, parcels = synthetic(seed=3)
    series[:, 0] = series[:, 0] * 50  # dominates the mean of parcel 1
    found = partial_correlations(series, parcels, N_PARCELS, 0.0)
    assert abs(found[0, 0]) < 0.9


def test_the_ledoit_wolf_coefficient_is_that_of_the_standardised_means():
    from sklearn.covariance import ledoit_wolf_shrinkage

    series, parcels = synthetic(seed=4)
    means = np.column_stack([series[:, parcels == j + 1].mean(axis=1)
                             for j in range(N_PARCELS)])
    alpha = ledoit_wolf_alpha(series, parcels, N_PARCELS)
    assert 0 < alpha < 1
    standard = (means - means.mean(0)) / means.std(0)
    assert alpha == pytest.approx(ledoit_wolf_shrinkage(standard))


def test_connectivity_is_the_fisher_z_of_the_partial_correlations():
    series, parcels = synthetic(seed=5)
    alpha = ledoit_wolf_alpha(series, parcels, N_PARCELS)
    expected = np.arctanh(np.clip(
        partial_correlations(series, parcels, N_PARCELS, alpha),
        -0.999999, 0.999999))
    found = connectivity(series, parcels, N_PARCELS)
    assert found.dtype == np.float32
    np.testing.assert_allclose(found, expected, atol=1e-5)


def test_centres_of_the_labelled_voxels_in_millimetres():
    labels = np.zeros((3, 3, 3), dtype=int)
    labels[0, 0, 0], labels[2, 1, 0], labels[1, 1, 1] = 1, 2, 3
    affine = np.array([[-2.0, 0, 0, 90], [0, 2, 0, -126], [0, 0, 2, -72],
                       [0, 0, 0, 1]])
    voxels, mm = centres(labels, affine)
    # C order of the labelled voxels
    assert voxels.tolist() == [[0, 0, 0], [1, 1, 1], [2, 1, 0]]
    assert mm.tolist() == [[90, -126, -72], [88, -124, -70],
                           [86, -124, -72]]


# The group: each voxel's z averaged over the subjects with signal in it,
# each family counted once, adjusted for age and sex.

def simulate(S=40, V=7, n=3, seed=0):
    """The z of S subjects (subjects, voxels, parcels), NaN for the voxels
    without signal in a subject, with their ages, sexes (1 male) and the
    family weights; z depends on age and sex."""
    rng = np.random.default_rng(seed)
    ages = rng.uniform(22, 37, S)
    male = rng.integers(0, 2, S).astype(float)
    family = rng.integers(0, 15, S)
    weights = 1.0 / np.bincount(family)[family]
    z = (rng.standard_normal((S, V, n)) + 0.05 * (ages - 30)[:, None, None]
         + 0.3 * male[:, None, None]).astype(np.float32)
    z[rng.random((S, V)) < 0.2] = np.nan  # a voxel without signal
    z[:, 0] = np.nan                       # never any signal
    z[1:, 1] = np.nan                      # only subject 0 has signal
    return z, ages, male, weights


def summed(z, ages, male, weights, V=7, n=3):
    sums = GroupSums(V, n)
    for s in range(len(z)):
        sums.add(f"s{s}", z[s], weights[s], ages[s], male[s])
    return sums


def regression_reference(z, ages, male, weights):
    """The weighted least squares of z on the age and the sex of the
    subjects with signal in a voxel, read at the weighted mean age of all
    the subjects and an equal mix of the sexes: tanh."""
    S, V, n = z.shape
    reference_age = np.average(ages, weights=weights)
    out = np.full((V, n), np.nan)
    for v in range(V):
        ok = ~np.isnan(z[:, v, :]).any(axis=1)
        if ok.sum() < 4:
            continue
        X = np.column_stack([np.ones(ok.sum()), ages[ok], male[ok]])
        root = np.sqrt(weights[ok])[:, None]
        beta = np.linalg.lstsq(X * root, z[ok, v, :] * root, rcond=None)[0]
        out[v] = np.array([1.0, reference_age, 0.5]) @ beta
    return np.tanh(out)


def test_the_group_is_the_weighted_regression_on_age_and_sex():
    z, ages, male, weights = simulate()
    sums = summed(z, ages, male, weights)
    found = sums.adjusted()
    expected = regression_reference(z, ages, male, weights)
    assert found.shape == expected.shape
    np.testing.assert_allclose(found, expected, atol=1e-4)
    # a voxel never with signal, or with it in fewer than 4 subjects: none
    assert np.isnan(found[0]).all() and np.isnan(found[1]).all()
    assert np.isfinite(found[2:]).all()
    # each family counts once: with a weight per subject the result differs
    unweighted = summed(z, ages, male, np.ones(len(z))).adjusted()
    assert np.nanmax(np.abs(found - unweighted)) > 1e-3


def test_each_voxel_is_averaged_over_the_subjects_with_signal_in_it():
    z, ages, male, weights = simulate()
    sums = summed(z, ages, male, weights)
    assert sums.n_valid.tolist() == (~np.isnan(z).any(axis=2)).sum(0).tolist()
    assert sums.n_valid[0] == 0 and sums.n_valid[1] == 1


def test_the_adjustment_reads_the_effects_at_the_mean_age_and_both_sexes():
    """z = 0.2 (age - 30) + 0.4 male, exactly, in a sample of mostly men
    of unequal ages: read at the mean age and an equal mix of the sexes."""
    rng = np.random.default_rng(1)
    S = 60
    ages = rng.uniform(22, 37, S)
    male = (rng.random(S) < 0.8).astype(float)
    weights = np.ones(S)
    z = (0.2 * (ages - 30) + 0.4 * male)[:, None, None] * np.ones((S, 2, 2))
    found = summed(z.astype(np.float32), ages, male, weights, 2, 2).adjusted()
    expected = np.tanh(0.2 * (ages.mean() - 30) + 0.2)
    np.testing.assert_allclose(found, expected, atol=1e-4)
    assert abs(np.tanh(z.mean()) - expected) > 0.01  # the plain mean differs


def test_the_weights_of_a_family_sum_to_one():
    family = pd.Series(["a", "b", "a", "c", "a", "b"],
                       index=[f"s{i}" for i in range(6)])
    weights = family_weights(family)
    assert list(weights.index) == list(family.index)
    assert weights.groupby(family).sum().tolist() == [1.0, 1.0, 1.0]
    assert weights["s3"] == 1.0 and weights["s1"] == 0.5


def test_the_sums_of_two_tasks_merge_into_those_of_one():
    z, ages, male, weights = simulate(seed=2)
    whole = summed(z, ages, male, weights)
    first = summed(z[:15], ages[:15], male[:15], weights[:15])
    second = GroupSums(7, 3)
    for s in range(15, len(z)):
        second.add(f"s{s}", z[s], weights[s], ages[s], male[s])
    first.merge(second)
    np.testing.assert_allclose(first.adjusted(), whole.adjusted(), atol=1e-5)
    assert first.n_valid.tolist() == whole.n_valid.tolist()
    assert len(first.subjects) == len(z)


def test_the_sums_are_saved_whole(tmp_path):
    z, ages, male, weights = simulate(seed=3)
    sums = summed(z, ages, male, weights)
    sums.save(tmp_path / "part-0.npz")
    assert (tmp_path / "part-0.subjects").read_text().split() == \
        sums.subjects
    again = GroupSums.load(tmp_path / "part-0.npz")
    assert again.subjects == sums.subjects
    np.testing.assert_allclose(again.weights, sums.weights)
    np.testing.assert_allclose(again.ages, sums.ages)
    np.testing.assert_allclose(again.male, sums.male)
    np.testing.assert_array_equal(again.adjusted(), sums.adjusted())


def test_a_part_is_float32_without_the_squares(tmp_path):
    z, ages, male, weights = simulate(seed=4)
    summed(z, ages, male, weights).save(tmp_path / "part-0.npz")
    stored = np.load(tmp_path / "part-0.npz")
    matrices = [k for k in stored.files if stored[k].ndim == 2]
    assert len(matrices) == 3
    assert all(stored[k].dtype == np.float32 for k in matrices)


def test_the_phenotypes_are_age_sex_and_family(tmp_path):
    (tmp_path / "hcp_restricted.csv").write_text(
        "Subject,Family_ID,Age_in_Yrs\n1,F1,25\n2,F1,29\n3,F2,31\n4,F3,22\n")
    (tmp_path / "hcp_behavioral.csv").write_text(
        "Subject,Gender\n1,M\n2,F\n3,F\n5,M\n")
    found = load_phenotypes(tmp_path)
    # the subjects of both files
    assert list(found.index) == ["1", "2", "3"]
    assert found.loc["2"].to_dict() == {"age": 29.0, "male": 0.0,
                                        "family": "F1"}
    assert found.loc["1", "male"] == 1.0


# a 4 x 4 x 4 grid of 2 mm voxels, 3 parcels and the rest unlabelled
SHAPE = (4, 4, 4)
AFFINE = np.array([[-2.0, 0, 0, 90], [0, 2, 0, -126], [0, 0, 2, -72],
                   [0, 0, 0, 1]])
NAMES = ["17Networks LH A 1", "17Networks LH B 1", "17Networks RH A 1"]
SUBJECTS = [f"10000{i}" for i in range(1, 9)]
FAMILY = ["F1", "F1", "F2", "F3", "F3", "F3", "F4", "F5"]
AGE = [24.0, 27.0, 31.0, 22.0, 26.0, 35.0, 29.0, 33.0]
MALE = ["M", "F", "M", "M", "F", "F", "M", "F"]
TIMES = 60
KEY = "schaefer400_17_voxel_partial"


def atlas():
    labels = np.zeros(SHAPE, dtype=int)
    labels[0, :, :2] = 1
    labels[1, :, :2] = 2
    labels[2, :2, :2] = 3
    return labels


@pytest.fixture
def hcp(tmp_path, monkeypatch):
    monkeypatch.setattr(voxel_functional, "schaefer_volume",
                        lambda folder: (atlas(), AFFINE, NAMES))
    root = tmp_path / "hcp"
    rng = np.random.default_rng(6)
    for subject in SUBJECTS:
        for run in voxel_functional.RUNS:
            folder = root / subject / "MNINonLinear" / "Results" / run
            folder.mkdir(parents=True)
            data = rng.standard_normal(SHAPE + (TIMES,)) + 50
            # a signal shared by all, and one of each parcel: without them
            # the parcels are noise, Ledoit-Wolf shrinks everything to 0
            data += 2 * rng.standard_normal(TIMES)
            for parcel in (1, 2, 3):
                data[atlas() == parcel] += 3 * rng.standard_normal(TIMES)
            data[0, 0, 0] = 7.0  # no signal in any subject
            if subject == "100002":
                data[0, 0, 1] = 7.0  # no signal in this one
            nib.save(nib.Nifti1Image(data.astype(np.float32), AFFINE),
                     folder / f"{run}_hp2000_clean.nii.gz")
    # a subject lacking a run, and one lacking a phenotype, are left out
    (root / "999999" / "MNINonLinear" / "Results").mkdir(parents=True)
    ids = SUBJECTS + ["999999"]
    (root / "hcp_restricted.csv").write_text(
        "Subject,Family_ID,Age_in_Yrs\n"
        + "".join(f"{s},{f},{a}\n" for s, f, a in zip(ids, FAMILY, AGE)))
    (root / "hcp_behavioral.csv").write_text(
        "Subject,Gender\n"
        + "".join(f"{s},{g}\n" for s, g in zip(ids, MALE)))
    return tmp_path


def test_subjects_have_the_four_volumetric_runs_and_a_phenotype(hcp):
    assert voxel_functional.subjects_with_all_runs(hcp / "hcp") == SUBJECTS
    assert voxel_functional.eligible(hcp / "hcp") == SUBJECTS
    # no phenotype: left out
    (hcp / "hcp" / "hcp_behavioral.csv").write_text(
        "Subject,Gender\n" + "".join(f"{s},M\n" for s in SUBJECTS[1:]))
    assert voxel_functional.eligible(hcp / "hcp") == SUBJECTS[1:]


def test_a_task_sums_the_connectivity_of_its_subjects(hcp):
    work, out = hcp / "work", hcp / "parts"
    voxel_functional.run(work, hcp / "hcp", out, 0, 2)
    voxel_functional.run(work, hcp / "hcp", out, 1, 2)
    sums = GroupSums.of(out)
    assert sorted(sums.subjects) == SUBJECTS
    n_voxels = int((atlas() > 0).sum())
    r = sums.adjusted()
    assert r.shape == (n_voxels, len(NAMES)) and r.dtype == np.float32
    # each family counts once: the weights of the subjects of the parts
    weights = dict(zip(sums.subjects, sums.weights))
    assert weights == {"100001": 0.5, "100002": 0.5, "100003": 1.0,
                       "100004": 1 / 3, "100005": 1 / 3, "100006": 1 / 3,
                       "100007": 1.0, "100008": 1.0}
    ages = dict(zip(sums.subjects, sums.ages))
    assert ages["100003"] == 31.0
    males = dict(zip(sums.subjects, sums.male))
    assert males["100001"] == 1.0 and males["100002"] == 0.0


def test_a_voxel_without_signal_is_left_out_not_taken_as_zero(hcp):
    out = hcp / "parts"
    voxel_functional.run(hcp / "work", hcp / "hcp", out, 0, 1)
    sums = GroupSums.of(out)
    # voxels in the order of the array: (0, 0, 0) then (0, 0, 1)
    assert sums.n_valid[0] == 0 and sums.n_valid[1] == len(SUBJECTS) - 1
    assert np.isnan(sums.adjusted()[0]).all()
    assert np.isfinite(sums.adjusted()[1:]).all()


def test_a_task_goes_on_where_it_stopped(hcp, capsys):
    work, out = hcp / "work", hcp / "parts"
    voxel_functional.run(work, hcp / "hcp", out, 0, 1)
    capsys.readouterr()
    voxel_functional.run(work, hcp / "hcp", out, 0, 1)
    assert capsys.readouterr().out.strip() == ""
    assert sorted(GroupSums.of(out).subjects) == SUBJECTS


def table(hcp, threshold):
    work, out, dataset = hcp / "work", hcp / "parts", hcp / "dataset"
    voxel_functional.run(work, hcp / "hcp", out, 0, 1)
    combine.voxel_functional(work, out, dataset, threshold)
    return dataset


def test_the_group_matrix_is_the_adjusted_connectivity(hcp):
    dataset = table(hcp, 0.0)
    adjusted = GroupSums.of(hcp / "parts").adjusted()
    full = pd.read_parquet(
        dataset / "parquet" / f"{KEY}_functional_mean.parquet")
    assert list(full.columns) == NAMES
    assert list(full.index.names) == ["x", "y", "z"]
    np.testing.assert_allclose(full.values, adjusted, atol=1e-6)
    # the subjects, with what the group was adjusted for
    subjects = pd.read_parquet(
        dataset / "parquet" / f"{KEY}_functional_subjects.parquet")
    assert list(subjects.columns) == ["subject", "family", "weight", "age",
                                      "male"]
    assert subjects.subject.tolist() == SUBJECTS


def test_the_engine_table_keeps_the_pairs_above_the_threshold(hcp):
    dataset = table(hcp, 0.05)
    frame = pd.read_parquet(
        dataset / "parquet" / f"{KEY}_functional_connectivity.parquet")
    assert list(frame.columns) == ["x", "y", "z", "label", "r"]
    full = pd.read_parquet(
        dataset / "parquet" / f"{KEY}_functional_mean.parquet")
    assert len(frame) == int((full.values > 0.05).sum()) > 0
    assert (frame.r > 0.05).all() and frame.r.between(-1, 1).all()
    assert set(frame.label) <= set(NAMES)
    # sorted, so that a table made again is the same file
    keys = list(zip(frame.x, frame.y, frame.z, frame.label))
    assert keys == sorted(keys)
    # a voxel's centre in millimetres; the voxel without signal has none
    centres_mm = {tuple(AFFINE[:3, :3] @ ijk + AFFINE[:3, 3])
                  for ijk in np.argwhere(atlas() > 0)}
    assert {tuple(row) for row in frame[["x", "y", "z"]].values} <= centres_mm
    assert tuple(AFFINE[:3, 3]) not in {
        tuple(row) for row in frame[["x", "y", "z"]].values}
    row = frame.iloc[0]
    column = full.loc[(row.x, row.y, row.z), row.label]
    assert row.r == pytest.approx(column, abs=1e-5)


def test_the_row_counts_at_several_thresholds_are_printed(hcp, capsys):
    table(hcp, 0.1)
    out = capsys.readouterr().out
    for cutoff in ("0.02", "0.03", "0.05", "0.1", "0.15", "0.2", "0.3"):
        assert f"r > {cutoff}:" in out


def test_the_default_threshold_is_005(monkeypatch):
    from hcp_connectivity.__main__ import main

    seen = {}
    monkeypatch.setattr(combine, "voxel_functional",
                        lambda *args: seen.update(threshold=args[3]))
    main(["combine", "voxel-functional", "w", "p", "d"])
    assert seen["threshold"] == 0.05
