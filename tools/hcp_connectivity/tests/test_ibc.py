import json

import nibabel as nib
import numpy as np
import pandas as pd

from hcp_connectivity import combine, ibc

from hcp_connectivity.tractography import combine_labels

from test_tractography import NAMES


def test_the_conditions_of_the_tasks():
    assert ibc.condition("RestingState") == "rest"
    assert ibc.condition("ClipsVal") == "clips"
    assert ibc.condition("HcpWm") == "hcp"
    assert ibc.condition("Stroop") == "tasks"


def test_combine_labels_offsets_tian_after_destrieux():
    cortex = np.array([0, 1, 148, 7])
    tian = np.array([0, 0, 0, 32])
    assert combine_labels(cortex, tian, offset=148).tolist() == \
        [0, 1, 148, 180]


def test_region_means_average_the_voxels_of_each_region():
    data = np.arange(2 * 2 * 1 * 3, dtype=float).reshape(2, 2, 1, 3)
    labels = np.array([[[1], [1]], [[2], [0]]])
    means = ibc.region_means(data, labels, 3)
    assert means.shape == (3, 3)
    np.testing.assert_allclose(means[:, 0], data[0, :, 0].mean(axis=0))
    np.testing.assert_allclose(means[:, 1], data[1, 0, 0])
    assert np.isnan(means[:, 2]).all()


def _events(tmp_path, tr, n):
    onsets = np.arange(10, n * tr - 30, 25.0)
    events = tmp_path / "events.tsv"
    pd.DataFrame({"onset": onsets, "duration": 10.0,
                  "trial_type": "task"}).to_csv(events, sep="\t",
                                                index=False)
    return events


def test_the_events_are_regressed_out_of_a_task(tmp_path):
    """Two regions sharing only the response to the task's events are
    correlated, and not once the events are regressed out."""
    tr, n = 2.0, 300
    events = _events(tmp_path, tr, n)
    response = ibc.event_regressors(events, n, tr)[:, 0]
    rng = np.random.default_rng(0)
    series = np.column_stack([3 * response + rng.normal(size=n)
                              for _ in range(2)])
    before = np.corrcoef(series, rowvar=False)[0, 1]
    cleaned = ibc.clean(series, ibc.event_regressors(events, n, tr), tr)
    after = np.corrcoef(cleaned, rowvar=False)[0, 1]
    assert before > 0.5 and abs(after) < 0.15


def test_a_run_of_a_subject_end_to_end(tmp_path):
    """A rest run and a task run of a fake IBC tree: their conditions,
    TRs, motion, events; two regions sharing a signal are connected."""
    root = tmp_path / "ibc"
    func = root / "derivatives" / "sub-01" / "ses-01" / "func"
    func.mkdir(parents=True)
    (root / "sourcedata").mkdir()
    rng = np.random.default_rng(1)
    shape, n = (12, 12, 12), 120
    brain = (slice(2, 10),) * 3  # a zero background, as IBC's images
    shared = rng.normal(size=n)
    for task, tr in (("RestingState", 0.76), ("HcpMotor", 2.0)):
        data = np.zeros((*shape, n))
        data[brain] = 100 + rng.normal(size=(8, 8, 8, n))
        data[2:6, 2:10, 2:10] += 2 * shared  # regions 1 and 2 share it
        stem = f"sub-01_ses-01_task-{task}_dir-ap"
        nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)),
                 func / f"wrdc{stem}_bold.nii.gz")
        np.savetxt(func / f"rp_dc{stem}_bold.txt",
                   rng.normal(scale=0.01, size=(n, 6)))
        (root / "sourcedata" / f"task-{task}_dir-ap_bold.json").write_text(
            json.dumps({"RepetitionTime": tr}))
        if task != "RestingState":
            _events(func, tr, n).rename(func / f"{stem}_events.tsv")
    labels = np.zeros(shape, dtype=int)
    labels[2:6, 2:6, 2:10] = 1
    labels[2:6, 6:10, 2:10] = 2
    labels[6:10, 2:6, 2:10] = 3
    assert [r[:2] for r in ibc.runs(root, "sub-01")] == \
        [("hcp", "HcpMotor"), ("rest", "RestingState")]
    matrices = ibc.subject_connectivity(
        root, "sub-01", {"atlas": (labels, ["a", "b", "c"])})
    assert matrices["runs:rest"] == 1 and matrices["runs:hcp"] == 1
    for cond in ("rest", "hcp"):
        r = np.tanh(matrices[f"atlas:{cond}"])
        assert r[0, 1] > 0.5 and abs(r[0, 2]) < 0.3, cond


def test_group_averages_the_subjects_of_each_condition(tmp_path):
    np.savez(tmp_path / "sub-01.npz", **{"a:rest": np.full((2, 2), 0.2),
                                         "a:hcp": np.full((2, 2), 0.4),
                                         "runs:rest": 4, "runs:hcp": 7})
    np.savez(tmp_path / "sub-02.npz", **{"a:hcp": np.full((2, 2), 0.6),
                                         "runs:hcp": 5})
    mean, counts = ibc.group(tmp_path)
    np.testing.assert_allclose(mean["a:rest"], 0.2)  # sub-02 has no rest
    np.testing.assert_allclose(mean["a:hcp"], 0.5)
    counts = counts.set_index("condition")
    assert counts.loc["rest"].tolist() == [1, 4]
    assert counts.loc["hcp"].tolist() == [2, 12]


def test_the_ibc_engine_tables_are_every_pair_of_each_condition(tmp_path):
    (tmp_path / "parquet").mkdir()
    (tmp_path / "tsv").mkdir()
    rng = np.random.default_rng(2)
    for cond in ("rest", "hcp"):
        values = rng.uniform(-0.9, 0.9, (len(NAMES), len(NAMES)))
        values = (values + values.T) / 2
        np.fill_diagonal(values, 1.0)
        combine.write_matrix(
            combine.matrix(values, NAMES), tmp_path / "parquet" /
            f"ibc_schaefer400_17_functional_mean_{cond}.parquet")
    combine.engine_tables(tmp_path)
    table = pd.read_csv(tmp_path / "tsv" /
                        "ibc_schaefer400_17_tian_functional_connectivity"
                        ".tsv.gz", sep="\t")
    assert list(table.columns) == ["condition", "region1", "region2", "r"]
    assert set(table.condition) == {"rest", "hcp"}
    for _, rows in table.groupby("condition"):
        assert len(rows) == len(NAMES) * (len(NAMES) - 1)
        both = rows.set_index(["region1", "region2"]).r
        swapped = rows.set_index(["region2", "region1"]).r
        swapped.index.names = both.index.names
        assert both.sort_index().equals(swapped.sort_index())
    assert not (tmp_path / "tsv" /
                "ibc_destrieux_tian_functional_connectivity.tsv.gz").exists()
