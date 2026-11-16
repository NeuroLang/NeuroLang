"""
The group functional connectivity of the Individual Brain Charting (IBC)
subjects, by condition: rest, each movie, and the task protocols in groups
(:data:`CONDITIONS`).

The runs are IBC's preprocessed ones, in MNI space (``derivatives/sub-*/
ses-*/func/wrdc*_bold.nii.gz``, 1.5 mm). Each run's time series are
averaged within each region of the two atlases of the connectivity of the
engines: the 400 Schaefer parcels (17 networks) then Tian's 32 subcortical
regions, and Destrieux's 148 parcels then Tian's, Tian's where both
(:func:`label_images`). The region means are cleaned of the confounds
(:func:`run_confounds`): the six motion parameters and their derivatives,
five components of the voxels of highest variance (tCompCor, as IBC's own
models), a cosine high-pass at 0.01 Hz, and, for a task, its events
convolved with the SPM HRF, so that the connectivity of a task is that of
the activity it does not evoke (background connectivity); rest and the
movies have none. Their Pearson correlations, Fisher-z transformed
(functional.connectivity), are averaged over the subject's runs of a
condition (:func:`group`; each run is ``OUT/<subject>/<condition>__
<run>.npz``, so that the workers share the runs), then over the
subjects of each condition.
"""

import json
import re
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from .functional import connectivity

# The conditions of the connectivity and their IBC tasks; any other task
# is of "tasks".
CONDITIONS = {
    "rest": ("RestingState",),
    "raiders": ("Raiders",),
    "good_bad_ugly": ("GoodBadUgly",),
    "monkey_kingdom": ("MonkeyKingdom",),
    "le_petit_prince": ("LePetitPrince",),
    "clips": ("ClipsTrn", "ClipsVal"),
    "archi": ("ArchiStandard", "ArchiSpatial", "ArchiSocial",
              "ArchiEmotional"),
    "hcp": ("HcpEmotion", "HcpGambling", "HcpLanguage", "HcpMotor",
            "HcpRelational", "HcpSocial", "HcpWm"),
    "language": ("RSVPLanguage", "MathLanguage", "LPPLocalizer", "Audio",
                 "Lec1", "Lec2"),
    "tasks": (),
}
# The conditions without events to regress out: rest and the movies.
WITHOUT_EVENTS = ("rest", "raiders", "good_bad_ugly", "monkey_kingdom",
                  "le_petit_prince", "clips")
ATLASES = ("schaefer400_17", "destrieux")
HIGH_PASS = 0.01
COMPCOR = 5
RUN = re.compile(r"wrdc(sub-\d+)_(ses-\d+)_task-([A-Za-z0-9]+)_"
                 r"(dir-[a-z]+)(?:_run-(\d+))?_bold\.nii\.gz$")


def condition(task: str) -> str:
    """The condition of the IBC *task*."""
    return next((c for c, tasks in CONDITIONS.items() if task in tasks),
                "tasks")


def subjects(ibc: Path) -> list:
    return sorted(p.name for p in (ibc / "derivatives").glob("sub-*"))


def runs(ibc: Path, subject: str) -> list:
    """The preprocessed runs of *subject*: (condition, task, direction,
    image) by session."""
    found = []
    for image in sorted((ibc / "derivatives" / subject).glob(
            "ses-*/func/wrdc*_bold.nii.gz")):
        match = RUN.search(image.name)
        if match:
            task, direction = match.group(3), match.group(4)
            found.append((condition(task), task, direction, image))
    return found


def repetition_time(ibc: Path, task: str, direction: str) -> float:
    """The TR of the runs of *task* (the header of the images says 1 s)."""
    sidecar = ibc / "sourcedata" / f"task-{task}_{direction}_bold.json"
    return float(json.loads(sidecar.read_text())["RepetitionTime"])


def motion(path: Path) -> np.ndarray:
    """The six motion parameters of SPM's ``rp_*.txt`` and their
    derivatives (0 at the first volume)."""
    rp = np.loadtxt(path, ndmin=2)
    return np.hstack([rp, np.vstack([np.zeros((1, 6)), np.diff(rp, axis=0)])])


def event_regressors(events: Path, n: int, tr: float) -> np.ndarray:
    """The events of a run (BIDS ``*_events.tsv``) convolved with the SPM
    HRF, one column per trial type, at its *n* volumes."""
    import pandas as pd
    from nilearn.glm.first_level import make_first_level_design_matrix

    frame = pd.read_csv(events, sep="\t")
    frame = frame[frame.duration.notna() & frame.onset.notna()]
    if "trial_type" not in frame:
        frame["trial_type"] = "task"
    design = make_first_level_design_matrix(
        np.arange(n) * tr, frame[["onset", "duration", "trial_type"]],
        hrf_model="spm", drift_model=None)
    return design.drop(columns="constant").to_numpy()


def run_confounds(data: np.ndarray, affine, rp: Path, events,
                  tr: float) -> np.ndarray:
    """The confounds of the run *data* (4D): motion, tCompCor of the
    voxels of its EPI mask and, for a task, its events (*events*, None
    for rest and the movies)."""
    from nilearn.masking import compute_epi_mask
    from nilearn.signal import high_variance_confounds

    n = data.shape[3]
    brain = compute_epi_mask(nib.Nifti1Image(data.mean(axis=3), affine))
    voxels = data[np.asarray(brain.dataobj, dtype=bool)].T
    columns = [motion(rp), high_variance_confounds(voxels,
                                                   n_confounds=COMPCOR)]
    if events is not None:
        columns.append(event_regressors(events, n, tr))
    return np.hstack(columns)


def clean(series: np.ndarray, confounds: np.ndarray, tr: float):
    """The region means *series* without the *confounds*, high-passed."""
    from nilearn.signal import clean as nilearn_clean

    return nilearn_clean(series, confounds=confounds, t_r=tr,
                         high_pass=HIGH_PASS, detrend=False,
                         standardize="zscore_sample",
                         standardize_confounds=True)


def destrieux_volume(folder: Path, names: list):
    """Nilearn's volume of Destrieux's atlas, its parcels numbered from 1
    in the order of *names* (the engines', as 'L G front sup'), and its
    affine."""
    from nilearn import datasets

    from .cifti import engine_name

    atlas = datasets.fetch_atlas_destrieux_2009(
        lateralized=True, data_dir=str(folder))
    image = nib.load(atlas["maps"])
    data = np.asarray(image.dataobj, dtype=int)
    number = {name: n for n, name in enumerate(names, start=1)}
    table = np.zeros(data.max() + 1, dtype=np.uint16)
    labels = atlas["labels"]
    for index, name in (zip(labels["index"], labels["name"])
                        if hasattr(labels, "columns") else enumerate(labels)):
        table[int(index)] = number.get(engine_name(name), 0)
    assert set(table[table > 0]) == set(number.values()), "parcels missing"
    return table[data], image.affine


def label_images(work: Path, grid) -> dict:
    """The label images of :data:`ATLASES` on the *grid* of the runs (an
    image), cortex then Tian, and the names of their regions; the atlases
    are those of ``WORK/atlases`` (atlases.build, the volume of
    tractography.volume_atlas)."""
    from .atlases import cortical, tian_names
    from .tractography import combine_labels, resample_nearest, volume_atlas

    shape, affine = grid.shape[:3], grid.affine
    atlas_folder = work / "atlases"
    tian = tian_names(atlas_folder)
    names = cortical(work)
    schaefer = nib.load(volume_atlas(atlas_folder))
    images = {"schaefer400_17": (resample_nearest(
        np.asarray(schaefer.dataobj, dtype=int), schaefer.affine, shape,
        affine), names["schaefer400_17"][1] + tian)}
    from .atlases import TIAN, fetch
    subcortex = nib.load(fetch(f"{TIAN}.nii.gz", atlas_folder))
    cortex, cortex_affine = destrieux_volume(atlas_folder,
                                             names["destrieux"][1])
    images["destrieux"] = (combine_labels(
        resample_nearest(cortex, cortex_affine, shape, affine),
        resample_nearest(np.asarray(subcortex.dataobj, dtype=int),
                         subcortex.affine, shape, affine),
        offset=len(names["destrieux"][1])),
        names["destrieux"][1] + tian)
    return images


def region_means(data: np.ndarray, labels: np.ndarray, n: int):
    """The mean time series of the regions 1..*n* of *labels* in the 4D
    *data*: (volumes, n); NaN for a region without voxels."""
    flat = labels.ravel()
    inside = flat > 0
    counts = np.bincount(flat[inside] - 1, minlength=n)[:n]
    series = data.reshape(-1, data.shape[3])[inside]
    sums = np.zeros((n, data.shape[3]))
    np.add.at(sums, flat[inside] - 1, series)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (sums / counts[:, None]).T


def run_connectivity(ibc: Path, run: tuple, images: dict):
    """The Fisher z of each atlas of *images* over the *run* (condition,
    task, direction, image), by atlas; None for a task run without
    events."""
    cond, task, direction, path = run
    func = path.parent
    stem = path.name[len("wrdc"):-len("_bold.nii.gz")]
    events = func / f"{stem}_events.tsv"
    if cond not in WITHOUT_EVENTS and not events.exists():
        return None
    image = nib.load(path)
    tr = repetition_time(ibc, task, direction)
    data = np.asarray(image.dataobj, dtype=np.float32)
    confounds = run_confounds(
        data, image.affine, func / f"rp_dc{stem}_bold.txt",
        None if cond in WITHOUT_EVENTS else events, tr)
    z = {}
    for atlas, (labels, names) in images.items():
        means = region_means(data, labels, len(names))
        empty = np.isnan(means).all(axis=0)
        means[:, empty] = 0  # a region without voxels: NaN in the end
        z[atlas] = connectivity(clean(means, confounds, tr))
    return z


def all_runs(ibc: Path) -> list:
    """Every run of every subject: (subject, run), in a fixed order."""
    return [(subject, run) for subject in subjects(ibc)
            for run in runs(ibc, subject)]


def run(work: Path, ibc: Path, out: Path, task: int, tasks: int) -> None:
    """The connectivity of the share *task* of *tasks* of the runs of
    every subject, ``OUT/<subject>/<condition>__<run>.npz``, those
    already there skipped (a run without events: an empty file)."""
    grids = {}  # the label images of each grid: the runs differ
    for subject, found in all_runs(ibc)[task::tasks]:
        cond, path = found[0], found[3]
        folder = out / subject
        target = folder / f"{cond}__{path.name[:-len('.nii.gz')]}.npz"
        if target.exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        start = time.time()
        image = nib.load(path)
        grid = (image.shape[:3], image.affine.round(4).tobytes())
        if grid not in grids:
            grids[grid] = label_images(work, image)
        z = run_connectivity(ibc, found, grids[grid])
        partial = target.with_name(target.stem + ".partial.npz")
        np.savez(partial, **(z or {}))
        partial.rename(target)
        print(f"{path.name}: {cond}, {time.time() - start:.0f} s"
              + ("" if z else ", no events: left out"), flush=True)


def group(folder: Path) -> tuple:
    """The mean Fisher z of each ``<atlas>:<condition>``: over the runs of
    a subject of the condition, then over its subjects; and the subjects
    and runs of each condition (a frame: condition, subjects, runs)."""
    import pandas as pd

    sums, counts = {}, []
    for subject in sorted(p for p in Path(folder).glob("sub-*")
                          if p.is_dir()):
        runs_of, own = {}, {}
        for path in sorted(subject.glob("*__*.npz")):
            if path.name.endswith(".partial.npz"):
                continue
            stored = np.load(path)
            if not stored.files:  # a task run without events
                continue
            cond = path.name.split("__")[0]
            runs_of[cond] = runs_of.get(cond, 0) + 1
            for atlas in stored.files:
                key = f"{atlas}:{cond}"
                own[key] = own.get(key, 0.0) + stored[atlas]
        for key, total in own.items():
            mean = total / runs_of[key.split(":")[1]]
            previous, n = sums.get(key, (0.0, 0))
            sums[key] = (previous + mean, n + 1)
        counts += [(cond, n) for cond, n in runs_of.items()]
    frame = pd.DataFrame(counts, columns=["condition", "runs"])
    summary = frame.groupby("condition", sort=False).runs.agg(
        subjects="count", runs="sum").reset_index()
    return {key: total / n for key, (total, n) in sums.items()}, summary
