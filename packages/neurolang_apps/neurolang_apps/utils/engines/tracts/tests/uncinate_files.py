"""A FreeSurfer-like atlas and streamlines for the uncinate fasciculus.

The atlas has the labels of aparc+aseg that ``uncinate.dl`` names, with
their FreeSurfer ids and names, as boxes in mm (RAS+), the right ones the
mirror of the left ones in x. :data:`DESIGNED` are streamlines of the left
hemisphere, each an uncinate or missing it by one of its conditions;
:func:`write_files` writes them, their mirrors and, optionally, random
streamlines between the regions, as a ``.trk``, with the atlas image and a
colour LUT.
"""

import nibabel as nib
import numpy as np

# id, name of the left label, of the right one, box of the left one (mm)
REGIONS = [
    (1012, 2012, "lateralorbitofrontal", [(-30, -10), (30, 50), (-10, 0)]),
    (1035, 2035, "insula", [(-40, -30), (0, 20), (-10, 10)]),
    (1030, 2030, "superiortemporal", [(-50, -40), (-20, 20), (-30, -10)]),
    (1011, 2011, "lateraloccipital", [(-30, -10), (-90, -70), (0, 10)]),
    (1029, 2029, "superiorparietal", [(-30, -10), (-60, -40), (30, 40)]),
    (1023, 2023, "posteriorcingulate", [(-8, -2), (-30, 30), (20, 30)]),
    (18, 54, "Amygdala", [(-25, -15), (-10, 0), (-20, -10)]),
    (12, 51, "Putamen", [(-28, -18), (-5, 15), (-5, 5)]),
]
AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])
AFFINE[:3, 3] = (-60, -100, -40)
SHAPE = (61, 81, 46)

_A = [(-45, 10, -20), (-35, 10, 0), (-20, 40, -5)]
#: Streamlines of the left hemisphere, as waypoints: name -> (waypoints,
#: whether it is an uncinate).
DESIGNED = {
    "uncinate": (_A, True),
    "reversed": (_A[::-1], True),  # the other end is the temporal one
    "temporal end not anterior": (
        [(-45, -4, -20), (-35, 10, 0), (-20, 40, -5)], False),
    "no insula": ([(-45, 10, -20), (-20, 40, -5)], False),
    "cingulate": (_A + [(-5, 20, 25)], False),
    "other hemisphere": (_A + [(20, 40, -5)], False),
    "posterior of the putamen": (_A + [(-20, -15, -5)], False),
    "occipital": (_A + [(-20, -80, 5)], False),
}


def _name(region, side):
    if region[0].isupper():
        return f"{'Left' if side == 'left' else 'Right'}-{region}"
    return f"ctx-{'lh' if side == 'left' else 'rh'}-{region}"


def labels():
    """The label image of the atlas."""
    volume = np.zeros(SHAPE, dtype=np.int32)
    centres = nib.affines.apply_affine(
        AFFINE, np.indices(SHAPE).reshape(3, -1).T
    ).reshape(SHAPE + (3,))
    for left, right, _, box in REGIONS:
        for label, sign in ((left, 1), (right, -1)):
            inside = np.ones(SHAPE, bool)
            for axis, (lo, hi) in enumerate(box):
                if axis == 0:
                    lo, hi = sorted((sign * lo, sign * hi))
                inside &= (centres[..., axis] >= lo) & (
                    centres[..., axis] <= hi)
            volume[inside] = label
    return volume


def lut():
    """The colour LUT of the atlas, as FreeSurfer writes it."""
    rows = ["0 Unknown 0 0 0 0"] + [
        f"{label} {_name(region, side)} 1 1 1 0"
        for left, right, region, _ in REGIONS
        for label, side in ((left, "left"), (right, "right"))
    ]
    return "\n".join(rows) + "\n"


def polyline(waypoints, step=1.0):
    """Points every ``step`` mm along the waypoints."""
    points = [np.asarray(waypoints[0], float)]
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        points.extend(a + (b - a) * t for t in np.arange(1, n + 1) / n)
    return np.array(points, dtype=np.float32)


def mirrored(waypoints):
    return [(-x, y, z) for x, y, z in waypoints]


def random_streamlines(n, seed=0):
    """``n`` streamlines: half through three waypoints near the centres of
    random regions of either hemisphere, half the designed ones, of either
    side, with their waypoints moved at random by a few mm, so that many
    are on the edge of a condition of the uncinate."""
    rng = np.random.default_rng(seed)
    centres = [
        np.mean(box, axis=1) * (sign, 1, 1)
        for _, _, _, box in REGIONS for sign in (1, -1)
    ]
    designed = [w for w, _ in DESIGNED.values()]
    designed += [mirrored(w) for w in designed]
    lines = []
    for i in range(n):
        if i % 2:
            waypoints = designed[rng.integers(len(designed))]
            lines.append([
                np.asarray(w) + rng.normal(scale=4, size=3)
                for w in waypoints
            ])
        else:
            lines.append([
                centres[j] + rng.normal(scale=4, size=3)
                for j in rng.choice(len(centres), 3, replace=False)
            ])
    return lines


def streamlines(n_random=0, seed=0):
    """The designed streamlines (``s`` 0, 1, ... in the order of
    :data:`DESIGNED`), their mirrors (then), and ``n_random`` random ones
    (last)."""
    designed = [w for w, _ in DESIGNED.values()]
    return [
        polyline(w) for w in designed + [mirrored(w) for w in designed]
        + random_streamlines(n_random, seed)
    ]


def write_files(directory, n_random=0, seed=0):
    """Write ``labels.nii.gz``, ``lut.txt`` and ``tracts.trk`` in
    ``directory``."""
    nib.save(nib.Nifti1Image(labels(), AFFINE), directory / "labels.nii.gz")
    (directory / "lut.txt").write_text(lut())
    nib.streamlines.save(
        nib.streamlines.Tractogram(
            streamlines(n_random, seed), affine_to_rasmm=np.eye(4)
        ),
        str(directory / "tracts.trk"),
    )
    return directory


def expected(side):
    """The positions ``s`` of the uncinates of a side, among the designed
    streamlines and their mirrors."""
    offset = 0 if side == "left" else len(DESIGNED)
    return {
        offset + i for i, (_, uf) in enumerate(DESIGNED.values()) if uf
    }
