"""From files to the answer: the whole path of ``nl-tracts``, against
tract_querier on the same files.

An atlas image and a LUT, two tractograms written by nibabel, a WMQL
program; ``datalog_file`` makes the ``.dl`` and a NeurolangPDL with the
commands of nl_tracts runs it. tract_querier indexes the same volume and
streamlines (``TractographySpatialIndexing``) and evaluates the same program
(``EvaluateQueries``). The two give the same streamlines.

One thing is fed from our side to the reference: its label boxes, which are
those of ``label_extent`` (the voxel edges) and not its own two-corner boxes,
so that what is tested is the query and not the convention of the boxes,
which differ by half a voxel (6.1.1).
"""

import ast
import warnings

import nibabel as nib
import numpy as np
import pytest

from nl_tracts.atlas import Atlas
from nl_tracts.dl import AtlasSource, TractogramSource, datalog_file
from nl_tracts.frontend import tracts_engine

from .conftest import AFFINES, synthetic_volume
from .reference_tq import (
    load_aabb,
    load_indexing,
    load_reference,
)

pytestmark = pytest.mark.slow

LUT = "".join(f"{i} region_{i} 1 1 1 0\n" for i in range(5))
PROGRAM = """\
a = 1
b = 2
c = 3
d = 4
x = a and not b
y = only(a or 0)
z = (c or d) not in posterior_of(a)
w = ((a or b or c) and both_endpoints_in(anterior_of(a))) not in only(c or 0)
u = inferior_of(d) not in a
v = superior_of(d)
"""


def lines(seed, n=60):
    """Streamlines between random labelled voxels, with noise: they start
    and end in regions, cross several, and pass through the background (a
    random walk almost never ends in a region, and the comparison would be
    between empty sets)."""
    rng = np.random.default_rng(seed)
    voxels = np.argwhere(synthetic_volume() > 0)
    world = (AFFINES["aligned"] @ np.c_[voxels, np.ones(len(voxels))].T).T
    out = []
    for _ in range(n):
        p, q = world[rng.integers(len(world), size=2), :3]
        k = rng.integers(3, 25)
        points = p + np.linspace(0, 1, k)[:, None] * (q - p)
        out.append((points + rng.normal(size=(k, 3)) * 1.5).astype(
            np.float32))
    return out


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("e2e")
    nib.save(nib.Nifti1Image(synthetic_volume(), AFFINES["aligned"]),
             d / "labels.nii.gz")
    (d / "lut.txt").write_text(LUT)
    for name, seed in (("a", 1), ("b", 2)):
        nib.streamlines.save(
            nib.streamlines.Tractogram(lines(seed),
                                       affine_to_rasmm=np.eye(4)),
            str(d / f"{name}.tck"),
        )
    return d


def run(dataset, answer, **kw):
    text = datalog_file(
        PROGRAM,
        AtlasSource("fs", str(dataset / "labels.nii.gz"),
                    str(dataset / "lut.txt"), space="s"),
        [TractogramSource("t1", str(dataset / "a.tck"), "s"),
         TractogramSource("t2", str(dataset / "b.tck"), "s")],
        answer=answer, **kw,
    )
    result = tracts_engine().execute_datalog_program(text)
    got = {"t1": set(), "t2": set()}
    for t, s in result.as_pandas_dataframe().itertuples(
        index=False, name=None
    ):
        got[t].add(int(s))
    return got


def reference(dataset, name, anatomical_vertical=False):
    ref = load_reference()
    indexing = load_indexing()
    BoundingBox = load_aabb().BoundingBox  # noqa: N806
    atlas = Atlas.from_freesurfer(
        "fs", dataset / "labels.nii.gz", dataset / "lut.txt", "s")
    boxes = atlas.label_extent_relation()
    out = {}
    for t, file in (("t1", "a.tck"), ("t2", "b.tck")):
        tracts = [
            np.asarray(s, np.float64) for s in nib.streamlines.load(
                str(dataset / file)).tractogram.to_world().streamlines
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            index = indexing.TractographySpatialIndexing(
                tracts, atlas.labels, atlas.affine, 0.0, 2.0)
        index.label_bounding_boxes = {
            int(label): BoundingBox(np.r_[g.sort_values("axis")["lo"],
                                          g.sort_values("axis")["hi"]])
            for label, g in boxes.groupby("l")
        }
        evaluator = ref.EvaluateQueries(index)
        evaluator.visit(ast.Module(
            ref.queries_preprocess(PROGRAM), type_ignores=[]))
        out[t] = set(evaluator.evaluated_queries_info[name].tracts)
    return out


@pytest.mark.parametrize("name", ["x", "y", "z", "w", "u", "v"])
def test_the_dl_file_gives_the_streamlines_of_tract_querier(dataset, name):
    got = run(dataset, name)
    want = reference(dataset, name)
    assert got == want
    # the comparison is not between empty sets or between everything
    assert any(got.values()), "the comparison would be of empty sets"
    assert not all(len(v) == 60 for v in got.values())


def test_every_construct_is_exercised():
    """The program has a complement, only, both_endpoints_in, a difference
    and the directional terms, the vertical ones as the reference has them."""
    for construct in ("not b", "only(", "both_endpoints_in", "anterior_of",
                      "posterior_of", "inferior_of", "superior_of", "not in"):
        assert construct in PROGRAM
