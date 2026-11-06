"""``examples/uncinate.dl`` end to end: its commands load an atlas and a
tractogram written here (``uncinate_files``), and its rules, over the
vocabulary, select the designed uncinates of each side, and only them."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from nl_tracts.frontend import tracts_engine

from . import uncinate_files

EXAMPLE = Path(__file__).parents[2] / "examples" / "uncinate.dl"


def example(directory, side="left"):
    """The example, reading the files of ``directory``."""
    if not EXAMPLE.exists():
        pytest.skip(f"{EXAMPLE} not found (an installed package)")
    text = EXAMPLE.read_text()
    for name, path in (
        ('"wmparc.mgz"', f'"{directory / "labels.nii.gz"}"'),
        ('"FreeSurferColorLUT.txt"', f'"{directory / "lut.txt"}"'),
        ('kind="hcp842")', f'"{directory / "tracts.trk"}", space="MNI152")'),
    ):
        assert name in text
        text = text.replace(name, path)
    return text.replace('uncinate(x, "left")', f'uncinate(x, "{side}")')


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    return uncinate_files.write_files(tmp_path_factory.mktemp("uncinate"))


def xs(result):
    return set(result.as_pandas_dataframe().iloc[:, 0])


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_example_selects_the_uncinates(files, side):
    result = tracts_engine().execute_datalog_program(example(files, side))
    # one tractogram: x is the position s of the streamline in the file
    assert xs(result) == uncinate_files.expected(side)


def test_each_condition_of_the_uncinate_excludes_a_streamline(files):
    """Every designed streamline but the uncinates is an uncinate of no
    side; the uncinates are of one side."""
    nl = tracts_engine()
    program = example(files).replace(
        'ans(x) :- uncinate(x, "left")', "ans(x, h) :- uncinate(x, h)")
    rows = set(nl.execute_datalog_program(program).as_pandas_dataframe()
               .itertuples(index=False, name=None))
    assert rows == {
        (x, side) for side in ("left", "right")
        for x in uncinate_files.expected(side)
    }


def test_the_hooks_give_the_points_of_the_streamlines(files):
    """``streamline_points`` of ``.view``, by id ``x``, and the label image
    of the atlas as ``streamline_background``."""
    nl = tracts_engine()
    assert nl.streamline_background() is None
    nl.execute_datalog_program(example(files))
    lines = uncinate_files.streamlines()
    points = nl.streamline_points([3, 0, 9])
    for got, s in zip(points, (3, 0, 9)):
        np.testing.assert_allclose(got, lines[s], atol=1e-4)
    background = nl.streamline_background()
    np.testing.assert_array_equal(
        np.asarray(background.dataobj), uncinate_files.labels())
    np.testing.assert_array_equal(background.affine, uncinate_files.AFFINE)


def test_the_hooks_with_two_tractograms(files, tmp_path):
    """The ids of the second tractogram are ``(1 << 32) | s``: its points
    are found by the load order of the store."""
    other = uncinate_files.streamlines()[::-1]
    nib.streamlines.save(
        nib.streamlines.Tractogram(other, affine_to_rasmm=np.eye(4)),
        str(tmp_path / "other.trk"),
    )
    nl = tracts_engine()
    nl.execute_datalog_program(example(files))  # the rules stay
    nl.execute_datalog_program(
        f'.load_tractogram("other", "{tmp_path}/other.trk", '
        'space="MNI152")\n.index_tracts("other", "fs")\n'
    )
    first, second = nl.streamline_points([2, (1 << 32) | 2])
    np.testing.assert_allclose(first, uncinate_files.streamlines()[2],
                               atol=1e-4)
    np.testing.assert_allclose(second, other[2], atol=1e-4)
    result = nl.execute_datalog_program(
        'ans(x) :- uncinate(x, "left"), streamline(x, "other", s)')
    n = len(other)
    assert {x & 0xFFFFFFFF for x in xs(result)} == {
        n - 1 - s for s in uncinate_files.expected("left")
    }
