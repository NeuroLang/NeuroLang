"""Tests for saving and viewing results in the interactive REPL."""

import io

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

import nibabel as nib  # noqa: E402
from rich.console import Console  # noqa: E402

from neurolang.frontend import NeurolangPDL  # noqa: E402
from neurolang.utils.tui import InteractiveTuiApp, TuiSettings  # noqa: E402
from neurolang.utils.tui import nifti  # noqa: E402
from neurolang.utils.tui.nifti import _result_to_nifti  # noqa: E402


class TestResultToNifti:
    def test_basic_nifti_from_ijk_columns(self):
        df = pd.DataFrame({
            "i": [10, 20], "j": [30, 40], "k": [50, 60],
            "value": [1.0, 2.0],
        })
        img = _result_to_nifti(df)
        assert isinstance(img, nib.Nifti1Image)
        assert img.shape == (256, 256, 256)
        data = img.get_fdata()
        assert data[10, 30, 50] == 1.0
        assert data[20, 40, 60] == 2.0

    def test_nifti_without_value_column(self):
        df = pd.DataFrame({"i": [5], "j": [5], "k": [5]})
        assert _result_to_nifti(df).get_fdata()[5, 5, 5] == 1.0

    def test_nifti_clips_out_of_bounds(self):
        df = pd.DataFrame({
            "i": [1000, 1], "j": [1000, 2], "k": [1000, 3],
            "value": [42.0, 7.0],
        })
        data = _result_to_nifti(df).get_fdata()
        assert data[255, 255, 255] == 0.0
        assert data[1, 2, 3] == 7.0

    def test_nifti_raises_with_no_coords(self):
        df = pd.DataFrame({"label": ["a"], "value": [1.0]})
        with pytest.raises(
            ValueError, match="Cannot determine voxel coordinates"
        ):
            _result_to_nifti(df)

    def test_nifti_uses_fallback_numeric_cols(self):
        df = pd.DataFrame({"x": [0], "y": [0], "z": [0], "val": [5.0]})
        assert _result_to_nifti(df).get_fdata()[0, 0, 0] == 5.0


def _output(app):
    return app.console.file.getvalue()


@pytest.fixture
def app():
    nl = NeurolangPDL()
    nl.add_tuple_set([(1, 2, 3), (4, 5, 6)], name="voxel")
    console = Console(file=io.StringIO(), width=100)
    return InteractiveTuiApp(
        nl, TuiSettings(), console=console, err_console=console
    )


def test_save_needs_a_result(app, tmp_path):
    app._handle_dot_command(f".save {tmp_path / 'out.csv'}")
    assert "Nothing to save" in _output(app)


def test_save_csv(app, tmp_path):
    app._execute_and_display("ans(i, j, k) :- voxel(i, j, k)")
    app._handle_dot_command(f".save {tmp_path / 'sub' / 'out.csv'}")
    saved = pd.read_csv(tmp_path / "sub" / "out.csv")
    assert saved.values.tolist() == [[1, 2, 3], [4, 5, 6]]


def test_save_nifti(app, tmp_path):
    app._execute_and_display("ans(i, j, k) :- voxel(i, j, k)")
    app._handle_dot_command(f".save {tmp_path / 'out.nii'}")
    data = nib.load(tmp_path / "out.nii").get_fdata()
    assert data[1, 2, 3] == 1.0 and data[4, 5, 6] == 1.0


def test_save_every_obtain_block(app, tmp_path):
    app.settings.squall = True
    app._execute_and_display(
        "obtain every voxel as First. obtain every voxel as Second."
    )
    app._handle_dot_command(f".save {tmp_path / 'out.csv'}")
    assert (tmp_path / "out_first.csv").exists()
    assert (tmp_path / "out_second.csv").exists()


def test_view(app, tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(nifti.subprocess, "Popen", opened.append)
    app._handle_dot_command(f".view {tmp_path / 'missing.nii'}")
    assert "File not found" in _output(app)
    path = tmp_path / "img.nii"
    nib.save(_result_to_nifti(pd.DataFrame(
        {"i": [100], "j": [100], "k": [100]}
    )), path)
    app._handle_dot_command(f".view {path}")
    assert "Saved visualisation to" in _output(app)
    assert opened and opened[0][1].endswith(".png")


# A tiny engine: a 4×4×4 template grid of 2 mm voxels, its origin away
# from the grid's corner, and the sorts declaring the voxel indices and
# the millimetre coordinates on it.
TINY_AFFINE = np.array([
    [2.0, 0.0, 0.0, -3.0],
    [0.0, 2.0, 0.0, -5.0],
    [0.0, 0.0, 2.0, -7.0],
    [0.0, 0.0, 0.0, 1.0],
])

TINY_ENGINE = """\
engines:
  tiny:
    sorts:
      study: {columns: [study_id], type: int}
      tfidf: {columns: [tfidf], type: float}
      voxel: {columns: [i, j, k], type: int, space: voxel}
      mni: {columns: [x, y, z], type: float, space: mm}
    predicates:
      voxel: {arguments: [voxel]}
      reports: {arguments: [study, voxel]}
      weight: {arguments: [voxel, tfidf]}
      peak: {arguments: [mni, study]}
"""


@pytest.fixture
def tiny(monkeypatch, tmp_path):
    """Return a REPL on the tiny engine, with its 4×4×4 template."""
    from neurolang.utils import engine_registry

    path = tmp_path / "engines.yaml"
    path.write_text(TINY_ENGINE)
    monkeypatch.setattr(engine_registry, "_ENGINES_YAML", path)
    nl = NeurolangPDL()
    nl.template = nib.Nifti1Image(
        np.ones((4, 4, 4), dtype=np.uint8), TINY_AFFINE
    )
    nl.add_tuple_set([(1, 2, 3), (0, 0, 0)], name="voxel")
    nl.add_tuple_set(
        [(10, 1, 2, 3), (11, 1, 2, 3), (12, 0, 0, 0)], name="reports"
    )
    nl.add_tuple_set(
        [(1, 2, 3, 0.25), (1, 2, 3, 0.75), (0, 0, 0, 0.5)], name="weight"
    )
    nl.add_tuple_set(
        [(-1.0, -1.0, -1.0, 10), (-3.2, -4.9, -7.0, 11)], name="peak"
    )
    console = Console(file=io.StringIO(), width=200)
    return InteractiveTuiApp(
        nl, TuiSettings(engine_name="tiny"), console=console,
        err_console=console,
    )


class TestNiftiPaths:
    """Check the paths ending in .nii or .nii.gz are saved as NIfTI."""

    @pytest.mark.parametrize("name", [
        "out.nii", "out.nii.gz", "OUT.NII.GZ", "Out.Nii",
    ])
    def test_nifti_suffixes(self, tiny, tmp_path, name):
        tiny._execute_and_display("ans(i, j, k) :- voxel(i, j, k)")
        tiny._handle_dot_command(f".save {tmp_path / name}")
        img = nib.load(tmp_path / name)
        assert isinstance(img, nib.Nifti1Image)
        assert "NIfTI" in _output(tiny) and "CSV" not in _output(tiny)

    def test_nii_gz_is_compressed(self, tiny, tmp_path):
        tiny._execute_and_display("ans(i, j, k) :- voxel(i, j, k)")
        tiny._handle_dot_command(f".save {tmp_path / 'out.nii.gz'}")
        with open(tmp_path / "out.nii.gz", "rb") as f:
            assert f.read(2) == b"\x1f\x8b"
        assert isinstance(nib.load(tmp_path / "out.nii.gz"), nib.Nifti1Image)

    def test_csv_gz_stays_csv(self, tiny, tmp_path):
        tiny._execute_and_display("ans(i, j, k) :- voxel(i, j, k)")
        tiny._handle_dot_command(f".save {tmp_path / 'out.csv.gz'}")
        saved = pd.read_csv(tmp_path / "out.csv.gz")
        assert list(saved.columns) == ["i", "j", "k"]
        assert "CSV (gzip)" in _output(tiny)

    def test_tsv_is_tab_separated(self, tiny, tmp_path):
        tiny._execute_and_display("ans(i, j, k) :- voxel(i, j, k)")
        tiny._handle_dot_command(f".save {tmp_path / 'out.tsv'}")
        saved = pd.read_csv(tmp_path / "out.tsv", sep="\t")
        assert list(saved.columns) == ["i", "j", "k"]
        assert "TSV" in _output(tiny)

    def test_message_tells_what_was_written(self, tiny, tmp_path):
        tiny._execute_and_display("ans(i, j, k) :- voxel(i, j, k)")
        tiny._handle_dot_command(f".save {tmp_path / 'out.csv'}")
        assert "Saved 2 rows as CSV to" in _output(tiny)

    def test_several_blocks_keep_the_nifti_suffix(self, tiny, tmp_path):
        tiny.settings.squall = True
        tiny._execute_and_display(
            "obtain every Voxel in 3D as First. "
            "obtain every Voxel in 3D as Second."
        )
        tiny._handle_dot_command(f".save {tmp_path / 'out.nii.gz'}")
        for name in ("out_first.nii.gz", "out_second.nii.gz"):
            assert isinstance(nib.load(tmp_path / name), nib.Nifti1Image)
