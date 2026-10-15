"""Tests for saving and viewing results in the interactive REPL."""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

import nibabel as nib  # noqa: E402
from rich.console import Console  # noqa: E402

from neurolang.frontend import NeurolangPDL  # noqa: E402
from neurolang.utils.engine_registry import Sort  # noqa: E402
from neurolang.utils.tui import InteractiveTuiApp, TuiSettings  # noqa: E402
from neurolang.utils.tui import nifti  # noqa: E402


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
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.float32), np.eye(4)), path)
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


def _saved(app, tmp_path, query, options="", name="out.nii.gz"):
    """Run the Datalog *query*, save it in *name* and load the image."""
    app._execute_and_display(query)
    app._handle_dot_command(f".save {tmp_path / name} {options}".strip())
    return nib.load(tmp_path / name)


class TestSaveOnTheTemplate:
    """Check .save writes the result on the grid of the engine's template."""

    def test_shape_and_affine_of_the_template(self, tiny, tmp_path):
        img = _saved(tiny, tmp_path, "ans(i, j, k) :- voxel(i, j, k)")
        assert img.shape == (4, 4, 4)
        assert np.array_equal(img.affine, TINY_AFFINE)

    def test_voxel_indices_written_as_they_are(self, tiny, tmp_path):
        img = _saved(tiny, tmp_path, "ans(i, j, k) :- voxel(i, j, k)")
        data = np.asanyarray(img.dataobj)
        assert data[1, 2, 3] == 1 and data[0, 0, 0] == 1
        assert np.count_nonzero(data) == 2

    def test_mask_is_uint8(self, tiny, tmp_path):
        img = _saved(tiny, tmp_path, "ans(i, j, k) :- voxel(i, j, k)")
        assert img.get_data_dtype() == np.uint8
        assert "mask" in _output(tiny)

    def test_millimetres_converted_to_voxels(self, tiny, tmp_path):
        img = _saved(
            tiny, tmp_path, "ans(x, y, z, s) :- peak(x, y, z, s)"
        )
        data = np.asanyarray(img.dataobj)
        # (-1, -1, -1) mm is the centre of voxel (1, 2, 3);
        # (-3.2, -4.9, -7.0) mm the nearest to voxel (0, 0, 0).
        assert data[1, 2, 3] == 1 and data[0, 0, 0] == 1
        assert np.count_nonzero(data) == 2

    def test_the_value_column_and_the_max_of_duplicates(
        self, tiny, tmp_path
    ):
        img = _saved(
            tiny, tmp_path, "ans(i, j, k, w) :- weight(i, j, k, w)"
        )
        data = np.asanyarray(img.dataobj)
        assert img.get_data_dtype() == np.float32
        assert data[1, 2, 3] == 0.75 and data[0, 0, 0] == 0.5
        assert "w" in _output(tiny) and "max" in _output(tiny)

    def test_an_entity_column_is_not_a_value(self, tiny, tmp_path):
        img = _saved(
            tiny, tmp_path, "ans(i, j, k, study_id) :- "
            "reports(study_id, i, j, k)"
        )
        data = np.asanyarray(img.dataobj)
        assert img.get_data_dtype() == np.uint8
        assert data[1, 2, 3] == 1 and data[0, 0, 0] == 1

    def test_an_explicit_value_column(self, tiny, tmp_path):
        img = _saved(
            tiny, tmp_path, "ans(i, j, k, s) :- reports(s, i, j, k)",
            "--value s",
        )
        data = np.asanyarray(img.dataobj)
        assert img.get_data_dtype() == np.int32
        assert data[1, 2, 3] == 11 and data[0, 0, 0] == 12

    def test_explicit_voxel_columns(self, tiny, tmp_path):
        img = _saved(
            tiny, tmp_path, "ans(a, b, c) :- voxel(a, b, c)",
            "--voxel a b c",
        )
        assert np.count_nonzero(np.asanyarray(img.dataobj)) == 2

    def test_explicit_mm_columns(self, tiny, tmp_path):
        img = _saved(
            tiny, tmp_path, "ans(a, b, c, s) :- peak(a, b, c, s)",
            "--mm a b c",
        )
        assert np.asanyarray(img.dataobj)[1, 2, 3] == 1

    def test_squall_voxels_in_3d(self, tiny, tmp_path):
        tiny.settings.squall = True
        img = _saved(
            tiny, tmp_path, "obtain every Voxel in 3D that a Study reports."
        )
        data = np.asanyarray(img.dataobj)
        assert data[1, 2, 3] == 1 and data[0, 0, 0] == 1

    def test_squall_probability(self, tiny, tmp_path):
        tiny._nl.add_uniform_probabilistic_choice_over_set(
            [(10,), (11,), (12,)], name="selected_study"
        )
        tiny.settings.squall = True
        img = _saved(
            tiny, tmp_path,
            "define as Map with inferred probability every Voxel in 3D "
            "that a Selected_study reports. obtain every Map.",
        )
        data = np.asanyarray(img.dataobj)
        assert data[1, 2, 3] == pytest.approx(2 / 3)
        assert data[0, 0, 0] == pytest.approx(1 / 3)
        assert "probability" in _output(tiny)


class TestSaveErrors:
    """Check .save tells why it cannot write a NIfTI image."""

    def _error(self, app, tmp_path, query, options=""):
        app._execute_and_display(query)
        app._handle_dot_command(f".save {tmp_path / 'out.nii'} {options}")
        assert not (tmp_path / "out.nii").exists()
        return _output(app)

    def test_no_template(self, tiny, tmp_path):
        tiny._nl.template = None
        out = self._error(tiny, tmp_path, "ans(i, j, k) :- voxel(i, j, k)")
        assert "no template" in out

    def test_no_voxel_columns(self, tiny, tmp_path):
        out = self._error(
            tiny, tmp_path, "ans(s, a, b, c) :- reports(s, a, b, c)"
        )
        assert "no voxel columns" in out
        assert "i, j, k" in out and "x, y, z" in out
        assert "--voxel" in out

    def test_unknown_value_column(self, tiny, tmp_path):
        out = self._error(
            tiny, tmp_path, "ans(i, j, k) :- voxel(i, j, k)", "--value p"
        )
        assert "no column 'p'" in out

    def test_unknown_option(self, tiny, tmp_path):
        out = self._error(
            tiny, tmp_path, "ans(i, j, k) :- voxel(i, j, k)", "--colour red"
        )
        assert "--colour" in out

    def test_option_missing_its_columns(self, tiny, tmp_path):
        out = self._error(
            tiny, tmp_path, "ans(a, b, c) :- voxel(a, b, c)", "--voxel a b"
        )
        assert "--voxel takes 3 columns" in out

    def test_options_of_a_table(self, tiny, tmp_path):
        tiny._execute_and_display("ans(i, j, k) :- voxel(i, j, k)")
        tiny._handle_dot_command(f".save {tmp_path / 'out.csv'} --value i")
        assert "NIfTI" in _output(tiny)
        assert not (tmp_path / "out.csv").exists()

    def test_non_integer_voxel_indices(self, tiny, tmp_path):
        out = self._error(
            tiny, tmp_path, "ans(a, b, c, s) :- peak(a, b, c, s)",
            "--voxel a b c",
        )
        assert "not integers" in out and "--mm" in out

    def test_voxels_outside_the_grid_are_counted(self, tiny, tmp_path):
        tiny._nl.add_tuple_set([(9, 9, 9), (1, 1, 1)], name="far")
        img = _saved(tiny, tmp_path, "ans(i, j, k) :- far(i, j, k)")
        assert np.count_nonzero(np.asanyarray(img.dataobj)) == 1
        assert "1 row outside the template grid" in _output(tiny)


class TestUnits:
    """Check the parts of the conversion on their own."""

    SORTS = {
        "voxel": Sort("voxel", ("i", "j", "k"), "int", "", "voxel"),
        "mni": Sort("mni", ("x", "y", "z"), "float", "", "mm"),
        "study": Sort("study", ("study_id",), "int"),
        "tfidf": Sort("tfidf", ("tfidf",), "float"),
    }

    def test_voxel_columns_from_the_sorts(self):
        df = pd.DataFrame(columns=["k", "s", "i", "j"])
        assert nifti.point_columns(df, self.SORTS) == nifti.PointColumns(
            ("i", "j", "k"), "voxel"
        )

    def test_mm_columns_from_the_sorts(self):
        df = pd.DataFrame(columns=["x", "y", "z"])
        assert nifti.point_columns(df, self.SORTS).space == "mm"

    def test_voxel_columns_preferred_to_mm(self):
        df = pd.DataFrame(columns=["x", "y", "z", "i", "j", "k"])
        assert nifti.point_columns(df, self.SORTS).space == "voxel"

    def test_mm_to_voxel_indices(self):
        df = pd.DataFrame({"x": [-1.0, 0.9], "y": [-1.0, 1.0],
                           "z": [-1.0, 2.6]})
        points = nifti.PointColumns(("x", "y", "z"), "mm")
        indices = nifti.voxel_indices(df, points, TINY_AFFINE)
        assert indices.tolist() == [[1, 2, 3], [2, 3, 5]]

    @pytest.mark.parametrize("columns, expected", [
        (["i", "j", "k", "probability", "tfidf"], "probability"),
        (["i", "j", "k", "PROB"], "PROB"),
        (["i", "j", "k", "tfidf"], "tfidf"),
        (["i", "j", "k", "count"], "count"),
        (["i", "j", "k", "study_id"], None),
        (["i", "j", "k", "study_id", "tfidf"], "tfidf"),
        (["i", "j", "k", "tfidf", "count"], None),
        (["i", "j", "k", "term"], None),
        (["i", "j", "k"], None),
    ])
    def test_value_column_rule(self, columns, expected):
        values = {
            "term": ["pain"], "study_id": [1], "count": [3],
            "tfidf": [0.5], "probability": [0.5], "PROB": [0.5],
        }
        df = pd.DataFrame({
            c: values.get(c, [0]) for c in columns
        })
        points = nifti.PointColumns(("i", "j", "k"), "voxel")
        assert nifti.value_column(df, points, self.SORTS) == expected

    def test_parse_save_arguments(self):
        request = nifti.parse_save_args(
            "out.nii.gz --value tfidf --mm a b c"
        )
        assert request == nifti.SaveRequest(
            Path("out.nii.gz"), "tfidf",
            nifti.PointColumns(("a", "b", "c"), "mm"),
        )
        assert nifti.parse_save_args("'my out.csv'").path == Path(
            "my out.csv"
        )
