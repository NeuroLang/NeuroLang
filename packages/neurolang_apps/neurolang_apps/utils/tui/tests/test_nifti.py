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
from neurolang_apps.utils.engine_registry import Sort  # noqa: E402
from neurolang_apps.utils.tui import (  # noqa: E402
    InteractiveTuiApp,
    TuiSettings,
)
from neurolang_apps.utils.tui import nifti  # noqa: E402


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


# A tiny engine: a 4×4×4 template grid of 2 mm voxels, its origin away
# from the grid's corner, and the sorts declaring the voxel indices and
# the millimetre coordinates on it.
TINY_AFFINE = np.array([
    [2.0, 0.0, 0.0, -3.0],
    [0.0, 2.0, 0.0, -5.0],
    [0.0, 0.0, 2.0, -7.0],
    [0.0, 0.0, 0.0, 1.0],
])

# The grid of the tiny engine's atlas, another than its template's:
# 3×5×2 voxels of 3 mm.
ATLAS_AFFINE = np.array([
    [3.0, 0.0, 0.0, -6.0],
    [0.0, 3.0, 0.0, -9.0],
    [0.0, 0.0, 3.0, -3.0],
    [0.0, 0.0, 0.0, 1.0],
])

TINY_ENGINE = """\
engines:
  tiny:
    atlases:
      schaefer: {predicate_name: schaefer}
    sorts:
      study: {columns: [study_id], type: int}
      tfidf: {columns: [tfidf], type: float}
      voxel: {columns: [i, j, k], type: int, space: voxel}
      mni: {columns: [x, y, z], type: float, space: mm}
      atlas_voxel:
        columns: [i, j, k]
        type: int
        space: voxel
        grid: schaefer
      label: {columns: [label], type: str}
      streamline: {columns: [streamline], type: int}
    predicates:
      voxel: {arguments: [voxel]}
      study: {arguments: [study]}
      reports: {arguments: [study, voxel]}
      weight: {arguments: [voxel, tfidf]}
      peak: {arguments: [mni, study]}
      mni: {arguments: [mni]}
      atlas_voxel: {arguments: [atlas_voxel]}
      labels: {arguments: [label, atlas_voxel]}
      tract: {arguments: [streamline]}
"""


@pytest.fixture
def tiny(monkeypatch, tmp_path):
    """Return a REPL on the tiny engine, with its 4×4×4 template."""
    from neurolang_apps.utils import engine_registry

    path = tmp_path / "engines.yaml"
    path.write_text(TINY_ENGINE)
    monkeypatch.setattr(engine_registry, "_ENGINES_YAML", path)
    nl = NeurolangPDL()
    nl.template = nib.Nifti1Image(
        np.ones((4, 4, 4), dtype=np.uint8), TINY_AFFINE
    )
    nl.add_tuple_set([(1, 2, 3), (0, 0, 0)], name="voxel")
    nl.add_tuple_set([(10,), (11,), (12,)], name="study")
    nl.add_tuple_set(
        [(10, 1, 2, 3), (11, 1, 2, 3), (12, 0, 0, 0)], name="reports"
    )
    nl.add_tuple_set(
        [(1, 2, 3, 0.25), (1, 2, 3, 0.75), (0, 0, 0, 0.5)], name="weight"
    )
    nl.add_tuple_set(
        [(-1.0, -1.0, -1.0, 10), (-3.2, -4.9, -7.0, 11)], name="peak"
    )
    nl.add_tuple_set([(-1.0, -1.0, -1.0), (-3.2, -4.9, -7.0)], name="mni")
    nl.grids = {
        "schaefer": nib.Nifti1Image(
            np.zeros((3, 5, 2), dtype=np.int16), ATLAS_AFFINE
        ),
    }
    nl.add_tuple_set([(2, 4, 1), (0, 0, 0)], name="atlas_voxel")
    nl.add_tuple_set([("a", 2, 4, 1), ("b", 0, 0, 0)], name="labels")
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
        # the file has exactly the name asked for, whatever its case, so
        # that it is found on case-sensitive file systems too
        assert name in [p.name for p in tmp_path.iterdir()]
        with open(tmp_path / name, "rb") as f:
            compressed = f.read(2) == b"\x1f\x8b"
        assert compressed == name.lower().endswith(".gz")
        # nib.load would look for a lowercase suffix on Linux (Out.nii)
        img = nifti.read_nifti(tmp_path / name)
        assert isinstance(img, nib.Nifti1Image)
        assert "NIfTI" in _output(tiny) and "CSV" not in _output(tiny)

    @pytest.mark.parametrize("name", ["Out.Nii", "OUT.NII.GZ", "v.nii"])
    def test_view_reads_any_case_of_suffix(
        self, tiny, tmp_path, view_calls, monkeypatch, name
    ):
        """
        Check .view reads the image itself, not through nib.load.

        nib.load looks for a lowercase suffix on case-sensitive file
        systems, so it cannot open 'Out.Nii' on Linux.
        """
        tiny._execute_and_display("ans(i, j, k) :- voxel(i, j, k)")
        tiny._handle_dot_command(f".save {tmp_path / name}")

        def no_load(*args, **kwargs):
            raise AssertionError("nib.load cannot open every NIfTI name")

        # nilearn loads the MNI background with nib.load: keep it out of
        # the stub, which would fail it and, with some versions of
        # nilearn, the backgrounds of the tests after this one.
        monkeypatch.setattr(
            nifti.brain_viewer, "mni_background", lambda: None
        )
        monkeypatch.setattr(nifti.nib, "load", no_load)
        tiny._handle_dot_command(f".view {tmp_path / name}")
        assert len(view_calls) == 1
        out = _output(tiny)
        Path(out.split("viewer to: ")[1].split()[0]).unlink()

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
            tiny, tmp_path, "ans(x, y, z, study_id) :- peak(x, y, z, study_id)"
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
            tiny, tmp_path,
            "ans(a, b, c, study_id) :- peak(a, b, c, study_id)",
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
        tiny._nl.add_tuple_set([(10, 1, 2, 3)], name="seen")
        out = self._error(
            tiny, tmp_path, "ans(s, a, b, c) :- seen(s, a, b, c)"
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


def test_help_save_tells_the_convention(tiny):
    """Check ``.help save`` documents the voxels and their value."""
    tiny._handle_dot_command(".help save")
    out = _output(tiny)
    assert "space: voxel" in out and "space: mm" in out
    assert "--value COLUMN" in out and "maximum" in out
    assert "mask" in out
    # the voxels follow from the last query, and a sort names its grid
    assert "ans(x, y, z) :- voxel(x, y, z)" in out
    assert "grid: schaefer" in out


def test_help_lists_the_save_options(tiny):
    tiny._handle_dot_command(".help")
    assert ".save PATH [--value COLUMN]" in _output(tiny)
    assert ".help save" in _output(tiny)


@pytest.fixture
def tiny_image(tiny, tmp_path):
    """Save the voxels of the tiny engine in a NIfTI image, its path."""
    _saved(tiny, tmp_path, "ans(i, j, k) :- voxel(i, j, k)", name="v.nii")
    return tmp_path / "v.nii"


def _terminal_app(app):
    """Return a REPL on the engine of *app* whose console is a terminal."""
    console = Console(file=io.StringIO(), width=200, force_terminal=True)
    return InteractiveTuiApp(
        app._nl, app.settings, console=console, err_console=console
    )


class TestView:
    """Check .view shows an image in the interactive brain viewer."""

    def test_missing_file(self, tiny, tmp_path):
        tiny._handle_dot_command(f".view {tmp_path / 'missing.nii'}")
        assert "File not found" in _output(tiny)

    def test_opens_the_viewer_in_a_browser(
        self, tiny, tiny_image, monkeypatch
    ):
        pytest.importorskip("nilearn.plotting")
        opened = []
        monkeypatch.setattr(
            nifti.webbrowser, "open", lambda url: opened.append(url) or True
        )
        app = _terminal_app(tiny)
        app._handle_dot_command(f".view {tiny_image}")
        assert len(opened) == 1 and opened[0].startswith("file://")
        html = Path(opened[0][len("file://"):])
        assert html.suffix == ".html" and html.stat().st_size > 0
        assert str(html) in _output(app)
        html.unlink()

    def test_prints_the_path_without_a_terminal(
        self, tiny, tiny_image, monkeypatch
    ):
        pytest.importorskip("nilearn.plotting")
        opened = []
        monkeypatch.setattr(nifti.webbrowser, "open", opened.append)
        tiny._handle_dot_command(f".view {tiny_image}")
        assert opened == []
        out = _output(tiny)
        path = out.split("viewer to: ")[1].split()[0]
        assert path.endswith(".html") and Path(path).exists()
        page = Path(path).read_text()
        assert 'id="threshold"' in page and 'id="cmap"' in page
        Path(path).unlink()

    def test_the_background_is_the_mni_t1(
        self, tiny, tiny_image, monkeypatch
    ):
        calls = []

        def viewer_html(img, background, **kwargs):
            calls.append((img, background, kwargs))
            return "<html></html>"

        monkeypatch.setattr(nifti.brain_viewer, "viewer_html", viewer_html)
        tiny._handle_dot_command(f".view {tiny_image}")
        img, background, kwargs = calls[0]
        assert img.shape == (4, 4, 4)
        assert np.array_equal(img.affine, TINY_AFFINE)
        assert background.shape == (197, 233, 189)
        assert kwargs["title"] == "v.nii" and "threshold" in kwargs
        path = _output(tiny).split("viewer to: ")[1].split()[0]
        Path(path).unlink()


@pytest.fixture
def view_calls(monkeypatch):
    """Record the options of the viewer page, which is left empty."""
    calls = []

    def viewer_html(img, background, **kwargs):
        calls.append(kwargs)
        return "<html></html>"

    monkeypatch.setattr(nifti.brain_viewer, "viewer_html", viewer_html)
    return calls


@pytest.fixture
def signed_image(tmp_path):
    """Return the path of a NIfTI image with negative and positive values."""
    data = np.zeros((4, 4, 4), dtype=np.float32)
    data[1, 2, 3], data[0, 0, 0] = -2.0, 3.0
    path = tmp_path / "signed.nii"
    nib.save(nib.Nifti1Image(data, TINY_AFFINE), path)
    return path


class TestViewOptions:
    """Check the threshold and colormap options of .view."""

    def _view(self, app, line, calls):
        app._handle_dot_command(line)
        out = _output(app)
        if "viewer to: " in out:
            Path(out.split("viewer to: ")[1].split()[0]).unlink()
        return (calls[-1] if calls else None), out

    def test_positive_image_gets_a_one_sided_colormap(
        self, tiny, tiny_image, view_calls
    ):
        kwargs, out = self._view(tiny, f".view {tiny_image}", view_calls)
        assert kwargs["symmetric"] is False
        assert "one-sided" in out and "--symmetric" in out

    def test_signed_image_gets_a_symmetric_colormap(
        self, tiny, signed_image, view_calls
    ):
        kwargs, out = self._view(tiny, f".view {signed_image}", view_calls)
        assert kwargs["symmetric"] is True
        assert "symmetric" in out and "--one-sided" in out

    @pytest.mark.parametrize(
        "flag, symmetric", [("--symmetric", True), ("--one-sided", False)]
    )
    def test_symmetry_can_be_chosen(
        self, tiny, signed_image, view_calls, flag, symmetric
    ):
        kwargs, _ = self._view(
            tiny, f".view {signed_image} {flag}", view_calls
        )
        assert kwargs["symmetric"] is symmetric

    @pytest.mark.parametrize(
        "value, threshold", [("0.5", 0.5), ("2", 2.0), ("95%", "95%")]
    )
    def test_threshold(self, tiny, tiny_image, view_calls, value, threshold):
        kwargs, _ = self._view(
            tiny, f".view {tiny_image} --threshold {value}", view_calls
        )
        assert kwargs["threshold"] == threshold

    def test_colormap(self, tiny, tiny_image, view_calls):
        kwargs, _ = self._view(
            tiny, f".view {tiny_image} --cmap viridis", view_calls
        )
        assert kwargs["cmap"] == "viridis"

    @pytest.mark.parametrize(
        "options, message",
        [
            ("--cmap no_such_map", "Unknown colormap"),
            ("--threshold -1", "--threshold"),
            ("--threshold abc", "--threshold"),
            ("--threshold 101%", "--threshold"),
            ("--symmetric --one-sided", "not both"),
            ("--zoom 2", "Unknown option"),
        ],
    )
    def test_bad_options(
        self, tiny, tiny_image, view_calls, options, message
    ):
        kwargs, out = self._view(
            tiny, f".view {tiny_image} {options}", view_calls
        )
        assert kwargs is None
        assert message in out and "File not found" not in out

    def test_help_lists_the_view_options(self, tiny):
        tiny._handle_dot_command(".help")
        out = _output(tiny)
        assert "--threshold" in out and "--cmap" in out
        assert "--symmetric" in out
        assert "nilearn" not in out.split(".view")[1].splitlines()[0]


class TestViewLastResult:
    """Check .view without a path shows the last result as an image."""

    def test_shows_the_last_result(self, tiny, view_calls):
        import tempfile

        tiny._execute_and_display("ans(i, j, k) :- voxel(i, j, k)")
        tiny._handle_dot_command(".view")
        assert len(view_calls) == 1
        assert view_calls[0]["title"] == "Last result"
        out = _output(tiny)
        image = Path(out.split("image to: ")[1].split()[0])
        assert image.parent == Path(tempfile.gettempdir())
        assert image.name.endswith(".nii.gz")
        img = nib.load(image)
        assert np.array_equal(img.affine, TINY_AFFINE)
        assert img.get_fdata()[1, 2, 3] == 1 and img.get_fdata().sum() == 2
        image.unlink()
        Path(out.split("viewer to: ")[1].split()[0]).unlink()

    def test_takes_the_save_and_view_options(self, tiny, view_calls):
        tiny._execute_and_display(
            "ans(i, j, k, n) :- reports(s, i, j, k), study(s), (n == 2.5)"
        )
        tiny._handle_dot_command(".view --value n --threshold 1 --cmap hot")
        kwargs = view_calls[-1]
        assert kwargs["threshold"] == 1.0 and kwargs["cmap"] == "hot"
        out = _output(tiny)
        image = Path(out.split("image to: ")[1].split()[0])
        assert nib.load(image).get_fdata().max() == 2.5
        image.unlink()
        Path(out.split("viewer to: ")[1].split()[0]).unlink()

    def test_without_a_result(self, tiny, view_calls):
        tiny._handle_dot_command(".view")
        assert view_calls == []
        assert "run a query" in _output(tiny)

    def test_result_without_voxels(self, tiny, view_calls):
        tiny._execute_and_display("ans(s) :- study(s)")
        tiny._handle_dot_command(".view")
        assert view_calls == []
        assert "voxel" in _output(tiny)

    def test_save_options_need_the_last_result(
        self, tiny, tiny_image, view_calls
    ):
        tiny._handle_dot_command(f".view {tiny_image} --value n")
        assert view_calls == []
        out = _output(tiny)
        assert "--value" in out and "File not found" not in out


@pytest.fixture
def view_layers(monkeypatch):
    """Record the images of the viewer page, which is left empty."""
    calls = []

    def viewer_html(img, background, **kwargs):
        calls.append((img, kwargs))
        return "<html></html>"

    monkeypatch.setattr(nifti.brain_viewer, "viewer_html", viewer_html)
    return calls


def _remove_view_files(out):
    """Remove the images and the page .view wrote, as *out* tells."""
    for line in out.split("image to: ")[1:]:
        Path(line.split()[0]).unlink()
    Path(out.split("viewer to: ")[1].split()[0]).unlink()


class TestViewColumns:
    """Check each value column of the last result is an overlay."""

    QUERY = "ans(i, j, k, a, b) :- voxel(i, j, k), (a == i + 1.5), (b == j - 0.5)"

    def test_one_overlay_per_value_column(self, tiny, view_layers):
        tiny._execute_and_display(self.QUERY)
        tiny._handle_dot_command(".view")
        layers, kwargs = view_layers[-1]
        assert [name for name, _ in layers] == ["a", "b"]
        a, b = (img.get_fdata() for _, img in layers)
        assert a[1, 2, 3] == 2.5 and a[0, 0, 0] == 1.5
        assert b[1, 2, 3] == 1.5 and b[0, 0, 0] == -0.5
        assert a[3, 3, 3] == 0 and b[3, 3, 3] == 0
        assert kwargs["symmetric"] == [False, True]
        out = _output(tiny)
        assert "'a'" in out and "'b'" in out
        _remove_view_files(out)

    def test_missing_values_take_the_fill(self, tiny, view_layers):
        tiny._execute_and_display(self.QUERY)
        tiny._handle_dot_command(".view --fill 1")
        layers, _ = view_layers[-1]
        a, b = (img.get_fdata() for _, img in layers)
        assert a[3, 3, 3] == 1 and b[3, 3, 3] == 1
        assert a[1, 2, 3] == 2.5
        _remove_view_files(_output(tiny))

    def test_value_picks_one_column(self, tiny, view_layers):
        tiny._execute_and_display(self.QUERY)
        tiny._handle_dot_command(".view --value b")
        img, _ = view_layers[-1]
        assert isinstance(img, nib.Nifti1Image)
        assert img.get_fdata()[1, 2, 3] == 1.5
        _remove_view_files(_output(tiny))

    def test_bad_fill(self, tiny, view_layers):
        tiny._execute_and_display(self.QUERY)
        tiny._handle_dot_command(".view --fill abc")
        assert view_layers == []
        assert "--fill" in _output(tiny)

    def test_help_tells_the_overlays_and_the_fill(self, tiny):
        tiny._handle_dot_command(".help view")
        out = _output(tiny)
        assert "--fill" in out and "overlay" in out


def test_nan_values_take_the_fill():
    frame = pd.DataFrame({
        "i": [0, 1], "j": [0, 0], "k": [0, 0], "v": [np.nan, 2.0],
    })
    sorts = {"voxel": Sort("voxel", ("i", "j", "k"), "int", space="voxel")}
    template = nib.Nifti1Image(np.ones((2, 2, 2), np.uint8), np.eye(4))
    saved = nifti.result_to_nifti(
        frame, template, sorts, nifti.SaveRequest(Path(), value="v", fill=-1.0)
    )
    data = saved.image.get_fdata()
    assert data[0, 0, 0] == -1 and data[1, 0, 0] == 2 and data[1, 1, 1] == -1

def test_view_keeps_nilearn_warnings_quiet(tiny, tiny_image):
    """Check the warnings of nilearn's viewer do not reach the prompt."""
    import warnings

    pytest.importorskip("nilearn.plotting")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tiny._handle_dot_command(f".view {tiny_image}")
    out = _output(tiny)
    assert "Saved the viewer to" in out
    Path(out.split("viewer to: ")[1].split()[0]).unlink()


class TestPathsWithSpaces:
    """Check .save and .view take a path with spaces, quoted or not."""

    def test_save_unquoted_nifti_path(self, tiny, tmp_path):
        img = _saved(
            tiny, tmp_path, "ans(i, j, k) :- voxel(i, j, k)",
            name="my brain map.nii.gz",
        )
        assert np.asanyarray(img.dataobj)[1, 2, 3] == 1

    def test_save_unquoted_path_and_options(self, tiny, tmp_path):
        img = _saved(
            tiny, tmp_path, "ans(i, j, k, w) :- weight(i, j, k, w)",
            "--value w", name="my map.nii",
        )
        assert np.asanyarray(img.dataobj)[1, 2, 3] == 0.75

    def test_save_unquoted_table_path(self, tiny, tmp_path):
        tiny._execute_and_display("ans(i, j, k) :- voxel(i, j, k)")
        path = tmp_path / "my sub dir" / "a b.csv"
        tiny._handle_dot_command(f".save {path}")
        saved = pd.read_csv(path)
        assert len(saved) == 2

    def test_save_quoted_path(self, tiny, tmp_path):
        tiny._execute_and_display("ans(i, j, k) :- voxel(i, j, k)")
        tiny._handle_dot_command(f".save '{tmp_path / 'a  b.csv'}'")
        assert (tmp_path / "a  b.csv").exists()

    def test_save_needs_a_path(self, tiny, tmp_path):
        tiny._execute_and_display("ans(i, j, k) :- voxel(i, j, k)")
        tiny._handle_dot_command(".save --value i")
        assert "Usage: .save PATH" in _output(tiny)
        tiny._handle_dot_command(".save")
        assert _output(tiny).count("Usage: .save PATH") == 2

    @pytest.mark.parametrize("arg, path", [
        ("my out.nii --value w", "my out.nii"),
        ("'my  out.nii' --value w", "my  out.nii"),
        ("a b c.csv", "a b c.csv"),
    ])
    def test_parse_save_path_with_spaces(self, arg, path):
        assert nifti.parse_save_args(arg).path == Path(path)

    def test_view_unquoted_path(self, tiny, tmp_path, view_calls):
        _saved(
            tiny, tmp_path, "ans(i, j, k) :- voxel(i, j, k)",
            name="my image.nii",
        )
        tiny._handle_dot_command(
            f".view {tmp_path / 'my image.nii'} --cmap hot"
        )
        out = _output(tiny)
        assert "File not found" not in out
        assert view_calls[-1]["cmap"] == "hot"
        assert view_calls[-1]["title"] == "my image.nii"
        Path(out.split("viewer to: ")[1].split()[0]).unlink()

    def test_parse_view_path_with_spaces(self):
        request = nifti.parse_view_args("my image.nii.gz --threshold 2")
        assert request.path == Path("my image.nii.gz")
        assert request.threshold == 2.0

    def test_view_default_threshold_is_the_top_5_percent(self):
        request = nifti.parse_view_args("my image.nii.gz")
        assert request.threshold == "95%"
        assert request.cmap is None


def _data(img):
    return np.asanyarray(img.dataobj)


class TestProvenance:
    """
    Check the voxels of a result follow from the query giving it.

    A column holds the values of the argument of a declared predicate
    its variable is in, whatever its name: indices of the template's
    grid, of an atlas's grid, or millimetres.
    """

    def test_voxels_named_as_millimetres(self, tiny, tmp_path):
        img = _saved(tiny, tmp_path, "ans(x, y, z) :- voxel(x, y, z)")
        assert _data(img)[1, 2, 3] == 1 and _data(img)[0, 0, 0] == 1
        assert np.count_nonzero(_data(img)) == 2

    def test_reordered_voxel_columns(self, tiny, tmp_path):
        img = _saved(tiny, tmp_path, "ans(k, j, i) :- voxel(i, j, k)")
        assert _data(img)[1, 2, 3] == 1 and _data(img)[0, 0, 0] == 1

    def test_voxels_of_a_relation_of_entities(self, tiny, tmp_path):
        img = _saved(
            tiny, tmp_path, "ans(s, a, b, c) :- reports(s, a, b, c)"
        )
        assert _data(img)[1, 2, 3] == 1 and _data(img)[0, 0, 0] == 1

    def test_millimetres(self, tiny, tmp_path):
        img = _saved(tiny, tmp_path, "ans(x, y, z) :- mni(x, y, z)")
        assert _data(img)[1, 2, 3] == 1 and _data(img)[0, 0, 0] == 1
        assert np.count_nonzero(_data(img)) == 2

    def test_millimetres_named_as_voxels(self, tiny, tmp_path):
        img = _saved(tiny, tmp_path, "ans(i, j, k) :- mni(i, j, k)")
        assert _data(img)[1, 2, 3] == 1 and _data(img)[0, 0, 0] == 1
        assert "outside" not in _output(tiny)

    def test_through_a_rule_of_the_program(self, tiny, tmp_path):
        img = _saved(
            tiny, tmp_path,
            "near(x, y, z) :- voxel(x, y, z)\nans(x, y, z) :- near(x, y, z)",
        )
        assert _data(img)[1, 2, 3] == 1 and _data(img)[0, 0, 0] == 1

    def test_through_a_rule_defined_before(self, tiny, tmp_path):
        tiny._execute_and_display("near(x, y, z) :- voxel(x, y, z)")
        img = _saved(tiny, tmp_path, "ans(a, b, c) :- near(a, b, c)")
        assert _data(img)[1, 2, 3] == 1 and _data(img)[0, 0, 0] == 1

    def test_atlas_grid(self, tiny, tmp_path):
        img = _saved(
            tiny, tmp_path, "ans(l, i, j, k) :- labels(l, i, j, k)"
        )
        assert img.shape == (3, 5, 2)
        assert np.array_equal(img.affine, ATLAS_AFFINE)
        assert _data(img)[2, 4, 1] == 1 and _data(img)[0, 0, 0] == 1
        assert "3×5×2 schaefer grid" in _output(tiny)

    def test_atlas_grid_outside(self, tiny, tmp_path):
        tiny._nl.add_tuple_set([("c", 3, 4, 3)], name="labels")
        _saved(tiny, tmp_path, "ans(l, i, j, k) :- labels(l, i, j, k)")
        assert "outside the schaefer grid" in _output(tiny)

    def test_explicit_voxels_are_on_the_template(self, tiny, tmp_path):
        img = _saved(
            tiny, tmp_path, "ans(l, i, j, k) :- labels(l, i, j, k)",
            "--voxel i j k",
        )
        assert img.shape == (4, 4, 4)
        assert np.array_equal(img.affine, TINY_AFFINE)

    def test_squall_in_3d(self, tiny, tmp_path):
        tiny.settings.squall = True
        img = _saved(tiny, tmp_path, "obtain every Atlas_voxel in 3D.")
        assert img.shape == (3, 5, 2)
        assert _data(img)[2, 4, 1] == 1 and _data(img)[0, 0, 0] == 1

    def test_squall_tuple_named_as_millimetres(self, tiny, tmp_path):
        tiny.settings.squall = True
        img = _saved(
            tiny, tmp_path,
            "obtain every Voxel (?x; ?y; ?z) that a Study reports.",
        )
        assert img.shape == (4, 4, 4)
        assert _data(img)[1, 2, 3] == 1 and _data(img)[0, 0, 0] == 1

    def test_squall_whole_relation(self, tiny, tmp_path):
        tiny.settings.squall = True
        img = _saved(tiny, tmp_path, "obtain every Labels.")
        assert img.shape == (3, 5, 2)

    def test_squall_blocks_have_their_own_grid(self, tiny, tmp_path):
        tiny.settings.squall = True
        tiny._execute_and_display(
            "obtain every Atlas_voxel in 3D as First. "
            "obtain every Voxel in 3D as Second."
        )
        tiny._handle_dot_command(f".save {tmp_path / 'out.nii'}")
        assert nib.load(tmp_path / "out_first.nii").shape == (3, 5, 2)
        assert nib.load(tmp_path / "out_second.nii").shape == (4, 4, 4)

    def test_unknown_provenance_falls_back_on_names(self, tiny, tmp_path):
        tiny._nl.add_tuple_set([(-1.0, -1.0, -1.0)], name="spots")
        img = _saved(tiny, tmp_path, "ans(x, y, z) :- spots(x, y, z)")
        assert img.shape == (4, 4, 4)
        assert _data(img)[1, 2, 3] == 1

    def test_engine_without_the_grid(self, tiny, tmp_path):
        tiny._nl.grids = {}
        tiny._execute_and_display("ans(l, i, j, k) :- labels(l, i, j, k)")
        tiny._handle_dot_command(f".save {tmp_path / 'out.nii'}")
        assert not (tmp_path / "out.nii").exists()
        assert "no grid 'schaefer'" in _output(tiny)

    def test_view_on_the_atlas_grid(self, tiny, view_calls):
        tiny._execute_and_display("ans(l, i, j, k) :- labels(l, i, j, k)")
        tiny._handle_dot_command(".view")
        out = _output(tiny)
        image = Path(out.split("image to: ")[1].split()[0])
        assert nib.load(image).shape == (3, 5, 2)
        assert "schaefer grid" in out
        image.unlink()
        Path(out.split("viewer to: ")[1].split()[0]).unlink()


class TestProvenanceUnits:
    """Check the sorts of the columns of a query and their voxels."""

    @staticmethod
    def _sorts():
        """Return the sorts of :class:`TestUnits` and of the atlas."""
        atlas = Sort(
            "atlas_voxel", ("i", "j", "k"), "int", "", "voxel", "schaefer"
        )
        return {**TestUnits.SORTS, "atlas_voxel": atlas}

    def test_column_sorts_of_a_datalog_query(self, tiny):
        from neurolang_apps.utils.tui.provenance import (
            ColumnSort,
            column_sorts,
        )

        sorts = column_sorts(
            tiny._nl, "tiny", "ans(s, x, y, z) :- reports(s, x, y, z)",
            squall=False,
        )
        assert list(sorts.values()) == [(
            ColumnSort("study", 0), ColumnSort("voxel", 0),
            ColumnSort("voxel", 1), ColumnSort("voxel", 2),
        )]

    def test_column_sorts_through_a_resampling(self, tiny):
        from neurolang_apps.utils.tui.provenance import (
            ColumnSort,
            column_sorts,
        )

        sorts = column_sorts(
            tiny._nl, "tiny",
            "measure(x, y, z, p) :- reports(s, x, y, z) & (p == 1)\n"
            "rep(b, x, y, z, p) :- "
            "RESAMPLE[boot](measure(x, y, z, p)) = b\n"
            "ans(x, y, z, mean(p)) :- rep(b, x, y, z, p)",
            squall=False,
        )
        assert list(sorts.values()) == [(
            ColumnSort("voxel", 0), ColumnSort("voxel", 1),
            ColumnSort("voxel", 2), None,
        )]

    def test_unknown_column_sorts(self, tiny):
        from neurolang_apps.utils.tui.provenance import (
            COMPUTED,
            ColumnSort,
            column_sorts,
        )

        sorts = column_sorts(
            tiny._nl, "tiny", "ans(a, n) :- far(a), (n == 2)", squall=False,
        )
        assert list(sorts.values()) == [(None, ColumnSort(COMPUTED, 0))]

    def test_a_computed_column_is_the_value_whatever_its_name(self, tiny):
        from neurolang_apps.utils.tui.provenance import column_sorts

        sorts = column_sorts(
            tiny._nl, "tiny",
            "ans(i, j, k, study_id) :- weight(i, j, k, w) & "
            "(study_id == w * 2)",
            squall=False,
        )
        provenance = next(iter(sorts.values()))
        df = pd.DataFrame(
            {"i": [1], "j": [2], "k": [3], "study_id": [0.5]}
        )
        points = nifti.point_columns(df, self._sorts(), provenance)
        assert nifti.value_column(
            df, points, self._sorts(), provenance=provenance
        ) == "study_id"
        # without the query, the name says it is a study: a mask
        assert nifti.value_column(df, points, self._sorts()) is None

    def test_points_from_the_column_sorts(self):
        from neurolang_apps.utils.tui.provenance import ColumnSort

        df = pd.DataFrame(columns=["l", "x", "y", "z"])
        provenance = (
            None, ColumnSort("atlas_voxel", 0), ColumnSort("atlas_voxel", 1),
            ColumnSort("atlas_voxel", 2),
        )
        assert nifti.point_columns(
            df, self._sorts(), provenance
        ) == nifti.PointColumns(("x", "y", "z"), "voxel", "schaefer")

    def test_names_only_of_columns_of_unknown_sort(self):
        from neurolang_apps.utils.tui.provenance import ColumnSort

        df = pd.DataFrame(columns=["i", "j", "k"])
        provenance = (ColumnSort("study", 0), None, None)
        with pytest.raises(nifti.SaveError, match="no voxel columns"):
            nifti.point_columns(df, self._sorts(), provenance)


# Streamlines on the tiny template, in millimetres: id s runs along x,
# at y = -5 + 2 (s % 4).
def _streamline(s):
    return np.array([
        [-3.0 + 2 * n, -5.0 + 2 * (s % 4), -7.0] for n in range(4)
    ])


@pytest.fixture
def tracts(tiny):
    """Give the tiny engine streamlines, ``uncinate(s, side)`` of them
    and the ``tract(s)`` of the sort streamline; record the ids asked."""
    nl = tiny._nl
    nl.add_tuple_set(
        [(1, "left"), (2, "left"), (3, "right")], name="uncinate"
    )
    nl.add_tuple_set([(1,), (3,)], name="tract")
    nl.asked = []

    def streamline_points(ids):
        nl.asked.append(list(ids))
        return [_streamline(s) for s in ids]

    nl.streamline_points = streamline_points
    nl.streamline_background = lambda: nl.template
    return tiny


class TestViewStreamlines:
    """Check .view draws a result of streamline ids as streamlines."""

    def _view(self, app, query, calls):
        app._execute_and_display(query)
        app._handle_dot_command(".view")
        out = _output(app)
        for prefix in ("image to: ", "viewer to: "):
            if prefix in out:
                Path(out.split(prefix)[-1].split()[0]).unlink()
        return (calls[-1] if calls else None), out

    def test_ids_of_a_rule_are_streamlines(self, tracts, view_calls):
        tracts._execute_and_display(
            'left_uncinate(s) :- uncinate(s, "left")'
        )
        kwargs, out = self._view(
            tracts, "ans(x) :- left_uncinate(x)", view_calls
        )
        assert tracts.results[-1].sorts in (None, (None,))
        assert tracts._nl.asked == [[1, 2]]
        state = kwargs["streamlines"]
        assert state["shown"] == 2 and state["total"] == 2
        assert kwargs["threshold"] == 0.0
        assert "Streamlines of the column 'x': 2 of 2 shown" in out
        assert "subsampled" not in out

    def test_a_column_of_the_sort_streamline(self, tracts, view_calls):
        kwargs, _ = self._view(tracts, "ans(s) :- tract(s)", view_calls)
        assert tracts.results[-1].sorts[0].sort == nifti.STREAMLINE_SORT
        assert tracts._nl.asked == [[1, 3]]
        assert kwargs["streamlines"]["total"] == 2

    def test_a_column_documented_of_the_sort_streamline(
        self, tracts, view_calls
    ):
        tracts._nl.document_predicate(
            "uncinate", "The uncinate fasciculus.",
            [("x", "streamline"), ("h", "hemisphere")],
        )
        self._view(tracts, 'ans(x) :- uncinate(x, "left")', view_calls)
        assert tracts.results[-1].sorts[0].sort == nifti.STREAMLINE_SORT
        assert tracts._nl.asked == [[1, 2]]

    def test_a_rule_documented_of_the_sort_streamline(
        self, tracts, view_calls
    ):
        tracts._execute_and_display(
            "%% unc(s: streamline, side)\n"
            "%% Renamed.\n"
            "unc(s, side) :- uncinate(s, side)"
        )
        self._view(tracts, "ans(a, b) :- unc(a, b)", view_calls)
        assert tracts.results[-1].sorts[0].sort == nifti.STREAMLINE_SORT
        assert tracts._nl.asked == [[1, 2, 3]]

    @pytest.mark.parametrize("query", [
        "ans(x, y, z) :- mni(x, y, z)",
        "ans(i, j, k) :- voxel(i, j, k)",
    ])
    def test_voxels_stay_voxels(self, tracts, view_calls, query):
        kwargs, _ = self._view(tracts, query, view_calls)
        assert "streamlines" not in kwargs and tracts._nl.asked == []
        assert kwargs["threshold"] == "95%"

    def test_pairs_of_points_stay_a_network(self, tracts):
        tracts._execute_and_display(
            "ans(x, y, z, x2, y2, z2) :- mni(x, y, z), mni(x2, y2, z2)"
        )
        assert nifti._result_kind(
            tracts, tracts.results[-1], nifti.SaveRequest(Path("x.nii"))
        ) == "network"

    def test_streamlines_between_pairs_stay_streamlines(
        self, tracts, view_calls
    ):
        """The streamlines joining two regions, ans(s, l1, l2), are drawn,
        not the network of their ends."""
        kwargs, out = self._view(
            tracts,
            "ans(s, x, y, z, x2, y2, z2) :- tract(s), mni(x, y, z), "
            "mni(x2, y2, z2)",
            view_calls,
        )
        assert tracts.results[-1].sorts[0].sort == nifti.STREAMLINE_SORT
        assert "connectome" not in kwargs
        assert kwargs["streamlines"]["total"] == 2
        assert "Streamlines of the column 's': 2 of 2 shown" in out

    def test_several_integer_columns_are_not_streamlines(self, tracts):
        tracts._nl.add_tuple_set([(1, 7), (2, 8)], name="pairs")
        tracts._execute_and_display("ans(a, b) :- pairs(a, b)")
        assert nifti._result_kind(
            tracts, tracts.results[-1], nifti.SaveRequest(Path("x.nii"))
        ) == "voxels"

    def test_the_overlay_counts_the_streamlines_per_voxel(
        self, tracts, view_calls, monkeypatch
    ):
        images = []
        monkeypatch.setattr(
            nifti, "_view_image_file",
            lambda app, image: images.append(image) or Path("x.nii.gz"),
        )
        self._view(tracts, 'ans(x) :- uncinate(x, "left")', view_calls)
        counts = np.asarray(images[0].image.dataobj)
        assert np.array_equal(images[0].image.affine, TINY_AFFINE)
        # ids 1 and 2 at y = -3 and -1 mm: voxel j = 1 and 2, all along i
        assert counts[:, 1, 0].tolist() == [1, 1, 1, 1]
        assert counts[:, 2, 0].tolist() == [1, 1, 1, 1]
        assert counts.sum() == 8

    def test_the_streamlines_are_capped(
        self, tracts, view_calls, monkeypatch
    ):
        monkeypatch.setattr(nifti.brain_viewer, "MAX_STREAMLINES", 2)
        kwargs, out = self._view(
            tracts, "ans(x) :- uncinate(x, s)", view_calls
        )
        assert tracts._nl.asked == [[1, 3]]
        assert kwargs["streamlines"]["shown"] == 2
        assert kwargs["streamlines"]["total"] == 3
        assert "2 of 3 shown, evenly subsampled to 2 at most" in out

    def test_the_engine_background(self, tracts, monkeypatch):
        calls = []
        monkeypatch.setattr(
            nifti.brain_viewer, "viewer_html",
            lambda img, background, **kw: calls.append((background, kw))
            or "",
        )
        _, out = self._view(tracts, "ans(s) :- tract(s)", [])
        assert calls[0][0] is tracts._nl.template
        # their space, not MNI's: the cursor's mm are named so
        assert calls[0][1]["space"] == "RAS"
        assert "MNI152" not in out

    def test_the_mni_background_otherwise(self, tracts, monkeypatch):
        pytest.importorskip("nilearn")
        calls = []
        monkeypatch.setattr(
            nifti.brain_viewer, "viewer_html",
            lambda img, background, **kw: calls.append((background, kw))
            or "",
        )
        del tracts._nl.streamline_background
        _, out = self._view(tracts, "ans(s) :- tract(s)", [])
        assert calls[0][0].shape == (197, 233, 189)
        assert calls[0][1].get("space", "MNI") == "MNI"
        # streamlines of a subject would be misplaced on it: say so
        assert "drawn on the MNI152 T1" in out

    def test_streamlines_in_mni_space_are_on_the_mni_t1(
        self, tracts, monkeypatch
    ):
        """An engine whose streamlines are in MNI space gives no background
        of its own (None): they are drawn on the MNI152 T1, not on the
        labels of its atlas, and that is not warned about."""
        pytest.importorskip("nilearn")
        calls = []
        monkeypatch.setattr(
            nifti.brain_viewer, "viewer_html",
            lambda img, background, **kw: calls.append((background, kw))
            or "",
        )
        tracts._nl.streamline_background = lambda: None
        _, out = self._view(tracts, "ans(s) :- tract(s)", [])
        assert calls[0][0].shape == (197, 233, 189)
        assert calls[0][1].get("space", "MNI") == "MNI"
        assert "drawn on the MNI152 T1" not in out

    def test_a_sort_streamline_without_the_hook(self, tracts, view_calls):
        del tracts._nl.streamline_points
        kwargs, out = self._view(tracts, "ans(s) :- tract(s)", view_calls)
        assert kwargs is None
        assert "engine has no streamlines" in out
        assert "streamline_points" in out

    def test_integers_without_the_hook_are_not_streamlines(
        self, tracts, view_calls
    ):
        del tracts._nl.streamline_points
        kwargs, out = self._view(
            tracts, 'ans(x) :- uncinate(x, "left")', view_calls
        )
        assert kwargs is None and "no voxel columns" in out

    def test_help_tells_the_hook(self, tiny):
        tiny._handle_dot_command(".help view")
        out = _output(tiny)
        assert "streamline_points" in out
        assert "streamline_background" in out


SEVERAL_QUERIES = """\
ans[seen](i, j, k) :- voxel(i, j, k)
ans[weighted](i, j, k, w) :- weight(i, j, k, w)
"""


class TestViewSeveralQueries:
    """Check .view draws the results of all the queries of a program."""

    @pytest.fixture
    def viewed(self, monkeypatch):
        """Record the images and the options of the viewer page."""
        calls = []

        def viewer_html(img, background, **kwargs):
            calls.append((img, kwargs))
            return "<html></html>"

        monkeypatch.setattr(nifti.brain_viewer, "viewer_html", viewer_html)
        return calls

    def _view(self, app, program):
        app._execute_and_display(program)
        app._handle_dot_command(".view")
        out = _output(app)
        rest = out
        for prefix in ("image to: ", "viewer to: "):
            while prefix in rest:
                _, _, rest = rest.partition(prefix)
                Path(rest.split()[0]).unlink(missing_ok=True)
        return out

    def test_a_result_block_per_query(self, tiny):
        tiny._execute_and_display(SEVERAL_QUERIES)
        assert [block.title for block in tiny.results] == [
            "seen", "weighted"
        ]
        assert [block.sorts[0].sort for block in tiny.results] == [
            "voxel", "voxel"
        ]

    def test_the_layers_of_all_the_queries_in_one_viewer(
        self, tiny, viewed
    ):
        self._view(tiny, SEVERAL_QUERIES)
        (images, kwargs), = viewed
        assert [name for name, _ in images] == [
            "seen: mask", "weighted: w"
        ]
        assert "Last results: seen, weighted" in kwargs["title"]

    def test_a_network_and_streamlines_beside_images(
        self, tracts, viewed
    ):
        self._view(tracts, SEVERAL_QUERIES + (
            "ans[tracts](s) :- tract(s)\n"
            "ans[pairs](x, y, z, x2, y2, z2, w) :- mni(x, y, z), "
            "mni(x2, y2, z2), weight(1, 2, 3, w)\n"
        ))
        (images, kwargs), = viewed
        names = [name for name, _ in images]
        assert names[:2] == ["seen: mask", "weighted: w"]
        assert "tracts: streamlines" in names
        assert "pairs: node strength" in names
        assert kwargs["streamlines"]["total"] == 2
        assert kwargs["connectome"]["nodes"]

    def test_one_set_of_streamlines_is_drawn(self, tracts, viewed):
        out = self._view(tracts, (
            "ans[a](s) :- tract(s)\nans[b](s) :- tract(s)\n"
        ))
        (images, kwargs), = viewed
        assert "streamlines" in kwargs
        assert isinstance(images, nib.Nifti1Image)  # a single layer
        assert "Only one streamlines is drawn: that of b" in out

    def test_a_result_that_cannot_be_drawn_is_left_out(self, tiny, viewed):
        """A query without voxels, regions or streamlines, as the terms of
        a Bayes factor beside the regions of the same program, is told and
        left out: the others are drawn."""
        out = self._view(
            tiny, SEVERAL_QUERIES + "ans[studies](s) :- study(s)\n"
        )
        (images, kwargs), = viewed
        assert [name for name, _ in images] == [
            "seen: mask", "weighted: w"
        ]
        assert "studies is not drawn:" in out
        assert "View failed" not in out
        assert "Last results: seen, weighted" in kwargs["title"]

    def test_the_view_fails_when_no_result_can_be_drawn(self, tiny, viewed):
        out = self._view(
            tiny, "ans[a](s) :- study(s)\nans[b](s) :- study(s)\n"
        )
        assert viewed == []
        assert "a is not drawn:" in out and "b is not drawn:" in out
        assert "View failed: none of the results can be drawn" in out

    def test_the_threshold_is_the_default_with_voxels(self, tiny, viewed):
        self._view(tiny, SEVERAL_QUERIES)
        assert viewed[0][1]["threshold"] == "95%"
