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
      study: {arguments: [study]}
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


def test_help_save_tells_the_convention(tiny):
    """Check ``.help save`` documents the voxels and their value."""
    tiny._handle_dot_command(".help save")
    out = _output(tiny)
    assert "space: voxel" in out and "space: mm" in out
    assert "--value COLUMN" in out and "maximum" in out
    assert "mask" in out


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
