"""The engine ``tracts`` of engines.yaml: the commands and the vocabulary of
nl_tracts, which neurolang_apps names in its YAML only (the package is the
extra ``tracts``)."""

import importlib
import io
import sys
from pathlib import Path

import pytest

from neurolang_apps.utils import engine_registry


def test_the_engine_is_declared():
    assert "tracts" in engine_registry.list_engine_names()
    cfg = engine_registry.get_engine_config("tracts")
    assert cfg["solver_mixins"] == ["nl_tracts.frontend:TractsCommandsMixin"]
    assert cfg["python_init"] == "nl_tracts.frontend"
    # nothing to download: no MNI template, no atlas of nilearn
    assert not cfg.get("requires_mni_mask")
    assert not {"atlases", "templates", "downloads"} & set(cfg)


def test_without_nl_tracts_the_engine_fails_clearly(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "nl_tracts", None)
    monkeypatch.setitem(sys.modules, "nl_tracts.frontend", None)
    with pytest.raises(ValueError, match=r"engine 'tracts' cannot be loaded"
                       r".*nl_tracts.*install the package"):
        engine_registry.build_engine("tracts", tmp_path)
    # the other engines are still listed and described
    assert "tracts" in engine_registry.list_engine_names()
    engine_registry.show_engines()


@pytest.fixture
def uncinate():
    """The example of nl_tracts and the files it reads, written by its
    tests; skipped if nl_tracts or its example is not there."""
    pytest.importorskip("nl_tracts")
    tests = importlib.import_module("nl_tracts.tests.test_uncinate")
    files = importlib.import_module("nl_tracts.tests.uncinate_files")
    if not tests.EXAMPLE.exists():
        pytest.skip("the examples of nl_tracts are not installed")
    return tests, files


def test_the_engine_runs_the_example(uncinate, tmp_path):
    tests, files = uncinate
    files.write_files(tmp_path)
    nl = engine_registry.build_engine("tracts", tmp_path / "data")
    assert nl.template is None
    result = nl.execute_datalog_program(tests.example(tmp_path))
    assert set(result.as_pandas_dataframe().iloc[:, 0]) == files.expected(
        "left")
    assert callable(nl.streamline_points)
    assert nl.streamline_background() is not None


def test_the_example_at_the_prompt_and_view(uncinate, tmp_path, monkeypatch):
    """Statement by statement, as typed or piped at the prompt, then
    ``.view``: the streamlines of the answer, on the atlas."""
    from rich.console import Console

    from neurolang_apps.utils.tui import InteractiveTuiApp, TuiSettings
    from neurolang_apps.utils.tui import nifti

    tests, files = uncinate
    files.write_files(tmp_path)
    nl = engine_registry.build_engine("tracts", tmp_path / "data")
    console = Console(file=io.StringIO(), width=200)
    app = InteractiveTuiApp(
        nl, TuiSettings(), console=console, err_console=console
    )
    calls = []
    monkeypatch.setattr(
        nifti.brain_viewer, "viewer_html",
        lambda img, background, **kwargs: calls.append(
            (background, kwargs)) or "<html></html>",
    )
    monkeypatch.setattr(nifti, "_open_in_browser", lambda *args: None)
    lines = io.StringIO(tests.example(tmp_path))
    status = app.run_batch(lines)
    out = console.file.getvalue()
    assert status == 0, out
    assert set(app.results[-1].frame.iloc[:, 0]) == files.expected("left")
    app._handle_dot_command(".view")
    out = console.file.getvalue()
    assert "Streamlines of the column" in out, out
    background, kwargs = calls[-1]
    assert kwargs["space"] == "RAS"
    assert kwargs["streamlines"]["total"] == len(files.expected("left"))
    assert background.shape == files.SHAPE
    for prefix in ("image to: ", "viewer to: "):
        if prefix in out:
            Path(out.split(prefix)[-1].split()[0]).unlink(missing_ok=True)


def test_the_example_as_the_program_of_the_command_line(uncinate, tmp_path):
    """``neurolang-query --engine tracts -i --file uncinate.dl``: the file
    is one submission, then the session goes on with what it loaded."""
    from rich.console import Console

    from neurolang_apps.utils.tui import InteractiveTuiApp, TuiSettings

    tests, files = uncinate
    files.write_files(tmp_path)
    nl = engine_registry.build_engine("tracts", tmp_path / "data")
    console = Console(file=io.StringIO(), width=200)
    app = InteractiveTuiApp(
        nl, TuiSettings(), console=console, err_console=console
    )
    app._run_initial_program(tests.example(tmp_path))
    assert not app.failed, console.file.getvalue()
    assert set(app.results[-1].frame.iloc[:, 0]) == files.expected("left")
    app._run_line('ans(x) :- streamline(x, t, s), (s < 2)')
    assert not app.failed, console.file.getvalue()
    assert set(app.results[-1].frame.iloc[:, 0]) == {0, 1}
