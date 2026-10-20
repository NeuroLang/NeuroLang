"""Tests of --save and --view, which run the REPL's .save and .view."""

import builtins
import sys

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

from neurolang.frontend import NeurolangPDL  # noqa: E402
from neurolang.utils import cli, engine_registry  # noqa: E402
from neurolang.utils.tui import brain_viewer  # noqa: E402

AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])
SHAPE = (4, 5, 6)
TEMPLATE = nib.Nifti1Image(np.ones(SHAPE, dtype=np.int8), AFFINE)


@pytest.fixture
def engine(monkeypatch):
    """A small engine with a template, for ``--engine neurosynth``."""
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [("s1", 0, 1, 2), ("s1", 3, 4, 5), ("s2", 0, 1, 2)],
        name="reports",
    )
    nl.add_tuple_set([("s1",), ("s2",)], name="study")
    nl.template = TEMPLATE
    nl.grids = {engine_registry.TEMPLATE_GRID: TEMPLATE}
    monkeypatch.setattr(
        engine_registry, "build_engine", lambda *args, **kwargs: nl
    )
    return nl


@pytest.fixture
def viewer(monkeypatch):
    """Record the images shown, rather than building and opening a page."""
    shown = []

    def viewer_html(images, background, **options):
        shown.append((images, options))
        return "<html></html>"

    monkeypatch.setattr(brain_viewer, "viewer_html", viewer_html)
    monkeypatch.setattr(brain_viewer, "mni_background", lambda: None)
    return shown


COUNTS = (
    "n_voxel(i, j, k, count(s)) :- reports(s, i, j, k)\n"
    "ans(i, j, k, n) :- n_voxel(i, j, k, n)"
)


def test_save_a_table(engine, tmp_path, capsys):
    path = tmp_path / "studies.csv"
    cli.main([
        "--engine", "neurosynth", "ans(s) :- study(s)", "--save", str(path),
        "--sort", "s:desc",
    ])
    assert path.read_text().splitlines() == ["s", "s2", "s1"]
    assert "Saved 2 rows as CSV" in capsys.readouterr().err


def test_save_a_tsv_after_the_extension(engine, tmp_path):
    path = tmp_path / "counts.tsv"
    cli.main(["--engine", "neurosynth", COUNTS, "--save", str(path)])
    assert path.read_text().splitlines()[0] == "i\tj\tk\tn"


def test_save_a_nifti_image(engine, tmp_path, capsys):
    path = tmp_path / "counts.nii.gz"
    cli.main(["--engine", "neurosynth", COUNTS, "--save", str(path)])
    image = nib.load(path)
    data = np.asanyarray(image.dataobj)
    assert image.shape == SHAPE and np.allclose(image.affine, AFFINE)
    assert data[0, 1, 2] == 2 and data[3, 4, 5] == 1 and data.sum() == 3
    assert "counts.nii.gz" in capsys.readouterr().err


def test_view_the_image_of_a_result(engine, viewer, capsys):
    cli.main(["--engine", "neurosynth", COUNTS, "--view"])
    [(images, options)] = viewer
    assert np.asanyarray(images.dataobj)[0, 1, 2] == 2
    assert "Saved the viewer to:" in capsys.readouterr().err


def test_save_and_view_with_ask(engine, viewer, tmp_path, monkeypatch):
    """--ask runs the query it writes for --save and --view."""
    from neurolang.utils.llm import agent as agent_module
    from neurolang.utils.llm.tests.fakes import ScriptedClient

    config = tmp_path / "llm.yaml"
    config.write_text(
        "default_provider: fake\nproviders:\n  fake: {api: openai, "
        "base_url: 'http://llm.test/v1', model: m, tool_calling: text}\n"
        "assistant: {strategy: react, few_shot_k: 0, verify_rounds: 0}\n"
    )
    answer = (
        'Action: final_answer\nAction Input: {"query": '
        '"ans(i, j, k) :- reports(s, i, j, k)", "explanation": "x"}'
    )
    client = ScriptedClient([answer])
    monkeypatch.setattr(
        agent_module, "make_client", lambda *args, **kwargs: client
    )
    path = tmp_path / "voxels.nii.gz"
    cli.main([
        "--engine", "neurosynth", "--ask", "Which voxels?", "--llm-config",
        str(config), "--save", str(path), "--view",
    ])
    assert np.asanyarray(nib.load(path).dataobj).sum() == 2
    assert len(viewer) == 1


def test_a_result_without_voxels_is_not_an_image(engine, tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli.main([
            "--engine", "neurosynth", "ans(s) :- study(s)", "--save",
            str(tmp_path / "x.nii.gz"),
        ])
    assert "Save failed" in capsys.readouterr().err


def test_unknown_formats_are_refused(engine, tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli.main([
            "--engine", "neurosynth", "ans(s) :- study(s)", "--save",
            str(tmp_path / "x.json"),
        ])
    assert "cannot tell the format" in capsys.readouterr().err
    assert not (tmp_path / "x.json").exists()


def test_without_the_tui_extra(engine, tmp_path, monkeypatch, capsys):
    real_import = builtins.__import__

    def no_prompt_toolkit(name, *args, **kwargs):
        if name.split(".")[0] == "prompt_toolkit":
            raise ModuleNotFoundError(name=name)
        return real_import(name, *args, **kwargs)

    for name in list(sys.modules):
        if name.startswith("neurolang.utils.tui"):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(builtins, "__import__", no_prompt_toolkit)
    with pytest.raises(SystemExit):
        cli.main([
            "--engine", "neurosynth", "ans(s) :- study(s)", "--save",
            str(tmp_path / "x.csv"),
        ])
    assert 'pip install "neurolang[tui]"' in capsys.readouterr().err


