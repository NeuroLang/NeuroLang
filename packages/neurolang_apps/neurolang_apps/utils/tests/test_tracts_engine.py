"""The engine ``tracts`` of engines.yaml: the commands and the vocabulary of
nl_tracts, which neurolang_apps names in its YAML only (the package is the
extra ``tracts``), over the atlases of ``mni_atlases``, those of the
neurosynth engine."""

import importlib
import io
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neurolang_apps.utils import engine_registry

# The raw names of the label files of the atlases, as nilearn and Tian's
# repository give them: value n is the name n of the list.
FAKE_LABELS = {
    "destrieux": ["Background", "L G_and_S_frontomargin", "R Pole_temporal",
                  "L Medial_wall", "R S_front_sup"],
    "schaefer": ["Background", "17Networks_LH_VisCent_ExStr_1",
                 "17Networks_RH_VisCent_ExStr_1",
                 "17Networks_LH_DefaultA_pCunPCC_1",
                 "17Networks_RH_DefaultA_pCunPCC_1"],
    "tian": ["Background", "lAMY-lh", "aPUT-rh", "NAc-shell-lh",
             "THA-DA-rh"],
}
# 2 mm voxels, around the origin of MNI152
AFFINE = np.array([[2.0, 0, 0, -12], [0, 2, 0, -10], [0, 0, 2, -8],
                   [0, 0, 0, 1]])


def fake_volume():
    """Four regions, the values 1 to 4, in a 12x10x8 volume."""
    volume = np.zeros((12, 10, 8), dtype=np.int16)
    volume[2:5, 1:4, 0:3] = 1
    volume[6:9, 5:8, 4:8] = 2
    volume[10, 9, 7] = 3
    volume[0:2, 8:10, 0:2] = 4
    return volume


@pytest.fixture
def offline_atlases(monkeypatch, tmp_path):
    """The fetchers of the atlases of the registry give small label images
    with the raw names of the real ones: the engines load them as they
    load the real ones, from the network."""
    path = tmp_path / "fake_atlas.nii.gz"
    nib.save(nib.Nifti1Image(fake_volume(), AFFINE), path)
    for name, labels in FAKE_LABELS.items():
        monkeypatch.setitem(
            engine_registry._ATLAS_REGISTRY[name], "fetch",
            lambda labels=labels, **kwargs: {
                "maps": str(path), "labels": list(labels)},
        )


def test_the_engine_is_declared():
    assert "tracts" in engine_registry.list_engine_names()
    cfg = engine_registry.get_engine_config("tracts")
    assert cfg["solver_mixins"] == ["nl_tracts.frontend:TractsCommandsMixin"]
    assert cfg["python_init"] == "nl_tracts.frontend"
    # the atlases of neurosynth, without the studies that activate them
    # (which need its peaks), and no MNI template
    assert not cfg.get("requires_mni_mask")
    assert set(cfg["atlases"]) == {"destrieux", "schaefer", "tian"}
    assert not any("activated" in a for a in cfg["atlases"].values())
    neurosynth = engine_registry.get_engine_config("neurosynth")
    for name, atlas in cfg["atlases"].items():
        assert {**atlas, "activated": neurosynth["atlases"][name][
            "activated"]} == neurosynth["atlases"][name]
    predicates = engine_registry.get_predicates("tracts")
    assert {"destrieux", "tian", "schaefer", "labels"} <= set(predicates)
    assert not {"destrieux_activated", "tian_activated",
                "schaefer_activated", "peak_reported"} & set(predicates)
    assert predicates["destrieux"] == engine_registry.get_predicates(
        "neurosynth")["destrieux"]


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


def test_the_engine_runs_the_example(uncinate, tmp_path, offline_atlases):
    tests, files = uncinate
    files.write_files(tmp_path)
    nl = engine_registry.build_engine("tracts", tmp_path / "data")
    assert nl.template is None
    result = nl.execute_datalog_program(tests.example(tmp_path))
    assert set(result.as_pandas_dataframe().iloc[:, 0]) == files.expected(
        "left")
    assert callable(nl.streamline_points)
    assert nl.streamline_background() is not None


def test_the_example_at_the_prompt_and_view(
    uncinate, tmp_path, monkeypatch, offline_atlases
):
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


def test_the_example_as_the_program_of_the_command_line(
    uncinate, tmp_path, offline_atlases
):
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


def rows(result):
    return {tuple(r) for r in result.as_pandas_dataframe().itertuples(
        index=False, name=None)}


@pytest.fixture
def tractogram(tmp_path):
    """Streamlines in MNI152, through the regions 1 and 2 of the fake
    atlases, and their ids: x is s, one tractogram."""
    pytest.importorskip("nl_tracts")
    centre = nib.affines.apply_affine(AFFINE, [[3, 2, 1], [7, 6, 5]])
    lines = [np.linspace(centre[0], centre[1], 20).astype(np.float32),
             np.linspace(centre[0], centre[0] + [0, -20, 0], 20).astype(
                 np.float32)]
    path = tmp_path / "t.tck"
    nib.streamlines.save(
        nib.streamlines.Tractogram(lines, affine_to_rasmm=np.eye(4)),
        str(path))
    return path


def index_program(path):
    return (f'.load_tractogram("t", "{path}", space="MNI152")\n'
            '.index_tracts("t", "destrieux")\n'
            '.index_tracts("t", "tian")\n')


def test_the_atlases_of_the_engine_index_tractograms(
    offline_atlases, tractogram, tmp_path
):
    """``.index_tracts(t, "destrieux")``: an atlas of the engine, whose label
    names are those of its relation ``destrieux(name, region)``."""
    nl = engine_registry.build_engine("tracts", tmp_path / "data")
    nl.execute_datalog_program(index_program(tractogram))
    names = rows(nl.execute_datalog_program(
        'ans(n) :- label("destrieux", l, n, h), destrieux(n, r)\n'))
    regions = rows(nl.execute_datalog_program('ans(n) :- destrieux(n, r)\n'))
    assert names == regions == {
        (engine_registry._label_name(n),)
        for n in FAKE_LABELS["destrieux"][1:]}
    assert rows(nl.execute_datalog_program(
        'ans(n, h) :- label("tian", l, n, h), tian(n, r)\n')) == {
        ("lAMY lh", "left"), ("aPUT rh", "right"),
        ("NAc shell lh", "left"), ("THA DA rh", "right")}
    # the terms of the vocabulary, of both atlases
    assert rows(nl.execute_datalog_program(
        'ans(x) :- traverses(x, "frontal", "left"), '
        'traverses(x, "amygdala", "left"), '
        'traverses(x, "temporalpole", "right"), '
        'traverses(x, "putamen", "right")\n')) == {(0,)}


def test_the_names_are_normalised_as_the_registry_does():
    """nl_tracts duplicates ``_label_name``, which it cannot import."""
    pytest.importorskip("nl_tracts")
    from nl_tracts.frontend import ATLAS_MAPPINGS, engine_label_name
    from nl_tracts.terms import load_term_mapping

    names = {n for labels in FAKE_LABELS.values() for n in labels}
    for key in ATLAS_MAPPINGS.values():
        names |= set(load_term_mapping(key)["atlas_label_name"])
    for name in names:
        assert engine_label_name(name) == engine_registry._label_name(name)


@pytest.mark.slow
def test_the_engine_with_the_real_atlases(tractogram, tmp_path):
    """Downloads Destrieux, Schaefer and Tian; every term of the uncinate
    of nl_tracts is in their term mappings."""
    try:
        nl = engine_registry.build_engine("tracts", tmp_path / "data")
    except Exception as e:  # offline, mirror down
        pytest.skip(f"cannot fetch the atlases: {e!r}")
    nl.execute_datalog_program(index_program(tractogram))
    terms = rows(nl.execute_datalog_program(
        'ans(t) :- label_term(a, l, t)\n'))
    assert {(t,) for t in (
        "orbitofrontalgyrus", "middle_frontal_gyrus",
        "inferior_frontal_gyrus", "temporal", "insula", "occipital",
        "parietal", "cingular", "amygdala", "putamen")} <= terms
    for atlas in ("destrieux", "tian"):
        assert rows(nl.execute_datalog_program(
            f'ans(n) :- label("{atlas}", l, n, h), {atlas}(n, r)\n'
        )) == rows(nl.execute_datalog_program(
            f'ans(n) :- {atlas}(n, r)\n'))
    hemis = rows(nl.execute_datalog_program('ans(h) :- label(a, l, n, h)\n'))
    assert hemis == {("left",), ("right",)}
    # nilearn's Destrieux, by the command, has the engine's names
    nl.execute_datalog_program(
        '.load_tract_atlas("d", kind="nilearn", fetcher="destrieux_2009")\n'
        '.index_tracts("t", "d")\n')
    assert rows(nl.execute_datalog_program(
        'ans(n) :- label("d", l, n, h), destrieux(n, r)\n'
    )) == rows(nl.execute_datalog_program('ans(n) :- destrieux(n, r)\n'))
