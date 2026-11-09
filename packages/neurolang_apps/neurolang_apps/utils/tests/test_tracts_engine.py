"""The engine ``tracts`` of engines.yaml: the commands and the vocabulary of
``neurolang_apps.utils.engines.tracts``, over the atlases of
``mni_atlases``, those of the neurosynth engine."""

import io
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neurolang.atlases import label_name
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
    frontend = "neurolang_apps.utils.engines.tracts.frontend"
    assert cfg["solver_mixins"] == [f"{frontend}:TractsCommandsMixin"]
    assert cfg["python_init"] == [frontend]
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


@pytest.fixture
def uncinate():
    """The example of the engine and the files it reads, written by its
    tests; skipped if its example is not there."""
    from ..engines.tracts.tests import test_uncinate as tests
    from ..engines.tracts.tests import uncinate_files as files

    if not tests.EXAMPLE.exists():
        pytest.skip("the examples of the engine tracts are not installed")
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
    pytest.importorskip("neurolang_apps.utils.engines.tracts")
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
        (label_name(n),)
        for n in FAKE_LABELS["destrieux"][1:]}
    assert rows(nl.execute_datalog_program(
        'ans(n, h) :- label("tian", l, n, h), tian(n, r)\n')) == {
        ("lAMY lh", "left"), ("aPUT rh", "right"),
        ("NAc shell lh", "left"), ("THA DA rh", "right")}
    # Schaefer, without terms, with the hemispheres of its names
    nl.execute_datalog_program('.index_tracts("t", "schaefer")\n')
    assert rows(nl.execute_datalog_program(
        'ans(n, h) :- label("schaefer", l, n, h), schaefer(n, r)\n')) == {
        ("17Networks LH VisCent ExStr 1", "left"),
        ("17Networks RH VisCent ExStr 1", "right"),
        ("17Networks LH DefaultA pCunPCC 1", "left"),
        ("17Networks RH DefaultA pCunPCC 1", "right")}
    # the terms of the vocabulary, of both atlases
    assert rows(nl.execute_datalog_program(
        'ans(x) :- traverses(x, "frontal", "left"), '
        'traverses(x, "amygdala", "left"), '
        'traverses(x, "temporalpole", "right"), '
        'traverses(x, "putamen", "right")\n')) == {(0,)}


@pytest.mark.slow
def test_the_engine_with_the_real_atlases(tractogram, tmp_path):
    """Downloads Destrieux, Schaefer and Tian; every term of the uncinate
    of the engine tracts is in their term mappings."""
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
    nl.execute_datalog_program('.index_tracts("t", "schaefer")\n')
    for atlas in ("destrieux", "tian", "schaefer"):
        assert rows(nl.execute_datalog_program(
            f'ans(n) :- label("{atlas}", l, n, h), {atlas}(n, r)\n'
        )) == rows(nl.execute_datalog_program(
            f'ans(n) :- {atlas}(n, r)\n'))
    hemis = rows(nl.execute_datalog_program('ans(h) :- label(a, l, n, h)\n'))
    assert hemis == {("left",), ("right",)}
    # nilearn's Destrieux, by the command, has the engine's names
    nl.execute_datalog_program(
        '.load_atlas("d", kind="nilearn", fetcher="destrieux_2009")\n'
        '.index_tracts("t", "d")\n')
    assert rows(nl.execute_datalog_program(
        'ans(n) :- label("d", l, n, h), destrieux(n, r)\n'
    )) == rows(nl.execute_datalog_program('ans(n) :- destrieux(n, r)\n'))


def test_the_points_of_the_atlases(offline_atlases, tmp_path):
    """``schaefer_points(x, y, z, label)``: the centre of each labelled
    voxel, in mm through the affine, with the name of its region."""
    pytest.importorskip("neurolang_apps.utils.engines.tracts")
    nl = engine_registry.build_engine("tracts", tmp_path / "data")
    volume = fake_volume()
    voxels = np.argwhere(volume)
    centres = nib.affines.apply_affine(AFFINE, voxels)
    for atlas, labels in FAKE_LABELS.items():
        result = nl.execute_datalog_program(
            f'ans(x, y, z, n) :- {atlas}_points(x, y, z, n)\n')
        assert rows(result) == {
            (*map(float, c), label_name(labels[v]))
            for c, v in zip(centres, volume[tuple(voxels.T)])}
        frame = result.as_pandas_dataframe()
        assert all(frame.dtypes.iloc[:3] == np.float64)


def test_the_parcels_near_the_ends_are_precomputed(
    offline_atlases, tractogram, tmp_path, recwarn
):
    """``EUCLIDEAN(...) <= 3`` between the ends of the streamlines and the
    points of an atlas, both stored, is joined by a tree, without the
    warning of the comparison of every pair."""
    nl = engine_registry.build_engine("tracts", tmp_path / "data")
    near = rows(nl.execute_datalog_program(
        f'.load_tractogram("t", "{tractogram}", space="MNI152")\n'
        '.index_tracts("t", "schaefer")\n'
        'near(x, e, n) :- endpoint_point(x, e, ex, ey, ez), '
        'schaefer_points(mx, my, mz, n), '
        '(EUCLIDEAN(ex, ey, ez, mx, my, mz) <= 3)\n'
        'ans(x, e, n) :- near(x, e, n)\n'))
    assert not [w for w in recwarn if "cannot be precomputed" in str(
        w.message)]
    ends = rows(nl.execute_datalog_program(
        'ans(x, e, px, py, pz) :- endpoint_point(x, e, px, py, pz)\n'))
    points = rows(nl.execute_datalog_program(
        'ans(x, y, z, n) :- schaefer_points(x, y, z, n)\n'))
    assert near == {
        (x, e, n) for x, e, *p in ends for *m, n in points
        if np.linalg.norm(np.subtract(p, m)) <= 3}
    # the first streamline ends in the parcels 1 and 2, the second in
    # parcel 1 and outside the atlas
    left, right = (label_name(n)
                   for n in FAKE_LABELS["schaefer"][1:3])
    assert {(x, n) for x, _, n in near} == {
        (0, left), (0, right), (1, left)}


def test_the_assistant_is_told_the_bundles_and_their_sorts(
    uncinate, tmp_path, offline_atlases
):
    """The %% documentation of the vocabulary and of a bundle module, and
    the sorts the engine declares, reach the catalog of the assistant."""
    from neurolang_apps.utils.llm.context import EngineContext
    from neurolang_apps.utils.engines.tracts.wmql_vocab import bundle

    tests, files = uncinate
    files.write_files(tmp_path)
    nl = engine_registry.build_engine("tracts", tmp_path / "data")
    nl.execute_datalog_program(tests.example(tmp_path).replace(
        'ans(x) :- uncinate(x, "left")', ""))
    nl.execute_datalog_program(bundle("freesurfer_queries"))
    context = EngineContext(nl, "tracts")
    catalog = context.catalog()
    assert catalog["uf"].columns == ("x", "h")
    assert [a.sort for a in catalog["cst"].arguments] == [
        "streamline", "hemisphere"]
    lines = context.schema_summary().splitlines()
    assert any(line.startswith(
        "- uf(x: streamline, h: hemisphere) — Uncinate fasciculus: ")
        for line in lines)
    assert any(line.startswith(
        "- traverses(x: streamline, term, h: hemisphere) — ")
        for line in lines)
    assert any(line.startswith('- hemisphere: a hemisphere: "left"')
               for line in lines)
    assert context.describe("uf").splitlines()[1].endswith("rows, e.g.:")
