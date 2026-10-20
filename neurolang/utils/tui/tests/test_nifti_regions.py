"""Tests for saving and viewing results on the regions of atlases."""

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

import nibabel as nib  # noqa: E402
from rich.console import Console  # noqa: E402

from neurolang.frontend import NeurolangPDL  # noqa: E402
from neurolang.utils import engine_registry  # noqa: E402
from neurolang.utils.tui import InteractiveTuiApp, TuiSettings  # noqa: E402
from neurolang.utils.tui import brain_viewer, nifti  # noqa: E402


def _output(app):
    return app.console.file.getvalue()


# The template of the engine, 1 mm voxels, and two atlases on grids of
# their own: "cortex", 2 mm voxels, and "deep", 1 mm voxels shifted.
TEMPLATE_AFFINE = np.diag([1.0, 1.0, 1.0, 1.0])
TEMPLATE_AFFINE[:3, 3] = -4.0
CORTEX_AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])
CORTEX_AFFINE[:3, 3] = -4.0
DEEP_AFFINE = np.diag([1.0, 1.0, 1.0, 1.0])
DEEP_AFFINE[:3, 3] = -2.0

ENGINE = """\
engines:
  tiny:
    atlases:
      cortex: {predicate_name: cortex}
      deep: {predicate_name: deep}
    sorts:
      study: {columns: [study_id], type: int}
      mni: {columns: [x, y, z], type: float, space: mm}
      cortex_label: {columns: [label], type: str, atlas: cortex}
      deep_label: {columns: [label], type: str, atlas: deep}
      probability: {columns: [p], type: float}
    predicates:
      cortex_activated: {arguments: [cortex_label, study]}
      deep_activated: {arguments: [deep_label, study]}
      cortex_p: {arguments: [cortex_label, probability]}
      cortex_pair: {arguments: [cortex_label, cortex_label, probability]}
      peak: {arguments: [mni, study]}
"""


def _cortex_labels():
    """The labels of the cortex atlas: 4×4×4 voxels of 2 mm, A at the
    voxels x < 2 (1 and 5, two values of one region), B at x ≥ 2 (2)."""
    data = np.zeros((4, 4, 4), dtype=np.int16)
    data[0, :, :] = 1
    data[1, :, :] = 5
    data[2:, :, :] = 2
    return data


@pytest.fixture
def tiny(monkeypatch, tmp_path):
    """Return a REPL on the tiny engine of two atlases."""
    path = tmp_path / "engines.yaml"
    path.write_text(ENGINE)
    monkeypatch.setattr(engine_registry, "_ENGINES_YAML", path)
    nl = NeurolangPDL()
    nl.template = nib.Nifti1Image(
        np.ones((8, 8, 8), dtype=np.uint8), TEMPLATE_AFFINE
    )
    deep = np.zeros((4, 4, 4), dtype=np.int16)
    deep[1:3, 1:3, 1:3] = 7
    nl.grids = {
        "template": nl.template,
        "cortex": nib.Nifti1Image(_cortex_labels(), CORTEX_AFFINE),
        "deep": nib.Nifti1Image(deep, DEEP_AFFINE),
    }
    engine_registry._register_atlas_labels(
        nl, "cortex", {1: "A", 5: "A", 2: "B"}
    )
    engine_registry._register_atlas_labels(nl, "deep", {7: "C"})
    nl.add_tuple_set(
        [("A", 1), ("B", 1), ("A", 2), ("B", 3)], name="cortex_activated"
    )
    nl.add_tuple_set([("C", 1), ("C", 2)], name="deep_activated")
    nl.add_tuple_set(
        [("A", 0.25), ("B", 0.75), ("Z", 0.5)], name="cortex_p"
    )
    nl.add_tuple_set(
        [("A", "B", 0.5), ("B", "A", 0.75), ("A", "A", 1.0)],
        name="cortex_pair",
    )
    console = Console(file=io.StringIO(), width=200)
    return InteractiveTuiApp(
        nl, TuiSettings(engine_name="tiny"), console=console,
        err_console=console,
    )


@pytest.fixture
def view_calls(monkeypatch):
    """Record the images and options of the viewer page, left empty."""
    calls = []

    def viewer_html(img, background, **kwargs):
        calls.append((img, kwargs))
        return "<html></html>"

    monkeypatch.setattr(nifti.brain_viewer, "viewer_html", viewer_html)
    monkeypatch.setattr(nifti.brain_viewer, "mni_background", lambda: None)
    return calls


def _remove_view_files(out):
    """Remove the images and the page .view wrote, as *out* tells."""
    for line in out.split("image to: ")[1:]:
        Path(line.split()[0]).unlink()
    Path(out.split("viewer to: ")[1].split()[0]).unlink()


def _data(img):
    return np.asanyarray(img.dataobj)


class TestAtlasSorts:
    """Check a sort of region labels names its atlas."""

    def test_the_atlas_of_a_sort(self, tiny):
        sorts = engine_registry.get_sorts("tiny")
        assert sorts["cortex_label"].atlas == "cortex"
        assert sorts["mni"].atlas == ""

    @pytest.mark.parametrize("sort, message", [
        ("{columns: [label], type: str, atlas: nowhere}", "the atlases"),
        ("{columns: [x, y], type: str, atlas: cortex}", "one column"),
        (
            "{columns: [label], type: str, space: mm, atlas: cortex}",
            "no space",
        ),
    ])
    def test_a_wrong_atlas_is_rejected(self, sort, message):
        cfg = {"atlases": {"cortex": {}}, "sorts": {}}
        info = yaml.safe_load(sort)
        with pytest.raises(ValueError, match=message):
            engine_registry._declared_sort(
                "bad", info, engine_registry.engine_grid_names(cfg)
            )

    def test_the_labels_of_an_atlas(self, tiny):
        values = engine_registry.atlas_label_values(tiny._nl, "cortex")
        assert values == {"A": [1, 5], "B": [2]}
        assert engine_registry.atlas_label_values(tiny._nl, "none") == {}

    def test_centroids(self, tiny):
        centroids = nifti.region_centroids(tiny._nl, "cortex")
        # A: x voxels 0 and 1, centres at -4 and -2 mm; y, z voxels 0..3
        np.testing.assert_allclose(centroids["A"], [-3.0, -1.0, -1.0])
        np.testing.assert_allclose(centroids["B"], [1.0, -1.0, -1.0])


class TestSaveRegions:
    """Check .save paints the regions of an atlas with their value."""

    def test_a_value_per_region(self, tiny, tmp_path):
        tiny._execute_and_display("ans(l, p) :- cortex_p(l, p)")
        tiny._handle_dot_command(f".save {tmp_path / 'r.nii.gz'}")
        img = nifti.read_nifti(tmp_path / "r.nii.gz")
        np.testing.assert_allclose(img.affine, CORTEX_AFFINE)
        data = _data(img)
        assert np.all(data[:2] == 0.25) and np.all(data[2:] == 0.75)
        out = _output(tiny)
        assert "2 regions, 64 voxels" in out and "cortex grid" in out
        assert "1 label of no region of the atlas skipped" in out

    def test_a_mask_without_a_value(self, tiny, tmp_path):
        tiny._execute_and_display(
            "ans(l) :- cortex_activated(l, s), (s == 3)"
        )
        tiny._handle_dot_command(f".save {tmp_path / 'r.nii'}")
        data = _data(nifti.read_nifti(tmp_path / "r.nii"))
        assert data.dtype == np.uint8
        assert np.all(data[2:] == 1) and np.all(data[:2] == 0)

    def test_regions_of_two_atlases_are_on_the_template(
        self, tiny, tmp_path
    ):
        tiny._execute_and_display("""
            region(l, s) :- cortex_activated(l, s)
            region(l, s) :- deep_activated(l, s)
            ans(l, count(s)) :- region(l, s)
        """)
        tiny._handle_dot_command(f".save {tmp_path / 'r.nii'}")
        img = nifti.read_nifti(tmp_path / "r.nii")
        np.testing.assert_allclose(img.affine, TEMPLATE_AFFINE)
        data = _data(img)
        # C, voxels 1..2 of the deep grid: -1 to 0 mm, voxels 3..4 of the
        # template; its value, 2 studies, over that of A and B
        assert data[3, 3, 3] == 2 and data[4, 4, 4] == 2
        # A: x < 0 mm (cortex voxels 0 and 1), 2 studies; B, x ≥ 0 mm up
        # to the centre of its last voxel, 2 mm: 2 studies
        assert data[0, 0, 0] == 2 and data[6, 0, 0] == 2
        assert "3 regions" in _output(tiny)

    def test_a_network_is_not_saved_as_an_image(self, tiny, tmp_path):
        tiny._execute_and_display("ans(a, b, p) :- cortex_pair(a, b, p)")
        tiny._handle_dot_command(f".save {tmp_path / 'n.nii'}")
        assert not (tmp_path / "n.nii").exists()
        assert "network" in _output(tiny) and ".view" in _output(tiny)
        tiny._handle_dot_command(f".save {tmp_path / 'n.csv'}")
        assert len(pd.read_csv(tmp_path / "n.csv")) == 3


class TestViewRegions:
    """Check .view shows the regions of a result, all of them."""

    def test_every_region_is_shown(self, tiny, view_calls):
        tiny._execute_and_display("ans(l, p) :- cortex_p(l, p)")
        tiny._handle_dot_command(".view")
        img, kwargs = view_calls[-1]
        assert kwargs["threshold"] == 0.0 and "connectome" not in kwargs
        assert _data(img).max() == 0.75
        _remove_view_files(_output(tiny))

    def test_a_threshold_given(self, tiny, view_calls):
        tiny._execute_and_display("ans(l, p) :- cortex_p(l, p)")
        tiny._handle_dot_command(".view --threshold 50%")
        assert view_calls[-1][1]["threshold"] == "50%"
        _remove_view_files(_output(tiny))

    def test_voxels_keep_the_top_5_percent(self, tiny, view_calls):
        tiny._nl.add_tuple_set([(0.0, 0.0, 0.0, 1)], name="peak")
        tiny._execute_and_display("ans(x, y, z) :- peak(x, y, z, s)")
        tiny._handle_dot_command(".view")
        assert view_calls[-1][1]["threshold"] == "95%"
        _remove_view_files(_output(tiny))


class TestViewNetworks:
    """Check .view shows pairs of regions or points as a network."""

    def test_pairs_of_regions(self, tiny, view_calls):
        tiny._execute_and_display("ans(a, b, p) :- cortex_pair(a, b, p)")
        tiny._handle_dot_command(".view")
        img, kwargs = view_calls[-1]
        network = kwargs["connectome"]
        names = [node["name"] for node in network["nodes"]]
        assert sorted(names) == ["A", "B"]
        centre = {
            n["name"]: [n["x"], n["y"], n["z"]] for n in network["nodes"]
        }
        np.testing.assert_allclose(centre["A"], [-3.0, -1.0, -1.0])
        # A–B given both ways keeps 0.75; A–A is left out
        assert [e["colorValue"] for e in network["edges"]] == [0.75]
        # the overlay is the strength of each node, on the atlas's grid
        np.testing.assert_allclose(_data(img)[0, 0, 0], 0.75)
        out = _output(tiny)
        assert "2 nodes and 1 edges" in out
        assert "1 pairs given both ways" in out
        _remove_view_files(out)

    def test_pairs_of_a_derived_relation(self, tiny, view_calls):
        tiny._execute_and_display("""
            co(a, b, s) :- cortex_activated(a, s), cortex_activated(b, s)
            ans(a, b, count(s)) :- co(a, b, s), (a < b)
        """)
        tiny._handle_dot_command(".view")
        network = view_calls[-1][1]["connectome"]
        assert [e["colorValue"] for e in network["edges"]] == [1.0]
        _remove_view_files(_output(tiny))

    def test_pairs_of_points_named_with_a_suffix(self, tiny, view_calls):
        tiny._nl.add_tuple_set(
            [(-2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.5)], name="pp"
        )
        tiny._execute_and_display(
            "ans(x1, y1, z1, x2, y2, z2, p) :- pp(x1, y1, z1, x2, y2, z2, p)"
        )
        tiny._handle_dot_command(".view")
        network = view_calls[-1][1]["connectome"]
        assert [n["name"] for n in network["nodes"]] == [
            "(-2, 0, 0)", "(2, 0, 0)",
        ]
        assert [n["x"] for n in network["nodes"]] == [-2.0, 2.0]
        _remove_view_files(_output(tiny))

    def test_unknown_regions(self, tiny):
        tiny._nl.add_tuple_set([("X", "Y", 0.5)], name="cortex_pair")
        tiny._execute_and_display(
            "ans(a, b, p) :- cortex_pair(a, b, p), (a == 'X')"
        )
        tiny._handle_dot_command(".view")
        assert "no pair of two known regions" in _output(tiny)


class TestConnectomeState:
    """Check the network NiiVue draws."""

    EDGES = [(0, 1, 0.1), (1, 2, 0.2), (0, 2, 0.4)]
    NAMES = ["a", "b", "c"]
    COORDINATES = np.array([[0.0, 0, 0], [10, 0, 0], [0, 10, 0]])

    def test_a_threshold_value(self):
        state = brain_viewer.connectome_state(
            self.NAMES, self.COORDINATES, self.EDGES, 0.15
        )
        assert [e["colorValue"] for e in state["edges"]] == [0.2, 0.4]
        assert state["edgeMin"] == 0.15 and state["edgeMax"] == 0.4
        # the edges index the nodes kept: b and c, and a
        strength = {n["name"]: n["colorValue"] for n in state["nodes"]}
        assert strength == pytest.approx({"a": 0.4, "b": 0.2, "c": 0.6})
        assert max(n["sizeValue"] for n in state["nodes"]) == 1.0

    def test_a_percentile_shows_a_few_edges_at_least(self):
        state = brain_viewer.connectome_state(
            self.NAMES, self.COORDINATES, self.EDGES, "95%"
        )
        assert len(state["edges"]) == 3

    def test_a_percentile_of_many_edges(self):
        edges = [(0, 1, float(v)) for v in range(1, 101)]
        assert brain_viewer.edge_threshold(
            [w for _, _, w in edges], "95%", 20
        ) == 81.0
        assert brain_viewer.edge_threshold(
            [w for _, _, w in edges], "50%"
        ) == pytest.approx(50.5)

    def test_the_page_draws_the_network(self):
        img = nib.Nifti1Image(np.ones((2, 2, 2), np.float32), np.eye(4))
        state = brain_viewer.connectome_state(
            self.NAMES, self.COORDINATES, self.EDGES, 0
        )
        page = brain_viewer.viewer_html(
            img, None, title="t", threshold=0, cmap=None, symmetric=False,
            connectome=state,
        )
        data = re.search(
            r'<script id="viewer-data" type="application/json">(.*?)'
            r"</script>", page, re.S,
        ).group(1)
        parsed = json.loads(data)
        assert parsed["connectome"]["edges"] == state["edges"]
        assert parsed["initial"]["display"]["renderTile"] is True
        assert parsed["initial"]["layers"][0]["opacity"] == 0.5
        assert 'id="edges"' in page
