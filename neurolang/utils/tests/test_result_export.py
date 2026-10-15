"""Tests of --save and --view: query results as tables and brain images."""

import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from neurolang.frontend import NeurolangPDL
from neurolang.regions import ExplicitVBR, ExplicitVBROverlay
from neurolang.utils import cli, engine_registry, result_export
from neurolang.utils.various import FrozenNDArray
from neurolang.utils.result_export import (
    ExportError,
    result_image,
    save_image,
    save_table,
)

AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])
SHAPE = (4, 5, 6)
MASK = nib.Nifti1Image(np.ones(SHAPE, dtype=np.int8), AFFINE)


def region(*voxels):
    return ExplicitVBR(np.array(voxels), AFFINE, image_dim=SHAPE)


def data(result):
    return np.asanyarray(result.image.dataobj)


def test_tables(tmp_path):
    df = pd.DataFrame({"name": ["a", "b"], "region": [
        region((0, 0, 0)), region((1, 1, 1), (1, 2, 1)),
    ]})
    save_table(df, tmp_path / "r.csv")
    assert (tmp_path / "r.csv").read_text().splitlines() == [
        "name,region", "a,<region of 1 voxels>", "b,<region of 2 voxels>",
    ]
    save_table(df[["name"]], tmp_path / "r.tsv.gz")
    assert list(pd.read_csv(tmp_path / "r.tsv.gz", sep="\t")["name"]) == [
        "a", "b",
    ]
    save_table(df[["name"]], tmp_path / "r.json")
    assert json.loads((tmp_path / "r.json").read_text()) == [
        {"name": "a"}, {"name": "b"},
    ]


def test_check_path():
    result_export.check_path("x.nii.gz")
    result_export.check_path("x.csv.gz")
    with pytest.raises(ExportError, match="cannot tell the format"):
        result_export.check_path("x.png")


def test_voxels_with_a_value(tmp_path):
    df = pd.DataFrame({"i": [0, 3], "j": [1, 4], "k": [2, 5],
                       "p": [0.25, 0.5]})
    image = result_image(df, MASK)
    assert image.labels is None
    assert np.allclose(image.image.affine, AFFINE)
    assert data(image)[0, 1, 2] == 0.25 and data(image)[3, 4, 5] == 0.5
    assert data(image).sum() == pytest.approx(0.75)
    [path] = save_image(image, tmp_path / "p.nii.gz")
    assert nib.load(path).get_fdata()[3, 4, 5] == 0.5


def test_voxels_without_a_value():
    df = pd.DataFrame({"i": [0], "j": [1], "k": [2]})
    image = result_image(df, MASK)
    assert data(image).sum() == 1 and data(image)[0, 1, 2] == 1


def test_voxels_outside_the_grid_are_left_out():
    df = pd.DataFrame({"i": [0, 9, -1], "j": [1, 1, 1], "k": [2, 2, 2],
                       "p": [0.5, 0.25, 0.25]})
    image = result_image(df, MASK)
    assert image.outside == 2 and data(image).sum() == pytest.approx(0.5)


@pytest.mark.parametrize("df, mask, message", [
    (pd.DataFrame({"i": [0], "j": [1], "k": [2], "s": ["x"]}), MASK,
     "besides the voxel"),
    (pd.DataFrame({"i": [0], "j": [1], "k": [2], "s": [1], "p": [.5]}),
     MASK, "besides the voxel"),
    (pd.DataFrame({"i": [9], "j": [1], "k": [2]}), MASK, "all the voxels"),
    (pd.DataFrame({"i": [0], "j": [1], "k": [2]}), None, "no voxel grid"),
    (pd.DataFrame({"s": [1]}), MASK, "no region column"),
    (pd.DataFrame({"i": []}), MASK, "empty"),
])
def test_results_that_are_not_images(df, mask, message):
    with pytest.raises(ExportError, match=message):
        result_image(df, mask)


def test_one_region_and_an_overlay():
    image = result_image(pd.DataFrame({"r": [region((0, 0, 0), (1, 1, 1))]}))
    assert data(image).sum() == 2 and image.labels is None
    overlay = ExplicitVBROverlay(
        np.array([(0, 0, 0), (1, 1, 1)]), AFFINE, [0.5, 0.25],
        image_dim=SHAPE,
    )
    image = result_image(pd.DataFrame({"r": [overlay]}))
    assert data(image)[1, 1, 1] == 0.25
    with pytest.raises(ExportError, match="2 overlays"):
        result_image(pd.DataFrame({"r": [overlay, overlay]}))


def test_regions_give_a_label_image(tmp_path):
    df = pd.DataFrame({
        "label": ["lh_a", "rh_b"],
        "r": [region((0, 0, 0)), region((1, 1, 1), (2, 2, 2))],
    })
    image = result_image(df)
    assert data(image)[0, 0, 0] == 1 and data(image)[2, 2, 2] == 2
    written = save_image(image, tmp_path / "atlas.nii.gz")
    assert written[1] == tmp_path / "atlas_labels.csv"
    assert written[1].read_text().splitlines() == [
        "value,label", "1,lh_a", "2,rh_b",
    ]
    with pytest.raises(ExportError, match="several region columns"):
        result_image(df.assign(other=df["r"]))


@pytest.fixture
def engine(monkeypatch):
    """Run neurolang-query on a small engine with voxels and regions."""
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [(0, 1, 2, "s1"), (3, 4, 5, "s1"), (0, 1, 2, "s2")],
        name="reports",
    )
    nl.add_tuple_set(
        [("lh_a", region((0, 0, 0))), ("rh_b", region((1, 1, 1)))],
        name="atlas",
    )
    monkeypatch.setattr(
        engine_registry, "build_engine", lambda *args, **kwargs: nl
    )
    monkeypatch.setattr(
        engine_registry, "engine_mask", lambda *args, **kwargs: MASK
    )
    viewed = []
    monkeypatch.setattr(
        result_export, "view_image",
        lambda image, title="": viewed.append((image, title)),
    )
    return viewed


def test_save_csv_from_the_command_line(engine, tmp_path, capsys):
    path = tmp_path / "studies.csv"
    cli.main(["ans(s) :- reports(i, j, k, s)", "--save", str(path),
              "--sort", "s:desc"])
    assert path.read_text().splitlines() == ["s", "s2", "s1"]
    assert f"Saved {path}" in capsys.readouterr().err


def test_save_and_view_nifti_from_the_command_line(engine, tmp_path):
    path = tmp_path / "voxels.nii.gz"
    cli.main(["ans(i, j, k) :- reports(i, j, k, s)", "--save", str(path),
              "--view"])
    image = nib.load(path).get_fdata()
    assert image.sum() == 2 and image[3, 4, 5] == 1
    assert len(engine) == 1 and engine[0][0].image.shape == SHAPE


def test_view_regions_from_the_command_line(engine):
    cli.main(["ans(l, r) :- atlas(l, r)", "--view"])
    [(image, title)] = engine
    assert sorted(image.labels["l"]) == ["lh_a", "rh_b"]
    assert sorted(np.unique(data(image))) == [0, 1, 2]


def test_save_errors_exit(engine, tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli.main(["ans(s) :- reports(i, j, k, s)", "--save",
                  str(tmp_path / "x.nii")])
    assert "no region column" in capsys.readouterr().err


def test_regions_stored_as_their_voxels():
    stored = (np.array([0, 0, 0]), np.array([1, 1, 1]))
    df = pd.DataFrame({"r": [stored]})
    image = result_image(df, grid_of=lambda value: (AFFINE, SHAPE))
    assert data(image).sum() == 2 and image.image.shape == SHAPE
    with pytest.raises(ExportError, match="grid is unknown"):
        result_image(df)


def test_atlas_regions_from_the_command_line(engine, tmp_path):
    nl = engine_registry.build_engine()
    voxels = (FrozenNDArray([0, 0, 0]), FrozenNDArray([1, 1, 1]))
    nl.add_tuple_set([("lh_c", voxels)], name="schaefer")
    nl._region_grids = {"schaefer": (AFFINE, SHAPE)}
    assert engine_registry.region_grids(nl) == {
        "schaefer": (AFFINE, SHAPE)
    }
    path = tmp_path / "parcel.nii.gz"
    cli.main(["ans(r) :- schaefer(l, r)", "--save", str(path)])
    assert nib.load(path).get_fdata().sum() == 2
