"""The atlases of NeuroLang: loaders, the label table, hemispheres, the
regions and points of the engine, the space and the hash."""

import subprocess
import sys
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from neurolang import atlases
from neurolang.atlases import (
    Atlas,
    HemisphereRule,
    SpaceMismatchError,
    label_name,
    load_label_table_bids_dseg,
    load_label_table_csv,
    load_label_table_freesurfer_lut,
    load_label_table_tsv,
)

AFFINES = {
    "aligned": np.array([[2.0, 0, 0, -10], [0, 3, 0, -20], [0, 0, 4, -30],
                         [0, 0, 0, 1]]),
    "flipped": np.array([[-2.0, 0, 0, 10], [0, -3, 0, 20], [0, 0, 4, -30],
                         [0, 0, 0, 1]]),
}

LUT = """\
#$Id: FreeSurferColorLUT.txt,v 1.70.2.7 2012/08/27 17:20:08 nicks Exp $

#No. Label Name:                            R   G   B   A
0   Unknown                                 0   0   0   0
1   Left-Putamen                            236 13  176 0

2   ctx-rh-precentral                       60  20  220 0
3   Brain-Stem                              119 159 176 0
4   ctx-lh-precuneus                        160 140 180 0
"""
DSEG = (
    "index\tname\tabbreviation\n"
    "0\tUnknown\tUNK\n1\tLeft-Putamen\tPut\n2\tctx-rh-precentral\tPC\n"
    "3\tBrain-Stem\tBS\n4\tctx-lh-precuneus\tPCun\n"
)
TSV = (
    "id\tname\themi\n"
    "0\tUnknown\t\n1\tLeft-Putamen\tleft\n2\tctx-rh-precentral\t\n"
    "3\tBrain-Stem\t\n4\tctx-lh-precuneus\t\n"
)
NAMES = ["Unknown", "Left-Putamen", "ctx-rh-precentral", "Brain-Stem",
         "ctx-lh-precuneus"]
TABLE = pd.DataFrame({"id": range(5), "name": NAMES})
HEMI = {
    "Unknown": "none", "Left-Putamen": "left", "ctx-rh-precentral": "right",
    "Brain-Stem": "none", "ctx-lh-precuneus": "left",
}


@pytest.fixture
def volume():
    """A 12x10x8 volume with four labels of known voxels: 1 and 2 boxes, 3
    a voxel, 4 a diagonal line."""
    vol = np.zeros((12, 10, 8), dtype=np.int32)
    vol[2:5, 1:4, 0:3] = 1
    vol[6:9, 5:8, 4:8] = 2
    vol[10, 9, 7] = 3
    for k in range(6):
        vol[k, k, k] = 4
    return vol


@pytest.fixture
def image(volume):
    return nib.Nifti1Image(volume, AFFINES["flipped"])


@pytest.fixture
def image_path(image, tmp_path):
    path = tmp_path / "labels.nii.gz"
    nib.save(image, path)
    return path


def make(volume, table=TABLE, **kw):
    kw.setdefault("name", "fs")
    kw.setdefault("space", "subject:1")
    kw.setdefault("affine", AFFINES["aligned"])
    return Atlas(labels=volume, table=table, **kw)


def check(atlas, volume, names=NAMES):
    assert np.array_equal(atlas.labels, volume)
    np.testing.assert_allclose(atlas.affine, AFFINES["flipped"])
    t = atlas.table
    assert t.columns.tolist() == ["id", "name", "hemi"]
    assert dict(zip(t["id"], t["name"])) == {
        i: n for i, n in enumerate(NAMES) if n in names
    }
    assert dict(zip(t["name"], t["hemi"])) == {
        n: h for n, h in HEMI.items() if n in names
    }


# The label tables.

def test_freesurfer_lut_table(tmp_path):
    path = tmp_path / "lut.txt"
    path.write_text(LUT)
    table = load_label_table_freesurfer_lut(path)
    assert table.columns.tolist() == ["id", "name"]
    assert table["id"].tolist() == [0, 1, 2, 3, 4]
    assert table["name"].tolist() == NAMES


def test_bids_dseg_table(tmp_path):
    path = tmp_path / "x_dseg.tsv"
    path.write_text(DSEG)
    table = load_label_table_bids_dseg(path)
    assert table.columns.tolist() == ["id", "name"]
    assert table["name"].tolist() == NAMES


def test_bids_dseg_needs_index_and_name(tmp_path):
    path = tmp_path / "x_dseg.tsv"
    path.write_text("id\tlabel\n0\tx\n")
    with pytest.raises(ValueError, match="index.*name"):
        load_label_table_bids_dseg(path)


def test_plain_tsv_table_keeps_the_hemi_column(tmp_path):
    path = tmp_path / "t.tsv"
    path.write_text(TSV)
    table = load_label_table_tsv(path)
    assert table.columns.tolist() == ["id", "name", "hemi"]
    assert table["hemi"].tolist() == ["", "left", "", "", ""]


def test_plain_tsv_without_hemi(tmp_path):
    path = tmp_path / "t.tsv"
    path.write_text("id\tname\n1\ta\n")
    assert load_label_table_tsv(path).columns.tolist() == ["id", "name"]


def test_csv_table_is_its_first_two_columns(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("value;label;other\n1;L S_central;x\n2;R S_central;y\n")
    table = load_label_table_csv(path, sep=";")
    assert table.values.tolist() == [[1, "L S_central"], [2, "R S_central"]]
    path.write_text("value\n1\n")
    with pytest.raises(ValueError, match="two columns"):
        load_label_table_csv(path)


# The loaders.

def test_from_freesurfer_with_a_path_and_with_an_image(
    volume, image, image_path, tmp_path
):
    lut = tmp_path / "lut.txt"
    lut.write_text(LUT)
    for img in (image_path, str(image_path), image):
        atlas = Atlas.from_freesurfer("fs", img, lut, "subject:1")
        assert atlas.name == "fs" and atlas.space == "subject:1"
        assert atlas.kind == "freesurfer"
        check(atlas, volume)
    # the image it was read from is kept
    assert atlas.image is image


def test_from_bids_dseg(volume, image_path, tmp_path):
    dseg = tmp_path / "x_dseg.tsv"
    dseg.write_text(DSEG)
    atlas = Atlas.from_bids_dseg("b", image_path, dseg, "s")
    check(atlas, volume)
    assert atlas.kind == "bids"


def test_from_tsv(volume, image, tmp_path):
    tsv = tmp_path / "t.tsv"
    tsv.write_text(TSV)
    atlas = Atlas.from_tsv("t", image, tsv)
    check(atlas, volume)
    assert (atlas.kind, atlas.space) == ("tsv", "unspecified")


def test_from_csv(volume, image, tmp_path):
    csv = tmp_path / "t.csv"
    csv.write_text("index,name\n" + "".join(
        f"{i},{n}\n" for i, n in enumerate(NAMES)
    ))
    atlas = Atlas.from_csv("c", image, csv)
    check(atlas, volume)
    assert atlas.kind == "csv"
    csv.write_text("".join(f"{i}\t{n}\n" for i, n in enumerate(NAMES)))
    atlas = Atlas.from_csv(
        "c", image, csv, read_csv={"sep": "\t", "header": None}
    )
    check(atlas, volume)


def test_loaders_forward_keyword_arguments(image, tmp_path):
    lut = tmp_path / "lut.txt"
    lut.write_text(LUT)
    atlas = Atlas.from_freesurfer("fs", image, lut, "s", background=3)
    assert atlas.background == 3
    assert 3 not in atlas.region_names()


def test_float_label_images_are_read_through_the_dataobj(volume, tmp_path):
    img = nib.Nifti1Image(volume.astype(np.float32), AFFINES["aligned"])
    lut = tmp_path / "lut.txt"
    lut.write_text(LUT)
    atlas = Atlas.from_freesurfer("fs", img, lut, "s")
    assert np.issubdtype(atlas.labels.dtype, np.integer)
    assert np.array_equal(atlas.labels, volume)


def test_a_trailing_singleton_dimension_is_dropped(volume, tmp_path):
    img = nib.Nifti1Image(volume[..., None], AFFINES["aligned"])
    lut = tmp_path / "lut.txt"
    lut.write_text(LUT)
    assert Atlas.from_freesurfer("fs", img, lut, "s").labels.ndim == 3


def test_a_4d_image_is_a_probabilistic_atlas(tmp_path):
    img = nib.Nifti1Image(np.zeros((4, 4, 4, 3), np.float32), np.eye(4))
    tsv = tmp_path / "t.tsv"
    tsv.write_text("id\tname\n1\ta\n")
    with pytest.raises(NotImplementedError, match="probabilistic"):
        Atlas.from_tsv("p", img, tsv, "s")


@pytest.mark.parametrize("labels", [
    pd.DataFrame({"index": range(5), "name": NAMES}),
    NAMES,
    [n.encode() for n in NAMES],
    dict(enumerate(NAMES)),
    list(enumerate(NAMES)),
    np.array(list(enumerate(NAMES)), dtype=[("id", int), ("name", "U20")]),
])
def test_from_labels_as_nilearn_gives_them(volume, image, labels):
    """A look-up table, a list (the id is the position), bytes, a dict,
    pairs: the background's row is dropped."""
    atlas = Atlas.from_labels("n", image, labels)
    check(atlas, volume, names=NAMES[1:])
    assert atlas.kind is None


def test_from_labels_with_a_path_and_normalised_names(volume, image_path):
    atlas = Atlas.from_labels(
        "n", str(image_path), NAMES, "MNI152", normalise=True
    )
    assert atlas.table["name"].tolist() == [
        "Left Putamen", "ctx rh precentral", "Brain Stem",
        "ctx lh precuneus",
    ]
    assert atlas.table["hemi"].tolist() == ["left", "right", "none", "left"]


def test_fetch_the_atlases_of_the_engines(image_path, monkeypatch, tmp_path):
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        return {"maps": str(image_path), "labels": NAMES}

    monkeypatch.setitem(atlases.FETCHERS, "schaefer", fetch)
    atlas = Atlas.fetch("s", "schaefer", tmp_path, n_rois=100)
    assert calls == [{"data_dir": str(tmp_path), "n_rois": 100}]
    assert (atlas.name, atlas.kind, atlas.space) == (
        "s", "schaefer", "MNI152"
    )
    assert atlas.table["name"].tolist() == [
        "Left Putamen", "ctx rh precentral", "Brain Stem",
        "ctx lh precuneus",
    ]
    with pytest.raises(ValueError, match="destrieux, schaefer, tian"):
        Atlas.fetch("x", "aal", tmp_path)


def test_the_tian_fetcher_reads_its_label_file(monkeypatch, tmp_path):
    (tmp_path / "labels.txt").write_text("lAMY-lh\naPUT-rh\n")
    # the authors' colours: a name, then its value, red, green, blue, alpha
    (tmp_path / "colours.txt").write_text(
        "aPUT-rh\n1 0 128 255 255\nlAMY-lh\n2 255 0 16 255\n"
        "17Networks_LH_Vis_1\n3 1 2 3 255\n"
    )
    calls = []

    def fetch_files(data_dir, files, **kwargs):
        calls.append((data_dir, [name for name, _, _ in files]))
        if len(files) == 2:
            return ["image.nii.gz", str(tmp_path / "labels.txt")]
        return [str(tmp_path / "colours.txt")]

    from neurolang.utils import nilearn_compat

    monkeypatch.setattr(nilearn_compat, "fetch_files", fetch_files)
    got = atlases.fetch_tian(scale=2, data_dir=str(tmp_path))
    assert got["maps"] == "image.nii.gz"
    assert got["labels"] == ["Background", "lAMY-lh", "aPUT-rh"]
    assert atlases.label_colours(got["lut"]) == {
        1: "#ff0010", 2: "#0080ff",
    }
    assert calls == [
        (str(tmp_path / "tian"), [
            "Tian_Subcortex_S2_3T_2009cAsym.nii.gz",
            "Tian_Subcortex_S2_3T_label.txt",
        ]),
        (str(tmp_path / "tian"), [
            "Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S2_"
            "label.txt",
        ]),
    ]


def test_the_tian_fetcher_goes_without_colours(monkeypatch, tmp_path):
    (tmp_path / "labels.txt").write_text("lAMY-lh\n")

    def fetch_files(data_dir, files, **kwargs):
        if len(files) == 1:
            raise OSError("offline")
        return ["image.nii.gz", str(tmp_path / "labels.txt")]

    from neurolang.utils import nilearn_compat

    monkeypatch.setattr(nilearn_compat, "fetch_files", fetch_files)
    got = atlases.fetch_tian(scale=2, data_dir=str(tmp_path))
    assert got == {"maps": "image.nii.gz",
                   "labels": ["Background", "lAMY-lh"]}


def test_the_colours_of_destrieux_are_freesurfers(monkeypatch, tmp_path):
    (tmp_path / "FreeSurferColorLUT.txt").write_text(
        "#No. Label Name: R G B A\n"
        "11101  ctx_lh_G_and_S_frontomargin  23 220  60   0\n"
        "12101  ctx_rh_G_and_S_frontomargin  23 220  61   0\n"
    )

    class Fetched(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    lut = pd.DataFrame({
        "index": [0, 1, 2, 3],
        "name": ["Background", "L G_and_S_frontomargin",
                 "R G_and_S_frontomargin", "L Medial_wall"],
    })
    maps = tmp_path / "destrieux.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), np.int16), np.eye(4)),
             str(maps))

    class Datasets:
        @staticmethod
        def fetch_atlas_destrieux_2009(**kwargs):
            return Fetched(maps=str(maps), labels=lut, lut=lut)

    from neurolang.utils import nilearn_compat

    monkeypatch.setattr(nilearn_compat, "nilearn_datasets", Datasets)
    monkeypatch.setattr(
        nilearn_compat, "fetch_files",
        lambda data_dir, files, **kwargs: [
            str(tmp_path / "FreeSurferColorLUT.txt")
        ],
    )
    got = atlases.fetch_destrieux(data_dir=str(tmp_path))
    assert atlases.label_colours(got.lut) == {1: "#17dc3c", 2: "#17dc3d"}
    assert atlases.label_colours(lut) == {}


def test_destrieux_is_registered_on_the_mni152_template(tmp_path):
    """Its voxels are those of nilearn's, on the affine moved by
    DESTRIEUX_TO_MNI152, in a file of its own written once."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [-76, -109, -64]
    labels = np.arange(8, dtype=np.int16).reshape(2, 2, 2)
    path = tmp_path / "destrieux2009_rois_lateralized.nii.gz"
    nib.save(nib.Nifti1Image(labels, affine), str(path))
    registered = atlases.registered_image(path, atlases.DESTRIEUX_TO_MNI152)
    assert registered.endswith("destrieux2009_rois_lateralized_mni152.nii.gz")
    image = nib.load(registered)
    np.testing.assert_array_equal(np.asanyarray(image.dataobj), labels)
    np.testing.assert_allclose(
        image.affine, atlases.DESTRIEUX_TO_MNI152 @ affine, atol=1e-5
    )
    # 1.3 to 2.6% larger, about a millimetre off
    scales = np.diag(atlases.DESTRIEUX_TO_MNI152)[:3]
    assert np.all((scales > 1.01) & (scales < 1.03))
    written = Path(registered).stat().st_mtime_ns
    assert atlases.registered_image(path, np.eye(4)) == registered
    assert Path(registered).stat().st_mtime_ns == written


def test_the_module_needs_nothing_else_of_neurolang():
    """Importing it loads neither nilearn, nor the core, the parsers, the
    regions or the frontends: the core of the engine tracts imports it."""
    code = (
        "import sys, neurolang.atlases;"
        "print(sorted(m for m in sys.modules if m.startswith("
        "('nilearn', 'nl_core', 'lark', 'neurolang.regions', "
        "'neurolang.frontend'))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        check=True,
    )
    assert result.stdout.strip() == "[]"


# nilearn's own fetchers: the data is cached after a first run; the tests
# skip if it cannot be fetched
def _from_nilearn_or_skip(*args, **kwargs):
    try:
        return Atlas.from_nilearn(*args, **kwargs)
    except Exception as e:  # offline, mirror down, cache unreadable
        pytest.skip(f"cannot fetch {args[1]}: {e!r}")


def test_from_nilearn_names_a_fetcher_that_nilearn_has():
    pytest.importorskip("nilearn")
    with pytest.raises(ValueError, match="fetch_atlas_nothing"):
        Atlas.from_nilearn("x", "nothing")


@pytest.mark.network
def test_nilearn_destrieux():
    with pytest.warns(UserWarning, match="Medial_wall"):
        atlas = _from_nilearn_or_skip(
            "destrieux", "destrieux_2009", lateralized=True
        )
    assert (atlas.kind, atlas.space) == ("nilearn:destrieux_2009", "MNI152")
    by_name = dict(zip(atlas.table["name"], atlas.table["hemi"]))
    # normalised as the engines name them
    assert by_name["L G precentral"] == "left"
    assert by_name["R G precentral"] == "right"
    assert set(by_name.values()) == {"left", "right"}  # no background


@pytest.mark.network
def test_nilearn_harvard_oxford():
    atlas = _from_nilearn_or_skip(
        "ho", "harvard_oxford", atlas_name="sub-maxprob-thr25-2mm"
    )
    by_name = dict(zip(atlas.table["name"], atlas.table["hemi"]))
    assert by_name["Left Thalamus"] == "left"
    assert by_name["Right Hippocampus"] == "right"


# The label names.

@pytest.mark.parametrize("raw,name", [
    ("L G_and_S_frontomargin", "L G and S frontomargin"),
    (b"aHIP-rh", "aHIP rh"),
    ("17Networks_LH_DefaultA_PCC_1", "17Networks LH DefaultA PCC 1"),
    (3, "3"),
])
def test_label_name(raw, name):
    assert label_name(raw) == name


# The table.

def test_attributes(volume):
    atlas = make(volume)
    assert atlas.name == "fs" and atlas.space == "subject:1"
    assert atlas.background == 0
    assert atlas.labels.shape == atlas.shape == volume.shape
    assert np.array_equal(atlas.labels, volume)
    assert np.array_equal(atlas.affine, AFFINES["aligned"])
    assert atlas.table.dtypes["id"] == np.int32
    assert atlas.present_labels.tolist() == [0, 1, 2, 3, 4]


def test_the_image_of_an_atlas_of_arrays(volume):
    image = make(volume).image
    assert np.array_equal(np.asanyarray(image.dataobj), volume)
    assert np.array_equal(image.affine, AFFINES["aligned"])


def test_inferred_hemispheres_can_be_configured(volume):
    rules = [HemisphereRule(r"Brain-.+", "right")]
    t = make(volume, hemisphere_rules=rules).table
    by_name = dict(zip(t["name"], t["hemi"]))
    assert by_name["Brain-Stem"] == "right"
    assert by_name["Left-Putamen"] == "none"


def test_a_hemi_column_beats_inference_and_warns_on_disagreement(volume):
    table = TABLE.assign(hemi=["", "right", "", "", ""])
    with pytest.warns(UserWarning, match="Left-Putamen.*hemi"):
        atlas = make(volume, table=table)
    by_name = dict(zip(atlas.table["name"], atlas.table["hemi"]))
    assert by_name["Left-Putamen"] == "right"
    # empty cells fall back to inference
    assert by_name["ctx-rh-precentral"] == "right"
    assert by_name["Brain-Stem"] == "none"


def test_an_explicit_none_is_a_decision(volume):
    table = TABLE.assign(hemi=["none"] * 5)
    with pytest.warns(UserWarning):
        atlas = make(volume, table=table)
    assert set(atlas.table["hemi"]) == {"none"}


def test_hints_come_between_the_table_and_the_inference(volume):
    table = TABLE.assign(hemi=["", "", "", "", "right"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atlas = make(volume, table=table)
    with pytest.warns(UserWarning, match="ctx-lh-precuneus.*hemi"):
        hemi = atlas.hemispheres({
            "Brain-Stem": "left", "ctx-lh-precuneus": "left",
            "Left-Putamen": "",
        })
    assert dict(zip(atlas.table["name"], hemi)) == {
        **HEMI, "Brain-Stem": "left", "ctx-lh-precuneus": "right",
    }


def test_invalid_hemi_in_the_table_is_rejected(volume):
    table = TABLE.assign(hemi=["none", "lft", "none", "none", "none"])
    with pytest.raises(ValueError, match="hemi"):
        make(volume, table=table)


def test_ids_in_the_volume_but_not_in_the_table_are_no_region(volume):
    table = TABLE[TABLE.id != 3]
    with pytest.warns(UserWarning, match="3"):
        atlas = make(volume, table=table)
    assert atlas.unlisted_labels == [3]
    assert 3 not in atlas.table["id"].tolist()
    assert 3 not in atlas.region_names()


def test_ids_in_the_table_but_not_in_the_volume_are_dropped_and_warned(volume):
    table = pd.concat(
        [TABLE, pd.DataFrame({"id": [99], "name": ["Left-Absent"]})]
    )
    with pytest.warns(UserWarning, match="99"):
        atlas = make(volume, table=table)
    assert 99 not in atlas.table["id"].tolist()


def test_duplicate_ids_in_the_table_are_rejected(volume):
    table = pd.concat([TABLE, TABLE.iloc[[1]]])
    with pytest.raises(ValueError, match="duplicate"):
        make(volume, table=table)


def test_the_table_is_sorted_by_id(volume):
    atlas = make(volume, table=TABLE.iloc[::-1])
    assert atlas.table["id"].tolist() == [0, 1, 2, 3, 4]
    assert dict(zip(atlas.table["name"], atlas.table["hemi"])) == HEMI


# The volume.

def test_float_volumes_with_integral_values_are_accepted(volume):
    atlas = make(volume.astype(np.float32))
    assert np.issubdtype(atlas.labels.dtype, np.integer)
    assert np.array_equal(atlas.labels, volume)


def test_float_volumes_with_fractions_are_rejected(volume):
    with pytest.raises(ValueError, match="integer"):
        make(volume.astype(np.float32) + 0.5)


def test_negative_labels_are_rejected(volume):
    volume[0, 0, 0] = -1
    with pytest.raises(ValueError, match="negative"):
        make(volume)


@pytest.mark.parametrize("shape", [(4, 4), (4, 4, 4, 3)])
def test_the_volume_must_be_3d(shape):
    with pytest.raises(ValueError, match="3D"):
        Atlas("x", np.zeros(shape, np.int32), np.eye(4), TABLE, "s")


def test_the_affine_must_be_4x4(volume):
    with pytest.raises(ValueError, match="4x4"):
        make(volume, affine=np.eye(3))


def test_a_singular_affine_is_rejected(volume):
    with pytest.raises(ValueError, match="singular"):
        make(volume, affine=np.zeros((4, 4)))


def test_the_table_needs_id_and_name(volume):
    with pytest.raises(ValueError, match="id.*name"):
        make(volume, table=TABLE.rename(columns={"id": "index"}))


# The regions and the points.

def test_regions_are_the_labels_of_the_table_but_the_background(volume):
    atlas = make(volume)
    assert atlas.region_names() == dict(enumerate(NAMES[1:], 1))
    assert atlas.label_values() == {
        n: [i] for i, n in enumerate(NAMES) if i
    }
    voxels = atlas.label_voxels()
    assert sorted(voxels) == [1, 2, 3, 4]
    for i, v in voxels.items():
        assert np.array_equal(v, np.argwhere(volume == i))


def test_several_ids_of_one_name(volume):
    atlas = make(volume, table=TABLE.assign(name=["U", "A", "A", "B", "C"]))
    assert atlas.label_values() == {"A": [1, 2], "B": [3], "C": [4]}


def test_points_are_the_centres_of_the_voxels_of_the_regions(volume):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 4 is in no region
        atlas = make(volume, table=TABLE[TABLE.id != 4])
    points = atlas.points()
    assert points.columns.tolist() == ["x", "y", "z", "label"]
    voxels = np.argwhere(np.isin(volume, [1, 2, 3]))
    np.testing.assert_allclose(
        points[["x", "y", "z"]].to_numpy(),
        nib.affines.apply_affine(AFFINES["aligned"], voxels),
    )
    assert points["label"].tolist() == [
        NAMES[v] for v in volume[tuple(voxels.T)]
    ]


def test_a_nonzero_background_is_no_region(volume):
    vol = np.where(volume == 0, 255, volume).astype(np.int32)
    table = pd.concat(
        [TABLE[TABLE.id != 0], pd.DataFrame({"id": [255], "name": ["BG"]})]
    )
    atlas = make(vol, table=table, background=255)
    assert 255 in atlas.table["id"].tolist()
    assert sorted(atlas.region_names()) == [1, 2, 3, 4]
    assert "BG" not in set(atlas.points()["label"])


# The hash.

def test_content_hash_changes_with_each_ingredient(volume):
    base = make(volume).content_hash()
    assert base == make(volume).content_hash()
    other_vol = volume.copy()
    other_vol[0, 0, 0] = 3
    other_aff = AFFINES["aligned"].copy()
    other_aff[0, 3] += 1
    renamed = TABLE.assign(name=TABLE.name.str.upper())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the hemi column disagrees
        variants = [
            make(other_vol),
            make(volume, affine=other_aff),
            make(volume, table=renamed),
            make(volume, table=TABLE.assign(hemi=["", "right", "", "", ""])),
            make(volume, space="subject:2"),
            make(volume, background=1),
        ]
    hashes = {v.content_hash() for v in variants} | {base}
    assert len(hashes) == len(variants) + 1


def test_content_hash_does_not_depend_on_the_name_or_the_kind(volume):
    a = make(volume, name="one").content_hash()
    assert a == make(volume, name="two", kind="tsv").content_hash()


# The spaces.

def test_two_atlases_in_different_spaces_do_not_combine(volume):
    a = make(volume, name="a", space="MNI152NLin2009cAsym")
    b = make(volume, name="b", space="subject:100307")
    with pytest.raises(SpaceMismatchError):
        a.check_same_space(b)
    with pytest.warns(UserWarning):
        a.check_same_space(b, allow_mismatch=True)


def test_two_atlases_in_the_same_space_combine(volume):
    make(volume, name="a").check_same_space(make(volume, name="b"))


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.int32, np.float32])
def test_label_voxel_indices_whatever_the_dtypes(dtype):
    """The labels are searched for in the dtype of the data, all at once;
    one it cannot hold has no voxel."""
    from neurolang.atlases import label_voxel_indices

    data = np.zeros((3, 4, 5), dtype=dtype)
    data[0, 1, 2] = data[2, 3, 4] = 7
    data[1, 1, 1] = 200
    got = label_voxel_indices(data, [7, 200, 3, 7.5, -1])
    assert list(got) == [7, 200, 3, 7.5, -1]
    assert got[7].tolist() == [[0, 1, 2], [2, 3, 4]]
    assert got[200].tolist() == [[1, 1, 1]]
    for label in (3, 7.5):
        assert got[label].shape == (0, 3)
    assert got[-1].shape == (0, 3) or dtype == np.float32


def test_the_regions_import_it_when_they_need_it():
    """It imports pandas, which the regions do not."""
    code = "import sys, neurolang.regions; print('pandas' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"
