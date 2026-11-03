import numpy as np
import nibabel as nib
import pandas as pd
import pytest

from nl_tracts.atlas import (
    Atlas,
    load_label_table_bids_dseg,
    load_label_table_freesurfer_lut,
    load_label_table_tsv,
)

from .conftest import AFFINES

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


@pytest.fixture
def image(volume):
    return nib.Nifti1Image(volume, AFFINES["flipped"])


@pytest.fixture
def image_path(image, tmp_path):
    path = tmp_path / "labels.nii.gz"
    nib.save(image, path)
    return path


def check(atlas, volume, affine="flipped"):
    assert np.array_equal(atlas.labels, volume)
    np.testing.assert_allclose(atlas.affine, AFFINES[affine])
    rel = atlas.label_relation()
    assert dict(zip(rel["l"], rel["name"])) == dict(enumerate(NAMES))
    assert dict(zip(rel["name"], rel["hemi"])) == {
        "Unknown": "none", "Left-Putamen": "left",
        "ctx-rh-precentral": "right", "Brain-Stem": "none",
        "ctx-lh-precuneus": "left",
    }


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


def test_from_freesurfer_with_a_path_and_with_an_image(
    volume, image, image_path, tmp_path
):
    lut = tmp_path / "lut.txt"
    lut.write_text(LUT)
    for img in (image_path, str(image_path), image):
        atlas = Atlas.from_freesurfer("fs", img, lut, "subject:1")
        assert atlas.name == "fs" and atlas.space == "subject:1"
        check(atlas, volume)


def test_from_bids_dseg(volume, image_path, tmp_path):
    dseg = tmp_path / "x_dseg.tsv"
    dseg.write_text(DSEG)
    check(Atlas.from_bids_dseg("b", image_path, dseg, "s"), volume)


def test_from_tsv(volume, image, tmp_path):
    tsv = tmp_path / "t.tsv"
    tsv.write_text(TSV)
    check(Atlas.from_tsv("t", image, tsv, "s"), volume)


def test_loaders_forward_keyword_arguments(image, tmp_path):
    lut = tmp_path / "lut.txt"
    lut.write_text(LUT)
    atlas = Atlas.from_freesurfer(
        "fs", image, lut, "s", term_mapping="freesurfer_aseg_aparc"
    )
    assert len(atlas.label_term_relation()) > 0


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
    with pytest.raises(NotImplementedError, match="DESIGN"):
        Atlas.from_tsv("p", img, tsv, "s")


class Bunch(dict):
    """As nilearn's: keys are attributes."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


def test_from_nilearn_with_a_lut(volume, image):
    lut = pd.DataFrame({"index": range(5), "name": NAMES})
    bunch = Bunch(maps=image, labels=NAMES, lut=lut)
    check(Atlas.from_nilearn("n", bunch, "s"), volume)


def test_from_nilearn_with_labels_only_ids_are_positions(volume, image):
    bunch = Bunch(maps=image, labels=NAMES)
    check(Atlas.from_nilearn("n", bunch, "s"), volume)


def test_from_nilearn_decodes_bytes_labels(volume, image):
    bunch = Bunch(maps=image, labels=[n.encode() for n in NAMES])
    check(Atlas.from_nilearn("n", bunch, "s"), volume)


def test_from_nilearn_with_a_path_to_the_maps(volume, image_path):
    bunch = Bunch(maps=str(image_path), labels=NAMES)
    check(Atlas.from_nilearn("n", bunch, "s"), volume)


def test_from_nilearn_does_not_import_nilearn(volume, image):
    import subprocess
    import sys

    code = (
        "import sys, nl_tracts.atlas;"
        "assert 'nilearn' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# nilearn's own fetchers: the data is cached after a first run; the tests
# skip if it cannot be fetched
@pytest.mark.network
def test_nilearn_destrieux(fetch_or_skip):
    from nilearn import datasets

    bunch = fetch_or_skip(
        datasets.fetch_atlas_destrieux_2009, lateralized=True
    )
    with pytest.warns(UserWarning, match="Medial_wall"):
        atlas = Atlas.from_nilearn(
            "destrieux", bunch, "MNI152NLin2009aSym",
            term_mapping="freesurfer_destrieux",
        )
    rel = atlas.label_relation()
    by_name = dict(zip(rel["name"], rel["hemi"]))
    assert by_name["L G_precentral"] == "left"
    assert by_name["R G_precentral"] == "right"
    assert set(by_name.values()) == {"left", "right", "none"}  # Background
    assert len(atlas.label_extent_relation()) == 3 * (len(rel) - 1)
    # what the mapping leaves out is the medial wall, and only that
    stems = {n.split(" ", 1)[1] for n in atlas.unmapped_labels()["name"]}
    assert stems <= {"Medial_wall", "Unknown"}
    assert "G_precentral" not in stems and "S_central" not in stems


@pytest.mark.network
def test_nilearn_harvard_oxford(fetch_or_skip):
    from nilearn import datasets

    bunch = fetch_or_skip(
        datasets.fetch_atlas_harvard_oxford, "sub-maxprob-thr25-2mm"
    )
    atlas = Atlas.from_nilearn("ho", bunch, "MNI152NLin6Asym")
    rel = atlas.label_relation()
    by_name = dict(zip(rel["name"], rel["hemi"]))
    assert by_name["Left Thalamus"] == "left"
    assert by_name["Right Hippocampus"] == "right"
