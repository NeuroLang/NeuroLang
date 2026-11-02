import warnings

import numpy as np
import pandas as pd
import pytest

from nl_tracts.atlas import Atlas, ProbabilisticAtlas, HemisphereRule
from nl_tracts.spaces import SpaceMismatchError

from .conftest import AFFINES

TABLE = pd.DataFrame(
    {
        "id": [0, 1, 2, 3, 4],
        "name": [
            "Unknown",
            "Left-Putamen",
            "ctx-rh-precentral",
            "Brain-Stem",
            "ctx-lh-precuneus",
        ],
    }
)
MAPPING = pd.DataFrame(
    {
        "atlas_label_name": ["Left-Putamen", "Left-Putamen", "Brain-Stem"],
        "term": ["putamen", "subcortical", "brain_stem"],
        "hemi": ["left", "left", "none"],
    }
)


def make(volume, table=TABLE, **kw):
    kw.setdefault("name", "fs")
    kw.setdefault("space", "subject:1")
    kw.setdefault("affine", AFFINES["aligned"])
    return Atlas(labels=volume, table=table, **kw)


def test_attributes(volume):
    atlas = make(volume, term_mapping=MAPPING)
    assert atlas.name == "fs" and atlas.space == "subject:1"
    assert atlas.background == 0
    assert atlas.labels.shape == volume.shape
    assert np.array_equal(atlas.labels, volume)
    assert np.array_equal(atlas.affine, AFFINES["aligned"])


def test_label_relation(volume):
    rel = make(volume).label_relation()
    assert rel.columns.tolist() == ["a", "l", "name", "hemi"]
    assert rel.dtypes["l"] == np.int32
    assert rel["a"].tolist() == ["fs"] * len(rel)
    cols = rel[["l", "name", "hemi"]]
    got = {tuple(r) for r in cols.itertuples(index=False)}
    assert got == {
        (0, "Unknown", "none"),
        (1, "Left-Putamen", "left"),
        (2, "ctx-rh-precentral", "right"),
        (3, "Brain-Stem", "none"),
        (4, "ctx-lh-precuneus", "left"),
    }


def test_the_atlas_column_is_a_constant_string_column(volume):
    rel = make(volume).label_relation()
    assert pd.api.types.is_string_dtype(rel["a"])
    assert rel["a"].nunique() == 1


def test_label_term_relation(volume):
    rel = make(volume, term_mapping=MAPPING).label_term_relation()
    assert rel.columns.tolist() == ["a", "l", "term"]
    assert rel.dtypes["l"] == np.int32
    assert {tuple(r) for r in rel[["l", "term"]].itertuples(index=False)} == {
        (1, "putamen"),
        (1, "subcortical"),
        (3, "brain_stem"),
    }


def test_label_term_relation_without_a_mapping_is_empty(volume):
    rel = make(volume).label_term_relation()
    assert rel.columns.tolist() == ["a", "l", "term"]
    assert len(rel) == 0 and rel.dtypes["l"] == np.int32


def test_shipped_mapping_by_key(volume):
    atlas = make(volume, term_mapping="freesurfer_aseg_aparc")
    terms = atlas.label_term_relation()
    assert {"precentral", "frontal"} == set(terms[terms.l == 2]["term"])


def test_unmapped_labels_are_reported(volume):
    atlas = make(volume, term_mapping=MAPPING)
    got = atlas.unmapped_labels()
    assert got.columns.tolist() == ["id", "name", "hemi"]
    # the background is not a region, it is never reported
    assert got.values.tolist() == [
        [2, "ctx-rh-precentral", "right"],
        [4, "ctx-lh-precuneus", "left"],
    ]


def test_without_a_mapping_every_label_is_unmapped(volume):
    assert make(volume).unmapped_labels()["id"].tolist() == [1, 2, 3, 4]


def test_nothing_unmapped_when_every_label_has_a_term(volume):
    atlas = make(volume, term_mapping="freesurfer_aseg_aparc")
    assert atlas.unmapped_labels()["id"].tolist() == []


# hemispheres: table > mapping > inference
def test_inferred_hemispheres_can_be_configured(volume):
    rules = [HemisphereRule(r"Brain-.+", "right")]
    rel = make(volume, hemisphere_rules=rules).label_relation()
    by_name = dict(zip(rel["name"], rel["hemi"]))
    assert by_name["Brain-Stem"] == "right"
    assert by_name["Left-Putamen"] == "none"


def test_a_hemi_column_beats_inference_and_warns_on_disagreement(volume):
    table = TABLE.assign(hemi=["", "right", "", "", ""])
    with pytest.warns(UserWarning, match="Left-Putamen.*hemi"):
        rel = make(volume, table=table).label_relation()
    by_name = dict(zip(rel["name"], rel["hemi"]))
    assert by_name["Left-Putamen"] == "right"
    # empty cells fall back to inference
    assert by_name["ctx-rh-precentral"] == "right"
    assert by_name["Brain-Stem"] == "none"


def test_an_explicit_none_is_a_decision(volume):
    table = TABLE.assign(hemi=["none"] * 5)
    with pytest.warns(UserWarning):
        rel = make(volume, table=table).label_relation()
    assert set(rel["hemi"]) == {"none"}


def test_mapping_hemi_beats_inference_for_mapped_labels(volume):
    mapping = pd.DataFrame(
        {"atlas_label_name": ["Brain-Stem"], "term": ["x"], "hemi": ["left"]}
    )
    rel = make(volume, term_mapping=mapping).label_relation()
    assert dict(zip(rel["name"], rel["hemi"]))["Brain-Stem"] == "left"


def test_disagreeing_sources_warn(volume):
    mapping = pd.DataFrame(
        {"atlas_label_name": ["Left-Putamen"], "term": ["x"],
         "hemi": ["right"]}
    )
    with pytest.warns(UserWarning, match="Left-Putamen.*hemi"):
        rel = make(volume, term_mapping=mapping).label_relation()
    assert dict(zip(rel["name"], rel["hemi"]))["Left-Putamen"] == "right"


def test_invalid_hemi_in_the_table_is_rejected(volume):
    table = TABLE.assign(hemi=["none", "lft", "none", "none", "none"])
    with pytest.raises(ValueError, match="hemi"):
        make(volume, table=table)


# ids of the volume and of the table
def test_ids_in_the_volume_but_not_in_the_table_are_named_and_warned(volume):
    table = TABLE[TABLE.id != 3]
    with pytest.warns(UserWarning, match="3"):
        rel = make(volume, table=table).label_relation()
    row = rel[rel.l == 3].iloc[0]
    assert (row["name"], row["hemi"]) == ("label_3", "none")


def test_ids_in_the_table_but_not_in_the_volume_are_dropped_and_warned(volume):
    table = pd.concat(
        [TABLE, pd.DataFrame({"id": [99], "name": ["Left-Absent"]})]
    )
    with pytest.warns(UserWarning, match="99"):
        rel = make(volume, table=table).label_relation()
    assert 99 not in rel["l"].tolist()


def test_the_background_has_a_label_row_only_if_the_table_has_one(volume):
    table = TABLE[TABLE.id != 0]
    rel = make(volume, table=table).label_relation()  # no warning for 0
    assert 0 not in rel["l"].tolist()


def test_a_nonzero_background(volume):
    vol = np.where(volume == 0, 255, volume).astype(np.int32)
    table = pd.concat(
        [TABLE[TABLE.id != 0], pd.DataFrame({"id": [255], "name": ["BG"]})]
    )
    atlas = make(vol, table=table, background=255)
    assert atlas.unmapped_labels()["id"].tolist() == [1, 2, 3, 4]
    assert 255 in atlas.label_relation()["l"].tolist()
    assert 255 not in atlas.label_extent_relation()["l"].tolist()


def test_duplicate_ids_in_the_table_are_rejected(volume):
    table = pd.concat([TABLE, TABLE.iloc[[1]]])
    with pytest.raises(ValueError, match="duplicate"):
        make(volume, table=table)


# the volume
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


# hash
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
        variants = _variants(volume, other_vol, other_aff, renamed)
    hashes = {v.content_hash() for v in variants} | {base}
    assert len(hashes) == len(variants) + 1


def _variants(volume, other_vol, other_aff, renamed):
    return [
        make(other_vol),
        make(volume, affine=other_aff),
        make(volume, table=renamed),
        make(volume, table=TABLE.assign(hemi=["", "right", "", "", ""])),
        make(volume, space="subject:2"),
        make(volume, background=1),
    ]


def test_content_hash_does_not_depend_on_the_name_or_the_mapping(volume):
    a = make(volume, name="one").content_hash()
    assert a == make(volume, name="two").content_hash()
    assert a == make(volume, term_mapping=MAPPING).content_hash()


# spaces
def test_two_atlases_in_different_spaces_do_not_combine(volume):
    a = make(volume, name="a", space="MNI152NLin2009cAsym")
    b = make(volume, name="b", space="subject:100307")
    with pytest.raises(SpaceMismatchError):
        a.check_same_space(b)
    with pytest.warns(UserWarning):
        a.check_same_space(b, allow_mismatch=True)


def test_two_atlases_in_the_same_space_combine(volume):
    a = make(volume, name="a")
    b = make(volume, name="b")
    a.check_same_space(b)


def test_probabilistic_atlases_are_not_implemented():
    with pytest.raises(NotImplementedError, match="DESIGN"):
        ProbabilisticAtlas()
