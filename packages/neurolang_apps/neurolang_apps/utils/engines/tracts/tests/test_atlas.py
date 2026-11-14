"""What tractography makes of an atlas: the term mapping of its labels and
the catalogue relations ``label`` and ``label_term`` (the atlas itself,
``neurolang.atlases.Atlas``, is tested in neurolang)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from neurolang.atlases import Atlas
from ..atlas import (
    ProbabilisticAtlas,
    atlas_term_mapping,
    label_extent_relation,
    label_relation,
    label_term_relation,
    unmapped_labels,
)
from ..terms import load_term_mapping

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


def by_name(rel):
    return dict(zip(rel["name"], rel["hemi"]))


def test_label_relation(volume):
    rel = label_relation(make(volume))
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
    rel = label_relation(make(volume))
    assert pd.api.types.is_string_dtype(rel["a"])
    assert rel["a"].nunique() == 1


def test_label_term_relation(volume):
    rel = label_term_relation(make(volume), MAPPING)
    assert rel.columns.tolist() == ["a", "l", "term"]
    assert rel.dtypes["l"] == np.int32
    assert {tuple(r) for r in rel[["l", "term"]].itertuples(index=False)} == {
        (1, "putamen"),
        (1, "subcortical"),
        (3, "brain_stem"),
    }


def test_label_term_relation_without_a_mapping_is_empty(volume):
    rel = label_term_relation(make(volume))
    assert rel.columns.tolist() == ["a", "l", "term"]
    assert len(rel) == 0 and rel.dtypes["l"] == np.int32


def test_shipped_mapping_by_key(volume):
    terms = label_term_relation(
        make(volume), load_term_mapping("freesurfer_aseg_aparc")
    )
    assert {"precentral", "ctx_precentral", "frontal"} == set(
        terms[terms.l == 2]["term"]
    )


def test_a_mapping_of_raw_names_serves_normalised_names(volume):
    """An atlas an engine fetches has the names of label_name: the raw
    names of a mapping match them so."""
    atlas = make(volume, table=TABLE.assign(
        name=TABLE["name"].str.replace("-", " ")
    ))
    assert set(label_term_relation(atlas, MAPPING)["l"]) == {1, 3}
    assert by_name(label_relation(atlas, MAPPING))["Left Putamen"] == "left"


def test_unmapped_labels_are_reported(volume):
    got = unmapped_labels(make(volume), MAPPING)
    assert got.columns.tolist() == ["id", "name", "hemi"]
    # the background is not a region, it is never reported
    assert got.values.tolist() == [
        [2, "ctx-rh-precentral", "right"],
        [4, "ctx-lh-precuneus", "left"],
    ]


def test_without_a_mapping_every_label_is_unmapped(volume):
    assert unmapped_labels(make(volume))["id"].tolist() == [1, 2, 3, 4]


def test_nothing_unmapped_when_every_label_has_a_term(volume):
    mapping = load_term_mapping("freesurfer_aseg_aparc")
    assert unmapped_labels(make(volume), mapping)["id"].tolist() == []


# The term mapping of an atlas: given, else the default of its kind.
@pytest.mark.parametrize("kind,key", [
    ("freesurfer", "freesurfer_aseg_aparc"),
    ("destrieux", "freesurfer_destrieux"),
    ("nilearn:destrieux_2009", "freesurfer_destrieux"),
    ("tian", "tian_subcortex_s2"),
])
def test_the_default_term_mapping_of_a_kind(volume, kind, key):
    got = atlas_term_mapping(make(volume, kind=kind))
    pd.testing.assert_frame_equal(got, load_term_mapping(key))


def test_a_given_term_mapping_beats_the_default(volume):
    got = atlas_term_mapping(make(volume, kind="freesurfer"), MAPPING)
    pd.testing.assert_frame_equal(got, load_term_mapping(MAPPING))


@pytest.mark.parametrize("kind", [None, "schaefer", "tsv"])
def test_no_default_term_mapping(volume, kind):
    assert len(atlas_term_mapping(make(volume, kind=kind))) == 0


# hemispheres: table > mapping > inference
def test_mapping_hemi_beats_inference_for_mapped_labels(volume):
    mapping = pd.DataFrame(
        {"atlas_label_name": ["Brain-Stem"], "term": ["x"], "hemi": ["left"]}
    )
    rel = label_relation(make(volume), mapping)
    assert by_name(rel)["Brain-Stem"] == "left"


def test_the_table_beats_the_mapping(volume):
    table = TABLE.assign(hemi=["", "", "", "none", ""])
    mapping = pd.DataFrame(
        {"atlas_label_name": ["Brain-Stem"], "term": ["x"], "hemi": ["left"]}
    )
    atlas = make(volume, table=table)
    with pytest.warns(UserWarning, match="Brain-Stem.*hemi"):
        rel = label_relation(atlas, mapping)
    assert by_name(rel)["Brain-Stem"] == "none"


def test_disagreeing_sources_warn(volume):
    mapping = pd.DataFrame(
        {"atlas_label_name": ["Left-Putamen"], "term": ["x"],
         "hemi": ["right"]}
    )
    with pytest.warns(UserWarning, match="Left-Putamen.*hemi"):
        rel = label_relation(make(volume), mapping)
    assert by_name(rel)["Left-Putamen"] == "right"


# ids of the volume and of the table
def test_ids_in_the_volume_but_not_in_the_table_are_labels(volume):
    """They are no region of the engine, but streamline_label has their
    streamlines: label names them."""
    table = TABLE[TABLE.id != 3]
    with pytest.warns(UserWarning, match="3"):
        atlas = make(volume, table=table)
    rel = label_relation(atlas)
    row = rel[rel.l == 3].iloc[0]
    assert (row["name"], row["hemi"]) == ("label_3", "none")
    assert 3 in unmapped_labels(atlas)["id"].tolist()


def test_the_background_has_a_label_row_only_if_the_table_has_one(volume):
    table = TABLE[TABLE.id != 0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning for 0
        rel = label_relation(make(volume, table=table))
    assert 0 not in rel["l"].tolist()


def test_a_nonzero_background(volume):
    vol = np.where(volume == 0, 255, volume).astype(np.int32)
    table = pd.concat(
        [TABLE[TABLE.id != 0], pd.DataFrame({"id": [255], "name": ["BG"]})]
    )
    atlas = make(vol, table=table, background=255)
    assert unmapped_labels(atlas)["id"].tolist() == [1, 2, 3, 4]
    assert 255 in label_relation(atlas)["l"].tolist()
    assert 255 not in label_extent_relation(atlas)["l"].tolist()


def test_probabilistic_atlases_are_not_implemented():
    with pytest.raises(NotImplementedError, match="DESIGN"):
        ProbabilisticAtlas()
