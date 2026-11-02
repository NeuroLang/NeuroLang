import os
import re

import pandas as pd
import pytest

from nl_tracts.atlas import load_label_table_freesurfer_lut
from nl_tracts.terms import (
    SHIPPED_MAPPINGS,
    load_term_mapping,
    shipped_mapping_path,
)

COLUMNS = ["atlas_label_name", "term", "hemi"]
# a hemisphere token in a term would make it atlas- and side-dependent
HEMISPHERE_TOKEN = re.compile(r"(^|_)(lh|rh|left|right)(_|$)|_[LR]$")


def write(tmp_path, text):
    path = tmp_path / "mapping.tsv"
    path.write_text(text)
    return path


def test_format_columns_and_comments(tmp_path):
    path = write(
        tmp_path,
        "# a comment\n"
        "atlas_label_name\tterm\themi\n"
        "\n"
        "Left-Putamen\tputamen\tleft\n"
        "Left-Putamen\tsubcortical\tleft\n"
        "# another\n"
        "Brain-Stem\tbrain_stem\tnone\n",
    )
    got = load_term_mapping(path)
    assert got.columns.tolist() == COLUMNS
    assert got.values.tolist() == [
        ["Left-Putamen", "putamen", "left"],
        ["Left-Putamen", "subcortical", "left"],
        ["Brain-Stem", "brain_stem", "none"],
    ]


def test_hemi_column_is_optional_and_may_be_empty(tmp_path):
    no_column = load_term_mapping(
        write(tmp_path, "atlas_label_name\tterm\nA\tx\n")
    )
    empty = load_term_mapping(
        write(tmp_path, "atlas_label_name\tterm\themi\nA\tx\t\n")
    )
    assert no_column.columns.tolist() == COLUMNS
    assert no_column["hemi"].tolist() == [""]
    assert empty["hemi"].tolist() == [""]


def test_accepts_a_dataframe():
    df = pd.DataFrame({"atlas_label_name": ["A"], "term": ["x"]})
    assert load_term_mapping(df)["hemi"].tolist() == [""]


@pytest.mark.parametrize(
    "text,match",
    [
        ("name\tterm\nA\tx\n", "atlas_label_name"),
        ("atlas_label_name\tterm\nA\t\n", "term"),
        ("atlas_label_name\tterm\themi\nA\tx\tlft\n", "hemi"),
        ("atlas_label_name\tterm\nA\tx\nA\tx\n", "duplicate"),
        (
            "atlas_label_name\tterm\themi\nA\tx\tleft\nA\ty\tright\n",
            "hemi",
        ),
    ],
)
def test_invalid_mappings_are_rejected(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        load_term_mapping(write(tmp_path, text))


def test_a_shipped_mapping_is_found_by_its_key():
    for key in SHIPPED_MAPPINGS:
        assert shipped_mapping_path(key).is_file()
        assert load_term_mapping(key).shape[0] > 0


def test_unknown_key_that_is_not_a_file_is_an_error():
    with pytest.raises(ValueError, match="freesurfer_aseg_aparc"):
        load_term_mapping("no_such_mapping")


def test_the_shipped_keys():
    assert set(SHIPPED_MAPPINGS) == {
        "freesurfer_aseg_aparc",
        "freesurfer_destrieux",
    }


@pytest.mark.parametrize("key", sorted(SHIPPED_MAPPINGS))
def test_shipped_mappings_are_valid_and_hemisphere_free(key):
    mapping = load_term_mapping(key)
    bad = [t for t in mapping["term"].unique() if HEMISPHERE_TOKEN.search(t)]
    assert bad == []
    assert mapping["term"].str.fullmatch(r"[a-z][a-z0-9_]*").all()
    assert set(mapping["hemi"]) <= {"left", "right", "none", ""}


def test_hemisphere_token_regex_does_not_hit_entorhinal():
    assert not HEMISPHERE_TOKEN.search("entorhinal")
    assert HEMISPHERE_TOKEN.search("temporal_left")


# The vocabulary is tract_querier's (data/freesurfer_queries.qry).
@pytest.mark.parametrize(
    "name,terms",
    [
        ("ctx-lh-precentral", {"precentral", "frontal"}),
        ("ctx-rh-temporalpole", {"temporalpole", "temporal"}),
        ("ctx-lh-lateralorbitofrontal",
         {"lateralorbitofrontal", "orbitofrontalgyrus", "frontal"}),
        ("ctx-rh-rostralmiddlefrontal",
         {"rostralmiddlefrontal", "middle_frontal_gyrus", "frontal"}),
        ("ctx-lh-parsopercularis",
         {"parsopercularis", "inferior_frontal_gyrus", "frontal"}),
        ("ctx-lh-precuneus",
         {"precuneus", "parietal", "superior_parietal_lobule"}),
        ("ctx-lh-supramarginal",
         {"supramarginal", "parietal", "inferior_parietal_lobule"}),
        ("ctx-lh-lingual", {"lingual", "occipital"}),
        ("ctx-lh-isthmuscingulate", {"isthmuscingulate", "cingular"}),
        ("ctx-lh-bankssts", {"bankssts", "temporal"}),
        ("Left-Thalamus-Proper", {"thalamus_proper", "subcortical"}),
        ("Left-Thalamus", {"thalamus_proper", "subcortical"}),
        ("Right-Putamen", {"putamen", "subcortical"}),
        ("Brain-Stem", {"brain_stem", "subcortical"}),
        ("Left-Cerebellum-Cortex", {"cerebellum_cortex", "cerebellum"}),
        ("Left-Lateral-Ventricle", {"lateral_ventricle", "ventricles"}),
        ("CC_Anterior", {"corpus_callosum"}),
    ],
)
def test_freesurfer_aseg_aparc_terms(name, terms):
    mapping = load_term_mapping("freesurfer_aseg_aparc")
    assert set(mapping.loc[mapping.atlas_label_name == name, "term"]) == terms


def test_freesurfer_aseg_aparc_covers_the_desikan_atlas():
    mapping = load_term_mapping("freesurfer_aseg_aparc")
    names = set(mapping["atlas_label_name"])
    desikan = (
        "bankssts caudalanteriorcingulate caudalmiddlefrontal cuneus "
        "entorhinal fusiform inferiorparietal inferiortemporal "
        "isthmuscingulate lateraloccipital lateralorbitofrontal lingual "
        "medialorbitofrontal middletemporal parahippocampal paracentral "
        "parsopercularis parsorbitalis parstriangularis pericalcarine "
        "postcentral posteriorcingulate precentral precuneus "
        "rostralanteriorcingulate rostralmiddlefrontal superiorfrontal "
        "superiorparietal superiortemporal supramarginal frontalpole "
        "temporalpole transversetemporal insula"
    ).split()
    assert len(desikan) == 34
    for hemi in ("lh", "rh"):
        assert {f"ctx-{hemi}-{n}" for n in desikan} <= names


def test_mapping_hemispheres_agree_with_the_names():
    mapping = load_term_mapping("freesurfer_aseg_aparc")
    starts = mapping.atlas_label_name.str.startswith
    lh = mapping[starts(("Left-", "ctx-lh-"))]
    rh = mapping[starts(("Right-", "ctx-rh-"))]
    assert set(lh["hemi"]) == {"left"} and set(rh["hemi"]) == {"right"}


def test_destrieux_has_both_naming_schemes():
    mapping = load_term_mapping("freesurfer_destrieux")
    names = set(mapping["atlas_label_name"])
    # FreeSurfer's colour LUT and nilearn's lateralised atlas
    for stem in ("G_precentral", "S_front_sup", "Pole_temporal"):
        assert {f"ctx_lh_{stem}", f"ctx_rh_{stem}"} <= names
        assert {f"L {stem}", f"R {stem}"} <= names
    left = mapping[mapping.atlas_label_name == "L G_precentral"]
    assert {"precentral", "frontal"} <= set(left["term"])
    assert set(left["hemi"]) == {"left"}
    pole = mapping[mapping.atlas_label_name == "ctx_rh_Pole_temporal"]
    assert {"temporalpole", "temporal"} <= set(pole["term"])


def test_destrieux_and_desikan_share_terms():
    d = set(load_term_mapping("freesurfer_destrieux")["term"])
    a = set(load_term_mapping("freesurfer_aseg_aparc")["term"])
    assert {"frontal", "temporal", "parietal", "occipital", "cingular",
            "precentral", "postcentral", "temporalpole"} <= d & a


def test_destrieux_boundaries_are_left_unmapped_on_purpose():
    names = set(load_term_mapping("freesurfer_destrieux")["atlas_label_name"])
    for stem in ("S_central", "Lat_Fis-post", "Medial_wall", "Unknown"):
        assert not {f"L {stem}", f"ctx_lh_{stem}"} & names


def _freesurfer_lut():
    for home in (os.environ.get("FREESURFER_HOME"), "/Applications/freesurfer",
                 "/usr/local/freesurfer"):
        path = home and os.path.join(home, "FreeSurferColorLUT.txt")
        if path and os.path.exists(path):
            return path
    return None


needs_freesurfer = pytest.mark.skipif(
    _freesurfer_lut() is None, reason="no FreeSurfer installation"
)


@needs_freesurfer
def test_shipped_names_exist_in_the_freesurfer_lut():
    """Against the real colour LUT: no typo, no outdated spelling."""
    lut = load_label_table_freesurfer_lut(_freesurfer_lut())
    names = set(lut["name"])
    for key, prefix in (("freesurfer_aseg_aparc", ""),
                        ("freesurfer_destrieux", "ctx_")):
        mapped = set(load_term_mapping(key)["atlas_label_name"])
        assert sorted(n for n in mapped if n.startswith(prefix)
                      and n not in names) == []


@needs_freesurfer
def test_the_standard_aseg_aparc_and_a2009s_labels_are_all_mapped():
    lut = load_label_table_freesurfer_lut(_freesurfer_lut())
    aseg = [
        0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28,
        30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62,
        63, 72, 77, 85, 251, 252, 253, 254, 255,
    ]
    wanted = (
        set(aseg)
        | set(range(1001, 1036)) | set(range(2001, 2036))
        | set(range(11101, 11176)) | set(range(12101, 12176))
    )
    by_id = dict(zip(lut["id"], lut["name"]))
    mapped = set()
    for key in SHIPPED_MAPPINGS:
        mapped |= set(load_term_mapping(key)["atlas_label_name"])
    # no term, on purpose: unknown, and the boundaries of Destrieux
    left_out = re.compile(
        r"Unknown|ctx-[lr]h-unknown|ctx_[lr]h_(Lat_Fis|S_central|"
        r"S_parieto_occipital|Medial_wall)"
    )
    missing = sorted(
        by_id[i] for i in wanted
        if by_id[i] not in mapped and not left_out.search(by_id[i])
    )
    assert missing == []
