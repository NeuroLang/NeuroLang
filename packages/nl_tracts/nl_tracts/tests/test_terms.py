import os
import re

import pandas as pd
import pytest

from neurolang.atlases import load_label_table_freesurfer_lut
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
        "tian_subcortex_s2",
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


# The vocabulary is tract_querier's (data/freesurfer_queries.qry): each
# Desikan name is the cortex and the white matter under it, with ctx_<name>
# and wm_<name> for either alone.
@pytest.mark.parametrize(
    "name,terms",
    [
        ("ctx-lh-precentral", {"precentral", "ctx_precentral", "frontal"}),
        ("wm-lh-precentral", {"precentral", "wm_precentral", "frontal"}),
        ("ctx-rh-temporalpole",
         {"temporalpole", "ctx_temporalpole", "temporal", "temporal_frontal"}),
        ("ctx-lh-lateralorbitofrontal",
         {"lateralorbitofrontal", "ctx_lateralorbitofrontal",
          "orbitofrontalgyrus", "frontal"}),
        ("ctx-rh-rostralmiddlefrontal",
         {"rostralmiddlefrontal", "ctx_rostralmiddlefrontal",
          "middle_frontal_gyrus", "middlefrontalgyrus", "frontal"}),
        ("ctx-lh-parsopercularis",
         {"parsopercularis", "ctx_parsopercularis", "inferior_frontal_gyrus",
          "inferiorfrontalgyrus", "frontal"}),
        ("ctx-lh-parsorbitalis",
         {"parsorbitalis", "ctx_parsorbitalis", "inferior_frontal_gyrus",
          "inferiorfrontalgyrus", "frontal", "frontal_frontal_section"}),
        ("ctx-lh-precuneus",
         {"precuneus", "ctx_precuneus", "parietal",
          "superior_parietal_lobule"}),
        ("ctx-lh-supramarginal",
         {"supramarginal", "ctx_supramarginal", "parietal",
          "inferior_parietal_lobule"}),
        ("ctx-lh-lingual", {"lingual", "ctx_lingual", "occipital"}),
        ("ctx-lh-isthmuscingulate",
         {"isthmuscingulate", "ctx_isthmuscingulate", "cingular",
          "cingular_cortex"}),
        ("wm-rh-isthmuscingulate",
         {"isthmuscingulate", "wm_isthmuscingulate", "cingular"}),
        ("ctx-lh-corpuscallosum",
         {"corpuscallosum", "ctx_corpuscallosum", "cingular",
          "cingular_cortex"}),
        ("ctx-lh-Corpus_callosum", {"corpus_callosum", "ctx_corpus_callosum"}),
        ("wm-rh-Corpus_callosum", {"corpus_callosum", "wm_corpus_callosum"}),
        ("ctx-lh-bankssts", {"bankssts", "ctx_bankssts", "temporal"}),
        ("Left-UnsegmentedWhiteMatter",
         {"unsegmentedwhitematter", "centrum_semiovale"}),
        ("Left-Cerebellum-Exterior", {"cerebellum_exterior", "cerebellum"}),
        ("Left-Thalamus-Proper", {"thalamus_proper", "subcortical"}),
        ("Left-Thalamus", {"thalamus_proper", "subcortical"}),
        ("Right-Putamen", {"putamen", "subcortical", "striatum"}),
        ("Left-Pallidum", {"pallidum", "subcortical", "striatum"}),
        ("Left-Caudate", {"caudate", "subcortical", "striatum"}),
        ("Right-Hippocampus", {"hippocampus", "subcortical"}),
        ("Left-Accumbens-area", {"accumbens_area", "subcortical"}),
        ("Left-Amygdala", {"amygdala", "subcortical"}),
        ("Right-VentralDC", {"ventraldc", "subcortical"}),
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
        assert {f"wm-{hemi}-{n}" for n in desikan} <= names


def test_mapping_hemispheres_agree_with_the_names():
    mapping = load_term_mapping("freesurfer_aseg_aparc")
    starts = mapping.atlas_label_name.str.startswith
    lh = mapping[starts(("Left-", "ctx-lh-", "wm-lh-"))]
    rh = mapping[starts(("Right-", "ctx-rh-", "wm-rh-"))]
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


def test_destrieux_leaves_out_only_the_medial_wall_and_unknown():
    names = set(load_term_mapping("freesurfer_destrieux")["atlas_label_name"])
    for stem in ("Medial_wall", "Unknown"):
        assert not {f"L {stem}", f"ctx_lh_{stem}"} & names


@pytest.mark.parametrize(
    "stem,terms",
    [
        ("S_central", {"precentral", "postcentral", "frontal", "parietal"}),
        ("S_parieto_occipital",
         {"precuneus", "cuneus", "parietal", "occipital"}),
        ("Lat_Fis-post",
         {"superiortemporal", "supramarginal", "temporal", "parietal"}),
        ("Lat_Fis-ant-Horizont",
         {"parsorbitalis", "parstriangularis", "inferior_frontal_gyrus",
          "frontal"}),
        ("Lat_Fis-ant-Vertical",
         {"parstriangularis", "parsopercularis", "inferior_frontal_gyrus",
          "frontal"}),
    ],
)
def test_destrieux_boundaries_have_the_terms_of_both_sides(stem, terms):
    """A boundary structure straddles two regions: it has the terms of both."""
    mapping = load_term_mapping("freesurfer_destrieux")
    for name in (f"ctx_rh_{stem}", f"R {stem}"):
        got = mapping[mapping.atlas_label_name == name]
        assert set(got["term"]) == terms
        assert set(got["hemi"]) == {"right"}


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
    # no term, on purpose: unknown, and Destrieux's medial wall
    left_out = re.compile(r"Unknown|ctx-[lr]h-unknown|ctx_[lr]h_Medial_wall")
    missing = sorted(
        by_id[i] for i in wanted
        if by_id[i] not in mapped and not left_out.search(by_id[i])
    )
    assert missing == []


# The 32 regions of Tian's subcortex at scale 2, as its label file has them
# (Tian_Subcortex_S2_3T_label.txt), and the structure of aseg they are part
# of: their terms are those of the structure in freesurfer_aseg_aparc.
TIAN_S2 = {
    "aHIP": "Hippocampus", "pHIP": "Hippocampus",
    "lAMY": "Amygdala", "mAMY": "Amygdala",
    "THA-DP": "Thalamus", "THA-VP": "Thalamus",
    "THA-VA": "Thalamus", "THA-DA": "Thalamus",
    "NAc-shell": "Accumbens-area", "NAc-core": "Accumbens-area",
    "pGP": "Pallidum", "aGP": "Pallidum",
    "aPUT": "Putamen", "pPUT": "Putamen",
    "aCAU": "Caudate", "pCAU": "Caudate",
}


def test_tian_has_the_terms_of_the_aseg_structures():
    tian = load_term_mapping("tian_subcortex_s2")
    aseg = load_term_mapping("freesurfer_aseg_aparc")
    assert set(tian["atlas_label_name"]) == {
        f"{region}-{h}" for region in TIAN_S2 for h in ("lh", "rh")
    }
    for region, structure in TIAN_S2.items():
        for h, side in (("lh", "Left"), ("rh", "Right")):
            rows = tian[tian.atlas_label_name == f"{region}-{h}"]
            same = aseg[aseg.atlas_label_name == f"{side}-{structure}"]
            assert set(rows["term"]) == set(same["term"]), region
            assert set(rows["hemi"]) == set(same["hemi"]) == {
                "left" if h == "lh" else "right"}
    # the uncinate's subcortical terms are there
    assert {"amygdala", "putamen", "striatum", "subcortical"} <= set(
        tian["term"])
