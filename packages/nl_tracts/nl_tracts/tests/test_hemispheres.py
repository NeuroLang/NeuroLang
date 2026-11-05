import pandas as pd
import pytest

from nl_tracts.atlas import (
    HEMISPHERE_RULES,
    HemisphereRule,
    infer_hemispheres,
)


@pytest.mark.parametrize(
    "name,hemi",
    [
        # FreeSurfer, subcortical and cortical, and a2009s
        ("Left-Thalamus-Proper", "left"),
        ("Right-Thalamus-Proper", "right"),
        ("ctx-lh-precentral", "left"),
        ("ctx-rh-precentral", "right"),
        ("ctx_lh_G_and_S_frontomargin", "left"),
        ("ctx_rh_G_and_S_frontomargin", "right"),
        # suffixes: AAL
        ("Precentral_L", "left"),
        ("Precentral_R", "right"),
        # lh. / rh.
        ("lh.superiorfrontal", "left"),
        ("rh.superiorfrontal", "right"),
        # .left / .right
        ("precentral.left", "left"),
        ("precentral.right", "right"),
        # nilearn's Destrieux, and Harvard-Oxford
        ("L G_and_S_frontomargin", "left"),
        ("R Medial_wall", "right"),
        ("Left Thalamus", "left"),
        ("Right Hippocampus", "right"),
        # Tian's subcortex, raw and as an engine of the apps names it
        ("aHIP-rh", "right"),
        ("THA-DA-lh", "left"),
        ("NAc shell lh", "left"),
        ("pPUT rh", "right"),
        # no hemisphere
        ("Brain-Stem", "none"),
        ("CC_Anterior", "none"),
        ("3rd-Ventricle", "none"),
        ("Background", "none"),
        # looks like a hemisphere token, is not one
        ("Lateral-Ventricle", "none"),
        ("Leftover", "none"),
        ("Rightmost_area", "none"),
        ("SPL", "none"),
    ],
)
def test_default_rules(name, hemi):
    got = infer_hemispheres([name], HEMISPHERE_RULES)
    assert got.tolist() == [hemi]


def test_returns_a_series_in_the_order_of_the_names():
    names = ["Left-Putamen", "Brain-Stem", "Right-Putamen"]
    got = infer_hemispheres(names)
    assert isinstance(got, pd.Series)
    assert got.tolist() == ["left", "none", "right"]


def test_the_first_matching_rule_wins():
    rules = [
        HemisphereRule(r"X_.+", "right"),
        HemisphereRule(r".+_Y", "left"),
    ]
    assert infer_hemispheres(["X_Y"], rules).tolist() == ["right"]


def test_rules_are_configurable():
    rules = [
        HemisphereRule(r"7Networks_LH_.+", "left"),
        HemisphereRule(r"7Networks_RH_.+", "right"),
    ]
    got = infer_hemispheres(
        ["7Networks_LH_Vis_1", "7Networks_RH_Vis_1", "Left-Putamen"], rules
    )
    # the default conventions are not applied when rules are given
    assert got.tolist() == ["left", "right", "none"]


def test_no_rules_means_no_hemispheres():
    assert infer_hemispheres(["Left-Putamen"], []).tolist() == ["none"]


def test_patterns_must_match_the_whole_name():
    got = infer_hemispheres(["xLeft-Putamen", "Left-Putamen "])
    assert got.tolist() == ["none", "left"]  # trailing text is after .+


def test_a_rule_with_a_bad_hemisphere_is_rejected():
    with pytest.raises(ValueError, match="hemi"):
        HemisphereRule(r".+_L", "lft")
