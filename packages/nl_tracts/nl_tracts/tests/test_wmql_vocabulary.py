"""The term vocabulary against tract_querier's own definitions.

``freesurfer_queries.qry`` (with the ``FreeSurfer.qry`` it imports) defines
each region name, and the groups of them (``frontal``, ``temporal``, ...), as
sets of FreeSurfer label ids. Transpiled to Datalog by ``nl_tracts.wmql`` and
evaluated by NeuroLang, its label sets are compared with those of the
shipped mapping ``freesurfer_aseg_aparc`` (``label_term``, over an atlas
built from the colour LUT). What differs is stated here, exactly: a change of
the mapping or of the file shows up as a failure.

Needs tract_querier's data files (``TRACT_QUERIER_DIR``) and the FreeSurfer
colour LUT the differences were recorded for; skipped otherwise. About a
minute, so ``slow`` (``pytest --runslow``).
"""

import ast
import os

import numpy as np
import pandas as pd
import pytest

from nl_tracts.atlas import Atlas, load_label_table_freesurfer_lut
from nl_tracts.wmql import transpile

from .reference_tq import data_file, reference_dir

pytestmark = pytest.mark.slow

# the colour LUT of FreeSurfer 6; the ids that differ below are those of it
LUT_VERSION = "1.105 2015/07/29"


def freesurfer_lut():
    for home in (os.environ.get("FREESURFER_HOME"), "/Applications/freesurfer",
                 "/usr/local/freesurfer"):
        path = home and os.path.join(home, "FreeSurferColorLUT.txt")
        if path and os.path.exists(path):
            with open(path) as f:
                if LUT_VERSION in f.readline():
                    return path
    pytest.skip(f"no FreeSurfer colour LUT {LUT_VERSION}")


def ids(*names):
    return set(names)


# The differences between the file's label sets and ours, ``name:
# (ids only the file has, ids only we have)``, by cause. Nothing else
# differs: the 155 terms, 34 Desikan regions with their cortex and white
# matter, the groups and the aseg structures, agree.
EXPECTED = {
    # 1. The file's unlateralised names are single labels of an old LUT
    # (ids 213-221: Amygdala, Cerebral_Cortex, ...), not the union of the
    # two sides. A term of ours without a side is the union of both. The
    # sided names, `.left` and `.right`, agree.
    "amygdala": (ids(218), ids(18, 54)),
    "cerebral_cortex": (ids(220), ids(3, 42)),
    "cerebral_white_matter": (ids(219), ids(2, 41)),
    "inf_lat_vent": (ids(221), ids(5, 44)),
    "lateral_ventricle": (ids(213), ids(4, 43)),
    "non_wm_hypointensities": (ids(), ids(81, 82)),
    "wm_hypointensities": (ids(), ids(78, 79)),
    # `ventricles` has the old unlateralised ventricle (213) and ids 75 and
    # 76, which this LUT does not have (Lateral_Ventricles.left/right).
    "ventricles": (ids(75, 76, 213), ids()),
    # 2. corpus_callosum: the file's is label 86 (a LUT that has since
    # given the id to something else: here Left_future_WMSA) and the labels
    # of aparc+aseg's extension, which we have; ours also has the five
    # subdivisions of the modern aseg, CC_Posterior ... CC_Anterior.
    "corpus_callosum": (ids(86), ids(251, 252, 253, 254, 255)),
    # 3. subcortical, completed (caudate, hippocampus, accumbens) and with
    # the legacy thalamus id; the file's `.left` also has the brain stem
    # (16), which has no side: in a rule, `subcortical.left` is the term
    # with hemi left *or none*.
    "subcortical.left": (ids(16), ids(9, 11, 17, 26)),
    "subcortical.right": (ids(16), ids(48, 50, 53, 58)),
    # 4. In this LUT `Left-Thalamus` (9) is the legacy whole thalamus; in
    # FreeSurfer 7 that name is the thalamus proper (10). Raw names being
    # the key, ours maps it to thalamus_proper, which no aseg volume uses
    # (it has 10) and which FreeSurfer 7 needs.
    "thalamus_proper.left": (ids(), ids(9)),
    "thalamus_proper.right": (ids(), ids(48)),
}

# What the file defines besides labels and groups of labels: the bundles
# (queries over streamlines), and `hemisphere`, which is `label.hemi`.
NOT_TERMS = {
    "af", "cb", "cc", "cc_1", "cc_2", "cc_3", "cc_4", "cc_5", "cc_6", "cc_7",
    "cst", "cst_post", "cst_pre", "ec", "extreme_capsule", "hemisphere",
    "ilf", "internal_capsule", "internal_capsule_new", "ioff", "mdlf",
    "mdlf_probable_sections", "precentral_postcentral", "slf_i", "slf_ii",
    "slf_iii", "soff", "striato_frontal", "striato_occipital",
    "striato_parietal", "thalamo_frontal", "thalamo_occipital",
    "thalamo_parietal", "uf",
}


@pytest.fixture(scope="module")
def qry():
    data = os.path.dirname(data_file("freesurfer_queries.qry"))
    return transpile(
        "import freesurfer_queries.qry\n", atlas="fs",
        include_folders=[data], skip_unsupported=True,
    )


@pytest.fixture(scope="module")
def ours():
    table = load_label_table_freesurfer_lut(freesurfer_lut())
    volume = table["id"].to_numpy().reshape(-1, 1, 1).astype(np.int32)
    atlas = Atlas(
        "fs", volume, np.eye(4), table, "lut",
        term_mapping="freesurfer_aseg_aparc",
    )
    terms = atlas.label_term_relation().merge(
        atlas.label_relation()[["l", "hemi"]], on="l"
    )
    return terms


@pytest.fixture(scope="module")
def differences(qry, ours):
    """{name: (ids of the file only, ids of ours only)}."""
    from neurolang.frontend import NeurolangDL

    terms = sorted(ours["term"].unique())
    names = [
        n for term in terms for n in (term, f"{term}.left", f"{term}.right")
        if n in qry.defined
    ]
    nl = NeurolangDL()
    # the label sets of a tractogram: one, and no streamlines
    nl.add_tuple_set(pd.DataFrame({"t": ["lut"]}), name="tractogram")
    for relation, columns in (
        ("streamline", "xts"), ("streamline_label", "axl"),
        ("endpoint_label", "axel"),
        ("label_extent", ["a", "l", "axis", "lo", "hi"]),
        ("streamline_extent", ["x", "axis", "lo", "hi"]),
        ("endpoint_coord", ["x", "e", "axis", "v"]),
    ):  # not read by label sets; the program may mention them
        nl.add_tuple_set(
            pd.DataFrame({c: pd.Series(dtype="int64") for c in columns}),
            name=relation,
        )
    nl.execute_datalog_program(qry.program(*names, components=("labels",)))
    solution = nl.solve_all()

    out = {}
    for name in names:
        term, _, side = name.partition(".")
        mine = ours[ours["term"] == term]
        if side:
            mine = mine[mine["hemi"] == side]
        pred = qry.defined[name].labels
        theirs = {
            l for (_, l) in solution[pred].as_pandas_dataframe().itertuples(
                index=False, name=None
            )
        } if pred in solution else set()
        mine = set(mine["l"])
        if theirs != mine:
            out[name] = (theirs - mine, mine - theirs)
    return out


def test_the_whole_file_is_transpiled(qry):
    """The four directional bundles (af, ec, ilf, uf) included."""
    assert qry.skipped == {}
    assert {"af.left", "ec.right", "ilf.left", "uf.right"} <= set(qry.defined)
    assert len(qry.defined) > 1500


def test_every_term_of_the_mapping_is_defined_by_the_file(qry, ours):
    terms = set(ours["term"])
    defined = {n.split(".")[0] for n in qry.defined}
    # the terms we have that the file does not know are none: all the
    # shipped terms are the file's
    assert sorted(terms - defined) == []


def test_every_label_group_of_the_file_is_a_term(qry, ours):
    """What the file defines is a term, a bundle or `hemisphere`."""
    data = reference_dir()
    tree = ast.parse(open(os.path.join(data, "data",
                                       "freesurfer_queries.qry")).read())
    defined_here = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            target = (node.targets[0] if isinstance(node, ast.Assign)
                      else node.target)
            if isinstance(target, ast.Name):
                defined_here.add(target.id.lower())
            elif isinstance(target, ast.Attribute):
                defined_here.add(target.value.id.lower())
    assert sorted(defined_here - set(ours["term"]) - NOT_TERMS) == []
    assert NOT_TERMS <= defined_here | {"cst"}


def test_the_vocabulary_differs_from_the_file_exactly_as_stated(differences):
    assert differences == EXPECTED


def test_the_aparc_regions_are_the_cortex_and_the_white_matter(qry, ours):
    """The point of the largest class of differences found: `precentral` of
    the file is the cortex 1024 and the white matter 3024 under it."""
    mine = ours[(ours["term"] == "precentral") & (ours["hemi"] == "left")]
    assert set(mine["l"]) == {1024, 3024}
    cortex = ours[
        (ours["term"] == "ctx_precentral") & (ours["hemi"] == "left")
    ]
    assert set(cortex["l"]) == {1024}
