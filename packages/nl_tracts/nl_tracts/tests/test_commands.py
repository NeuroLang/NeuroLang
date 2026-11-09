"""The commands of a ``.dl`` file that build the EDB from files:
``.load_tractogram`` and ``.index_tracts``, over an atlas of the engine
(``.load_atlas``, ``add_atlas_set``), run by a NeurolangPDL with the solver
of nl_tracts."""

from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from nl_expr.exceptions import (
    InvalidCommandExpression,
    UnsupportedProgramError,
)
from neurolang.atlases import Atlas
from nl_tracts.frontend import tracts_engine
from nl_tracts.indexing import (
    IndexingParams,
    endpoint_relation,
    traverses_relation,
)
from nl_tracts.tractogram import Tractogram

from .conftest import AFFINES, synthetic_volume

LUT = """\
0   Unknown          0 0 0 0
1   Left-Putamen     1 1 1 0
2   Right-Thalamus   1 1 1 0
3   Brain-Stem       1 1 1 0
4   ctx-lh-precuneus 1 1 1 0
"""


def lines(seed=0, n=40):
    rng = np.random.default_rng(seed)
    return [
        # around label 1 of the synthetic volume (x -7..-1, y -18.5..-9.5)
        (rng.normal(size=(rng.integers(2, 20), 3)) * 6
         + [-4, -14, -26]).astype(np.float32)
        for _ in range(n)
    ]


@pytest.fixture
def files(tmp_path):
    image = nib.Nifti1Image(synthetic_volume(), AFFINES["aligned"])
    nib.save(image, tmp_path / "labels.nii.gz")
    (tmp_path / "lut.txt").write_text(LUT)
    for name, seed in (("a", 0), ("b", 1)):
        nib.streamlines.save(
            nib.streamlines.Tractogram(lines(seed),
                                       affine_to_rasmm=np.eye(4)),
            str(tmp_path / f"{name}.tck"),
        )
    return tmp_path


def atlas_command(files, **extra):
    options = "".join(f', {k}="{v}"' for k, v in extra.items())
    return (f'.load_atlas("fs", "{files}/labels.nii.gz", '
            f'"{files}/lut.txt", kind="freesurfer", space="s"{options})\n')


def tractogram_command(files, name="a", ident="t1"):
    return f'.load_tractogram("{ident}", "{files}/{name}.tck", space="s")\n'


def rows(result):
    return {tuple(r) for r in result.as_pandas_dataframe().itertuples(
        index=False, name=None)}


def expected(files, ident="t1", name="a", **params):
    atlas = Atlas.from_freesurfer(
        "fs", files / "labels.nii.gz", files / "lut.txt", "s")
    tg = Tractogram.from_file(ident, files / f"{name}.tck", "s")
    p = IndexingParams(**params)
    return traverses_relation(tg, atlas, p), endpoint_relation(tg, atlas, p)


def test_the_commands_build_the_relations(files):
    nl = tracts_engine()
    result = nl.execute_datalog_program(
        atlas_command(files) + tractogram_command(files)
        + '.index_tracts("t1", "fs", crossing_threshold=2.0)\n'
        + 'ans(s, l) :- streamline_label("fs", x, l), '
        + 'streamline(x, "t1", s)\n'
    )
    traverses, _ = expected(files, crossing_threshold=2.0)
    assert rows(result) == {(s, l) for s, l in zip(traverses["s"],
                                                   traverses["l"])}
    assert len(rows(result)) > 20


def test_every_relation_of_the_schema_is_there(files):
    nl = tracts_engine()
    nl.execute_datalog_program(
        atlas_command(files) + tractogram_command(files)
        + '.index_tracts("t1", "fs")\n'
    )
    names = {s.name for s in nl.program_ir.extensional_database()}
    assert {
        "atlas", "label", "label_term", "label_extent", "tractogram",
        "streamline", "streamline_extent", "endpoint_coord",
        "streamline_length", "streamline_label", "endpoint_label",
        "opposite_hemisphere",
    } <= names
    # the names of a vocabulary of rules, not of stored relations
    assert not {"traverses", "endpoint"} & names


def test_the_hemispheres_and_terms_of_the_atlas_are_queryable(files):
    """The terms of a FreeSurfer atlas: freesurfer_aseg_aparc, the default
    term mapping of its kind."""
    nl = tracts_engine()
    result = nl.execute_datalog_program(
        atlas_command(files) + tractogram_command(files)
        + '.index_tracts("t1", "fs")\n'
        + 'ans(n, h) :- label("fs", l, n, h), label_term("fs", l, "putamen")\n'
    )
    assert rows(result) == {("Left-Putamen", "left")}


def test_a_term_mapping_of_index_tracts(files):
    (files / "terms.tsv").write_text(
        "atlas_label_name\tterm\themi\nBrain-Stem\tstem\tleft\n"
    )
    nl = tracts_engine()
    result = nl.execute_datalog_program(
        atlas_command(files) + tractogram_command(files, "a", "t1")
        + tractogram_command(files, "b", "t2")
        + f'.index_tracts("t1", "fs", term_mapping="{files}/terms.tsv")\n'
        + 'ans(n, h, t) :- label("fs", l, n, h), label_term("fs", l, t)\n'
    )
    # its hemi is a hint, over the inference
    assert rows(result) == {("Brain-Stem", "left", "stem")}
    nl.execute_datalog_program('.index_tracts("t2", "fs")\n')
    with pytest.raises(InvalidCommandExpression, match="term_mapping"):
        nl.execute_datalog_program(
            '.index_tracts("t2", "fs", term_mapping="freesurfer_destrieux")'
            "\n"
        )


def test_a_term_mapping_that_is_not_one(files):
    with pytest.raises(InvalidCommandExpression, match="term mapping"):
        tracts_engine().execute_datalog_program(
            atlas_command(files) + tractogram_command(files)
            + '.index_tracts("t1", "fs", term_mapping="no_such_mapping")\n'
        )


def test_the_parameters_reach_the_indexing(files):
    nl = tracts_engine()
    result = nl.execute_datalog_program(
        atlas_command(files) + tractogram_command(files)
        + '.index_tracts("t1", "fs", crossing_threshold=30.0, '
        + "emit_background=1)\n"
        + 'ans(s, l) :- streamline_label("fs", x, l), '
        + 'streamline(x, "t1", s)\n'
    )
    traverses, _ = expected(
        files, crossing_threshold=30.0, emit_background=True)
    assert rows(result) == {(s, l) for s, l in zip(traverses["s"],
                                                   traverses["l"])}
    assert any(l == 0 for _, l in rows(result))  # the background


def test_two_tractograms_and_the_length_threshold(files):
    nl = tracts_engine()
    result = nl.execute_datalog_program(
        atlas_command(files)
        + tractogram_command(files, "a", "t1")
        + tractogram_command(files, "b", "t2")
        + '.index_tracts("t1", "fs", length_threshold=10.0)\n'
        + '.index_tracts("t2", "fs", length_threshold=10.0)\n'
        + "ans(x, t, s) :- streamline(x, t, s)\n"
    )
    got = rows(result)
    for index, (ident, name) in enumerate((("t1", "a"), ("t2", "b"))):
        tg = Tractogram.from_file(ident, files / f"{name}.tck", "s")
        kept = np.flatnonzero(tg.lengths_in_mm() >= 10.0)
        assert {s for x, t, s in got if t == ident} == set(kept.tolist())
        # x is (index << 32) | s, the index the order of .index_tracts
        assert {x for x, t, s in got if t == ident} == {
            index * 2 ** 32 + s for s in kept.tolist()}


def test_a_cache_directory_is_used(files, tmp_path):
    cache = tmp_path / "cache"
    program = (
        atlas_command(files) + tractogram_command(files)
        + f'.index_tracts("t1", "fs", cache="{cache}")\n'
        + 'ans(s, l) :- streamline_label("fs", x, l), '
        + 'streamline(x, "t1", s)\n'
    )
    first = rows(tracts_engine().execute_datalog_program(program))
    assert list(cache.glob("*/traverses.parquet"))
    assert rows(tracts_engine().execute_datalog_program(program)) == first


def test_an_unknown_command_is_still_an_error(files):
    """The mixin is before the catch-all of ``CommandsMixin``, which stays."""
    with pytest.raises(UnsupportedProgramError, match="not supported"):
        tracts_engine().execute_datalog_program(".no_such_command(1)\n")


def test_the_other_commands_still_work(files):
    (files / "t.csv").write_text("x,y\n1,a\n2,b\n")
    nl = tracts_engine()
    result = nl.execute_datalog_program(
        f'.load_csv(Pairs, "{files}/t.csv")\n'
        "ans(x) :- Pairs(x, y)\n"
    )
    assert rows(result) == {(1,), (2,)}


@pytest.mark.parametrize(
    "program,match",
    [
        ('.load_tractogram("t", "/no/such/file.tck", space="s")\n',
         "tractogram"),
        ('.load_atlas("a", "/no/such.nii.gz", "/no/lut.txt", '
         'kind="freesurfer")\n', "atlas"),
        ('.index_tracts("t1", "fs")\n', "t1"),
    ],
)
def test_commands_fail_with_a_message(files, program, match):
    with pytest.raises(InvalidCommandExpression, match=match):
        tracts_engine().execute_datalog_program(program)


def test_an_unknown_kind_of_atlas_is_an_error(files):
    with pytest.raises(InvalidCommandExpression, match="kind"):
        tracts_engine().execute_datalog_program(
            atlas_command(files, kind="nonsense"))


def test_the_spaces_must_agree(files):
    program = (
        atlas_command(files)
        + f'.load_tractogram("t1", "{files}/a.tck", space="other")\n'
        + '.index_tracts("t1", "fs")\n'
    )
    with pytest.raises(InvalidCommandExpression, match="space"):
        tracts_engine().execute_datalog_program(program)


def test_a_pair_is_indexed_once_and_the_length_threshold_is_fixed(files):
    nl = tracts_engine()
    nl.execute_datalog_program(
        atlas_command(files) + tractogram_command(files)
        + '.index_tracts("t1", "fs")\n'
    )
    with pytest.raises(InvalidCommandExpression, match="already"):
        nl.execute_datalog_program('.index_tracts("t1", "fs")\n')


def two_tractograms(files):
    return (
        atlas_command(files)
        + tractogram_command(files, "a", "t1")
        + tractogram_command(files, "b", "t2")
        + '.index_tracts("t1", "fs")\n'
        + '.index_tracts("t2", "fs")\n'
    )


def test_subject_of_decodes_x_in_a_rule(files):
    """The engine's functions read the tractograms its commands indexed."""
    result = tracts_engine().execute_datalog_program(
        two_tractograms(files)
        + 'ans(t) :- streamline_label("fs", x, 1), (t == subject_of(x))\n'
    )
    want = {
        ident for ident, name in (("t1", "a"), ("t2", "b"))
        if (expected(files, ident, name)[0]["l"] == 1).any()
    }
    assert want == {"t1", "t2"}
    assert rows(result) == {(t,) for t in want}


def test_the_functions_agree_with_the_streamline_relation(files):
    result = tracts_engine().execute_datalog_program(
        two_tractograms(files)
        + "ans(x, t, s, y, u, i) :- streamline(x, t, s), "
        + "(y == streamline_id(t, s)), (u == subject_of(x)), "
        + "(i == index_of(x))\n"
    )
    got = rows(result)
    assert len(got) == 80
    assert all(x == y and t == u and s == i for x, t, s, y, u, i in got)
    assert (2 ** 32 + 3, "t2", 3, 2 ** 32 + 3, "t2", 3) in got


# --- an atlas of the engine (the atlases of an engine of the apps) ---------

# The raw names of the labels as nilearn and Tian's repository give them:
# the engine names its regions by them, "-" and "_" as spaces.
ENGINE_ATLASES = {
    "destrieux": ["Background", "L G_and_S_frontomargin", "R Pole_temporal",
                  "L Medial_wall", "R S_front_sup"],
    "tian": ["Background", "lAMY-lh", "aPUT-rh", "NAc-shell-lh", "THA-DA-rh"],
}


def engine_with_atlases():
    """A tracts engine given atlases as the apps' registry gives them: an
    Atlas of their kind, its names normalised, in MNI152, by
    ``add_atlas_set``."""
    nl = tracts_engine()
    image = nib.Nifti1Image(synthetic_volume(), AFFINES["aligned"])
    for name, labels in ENGINE_ATLASES.items():
        nl.add_atlas_set(name, Atlas.from_labels(
            name, image, labels, "MNI152", normalise=True, kind=name,
        ))
    return nl


def index_engine_atlases(files):
    return (
        f'.load_tractogram("t1", "{files}/a.tck", space="MNI152")\n'
        '.index_tracts("t1", "destrieux")\n'
        '.index_tracts("t1", "tian")\n'
    )


def test_an_atlas_of_the_engine_is_indexed_by_its_name(files):
    nl = engine_with_atlases()
    result = nl.execute_datalog_program(
        index_engine_atlases(files)
        + 'ans(s, l) :- streamline_label("destrieux", x, l), '
        + 'streamline(x, "t1", s)\n'
    )
    traverses, _ = expected(files)  # the same volume, the same labels
    assert rows(result) == {(s, l) for s, l in zip(traverses["s"],
                                                   traverses["l"])}


def test_the_labels_of_an_engine_atlas_have_its_names_terms_and_sides(files):
    nl = engine_with_atlases()
    nl.execute_datalog_program(index_engine_atlases(files))
    labels = rows(nl.execute_datalog_program(
        'ans(a, l, n, h) :- label(a, l, n, h)\n'))
    # the names of the relations of the engine, destrieux(name, region)
    regions = {
        (a, n) for a in ENGINE_ATLASES
        for (n,) in rows(nl.execute_datalog_program(
            f"ans(n) :- {a}(n, r)\n"))
    }
    assert {(a, n) for a, _, n, _ in labels} == regions
    assert {(n, h) for _, _, n, h in labels} == {
        ("L G and S frontomargin", "left"), ("R Pole temporal", "right"),
        ("L Medial wall", "left"), ("R S front sup", "right"),
        ("lAMY lh", "left"), ("aPUT rh", "right"),
        ("NAc shell lh", "left"), ("THA DA rh", "right"),
    }
    terms = rows(nl.execute_datalog_program(
        'ans(n, t) :- label(a, l, n, h), label_term(a, l, t)\n'))
    # the shipped mappings of their kinds, keyed by the raw names, match
    # the engine's
    assert {("L G and S frontomargin", "frontal"),
            ("R Pole temporal", "temporalpole"),
            ("lAMY lh", "amygdala"), ("aPUT rh", "putamen"),
            ("aPUT rh", "striatum"), ("NAc shell lh", "accumbens_area"),
            ("THA DA rh", "thalamus_proper")} <= terms
    assert "L Medial wall" not in {n for n, _ in terms}  # no term, on purpose
    # label 1 of both atlases, which the streamlines cross
    both = rows(nl.execute_datalog_program(
        'ans(x) :- traverses(x, "amygdala", "left"), '
        'traverses(x, "frontal", "left")\n'))
    assert both and both == rows(nl.execute_datalog_program(
        'ans(x) :- streamline_label("tian", x, 1)\n'))


def test_an_atlas_the_engine_does_not_have_is_an_error(files):
    nl = engine_with_atlases()
    with pytest.raises(InvalidCommandExpression,
                       match=r"'schaefer'.*\['destrieux', 'tian'\]"):
        nl.execute_datalog_program(
            f'.load_tractogram("t1", "{files}/a.tck", space="MNI152")\n'
            '.index_tracts("t1", "schaefer")\n')


def test_an_engine_atlas_is_in_mni152(files):
    nl = engine_with_atlases()
    with pytest.raises(InvalidCommandExpression, match="space"):
        nl.execute_datalog_program(
            tractogram_command(files) + '.index_tracts("t1", "destrieux")\n')


def test_the_atlas_indexed_is_the_one_loaded_then(files):
    """An atlas loaded again under its name after it was indexed: the
    relations keep the first, and so does the background of .view."""
    nl = tracts_engine()
    nl.execute_datalog_program(
        atlas_command(files) + tractogram_command(files, "a", "t1")
        + tractogram_command(files, "b", "t2")
        + '.index_tracts("t1", "fs")\n')
    first = nl.atlases["fs"]
    nl.execute_datalog_program(
        atlas_command(files) + '.index_tracts("t2", "fs")\n')
    assert nl.atlases["fs"] is not first
    assert nl.program_ir._tracts.atlases["fs"] is first


# --- an atlas of nilearn, by the name of its fetcher -----------------------

@pytest.fixture
def fetchers(monkeypatch):
    """nilearn's fetchers of Destrieux (a ``lut``) and Harvard-Oxford (a
    list of ``labels``), giving the synthetic volume; the options they are
    called with are recorded."""
    datasets = pytest.importorskip("nilearn.datasets")
    image = nib.Nifti1Image(synthetic_volume(), AFFINES["aligned"])
    calls = []

    def destrieux(**options):
        calls.append(("destrieux_2009", options))
        return SimpleNamespace(maps=image, labels=[], lut=pd.DataFrame({
            "index": [0, 1, 2, 3, 4],
            "name": ["Unknown", b"L G_and_S_frontomargin", "R Pole_temporal",
                     "L Medial_wall", "R S_front_sup"]}))

    def harvard_oxford(atlas_name, **options):
        calls.append(("harvard_oxford", {"atlas_name": atlas_name,
                                         **options}))
        return SimpleNamespace(maps=image, labels=[
            "Background", "Left Amygdala", "Right Putamen", "Brain-Stem",
            "Left Thalamus"])

    monkeypatch.setattr(datasets, "fetch_atlas_destrieux_2009", destrieux)
    monkeypatch.setattr(datasets, "fetch_atlas_harvard_oxford",
                        harvard_oxford)
    return calls


def test_an_atlas_of_nilearn_by_its_fetcher(files, fetchers):
    nl = engine_with_atlases()
    nl.execute_datalog_program(
        '.load_atlas("d", kind="nilearn", fetcher="destrieux_2009", '
        'lateralized=1)\n'
        '.load_atlas("ho", kind="nilearn", fetcher="harvard_oxford", '
        'atlas_name="sub-maxprob-thr25-1mm", data_dir="/no/where")\n'
        + index_engine_atlases(files)
        + '.index_tracts("t1", "d")\n.index_tracts("t1", "ho")\n')
    assert fetchers == [
        ("destrieux_2009", {"lateralized": 1}),
        ("harvard_oxford", {"atlas_name": "sub-maxprob-thr25-1mm",
                            "data_dir": "/no/where"})]
    labels = rows(nl.execute_datalog_program(
        'ans(a, n, h) :- label(a, l, n, h)\n'))
    # the names of the engine's Destrieux
    assert {(n, h) for a, n, h in labels if a == "d"} == {
        (n, h) for a, n, h in labels if a == "destrieux"}
    assert {(n, h) for a, n, h in labels if a == "ho"} == {
        ("Left Amygdala", "left"), ("Right Putamen", "right"),
        ("Brain Stem", "none"), ("Left Thalamus", "left")}
    # the term mapping of the fetcher, as for the engine's atlas
    terms = rows(nl.execute_datalog_program(
        'ans(a, l, t) :- label_term(a, l, t)\n'))
    assert {(l, t) for a, l, t in terms if a == "d"} == {
        (l, t) for a, l, t in terms if a == "destrieux"}
    assert not {t for a, _, t in terms if a == "ho"}  # no mapping of it


@pytest.mark.slow
@pytest.mark.network
def test_the_real_destrieux_of_nilearn(files, tmp_path):
    """If nilearn has it in its cache: every term of the uncinate in the
    cortex, and the hemispheres of its names."""
    datasets = pytest.importorskip("nilearn.datasets")
    if not (Path(datasets.get_data_dirs()[0]) / "destrieux_2009").is_dir():
        pytest.skip("nilearn's cache has no Destrieux")
    nl = tracts_engine()
    nl.execute_datalog_program(
        '.load_atlas("d", kind="nilearn", fetcher="destrieux_2009")\n'
        f'.load_tractogram("t1", "{files}/a.tck", space="MNI152")\n'
        '.index_tracts("t1", "d")\n')
    terms = rows(nl.execute_datalog_program(
        'ans(t) :- label_term("d", l, t)\n'))
    assert {(t,) for t in (
        "orbitofrontalgyrus", "middle_frontal_gyrus",
        "inferior_frontal_gyrus", "temporal", "insula", "occipital",
        "parietal", "cingular")} <= terms
    names = rows(nl.execute_datalog_program(
        'ans(n, h) :- label("d", l, n, h)\n'))
    assert {h for _, h in names} == {"left", "right"}
    assert all(n[:2] == {"left": "L ", "right": "R "}[h] for n, h in names)
