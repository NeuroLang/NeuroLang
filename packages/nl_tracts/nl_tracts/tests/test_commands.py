"""The commands of a ``.dl`` file that build the EDB from files:
``.load_tract_atlas``, ``.load_tractogram`` and ``.index_tracts``, run by a
NeurolangPDL with the solver of nl_tracts."""

import nibabel as nib
import numpy as np
import pytest

from nl_expr.exceptions import (
    InvalidCommandExpression,
    UnsupportedProgramError,
)
from nl_tracts.atlas import Atlas
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
    return (f'.load_tract_atlas("fs", "{files}/labels.nii.gz", '
            f'"{files}/lut.txt", space="s"{options})\n')


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
        + 'ans(s, l) :- traverses("fs", "t1", s, l)\n'
    )
    traverses, _ = expected(files, crossing_threshold=2.0)
    assert rows(result) == {(s, l) for s, l in zip(traverses["s"],
                                                   traverses["l"])}
    assert len(rows(result)) > 20


def test_every_relation_of_the_schema_is_there(files):
    nl = tracts_engine()
    nl.execute_datalog_program(
        atlas_command(files, term_mapping="freesurfer_aseg_aparc")
        + tractogram_command(files)
        + '.index_tracts("t1", "fs")\n'
    )
    names = {s.name for s in nl.program_ir.extensional_database()}
    assert {
        "atlas", "label", "label_term", "label_extent", "tractogram",
        "streamline", "streamline_extent", "endpoint_coord",
        "streamline_length", "traverses", "endpoint", "opposite_hemisphere",
    } <= names


def test_the_hemispheres_and_terms_of_the_atlas_are_queryable(files):
    nl = tracts_engine()
    result = nl.execute_datalog_program(
        atlas_command(files, term_mapping="freesurfer_aseg_aparc")
        + tractogram_command(files) + '.index_tracts("t1", "fs")\n'
        + 'ans(n, h) :- label("fs", l, n, h), label_term("fs", l, "putamen")\n'
    )
    assert rows(result) == {("Left-Putamen", "left")}


def test_the_parameters_reach_the_indexing(files):
    nl = tracts_engine()
    result = nl.execute_datalog_program(
        atlas_command(files) + tractogram_command(files)
        + '.index_tracts("t1", "fs", crossing_threshold=30.0, '
        + "emit_background=1)\n"
        + 'ans(s, l) :- traverses("fs", "t1", s, l)\n'
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
        + "ans(t, s) :- streamline(t, s)\n"
    )
    got = rows(result)
    for ident, name in (("t1", "a"), ("t2", "b")):
        tg = Tractogram.from_file(ident, files / f"{name}.tck", "s")
        kept = np.flatnonzero(tg.lengths_in_mm() >= 10.0)
        assert {s for t, s in got if t == ident} == set(kept.tolist())


def test_a_cache_directory_is_used(files, tmp_path):
    cache = tmp_path / "cache"
    program = (
        atlas_command(files) + tractogram_command(files)
        + f'.index_tracts("t1", "fs", cache="{cache}")\n'
        + 'ans(s, l) :- traverses("fs", "t1", s, l)\n'
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
        ('.load_tract_atlas("a", "/no/such.nii.gz", "/no/lut.txt")\n',
         "atlas"),
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
