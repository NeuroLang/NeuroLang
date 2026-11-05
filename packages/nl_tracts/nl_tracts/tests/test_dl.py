"""The ``.dl`` file written from a WMQL program: the commands that build the
EDB from the files, the rules of the query with a comment on each statement
they come from, and the ``ans`` rule."""

import re

import pytest

from nl_tracts.dl import AtlasSource, TractogramSource, datalog_file
from nl_tracts.wmql import WMQLError

ATLAS = AtlasSource("fs", "labels.nii.gz", "lut.txt", space="subject:1")
TRACTS = [TractogramSource("100307", "tracts.trk", space="subject:1")]
SOURCE = (
    "amy.left = 18\namy.right = 54\ntmp.left = 5\ntmp.right = 6\n"
    "uf.side = endpoints_in(tmp.side and anterior_of(amy.side))"
    " not in only(tmp.opposite)\n"
)


def make(source=SOURCE, **kw):
    kw.setdefault("atlas", ATLAS)
    kw.setdefault("tractograms", TRACTS)
    return datalog_file(source, **kw)


def lines(text):
    return text.strip().split("\n")


def test_the_commands_come_first_then_the_rules_then_the_answer():
    out = lines(make())
    commands = [i for i, x in enumerate(out) if x.startswith(".")]
    rules = [i for i, x in enumerate(out) if re.match(r"q[i_]", x)]
    assert len(commands) == 3
    assert commands[-1] < rules[0]
    assert out[-1] == "ans(t, s) :- q_uf_right(t, s)"  # the last saved name


def test_the_commands_load_the_atlas_and_the_tractogram_and_index():
    out = make()
    assert (
        '.load_tract_atlas("fs", "labels.nii.gz", "lut.txt", '
        'kind="freesurfer", space="subject:1")\n' in out
    )
    assert (
        '.load_tractogram("100307", "tracts.trk", space="subject:1")\n' in out
    )
    assert (
        '.index_tracts("100307", "fs", crossing_threshold=2.0, '
        "length_threshold=0.0, emit_background=1)\n" in out
    )


def test_the_background_is_in_traverses_unless_asked_otherwise():
    """WMQL's ``not`` has the background in its universe (6.3, 11.2)."""
    assert "emit_background=1" in make()
    assert "emit_background=0" in make(emit_background=False)


def test_the_options_reach_the_commands():
    out = make(
        crossing_threshold=25.0, length_threshold=40.0, cache="/tmp/c",
        atlas=AtlasSource("fs", "i", "t", kind="bids", space="s",
                          term_mapping="freesurfer_aseg_aparc",
                          background=3),
    )
    assert 'kind="bids"' in out
    assert 'term_mapping="freesurfer_aseg_aparc"' in out
    assert "background=3" in out
    assert "crossing_threshold=25.0, length_threshold=40.0" in out
    assert 'cache="/tmp/c"' in out


def test_several_tractograms_are_loaded_and_indexed_each():
    out = make(tractograms=[
        TractogramSource("a", "a.trk", "s"),
        TractogramSource("b", "b.tck", "s"),
    ])
    assert out.count(".load_tractogram(") == 2
    assert out.count(".index_tracts(") == 2
    assert '.index_tracts("b", "fs"' in out


def test_the_answer_is_chosen_by_name():
    assert lines(make(answer="uf.left"))[-1] == (
        "ans(t, s) :- q_uf_left(t, s)")
    assert lines(make(answer="amy.left"))[-1] == (
        "ans(t, s) :- q_amy_left(t, s)")
    with pytest.raises(WMQLError, match="nope"):
        make(answer="nope")


def test_a_program_that_saves_nothing_has_no_default_answer():
    with pytest.raises(ValueError, match="answer"):
        make("a |= 1\n")
    assert "q_a(t, s)" in make("a |= 1\n", answer="a")


def test_each_statement_is_a_comment_over_the_rules_it_gives():
    out = lines(make(answer="uf.left"))
    # the statement as it is run, ``.side`` expanded
    assert "% amy.left = 18" in out
    expanded = [x for x in out if x.startswith("% uf.left = ")]
    assert expanded == [
        "% uf.left = endpoints_in(tmp.left and anterior_of(amy.left)) "
        "not in only(tmp.right)"
    ]
    i = out.index(expanded[0])
    assert out[i + 1].startswith("qi_")  # the rules of it follow


def test_only_what_the_answer_needs_is_written():
    out = make("a = 1\nb = 2\nx = a or b\ny = 3\n", answer="x")
    assert "q_y" not in out
    assert 'traverses("fs", t, s, 3)' not in out
    # the streamlines only: no rule for the labels of a, b or x
    assert "__label" not in out


def test_the_file_parses_with_the_frontends_syntax():
    from nl_expr.expressions import Command
    from neurolang.frontend.datalog.standard_syntax import parser

    program = parser(make())
    commands = [f for f in program.formulas if isinstance(f, Command)]
    assert [c.name.name for c in commands] == [
        "load_tract_atlas", "load_tractogram", "index_tracts"]
    assert len(program.formulas) > 15


def test_the_header_says_what_it_is_and_how_to_run_it():
    out = make(title="uf.qry")
    head = [x for x in lines(out) if x.startswith("%")][:6]
    assert any("uf.qry" in x for x in head)
    assert any("nl-tracts run" in x for x in head)
    assert any("fs" in x for x in head)


@pytest.mark.parametrize(
    "bad", ['quo"te', "back\\slash", "new\nline"],
)
def test_values_that_would_break_a_string_are_refused(bad):
    with pytest.raises(ValueError, match="string"):
        make(tractograms=[TractogramSource("t", bad, "s")])
    with pytest.raises(ValueError, match="string"):
        make(atlas=AtlasSource("fs", bad, "lut", space="s"))


def test_directional_terms_and_every_construct_are_written():
    out = make("a = 1\nx = anterior_of(a)\ny = only(a or x)\n", answer="y")
    assert "label_extent(" in out
    assert "(hi > box)" in out


def test_comments_are_off_by_default_in_the_transpiled_text():
    from nl_tracts.wmql import transpile

    p = transpile(SOURCE, atlas="fs")
    assert "%" not in p.text
    assert "%" not in p.program("uf.left")
    assert "% uf.left" in p.program("uf.left", comments=True)
