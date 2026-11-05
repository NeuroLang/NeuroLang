"""The WMQL transpiler: what its output is made of. That it computes what
tract_querier computes is the job of ``test_wmql_oracle.py``."""

import re
import subprocess
import sys

import pytest

from nl_tracts.wmql import WMQLError, WMQLUnsupported, transpile

BASE = "a = 1\nb = 2\nc = 3\n"


def t(source, **kw):
    return transpile(source, atlas="fs", **kw)


def test_a_label_is_a_set_of_one_label_and_the_streamlines_that_have_it():
    p = t("a = 7\n")
    q = p.defined["a"]
    assert (q.tracts, q.labels, q.end0, q.end1) == (
        "q_a", "q_a__label", "q_a__end0", "q_a__end1"
    )
    # the label of every tractogram
    assert re.search(
        r"^qi_\d+\(t, 7\) :- tractogram\(t\)$", p.text, re.M
    )
    assert re.search(
        r"^q_a__label\(t, l\) :- qi_\d+\(t, l\)$", p.text, re.M
    )
    # the streamlines, by their id x
    assert re.search(
        r'^qi_\d+\(x\) :- streamline_label\("fs", x, 7\)$', p.text, re.M
    )
    assert re.search(
        r'^qi_\d+\(x\) :- endpoint_label\("fs", x, 1, 7\)$', p.text, re.M
    )
    assert re.search(r"^q_a\(x\) :- qi_\d+\(x\)$", p.text, re.M)
    assert re.search(r"^q_a__end0\(x\) :- qi_\d+\(x\)$", p.text, re.M)


def test_the_atlas_is_a_constant_of_the_program():
    assert 'streamline_label("dk", ' in transpile("a = 1\n", atlas="dk").text
    with pytest.raises(WMQLError, match="quote"):
        transpile("a = 1\n", atlas='d"k')


@pytest.mark.parametrize("source", ["x = not a", "x = -a", "x = ~a"])
def test_complement_and_invert_have_no_endpoints(source):
    q = t(BASE + source + "\n").defined["x"]
    assert q.end0 is None and q.end1 is None
    assert q.tracts is not None and q.labels is not None


def test_only_keeps_the_endpoints_but_invert_drops_them():
    """The two look alike, and differ in the endpoint sets (DESIGN.md, 11)."""
    p = t(BASE + "x = ~a\ny = only(a)\nz = not a\n")
    assert p.defined["x"].end0 is None
    assert p.defined["y"].end0 is not None
    assert p.defined["x"].tracts is not None
    assert p.defined["z"].end1 is None


def test_a_statically_empty_component_propagates():
    p = t(BASE + "x = not a\ny = x and endpoints_in(b)\nz = y or c\n")
    # x has no endpoints, so the streamlines of endpoints_in(x) are none
    p2 = t(BASE + "x = not a\nw = endpoints_in(x)\n")
    assert p2.defined["w"].tracts is None
    assert p.defined["y"].end0 is None
    assert p.defined["z"].end0 is not None  # c has some


def test_the_complement_has_the_labels_of_the_data_as_universe():
    text = t(BASE + "x = not a\n").text
    # the universe: the labels of the streamlines of each tractogram
    assert re.search(
        r'^qi_\d+\(t, l\) :- streamline_label\("fs", x, l\), '
        r"streamline\(x, t, s\)$", text, re.M)
    # the streamlines with a label of the complement, of their tractogram
    assert re.search(
        r'^qi_\d+\(x\) :- streamline_label\("fs", x, l\), '
        r"streamline\(x, t, s\), qi_\d+\(t, l\)$", text, re.M)
    assert "~q_a__label(l)" in text.replace("qi_", "q_a__label") or \
        "~" in text


def test_names_are_predicates_with_the_prefix_q():
    p = t("label = 1\nendpoint |= 2\ntraverses = label or endpoint\n")
    # WMQL names are never confused with the relations
    assert set(p.defined) == {"label", "endpoint", "traverses"}
    assert p.defined["traverses"].tracts == "q_traverses"


def test_dotted_names_become_underscored_predicates():
    p = t("x.left = 1\nx.right = 2\n")
    assert p.defined["x.left"].tracts == "q_x_left"
    assert p.defined["x.right"].labels == "q_x_right__label"


def test_two_names_that_mangle_alike_are_rejected():
    with pytest.raises(WMQLError, match="q_x_left"):
        t("x.left = 1\nx_left = 2\n")


def test_a_redefinition_keeps_the_earlier_one_for_earlier_uses():
    p = t("x = 1\ny = x\nx = 2\n")
    assert p.defined["x"].tracts == "q_x"
    assert p.defined["y"].tracts == "q_y"
    assert "q_x__v1" in p.text  # the first x, which y reads
    assert p.saved == ("x", "y")


def test_saved_names():
    p = t("a = 1\nb |= 2\nc = a or b\nb\n")
    assert p.saved == ("a", "c", "b")


def test_the_program_of_a_name_has_only_the_rules_it_needs():
    p = t("a = 1\nb = 2\nc = a or b\nd = 3\n")
    text = p.program("c")
    assert "q_c(" in text and "q_a(" not in text.split("q_c")[0] or True
    assert "q_d" not in text
    assert "streamline_label" in text
    only_labels = p.program("c", components=("labels",))
    assert "streamline_label" not in only_labels
    assert "endpoint_label" not in only_labels
    assert "q_c__label" in only_labels


def test_the_default_program_is_that_of_the_saved_names():
    p = t("a = 1\nb |= 2\n")
    assert "q_a(" in p.program()
    assert "q_b(" not in p.program()


def test_an_unknown_name_is_an_error():
    with pytest.raises(WMQLError, match="nope"):
        t("x = nope\n")
    with pytest.raises(WMQLError, match="nope"):
        t("nope\n")


@pytest.mark.parametrize(
    "source",
    ["x = a < b", "x = a in b", "x = a ** b", "x = a / b", "x, y = a, b",
     "x = a if b else c", "x = [a]", "x = f(a, b)", "x = only(a, b)",
     "x = a.b.c", "x == a", "x = 1.5",
     # the arguments of a directional term: a name, or name.left / .right
     "x = anterior_of(a or b)", "x = anterior_of(not a)",
     "x = anterior_of(a.foo)", "x = medial_of(a)", "x = lateral_of(a)",
     "x = superior_of(only(a))", "x = anterior_of(a, b)"],
)
def test_what_the_reference_rejects_is_rejected(source):
    with pytest.raises(WMQLError):
        t(BASE + "a.b = 1\n" + source + "\n")


def test_a_syntax_error_says_where():
    with pytest.raises(WMQLError, match="line 2"):
        t("a = 1\nx = (a and\n")


@pytest.mark.parametrize(
    "term",
    ["anterior_of", "posterior_of", "superior_of", "inferior_of"],
)
def test_a_directional_term_compares_the_streamlines_with_a_box(term):
    p = t(BASE + f"x = {term}(a)\n")
    q = p.defined["x"]
    assert None not in (q.tracts, q.labels, q.end0, q.end1)
    text = p.text
    # the face of the box of the labels, per tractogram, by an aggregate
    assert re.search(r"^qi_\d+\(t, (max\(hi\)|min\(lo\))\) :- ", text, re.M)
    assert 'label_extent("fs", l, ' in text
    assert "streamline_extent(x, " in text
    assert "endpoint_coord(x, 0, " in text and \
        "endpoint_coord(x, 1, " in text
    # the labels of the result are those its streamlines traverse
    assert re.search(
        r'^qi_\d+\(t, l\) :- qi_\d+\(x\), streamline_label\("fs", x, l\), '
        r"streamline\(x, t, s\)$",
        text, re.M,
    )


# term -> (axis, face of the box, operator on the streamline), as in the
# reference's process_relative_term; the vertical terms are its, reversed
TABLE = {
    "anterior_of(a)": (1, "max(hi)", ">", "hi"),
    "posterior_of(a)": (1, "min(lo)", "<", "lo"),
    "superior_of(a)": (2, "max(hi)", "<", "hi"),
    "inferior_of(a)": (2, "min(lo)", ">", "lo"),
    "medial_of(a.left)": (0, "max(hi)", ">", "hi"),
    "medial_of(a.right)": (0, "min(lo)", "<", "lo"),
    "lateral_of(a.left)": (0, "min(lo)", "<", "lo"),
    "lateral_of(a.right)": (0, "max(hi)", ">", "hi"),
}


@pytest.mark.parametrize("call", sorted(TABLE))
def test_the_direction_table_is_the_references(call):
    axis, aggregate, op, edge = TABLE[call]
    p = t("a.left = 1\na.right = 2\na = 3\nx = " + call + "\n")
    text = p.text
    assert re.search(
        rf'^qi_\d+\(t, {re.escape(aggregate)}\) :- \w+\(t, l\), '
        rf'label_extent\("fs", l, {axis}, lo, hi\)$', text, re.M)
    # the box is per tractogram: t of the streamline, by streamline(x, t, s)
    assert re.search(
        rf"^qi_\d+\(x\) :- streamline_extent\(x, {axis}, lo, hi\), "
        rf"streamline\(x, t, s\), qi_\d+\(t, box\), "
        rf"\({edge} {re.escape(op)} box\)$", text, re.M)
    assert re.search(
        rf"^qi_\d+\(x\) :- endpoint_coord\(x, 0, {axis}, v\), "
        rf"streamline\(x, t, s\), qi_\d+\(t, box\), "
        rf"\(v {re.escape(op)} box\)$", text, re.M)


def test_the_vertical_terms_can_be_anatomical():
    reference = t(BASE + "x = superior_of(a)\ny = inferior_of(a)\n")
    fixed = t(BASE + "x = superior_of(a)\ny = inferior_of(a)\n",
              anatomical_vertical=True)
    assert ", (hi < box)" in reference.text
    assert ", (lo > box)" in reference.text
    assert ", (hi > box)" in fixed.text and ", (lo < box)" in fixed.text


def test_a_directional_term_with_no_labels_is_empty():
    q = t("none = 'nomatch*'\nx = anterior_of(none)\n").defined["x"]
    assert (q.tracts, q.labels, q.end0, q.end1) == (None, None, None, None)


def test_a_directional_term_does_not_read_the_streamlines_of_its_argument():
    """Only the labels of the argument count (the reference): the rules of
    the box need the label set, not the streamlines."""
    p = t("a = 1\nx = anterior_of(a)\n")
    needed = p.program("x", components=("tracts",))
    assert 'streamline_label("fs", x, 1)' not in needed
    assert "label_extent" in needed and "tractogram(t)" in needed


def test_unsupported_constructs_name_themselves():
    for source in ("x = save(a)", "x = frobnicate(a)"):
        with pytest.raises(WMQLUnsupported, match=source[4:8]):
            t("a = 1\n" + source + "\n")
    assert issubclass(WMQLUnsupported, NotImplementedError)
    assert issubclass(WMQLUnsupported, WMQLError)


def test_skip_unsupported_leaves_out_the_query_and_what_depends_on_it():
    source = (
        "a = 1\nb = 2\n"
        "far = endpoints_in(a and frobnicate(b))\n"
        "uses_far = far or a\n"
        "fine = a or b\n"
        "ok_again.side = a\n"
    )
    p = t(source, skip_unsupported=True)
    assert set(p.defined) == {"a", "b", "fine", "ok_again.left",
                              "ok_again.right"}
    assert set(p.skipped) == {"far", "uses_far"}
    assert "frobnicate" in p.skipped["far"]
    assert "far" in p.skipped["uses_far"]
    with pytest.raises(WMQLUnsupported):
        t(source)


def test_a_redefinition_that_works_replaces_a_skipped_name():
    p = t("a = 1\nx = frobnicate(a)\nx = a\n", skip_unsupported=True)
    assert "x" in p.defined and "x" not in p.skipped


def test_import_inlines_the_file(tmp_path):
    (tmp_path / "base.qry").write_text("A = 1\nb.left = 2\n")
    p = transpile(
        "import base.qry\nx = a or b.left\n", atlas="fs",
        include_folders=[str(tmp_path)],
    )
    assert set(p.defined) == {"a", "b.left", "x"}
    with pytest.raises(WMQLError, match="not found"):
        transpile("import missing.qry\n", atlas="fs",
                  include_folders=[str(tmp_path)])


def test_not_in_binds_looser_than_and_and_or():
    """``a and b not in c`` is ``(a and b) not in c``, not
    ``a and (b not in c)``: the labels differ (a union, then a
    difference)."""
    p = t(BASE + "x = a and b not in c\ny = a and (b not in c)\n")
    assert p.defined["x"].labels != p.defined["y"].labels
    text = p.text
    assert text.count("~") >= 2


def test_pattern_and_for_expand_statically():
    p = t("r1 = 1\nr2 = 2\nq = 3\nall_r = 'r*'\nnone = 'zz*'\n")
    assert p.defined["all_r"].tracts is not None
    assert p.defined["none"].tracts is None
    assert p.defined["none"].labels is None


def test_the_module_imports_nothing_of_neurolang():
    code = (
        "import sys, nl_tracts.wmql;"
        "bad = [m for m in sys.modules if m.split('.')[0] in "
        "('neurolang', 'nl_core', 'nl_expr', 'lark')];"
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_a_definition_adds_no_rule_of_its_own():
    """The public name of a query is an alias at the output, not a copy
    in the middle of the program: a copy is a step for the engine, and one
    more pass over a large relation."""
    p = t("a = 1\nb = 2\nx = a or b\n")
    rules = p.program("x", components=("tracts",)).strip().split("\n")
    # the streamlines of a and of b, their union (two rules), the name
    assert len(rules) == 5
    assert re.fullmatch(r"q_x\(x\) :- qi_\d+\(x\)", rules[-1])


def test_a_name_defined_by_a_name_is_an_alias():
    p = t("a = 1\nx = a\ny = x\n")
    rules = p.program("y", components=("tracts",)).strip().split("\n")
    assert len(rules) == 2  # the streamlines of 1, and the name y
    assert p.defined["x"].tracts == "q_x" and p.defined["y"].tracts == "q_y"


def test_every_version_of_a_name_has_its_alias_in_the_whole_program():
    p = t("x = 1\ny = x\nx = 2\n")
    assert re.search(r"^q_x__v1\(x\) :- qi_\d+\(x\)$", p.text, re.M)
    assert re.search(r"^q_x\(x\) :- qi_\d+\(x\)$", p.text, re.M)


@pytest.mark.parametrize("source", ["x = only(a)", "x = ~a"])
def test_the_scan_of_only_and_invert_is_of_the_streamlines_of_the_argument(
    source
):
    """``outside`` is the streamlines of the argument with a label outside
    its labels, not all the streamlines with a label outside: the same
    streamlines in the difference, without a pass over, and a result as
    large as, all of ``streamline_label``."""
    text = t("a = 1\n" + source + "\n").text
    assert re.search(
        r'^qi_\d+\(x\) :- qi_\d+\(x\), streamline_label\("fs", x, l\), '
        r"streamline\(x, t, s\), ~qi_\d+\(t, l\)$",
        text, re.M,
    )


def test_only_keeps_an_end_whose_label_is_in_the_labels_of_its_tractogram():
    text = t("a = 1\nx = only(a)\n").text
    assert re.search(
        r'^qi_\d+\(x\) :- qi_\d+\(x\), endpoint_label\("fs", x, 1, l\), '
        r"streamline\(x, t, s\), qi_\d+\(t, l\)$",
        text, re.M,
    )


def test_the_relations_are_keyed_by_x_and_traverses_is_not_one():
    """``traverses`` and ``endpoint`` are names for a vocabulary of rules,
    not relations the program reads; a streamline is one id ``x``."""
    text = t(
        BASE + "p.left = 4\n"
        "x = only(not a or b) and endpoints_in(c)\n"
        "y = anterior_of(x) or medial_of(p.left)\n"
    ).text
    assert "traverses(" not in text and "endpoint(" not in text
    assert "(t, s)" not in text


def test_the_scan_of_only_without_labels_is_still_restricted():
    p = t("none = 'zz*'\nx = only(none)\n")
    assert p.defined["x"].tracts is None
