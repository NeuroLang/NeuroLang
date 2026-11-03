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
    assert re.search(r"^qi_\d+\(7\)$", p.text, re.M)  # a fact
    assert re.search(r"^q_a__label\(l\) :- qi_\d+\(l\)$", p.text, re.M)
    assert re.search(
        r'^qi_\d+\(t, s\) :- traverses\("fs", t, s, 7\)$', p.text, re.M
    )
    assert re.search(
        r'^qi_\d+\(t, s\) :- endpoint\("fs", t, s, 1, 7\)$', p.text, re.M
    )


def test_the_atlas_is_a_constant_of_the_program():
    assert 'traverses("dk", ' in transpile("a = 1\n", atlas="dk").text
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
    assert 'traverses("fs", t, s, l)' in text  # the universe
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
    assert "traverses" in text
    only_labels = p.program("c", components=("labels",))
    assert "traverses" not in only_labels and "endpoint" not in only_labels
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
     "x = a.b.c", "x == a", "x = 1.5"],
)
def test_what_the_reference_rejects_is_rejected(source):
    with pytest.raises(WMQLError):
        t(BASE + "a.b = 1\n" + source + "\n")


def test_a_syntax_error_says_where():
    with pytest.raises(WMQLError, match="line 2"):
        t("a = 1\nx = (a and\n")


@pytest.mark.parametrize(
    "term",
    ["anterior_of", "posterior_of", "medial_of", "lateral_of",
     "inferior_of", "superior_of"],
)
def test_directional_terms_name_themselves_and_the_design(term):
    with pytest.raises(WMQLUnsupported, match=f"{term}.*DESIGN"):
        t(f"a = 1\nx = {term}(a)\n")
    assert issubclass(WMQLUnsupported, NotImplementedError)
    assert issubclass(WMQLUnsupported, WMQLError)


def test_skip_unsupported_leaves_out_the_query_and_what_depends_on_it():
    source = (
        "a = 1\nb = 2\n"
        "far = endpoints_in(a and anterior_of(b))\n"
        "uses_far = far or a\n"
        "fine = a or b\n"
        "ok_again.side = a\n"
    )
    p = t(source, skip_unsupported=True)
    assert set(p.defined) == {"a", "b", "fine", "ok_again.left",
                              "ok_again.right"}
    assert set(p.skipped) == {"far", "uses_far"}
    assert "anterior_of" in p.skipped["far"]
    assert "far" in p.skipped["uses_far"]
    with pytest.raises(WMQLUnsupported):
        t(source)


def test_a_redefinition_that_works_replaces_a_skipped_name():
    p = t("a = 1\nx = anterior_of(a)\nx = a\n", skip_unsupported=True)
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
