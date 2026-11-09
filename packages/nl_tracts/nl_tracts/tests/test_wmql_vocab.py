"""The translation of WMQL to the vocabulary, construct by construct: the
text of the rules (``nl_tracts.wmql_vocab``, DESIGN.md, 17)."""

import pytest

from nl_tracts.wmql_vocab import (
    BUNDLES, WMQLError, WMQLUnsupported, bundle, translate,
)

LABELS = """
a.left |= 1
a.right |= 2
b.left |= 3
b.right |= 4
c.left |= 5
c.right |= 6
u |= 7
HEMISPHERE.left |= '*.left'
HEMISPHERE.right |= '*.right'
"""


def rules(source, **kwargs):
    """The rules of the translation, without the comments."""
    text = translate(LABELS + source, **kwargs).text
    return "\n".join(
        line for line in text.splitlines()
        if line and not line.startswith("%"))


@pytest.mark.parametrize("wmql, datalog", [
    ("x.side = a.side or b.side",
     'x(x, h) :-\n    (traverses(x, "a", h) | traverses(x, "b", h))'),
    ("x.side = a.side and b.side",
     'x(x, h) :-\n    traverses(x, "a", h), traverses(x, "b", h)'),
    ("x.side = a.side + b.side * c.side",
     'x(x, h) :-\n    (traverses(x, "a", h) | '
     '(traverses(x, "b", h), traverses(x, "c", h)))'),
    ("x.side = a.side not in b.side not in c.side",
     'x(x, h) :-\n    traverses(x, "a", h), ~traverses(x, "b", h), '
     '~traverses(x, "c", h)'),
    ("x.side = a.side - b.side",
     'x(x, h) :-\n    traverses(x, "a", h), ~traverses(x, "b", h)'),
    # not in binds looser than or: (a or b) not in c
    ("x.side = a.side or b.side not in c.side",
     'x(x, h) :-\n    (traverses(x, "a", h) | traverses(x, "b", h)), '
     '~traverses(x, "c", h)'),
    ("x.side = not a.side and b.side",
     'x(x, h) :-\n    traverses(x, "b", h), ~traverses(x, "a", h)'),
    ("x.side = a.side not in (b.side and c.side)",
     'x(x, h) :-\n    traverses(x, "a", h), '
     '(~traverses(x, "b", h) | ~traverses(x, "c", h))'),
    ("x.side = a.side not in hemisphere.opposite",
     'x(x, h) :-\n    traverses(x, "a", h), opposite_hemisphere(h, o), '
     '~in_hemisphere(x, o)'),
    ("x.side = hemisphere.side and a.opposite",
     'x(x, h) :-\n    in_hemisphere(x, h), opposite_hemisphere(h, o), '
     'traverses(x, "a", o)'),
    ("x.side = a.side and u",
     'x(x, h) :-\n    traverses(x, "a", h), traverses(x, "u", _)'),
    ("x.side = a.side not in u",
     'x(x, h) :-\n    traverses(x, "a", h), ~traverses(x, "u", _)'),
    ("x = a.left and a.right",
     'x(x) :-\n    traverses(x, "a", "left"), traverses(x, "a", "right")'),
    ("x.left = a.left not in b.right",
     'x(x, "left") :-\n    traverses(x, "a", "left"), '
     '~traverses(x, "b", "right")'),
    ("x.side = endpoints_in(a.side or b.side)",
     'x(x, h) :-\n    (ends_in(x, "a", h) | ends_in(x, "b", h))'),
    ("x.side = endpoints_in(a.side and anterior_of(b.side))",
     'x(x, h) :-\n    endpoint_in(x, e, "a", h), '
     'endpoint_anterior_of(x, e, "b", h)'),
    ("x.side = endpoints_in(a.side not in posterior_of(b.side))"
     " and endpoints_in(c.side and superior_of(b.side))",
     'x(x, h) :-\n    endpoint_in(x, e, "a", h), endpoint_in(x, e2, "c", h),'
     '\n    endpoint_superior_of(x, e2, "b", h), '
     '~endpoint_posterior_of(x, e, "b", h)'),
    ("x.side = a.side not in endpoints_in(b.side and medial_of(c.side))",
     'x(x, h) :-\n    traverses(x, "a", h),\n'
     '    ~exists(e; endpoint_in(x, e, "b", h), '
     'endpoint_medial_of(x, e, "c", h))'),
    ("x.side = both_endpoints_in(a.side or b.side)",
     'x(x, h) :-\n    (endpoint_in(x, 0, "a", h) | endpoint_in(x, 0, "b", h)),'
     '\n    (endpoint_in(x, 1, "a", h) | endpoint_in(x, 1, "b", h))'),
    ("x.side = only(a.side)", 'x(x, h) :-\n    only_in(x, "a", h)'),
    ("x.side = ~a.side", 'x(x, h) :-\n    only_in(x, "a", h)'),
    ("x.side = only(a.side and (b.side or u))",
     'x_leaves(x, h) :-\n    in_hemisphere(x, h), streamline_label(a, x, l),'
     ' ~term_label(a, l, "a", h),\n    ~term_label(a, l, "b", h), '
     '~term_label(a, l, "u", _)\n'
     'x(x, h) :-\n    traverses(x, "a", h), '
     '(traverses(x, "b", h) | traverses(x, "u", _)),\n    ~x_leaves(x, h)'),
    ("x.side = anterior_of(a.side) not in posterior_of(b.side)",
     'x(x, h) :-\n    anterior_of(x, "a", h), ~posterior_of(x, "b", h)'),
    ("x.side = superior_of(a.side) and inferior_of(b.side)",
     'x(x, h) :-\n    superior_of(x, "a", h), inferior_of(x, "b", h)'),
    ("x.side = medial_of(a.side) and lateral_of(b.opposite)",
     'x(x, h) :-\n    medial_of(x, "a", h), opposite_hemisphere(h, o), '
     'lateral_of(x, "b", o)'),
    # a query of labels saved with =: a predicate, of its terms
    ("x.side = a.side or b.side\ny.side = x.side not in x.opposite",
     'x(x, h) :-\n    (traverses(x, "a", h) | traverses(x, "b", h))\n'
     'y(x, h) :-\n    x(x, h), opposite_hemisphere(h, o), ~x(x, o)'),
    # |= of labels: a term
    ("g.side |= a.side or b.side\nx.side = g.side and c.side",
     'x(x, h) :-\n    traverses(x, "g", h), traverses(x, "c", h)'),
    ("x = '*.left' and u", 'x(x) :-\n    in_hemisphere(x, "left"), '
     'traverses(x, "u", _)'),
])
def test_construct(wmql, datalog):
    assert rules(wmql) == datalog


def test_the_uncinate_of_freesurfer_queries():
    source = LABELS.replace("a.", "orbitofrontalgyrus.").replace(
        "b.", "temporal.").replace("c.", "amygdala.") + """
insula.side |= 8
UF.side = ((orbitofrontalgyrus.side
            and endpoints_in(temporal.side and anterior_of(amygdala.side))
            and insula.side)
           not in posterior_of(amygdala.side) not in HEMISPHERE.opposite)
"""
    text = translate(source).text
    assert text.endswith("""
%% uf(x: streamline, h: hemisphere)
%% The query uf of WMQL.
%%
%% Translated from WMQL:
%% UF.side = ((orbitofrontalgyrus.side
%%             and endpoints_in(temporal.side and anterior_of(amygdala.side))
%%             and insula.side)
%%            not in posterior_of(amygdala.side) not in HEMISPHERE.opposite)
uf(x, h) :-
    traverses(x, "orbitofrontalgyrus", h), endpoint_in(x, e, "temporal", h),
    endpoint_anterior_of(x, e, "amygdala", h), traverses(x, "insula", h),
    ~posterior_of(x, "amygdala", h), opposite_hemisphere(h, o),
    ~in_hemisphere(x, o)
""")
    assert "% The terms it uses: amygdala, insula, orbitofrontalgyrus, " \
        "temporal." in text


def test_a_long_disjunction_is_broken_at_its_bars():
    names = [f"region_number_{i}" for i in range(6)]
    source = "".join(f"{n}.left |= {i}\n{n}.right |= {i + 10}\n"
                     for i, n in enumerate(names))
    source += "x.side = " + " or ".join(f"{n}.side" for n in names) + "\n"
    text = translate(source).text
    body = text[text.index("x(x, h) :-"):].splitlines()
    assert body == [
        "x(x, h) :-",
        '    (traverses(x, "region_number_0", h)'
        ' | traverses(x, "region_number_1", h)',
        '     | traverses(x, "region_number_2", h)'
        ' | traverses(x, "region_number_3", h)',
        '     | traverses(x, "region_number_4", h)'
        ' | traverses(x, "region_number_5", h))',
    ]
    assert all(len(line) <= 79 for line in text.splitlines()
               if not line.startswith("%"))


def test_a_redefined_query_keeps_its_earlier_rules():
    """A name defined again is a new predicate for what follows; what
    used the earlier definition keeps it, renamed ``x_1``."""
    source = ("x.side = a.side and b.side\ny.side = x.side and c.side\n"
              "x.side |= a.side and c.side\n")
    assert rules(source) == (
        'x_1(x, h) :-\n    traverses(x, "a", h), traverses(x, "b", h)\n'
        'y(x, h) :-\n    x_1(x, h), traverses(x, "c", h)\n'
        'x(x, h) :-\n    traverses(x, "a", h), traverses(x, "c", h)')
    assert translate(LABELS + source).queries["x.left"] == 'x(x, "left")'


def test_names_of_the_vocabulary_are_prefixed():
    lines = rules("traverses = a.left and b.left\n_x = a.left and c.left")
    assert lines.splitlines()[0] == "wmql_traverses(x) :-"
    assert lines.splitlines()[2] == "wmql__x(x) :-"
    with pytest.raises(WMQLError, match="syntax error"):
        translate("1x = a")


def test_the_queries_and_terms():
    t = translate(LABELS + "x.side = a.side and b.side\ny = endpoints_in("
                  "c.left)\nz.left = a.left and u\n")
    assert t.queries == {
        "x.left": 'x(x, "left")', "x.right": 'x(x, "right")',
        "y": "y(x)", "z.left": 'z(x, "left")',
    }
    assert t.terms == ("a", "b", "c", "u")


@pytest.mark.parametrize("wmql, reason", [
    ("x.side = anterior_of(u)", "a name without a side"),
    ("x.side = anterior_of(a.side or b.side)", "must be a name"),
    ("g.side = a.side or b.side\nx.side = anterior_of(g.side)",
     "not one term"),
    ("y.side = a.side and b.side\nx.side = endpoints_in(y.side)",
     "not made of terms"),
    ("y.side = a.side and b.side\nx.side = only(y.side)", "not made of terms"),
    ("x.side = only(a.side not in b.side)", "not made of terms"),
    ("x.side = not a.side", "only negations"),
    ("x.side = u and b.left", "no condition on the hemisphere"),
    ("x.side = a.side and 7", "a label id is not a term"),
    ("x.side = save(a.side)", "save"),
    ("x.side = a.side and '*.nothing'", "matches no name"),
    ("x.side = a.side and nothing.side", "not defined"),
])
def test_what_the_vocabulary_cannot_say(wmql, reason):
    with pytest.raises(WMQLError, match=reason):
        translate(LABELS + wmql)
    t = translate(LABELS + wmql + "\nz.side = x.side or a.side\n"
                  "w.side = a.side\n", skip_unsupported=True)
    skipped = {k for k in t.skipped if k[0] in "xz"}
    assert skipped >= {k for k in t.skipped if k.startswith("x")}
    assert reason in t.skipped[next(k for k in t.skipped if k[0] == "x")]
    assert t.skipped["z.left"].startswith("uses x.")
    assert "w.left" in t.queries
    assert f"%   {next(k for k in t.skipped if k[0] == 'x')}" in t.text \
        or "%   x.side:" in t.text or "%   x:" in t.text


def test_a_side_in_a_query_without_side():
    with pytest.raises(WMQLError, match="invalid name"):
        translate(LABELS + "x = a.side")


def test_label_ids_are_terms_only_if_the_mapping_has_them():
    source = LABELS + "g.side |= a.side or b.side\nx.side = g.side and u\n"
    assert rules(source, terms={"a", "b", "u"}) == (
        'x(x, h) :-\n    (traverses(x, "a", h) | traverses(x, "b", h)), '
        'traverses(x, "u", _)')
    assert rules(source, terms={"g", "u"}) == (
        'x(x, h) :-\n    traverses(x, "g", h), traverses(x, "u", _)')
    with pytest.raises(WMQLUnsupported, match="a is not a term"):
        translate(source, terms={"u"})
    source = LABELS + "g.side |= 8\nx.side = g.side and u\n"
    with pytest.raises(WMQLUnsupported, match="g is made of label ids"):
        translate(source, terms={"u"})


def test_import_and_for(tmp_path):
    (tmp_path / "labels.qry").write_text(LABELS)
    source = ("import labels.qry\nfor n in [a, b]:\n"
              "    n_and_c.side = n.side and c.side\n")
    with pytest.raises(WMQLError, match="not found"):
        translate(source)
    text = translate(source, include_folders=[str(tmp_path)]).text
    assert 'n_and_c(x, h) :-\n    traverses(x, "b", h), ' \
        'traverses(x, "c", h)' in text


def _tool():
    """The script tools/wmql_to_datalog.py, the command line."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[2] / "tools" / "wmql_to_datalog.py"
    if not path.is_file():
        pytest.skip("the tools of nl_tracts are not installed")
    spec = importlib.util.spec_from_file_location("wmql_to_datalog", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_command_line(tmp_path, capsys):
    cli = _tool()
    path = tmp_path / "q.qry"
    path.write_text(LABELS + "x.side = a.side\ny.side = anterior_of(u)\n")
    assert cli.main([str(path)]) == 0
    out, err = capsys.readouterr()
    assert out == translate(path.read_text(), skip_unsupported=True,
                            title="q.qry", qry_file="q.qry").text
    assert "%%\n%% Translated from q.qry:\n%% x.side = a.side\n" in out
    assert "wmql_to_datalog: 2 queries not translated" in err  # y.*
    assert cli.main(["--strict", str(path)]) == 1
    assert "y.left" in capsys.readouterr().err
    assert cli.main(
        ["--terms", "freesurfer_aseg_aparc", str(path)]) == 0
    assert "%   x.side: a is not a term" in capsys.readouterr().out


@pytest.mark.parametrize("name", BUNDLES)
def test_the_shipped_modules_are_self_contained_rules(name):
    text = bundle(name)
    assert "ans(" not in text
    assert not any(line.startswith(".") for line in text.splitlines())
    with pytest.raises(ValueError, match="unknown bundle"):
        bundle("nothing")


def docs(text):
    """The ``%%`` documentation of each predicate of ``text``, as the
    Datalog parser reads it: ``{name: (description, arguments)}``, the
    blocks of a predicate one after the other."""
    from neurolang.frontend.datalog.standard_syntax import (
        DOCUMENT_COMMAND, parser,
    )

    out = {}
    for f in parser(text).formulas:
        if getattr(f, "name", None) == DOCUMENT_COMMAND:
            name, description, arguments = f.args
            old = out.get(name.name, ("", ()))[0]
            out[name.name] = (
                "\n\n".join(filter(None, (old, description.value))),
                arguments.value)
    return out


def test_each_predicate_is_documented():
    source = ("x.left = a.left and b.left\nx.right = a.right\n"
              "y = endpoints_in(c.left)\nz.side = only(a.side or b.side)\n")
    documented = docs(translate(LABELS + source, qry_file="q.qry").text)
    assert documented["x"] == (
        "The query x of q.qry.\n\nTranslated from q.qry:\n"
        "x.left = a.left and b.left\n\n"
        "Translated from q.qry:\nx.right = a.right",
        (("x", "streamline"), ("h", "hemisphere")),
    )
    assert documented["y"][1] == (("x", "streamline"),)
    assert documented["z_leaves"] == (
        'Auxiliary of z: x goes through a label of hemisphere h that is of '
        'none of\nthe terms of its only(...), "a", "b".',
        (("x", "streamline"), ("h", "hemisphere")),
    )


def test_an_imported_statement_cites_its_file(tmp_path):
    (tmp_path / "labels.qry").write_text(
        LABELS + "w.side = a.side and b.side\n")
    text = translate("import labels.qry\nv.side = w.side\n",
                     include_folders=[str(tmp_path)], qry_file="q.qry").text
    assert "%% Translated from labels.qry:\n%% w.side = a.side" in text
    assert "%% Translated from q.qry:\n%% v.side = w.side" in text


@pytest.mark.parametrize("name", BUNDLES)
def test_each_bundle_of_the_shipped_modules_is_named(name):
    """Every predicate of a shipped module has a full name, from
    ``tract_names.tsv``, and the sorts of its arguments."""
    documented = docs(bundle(name))
    for predicate, (description, arguments) in documented.items():
        assert arguments[0] == ("x", "streamline"), predicate
        assert "The query" not in description, predicate
        if "_leaves" not in predicate:
            assert "Translated from tract_querier's " in description


def test_the_names_of_the_bundles():
    fs = docs(bundle("freesurfer_queries"))
    assert fs["uf"][0].startswith("Uncinate fasciculus: streamlines joining")
    assert fs["uf"][1] == (("x", "streamline"), ("h", "hemisphere"))
    assert fs["cc_7"][0].startswith("Corpus callosum, splenium: ")
    assert fs["ec"][0].startswith("tract_querier's ec: ")
    toad = docs(bundle("freesurfer_queries_bsf2016_toad"))
    assert "freesurfer_queries_bsf2016.qry:\nUF.side" in toad["uf_1"][0]
    assert "freesurfer_queries_bsf2016_toad.qry:\nUF.side" in toad["uf"][0]
    assert docs(bundle("mori_queries"))["ifo"][0].startswith(
        "Inferior fronto-occipital fasciculus: ")
