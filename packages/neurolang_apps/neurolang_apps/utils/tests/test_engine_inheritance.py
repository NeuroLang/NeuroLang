import textwrap

import pytest

from neurolang_apps.utils.engine_registry import (
    get_engine_config,
    list_engine_names,
    resolve_engine_config,
)


def _write(tmp_path, text):
    path = tmp_path / "engines.yaml"
    path.write_text(textwrap.dedent(text))
    return path


@pytest.fixture
def yaml_path(tmp_path):
    return _write(
        tmp_path,
        """
        engines:
          base:
            abstract: true
            description: base
            builtins: [exp, log]
            datalog_init: |
              a(x) :- b(x)
            atlases:
              schaefer: {n_rois: 100, yeo_networks: 7}
              destrieux: {predicate_name: destrieux}
            predicates:
              p: {arity: 1, columns: [x]}
              q: {arity: 1, columns: [x]}
          other:
            abstract: true
            builtins: [startswith]
            predicates:
              r: {arity: 1, columns: [x]}
          child:
            extends: base
            description: child
            builtins: [log, exp]
            datalog_init: |
              c(x) :- a(x)
            atlases:
              schaefer: {n_rois: 200}
              destrieux: null
            predicates:
              p: {columns: [y]}
              q: null
          multi:
            extends: [base, other]
        """,
    )


def test_child_overrides_and_merges(yaml_path):
    cfg = get_engine_config("child", yaml_path)
    assert cfg["description"] == "child"
    assert cfg["builtins"] == ["exp", "log"]
    assert cfg["datalog_init"].splitlines() == [
        "a(x) :- b(x)",
        "c(x) :- a(x)",
    ]
    assert cfg["atlases"] == {"schaefer": {"n_rois": 200, "yeo_networks": 7}}
    assert cfg["predicates"] == {"p": {"arity": 1, "columns": ["y"]}}
    assert "extends" not in cfg and "abstract" not in cfg


def test_multiple_bases(yaml_path):
    cfg = get_engine_config("multi", yaml_path)
    assert cfg["builtins"] == ["exp", "log", "startswith"]
    assert set(cfg["predicates"]) == {"p", "q", "r"}


def test_abstract_engines_hidden_and_unbuildable(yaml_path):
    assert list_engine_names(yaml_path) == ["child", "multi"]
    with pytest.raises(ValueError, match="abstract"):
        get_engine_config("base", yaml_path)


def test_unknown_base_and_cycle():
    with pytest.raises(ValueError, match="Unknown engine"):
        resolve_engine_config({"a": {"extends": "nope"}}, "a")
    engines = {"a": {"extends": "b"}, "b": {"extends": "a"}}
    with pytest.raises(ValueError, match="Cyclic"):
        resolve_engine_config(engines, "a")


def test_builtin_engines():
    assert list_engine_names() == [
        "destrieux", "functional_connectivity", "neurostore",
        "neurostore_disconnectome7t", "neurostore_iit", "neurostore_tracts",
        "neurosynth", "neurosynth_disconnectome7t", "neurosynth_iit",
        "neurosynth_tracts", "structural_connectivity", "tracts",
    ]
    for name in ("neurosynth", "neurostore"):
        cfg = get_engine_config(name)
        assert cfg["requires_mni_mask"] and cfg["use_base_symbols"]
        assert "study_with_peaks" in cfg["datalog_init"]
        assert {"peak_reported", "study", "term_in_study_tfidf"} <= set(
            cfg["predicates"]
        )
        assert "extends" not in cfg and "abstract" not in cfg
    assert get_engine_config("neurostore")["python_init"] == [
        "neurolang_apps.utils.engines.neurostore.init"
    ]
    with pytest.raises(ValueError, match="abstract"):
        get_engine_config("coordinate_meta_analysis")


def test_engine_files_are_merged_into_their_engine(
    yaml_path, tmp_path, monkeypatch
):
    from neurolang_apps.utils import engine_registry

    (tmp_path / "base").mkdir()
    (tmp_path / "base" / "examples.yaml").write_text(
        "assistant_examples: [{question: q1, datalog: a}]\n"
        "examples: {datalog: [b]}\n"
    )
    (tmp_path / "child").mkdir()
    (tmp_path / "child" / "examples.yaml").write_text(
        "assistant_examples: [{question: q2, datalog: c}]\n"
    )
    (tmp_path / "child" / "guidance.yaml").write_text(
        "assistant_guidance: Answer on ${engines.base.description}.\n"
    )
    monkeypatch.setattr(engine_registry, "_ENGINES_YAML", yaml_path)
    cfg = get_engine_config("child")
    assert cfg["assistant_guidance"] == "Answer on base."
    assert cfg["examples"] == {"datalog": ["b"]}
    assert [e["question"] for e in cfg["assistant_examples"]] == ["q1", "q2"]
    assert cfg["description"] == "child" and "c(x) :- a(x)" in cfg[
        "datalog_init"
    ]


@pytest.fixture
def init_modules(tmp_path, monkeypatch):
    """Two modules ``init_a`` and ``init_b`` whose ``init_engine`` appends
    their name and options to the list they return."""
    calls = []
    for name in ("init_a", "init_b"):
        (tmp_path / f"{name}.py").write_text(textwrap.dedent(f"""
            def init_engine(nl, mask, data_dir, **options):
                nl.append(({name!r}, options))
        """))
    monkeypatch.syspath_prepend(str(tmp_path))
    return calls


def test_python_init_is_a_list_parent_first():
    engines = {
        "one": {"python_init": "init_a"},
        "two": {"python_init": ["init_b"]},
        "child": {"extends": "one", "python_init": "init_b"},
        "both": {"extends": ["one", "two"], "python_init": ["init_a"]},
    }
    assert resolve_engine_config(engines, "one")["python_init"] == ["init_a"]
    assert resolve_engine_config(engines, "child")["python_init"] == [
        "init_a", "init_b"
    ]
    assert resolve_engine_config(engines, "both")["python_init"] == [
        "init_a", "init_b"
    ]


def test_python_inits_run_in_order_with_the_options(init_modules):
    from neurolang_apps.utils.engine_registry import _phase_python_init

    cfg = {
        "python_init": ["init_b", "init_a"],
        "python_init_options": {"opt": 1},
    }
    _phase_python_init(init_modules, cfg, "mask", "data")
    assert init_modules == [("init_b", {"opt": 1}), ("init_a", {"opt": 1})]
    del init_modules[:]
    _phase_python_init(init_modules, {"python_init": "init_a"}, None, None)
    assert init_modules == [("init_a", {})]


def test_a_base_reached_twice_is_merged_once():
    engines = {
        "root": {"abstract": True, "datalog_init": "r(x) :- s(x)\n",
                 "atlases": {"a": {"n": 1}}},
        "left": {"extends": "root", "datalog_init": "l(x) :- r(x)\n"},
        "right": {"extends": "root", "datalog_init": "q(x) :- r(x)\n",
                  "atlases": {"a": {"n": 2}}},
        "both": {"extends": ["left", "right"]},
    }
    cfg = resolve_engine_config(engines, "both")
    assert cfg["datalog_init"].splitlines() == [
        "r(x) :- s(x)", "l(x) :- r(x)", "q(x) :- r(x)",
    ]
    assert cfg["atlases"] == {"a": {"n": 2}}


def test_hidden_predicates_are_concatenated():
    engines = {
        "a": {"hidden_predicates": ["p", "q"]},
        "b": {"hidden_predicates": ["q", "r"]},
        "both": {"extends": ["a", "b"]},
    }
    assert resolve_engine_config(engines, "both")["hidden_predicates"] == [
        "p", "q", "r"
    ]


def test_acknowledgements_are_concatenated():
    engines = {
        "atlas": {"acknowledgements": ["Atlas: Someone (2010)."]},
        "data": {"extends": "atlas",
                 "acknowledgements": ["Data: Others (2020)."]},
    }
    assert resolve_engine_config(engines, "data")["acknowledgements"] == [
        "Atlas: Someone (2010).", "Data: Others (2020)."]


def test_the_engines_acknowledge_the_hcp_and_their_atlases():
    from neurolang_apps.utils.engine_registry import get_acknowledgements

    texts = get_acknowledgements("neurosynth_iit")
    for source in ("Schaefer", "Destrieux", "Tian", "MRtrix3",
                   "Human Connectome Project", "Neurosynth", "IIT"):
        assert any(source in text for text in texts), source
    hcp = next(t for t in texts if "1U54MH091657" in t)
    assert "McDonnell Center for Systems Neuroscience" in hcp
    assert all("\n" not in text for text in texts)


@pytest.mark.parametrize("engine", list_engine_names())
def test_every_engine_names_the_sources_of_its_data(engine):
    """.credits and the REPL's credits page have something to say of every
    engine: the template, atlases and databases it loads."""
    from neurolang_apps.utils.engine_registry import get_acknowledgements

    texts = get_acknowledgements(engine)
    assert texts and len(texts) == len(set(texts))


@pytest.mark.parametrize("engine", list_engine_names())
def test_the_questions_shown_are_those_of_the_assistant(engine):
    """The questions the REPL lists first are in the assistant's bank,
    whose queries the tests check on the engines."""
    cfg = get_engine_config(engine)
    bank = {e["question"] for e in cfg.get("assistant_examples", [])}
    assert set(cfg.get("questions") or []) <= bank


def test_a_relation_file_downloaded_is_checked_by_its_md5(
    tmp_path, monkeypatch
):
    from neurolang_apps.utils import engine_registry

    asked = []
    monkeypatch.setattr(
        engine_registry, "_fetch_files",
        lambda folder, files, verbose: asked.append(files) or ["x.tsv"])
    engine_registry._resolve_relation_file(
        {"file": "https://example.org/v1/x.tsv", "md5": "abc"}, "e",
        tmp_path)
    assert asked == [[("x.tsv", "https://example.org/v1/x.tsv",
                       {"md5sum": "abc"})]]


def _fetched(relation, tmp_path):
    """The table of *relation* of the engines, at its tag on GitLab,
    checked by its md5, as the engines download it (skip if offline)."""
    import pandas as pd

    from neurolang_apps.utils import engine_registry

    entry = get_engine_config("neurosynth")["relations"][relation]
    assert "/-/raw/v" in entry["file"] and entry["md5"]  # pinned
    try:
        path = engine_registry._resolve_relation_file(entry, "e", tmp_path)
    except Exception as error:  # offline, GitLab down
        pytest.skip(f"cannot fetch {entry['file']}: {error!r}")
    return pd.read_csv(path, sep="\t")


def test_the_guidance_of_the_assistant_is_inherited():
    engines = {
        "connectivity": {"assistant_guidance": "Connectivity is stored."},
        "meta": {"extends": "connectivity",
                 "assistant_guidance": "Use PROB over studies."},
    }
    assert resolve_engine_config(engines, "meta")["assistant_guidance"] == (
        "Connectivity is stored.\nUse PROB over studies.")
    guidance = get_engine_config("neurosynth")["assistant_guidance"]
    # the connectivities' guidance first, then the meta-analyses'
    assert guidance.index("Three connectivities") < \
        guidance.index("The resting-state functional connectivity is") < \
        guidance.index("A question asking how many")


def test_the_schaefer_400_atlas_is_its_own_atlas_of_the_connectivity():
    cfg = get_engine_config("neurosynth")
    atlases = cfg["atlases"]
    assert atlases["schaefer"]["n_rois"] == 100
    assert atlases["schaefer400"] == {
        "kind": "schaefer", "predicate_name": "schaefer400", "n_rois": 400,
        "resolution_mm": 2, "yeo_networks": 17}
    assert cfg["sorts"]["schaefer400_label"]["atlas"] == "schaefer400"
    assert "schaefer400" not in get_engine_config("destrieux")["atlases"]


RELATIONS = {
    "schaefer400_tian_structural_connectivity":
        ("schaefer400_tian_label", "streamline_count", "schaefer400", 400),
    "destrieux_tian_structural_connectivity":
        ("destrieux_tian_label", "streamline_count", "destrieux", 148),
    "schaefer400_tian_functional_connectivity":
        ("schaefer400_tian_label", "correlation", "schaefer400", 400),
    "destrieux_tian_functional_connectivity":
        ("destrieux_tian_label", "correlation", "destrieux", 148),
}


def test_the_connectivity_is_four_relations_of_cortex_and_subcortex():
    """The connectivity engines have the four relations, structural and
    functional over Schaefer 400 + Tian and Destrieux + Tian, and no
    other; their descriptions tell them apart and name the subcortex."""
    from neurolang_apps.utils.engine_registry import (
        get_acknowledgements, get_predicates)

    for engine in ("structural_connectivity", "functional_connectivity"):
        cfg = get_engine_config(engine)
        kind = engine.split("_")[0]
        assert {r for r in cfg["relations"]
                if r.endswith("_connectivity")} == {
            r for r in RELATIONS if f"_{kind}_" in r}
    predicates = get_predicates("neurosynth")
    connectivity = {p for p in predicates if p.endswith("_connectivity")}
    assert connectivity == set(RELATIONS)
    for name, (label, value, atlas, _) in RELATIONS.items():
        assert [a["sort"] for a in predicates[name]["arguments"]] == \
            [label, label, value]
        text = " ".join(predicates[name]["description"].split())
        assert "Tian" in text and f"{atlas}(l, r)" in text, name
        if "structural" in name:
            assert "100 HCP subjects" in text and "not functional" in text
        else:
            assert "Resting-state functional" in text
            assert "not structural" in text and "co-activation" in text
        entry = get_engine_config("neurosynth")["relations"][name]
        assert "/-/raw/v2.4/tsv/" in entry["file"] and entry["md5"]
    guidance = " ".join(
        get_engine_config("neurosynth")["assistant_guidance"].split())
    assert "Three connectivities" in guidance
    assert "meta-analytic co-activation" in guidance
    texts = get_acknowledgements("neurosynth")
    for doi in ("10.1016/j.neuroimage.2019.116137",  # MRtrix3
                "10.1016/j.neuroimage.2015.06.092",  # SIFT2
                "10.1016/j.neuroimage.2013.05.039",  # HCP resting state
                "10.1038/s41593-020-00711-6"):  # Tian
        assert any(doi in t for t in texts), doi
    # the connectivity engines share the atlas of 400, others do not load it
    assert "schaefer400" in get_predicates("functional_connectivity")
    assert "schaefer400" not in get_predicates("tracts")


@pytest.mark.network
@pytest.mark.parametrize("relation", list(RELATIONS))
def test_the_connectivity_files_are_every_pair_of_engine_regions(
    relation, tmp_path
):
    _, value, _, cortex = RELATIONS[relation]
    table = _fetched(relation, tmp_path)
    column = "streamline_count" if value == "streamline_count" else "r"
    assert list(table.columns) == ["region1", "region2", column]
    regions = set(table.region1)
    tian = {r for r in regions if r.endswith((" lh", " rh"))}
    assert len(tian) == 32 and len(regions) == cortex + 32
    assert not any("-" in r or "_" in r for r in regions)
    assert (table.region1 != table.region2).all()
    assert table[column].notna().all()
    if column == "r":
        assert table.r.between(-1, 1).all()
    else:
        assert (table.streamline_count > 0).all()
    both = table.set_index(["region1", "region2"])[column]
    swapped = table.set_index(["region2", "region1"])[column]
    swapped.index.names = both.index.names
    assert both.sort_index().equals(swapped.sort_index())


def test_the_destrieux_and_tian_regions_have_their_full_names():
    """destrieux_name and tian_name, files of the package, name each of the
    148 Destrieux and 32 Tian regions once, by the labels of the engines,
    with the hemisphere; every engine of the MNI atlases has them."""
    import pandas as pd

    from neurolang_apps.utils.engine_registry import (
        _ENGINES_DIR, get_predicates)

    for relation, regions, sides in (
            ("destrieux_name", 148, ("L ", "R ")),
            ("tian_name", 32, (" lh", " rh"))):
        entry = get_engine_config("neurosynth")["relations"][relation]
        table = pd.read_csv(_ENGINES_DIR / entry["file"], sep="\t")
        assert list(table.columns) == ["label", "name"]
        assert table.label.is_unique and len(table) == regions
        assert table.name.is_unique
        for side, word in zip(sides, ("left", "right")):
            mine = table[table.label.str.startswith(side)
                         | table.label.str.endswith(side)]
            assert len(mine) == regions // 2
            assert mine.name.str.endswith(f", {word}").all()
        assert not table.label.str.contains("[-_]").any()
        for engine in ("neurosynth", "structural_connectivity", "tracts"):
            assert relation in get_predicates(engine)
            guidance = get_engine_config(engine)["assistant_guidance"]
            assert relation in guidance, engine
    names = pd.read_csv(_ENGINES_DIR / "mni_atlases/destrieux_names.tsv",
                        sep="\t").set_index("label").name
    assert names["L G front sup"] == "Superior frontal gyrus (F1), left"
    # the engine of Destrieux alone has its names too
    assert "destrieux_name" in get_predicates("destrieux")


@pytest.mark.network
def test_the_destrieux_engine_has_both_hemispheres_and_their_names(
        tmp_path):
    """The engine destrieux builds (its atlas's description is not an
    argument of the fetcher), left_region and right_region hold the 74
    regions of each hemisphere, and every region has its full name."""
    from neurolang_apps.utils.engine_registry import build_engine

    try:
        nl = build_engine("destrieux", tmp_path, resolution=2)
    except OSError as error:  # offline
        pytest.skip(f"cannot fetch the atlas: {error!r}")
    for side in ("left_region", "right_region"):
        assert len(nl.execute_datalog_program(
            f"ans(n) :- {side}(n, r)")) == 74
    named = nl.execute_datalog_program(
        "ans(n, m) :- destrieux(n, r), destrieux_name(n, m)")
    assert len(named) == 148


def test_disconnectome7t_engines_load_the_files_they_download():
    """Both engines of Disconnectome7T load the tractogram and the
    segmentation they download, and acknowledge it and the HCP."""
    from neurolang_apps.utils.engine_registry import get_acknowledgements

    ns = get_engine_config("neurosynth_disconnectome7t")
    store = get_engine_config("neurostore_disconnectome7t")
    for key in ("downloads", "tractogram", "questions"):
        assert ns[key] == store[key]
    files = {f"{d['dest']}/{d.get('name', d['url'].rsplit('/', 1)[-1])}"
             for d in ns["downloads"]}
    tail = ns["datalog_init"].rstrip().splitlines()[-4:]
    assert store["datalog_init"].rstrip().splitlines()[-4:] == tail
    for path in ("tracts/disconnectome7t/disconnectome7t.tck",
                 "tracts/disconnectome7t/disconnectome7t_wmparc.nii.gz",
                 "tracts/fs/FreeSurferColorLUT.txt"):
        assert path in files and f'"{path}"' in "\n".join(tail)
    assert ns["tractogram"]["path"] in files
    texts = " ".join(get_acknowledgements("neurosynth_disconnectome7t"))
    assert "Disconnectome7T" in texts and "Human Connectome" in texts
