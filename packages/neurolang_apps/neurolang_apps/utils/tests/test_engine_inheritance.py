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
        "destrieux", "neurostore", "neurostore_iit", "neurostore_tracts",
        "neurosynth", "neurosynth_iit", "neurosynth_tracts", "tracts",
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
    for source in ("Schaefer", "Destrieux", "Tian", "Mansour",
                   "Human Connectome Project", "Neurosynth", "IIT"):
        assert any(source in text for text in texts), source
    hcp = next(t for t in texts if "1U54MH091657" in t)
    assert "McDonnell Center for Systems Neuroscience" in hcp
    assert all("\n" not in text for text in texts)


@pytest.mark.parametrize("engine", list_engine_names())
def test_the_questions_shown_are_those_of_the_assistant(engine):
    """The questions the REPL lists first are in the assistant's bank,
    whose queries the tests check on the engines."""
    cfg = get_engine_config(engine)
    bank = {e["question"] for e in cfg.get("assistant_examples", [])}
    assert set(cfg.get("questions") or []) <= bank


def test_the_assistant_is_told_the_connectivity_has_no_subcortex():
    from neurolang_apps.utils.engine_registry import get_predicates

    predicates = get_predicates("neurosynth")
    for name in ("schaefer_structural_connectivity",
                 "destrieux_structural_connectivity"):
        text = " ".join(predicates[name]["description"].split())
        assert "Cortex only" in text and "thalamus" in text, name


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


@pytest.mark.network
@pytest.mark.parametrize("relation", ["schaefer_structural_connectivity",
                                      "destrieux_structural_connectivity"])
def test_the_connectivity_files_are_pairs_of_engine_regions(
    relation, tmp_path
):
    """The tables of the HCP connectivity, at their tag on GitLab, checked
    by their md5, as the engines download them."""
    import pandas as pd

    from neurolang_apps.utils import engine_registry

    entry = get_engine_config("neurosynth")["relations"][relation]
    assert "/-/raw/v" in entry["file"] and entry["md5"]  # pinned
    try:
        path = engine_registry._resolve_relation_file(entry, "e", tmp_path)
    except Exception as error:  # offline, GitLab down
        pytest.skip(f"cannot fetch {entry['file']}: {error!r}")
    table = pd.read_csv(path, sep="\t")
    assert list(table.columns) == ["region1", "region2", "streamlines"]
    regions = set(table.region1)
    assert len(regions) == (100 if "schaefer" in relation else 148)
    # named as the engines name the parcels (neurolang.atlases.label_name)
    assert not any("-" in r or "_" in r for r in regions)
    assert (table.region1 != table.region2).all()
    assert (table.streamlines > 0).all()
    both = table.set_index(["region1", "region2"]).streamlines
    swapped = table.set_index(["region2", "region1"]).streamlines
    swapped.index.names = both.index.names
    assert both.sort_index().equals(swapped.sort_index())  # both ways round
