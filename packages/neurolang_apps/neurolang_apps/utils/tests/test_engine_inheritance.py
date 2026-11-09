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
        "destrieux", "neurostore", "neurosynth", "neurosynth_iit",
        "neurosynth_tracts", "tracts",
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
