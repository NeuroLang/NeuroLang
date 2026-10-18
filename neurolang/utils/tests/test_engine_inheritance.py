import textwrap

import pytest

from neurolang.utils.engine_registry import (
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
    assert list_engine_names() == ["destrieux", "neurostore", "neurosynth"]
    for name in ("neurosynth", "neurostore"):
        cfg = get_engine_config(name)
        assert cfg["requires_mni_mask"] and cfg["use_base_symbols"]
        assert "study_with_peaks" in cfg["datalog_init"]
        assert {"peak_reported", "study", "term_in_study_tfidf"} <= set(
            cfg["predicates"]
        )
        assert "extends" not in cfg and "abstract" not in cfg
    assert get_engine_config("neurostore")["python_init"].endswith(
        "neurostore.init"
    )
    with pytest.raises(ValueError, match="abstract"):
        get_engine_config("coordinate_meta_analysis")

