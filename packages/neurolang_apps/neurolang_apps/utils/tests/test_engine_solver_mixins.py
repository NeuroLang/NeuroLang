"""``solver_mixins``: the commands, or other pattern matches, an engine adds
to the solver of its NeurolangPDL, from a module named in engines.yaml."""

import textwrap

import pytest

from neurolang_apps.utils.engine_registry import (
    build_engine, get_engine_config,
)

MIXIN = "neurolang_apps.utils.tests.toy_commands:GreetingMixin"


def _write(tmp_path, text):
    path = tmp_path / "engines.yaml"
    path.write_text(textwrap.dedent(text))
    return path


def _rows(result):
    return {tuple(r) for r in result.as_pandas_dataframe().itertuples(
        index=False, name=None)}


def test_an_engine_adds_a_command(tmp_path):
    path = _write(tmp_path, f"""
        engines:
          toy:
            description: an engine with a command
            solver_mixins: ["{MIXIN}"]
    """)
    nl = build_engine("toy", tmp_path, yaml_path=path)
    result = nl.execute_datalog_program(
        '.greet("ada")\n.greet("alan")\nans(x) :- greeted(x)'
    )
    assert _rows(result) == {("ada",), ("alan",)}


def test_the_other_commands_are_still_there(tmp_path):
    path = _write(tmp_path, f"""
        engines:
          toy:
            description: an engine with a command
            solver_mixins: ["{MIXIN}"]
    """)
    (tmp_path / "t.csv").write_text("x,y\n1,a\n2,b\n")
    nl = build_engine("toy", tmp_path, yaml_path=path)
    result = nl.execute_datalog_program(
        f'.load_csv(pairs, "{tmp_path}/t.csv")\nans(x) :- pairs(x, y)'
    )
    assert _rows(result) == {(1,), (2,)}


def test_an_engine_without_mixins_has_no_such_command(tmp_path):
    from nl_expr.exceptions import UnsupportedProgramError

    path = _write(tmp_path, """
        engines:
          plain:
            description: no command
    """)
    nl = build_engine("plain", tmp_path, yaml_path=path)
    with pytest.raises(UnsupportedProgramError, match="not supported"):
        nl.execute_datalog_program('.greet("ada")')


def test_mixins_are_inherited_parent_first(tmp_path):
    path = _write(tmp_path, f"""
        engines:
          base:
            abstract: true
            solver_mixins: ["{MIXIN}"]
          child:
            extends: base
            description: inherits the command
            solver_mixins: ["x.y:Z"]
    """)
    assert get_engine_config("child", path)["solver_mixins"] == [
        MIXIN, "x.y:Z"]


@pytest.mark.parametrize(
    "entry,match",
    [
        ("no.such.module:Mixin", "no.such.module"),
        ("neurolang_apps.utils.tests.toy_commands:Nope", "Nope"),
        ("neurolang_apps.utils.tests.toy_commands", "module:Class"),
        ("neurolang_apps.utils.tests.toy_commands:NOT_A_CLASS", "class"),
    ],
)
def test_a_bad_entry_is_a_clear_error(tmp_path, entry, match):
    path = _write(tmp_path, f"""
        engines:
          toy:
            description: a bad mixin
            solver_mixins: ["{entry}"]
    """)
    with pytest.raises(ValueError, match=match) as raised:
        build_engine("toy", tmp_path, yaml_path=path)
    assert "solver_mixins" in str(raised.value)
