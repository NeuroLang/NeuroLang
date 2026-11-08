from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ..frontend.probabilistic_frontend import NeurolangPDL

from nl_core.datalog.basic_representation import DatalogProgram
from nl_expr.exceptions import (
    InvalidCommandExpression,
    UnsupportedProgramError,
)
from nl_expr.expression_walker import ExpressionBasicEvaluator
from nl_expr.expressions import Command, Constant, Symbol
from nl_core.probabilistic.cplogic.program import CPLogicMixin
from ..commands import CommandsMixin


class Datalog(
    CommandsMixin, DatalogProgram, ExpressionBasicEvaluator, CPLogicMixin,
):
    pass


def test_invalid_commands_raise_exceptions():
    cmd = Command(
        Symbol("load_csv"), (Constant("https://somerandomurl.csv"),), ()
    )

    datalog = Datalog()
    with pytest.raises(InvalidCommandExpression):
        datalog.walk(cmd)

    cmd = Command(Symbol("print"), (Symbol("A"),), ())
    with pytest.raises(UnsupportedProgramError):
        datalog.walk(cmd)


@patch("neurolang.commands.pd.read_csv")
def test_load_csv_command_adds_tuple_set(mock_pd_readcsv):
    mock_study_ids = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    mock_pd_readcsv.return_value = mock_study_ids

    url = "https://github.com/neuroquery/.../data-neuroquery_version-1_metadata.tsv.gz"

    studies = Symbol("Studies")
    cmd = Command(
        Symbol("load_csv"),
        (studies, Constant(url),),
        ((Symbol("sep"), Constant("\t")),),
    )

    datalog = Datalog()
    datalog.walk(cmd)

    mock_pd_readcsv.assert_called_with(url, sep="\t")
    assert studies in datalog.symbol_table
    assert (
        datalog.symbol_table[studies].value
        == mock_study_ids.to_records(index=False).tolist()
    )


@patch("neurolang.commands.nibabel.load")
@patch("neurolang.commands.pd.read_csv")
@patch("neurolang.commands._fetch_files")
def test_load_atlas_command(mock_fetch_files, mock_read_csv, mock_nib_load):
    """
    Test that load_atlas command can load destrieux atlas in neurolang
    """
    # setup mocks for fetch methods
    mock_fetch_files.return_value = [
        "/home/users/mock_atlas_file",
        "/home/users/mock_labels_file",
    ]
    mock_labels = pd.DataFrame(
        {"id": [1, 2], "labels": ["L S_central", "L S_postcentral"]}
    )
    mock_read_csv.return_value = mock_labels
    shape = (3, 3, 3)
    dataobj = np.arange(27).reshape(shape)
    affine = np.eye(4)
    mock_image = namedtuple("SpatialImage", ["shape", "affine", "dataobj"])(
        shape, affine, dataobj
    )
    mock_nib_load.return_value = mock_image

    # Create command and walk it
    url = "https://www.nitrc.org/frs/download.php/11942/destrieux2009.tgz"
    labels_url = "destrieux2009_rois_labels_lateralized.csv"
    atlas_url = "destrieux2009_rois_lateralized.nii.gz"
    destrieux = Symbol("Destrieux")
    cmd = Command(
        Symbol("load_atlas"),
        (destrieux, Constant(atlas_url), Constant(labels_url), Constant(url),),
        (),
    )

    datalog = Datalog()
    datalog.walk(cmd)

    mock_fetch_files.assert_called_with(
        Path.home() / "neurolang_data",
        [
            (atlas_url, url, {"uncompress": True}),
            (labels_url, url, {"uncompress": True}),
        ],
    )
    assert destrieux in datalog.symbol_table
    assert len(datalog.symbol_table[destrieux].value) == 2


def test_command_end2end():
    """
    Create a PDL engine and use the .load_atlas command to load the data.
    """
    nl = NeurolangPDL()

    @nl.add_symbol
    def startswith(prefix: str, s: str) -> bool:
        return s.startswith(prefix)

    query = """.load_atlas(Destrieux, "destrieux2009_rois_lateralized.nii.gz", "destrieux2009_rois_labels_lateralized.csv", "https://www.nitrc.org/frs/download.php/11942/destrieux2009.tgz")
    LeftSulcus(name_, region) :- Destrieux(name_, region) , startswith("L S", name_)"""

    with nl.scope:
        nl.execute_datalog_program(query)
        res = nl.solve_all()

    assert len(res["LeftSulcus"]) == 31
    assert "LeftSulcus" not in nl.symbol_table


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _rows(result):
    return set(result.as_pandas_dataframe().itertuples(index=False, name=None))


def test_import_runs_the_module_where_it_is_imported(tmp_path):
    """The facts, rules and commands of a module are those of the program
    that imports it."""
    _write(tmp_path / "pairs.csv", "x,y\n1,a\n2,b\n")
    module = _write(
        tmp_path / "module.dl",
        f'.load_csv(pair, "{tmp_path / "pairs.csv"}")\n'
        'big(2)\nbig(3)\nboth(x) :- pair(x, y), big(x)\n'
    )
    nl = NeurolangPDL()
    result = nl.execute_datalog_program(
        f'.import("{module}")\nans(x) :- both(x)'
    )
    assert _rows(result) == {(2,)}
    assert {"big", "both", "pair"} <= set(nl.symbol_table)


def test_import_paths_are_relative_to_the_importing_file(tmp_path):
    """A module imports the modules beside it, and the program read from a
    file the modules beside the file; without a file, they are relative
    to the working directory."""
    _write(tmp_path / "lib" / "base.dl", "base(1)\n")
    _write(
        tmp_path / "lib" / "more.dl",
        '.import("base.dl")\nmore(x) :- base(x)\n',
    )
    main = _write(
        tmp_path / "main.dl", '.import("lib/more.dl")\nans(x) :- more(x)\n'
    )
    nl = NeurolangPDL()
    with nl.program_ir.importing_from(main):
        result = nl.execute_datalog_program(main.read_text())
    assert _rows(result) == {(1,)}

    nl = NeurolangPDL()
    with pytest.raises(InvalidCommandExpression, match="No Datalog module"):
        nl.execute_datalog_program(main.read_text())


def test_import_relative_to_the_working_directory(tmp_path, monkeypatch):
    _write(tmp_path / "lib" / "base.dl", "base(1)\n")
    monkeypatch.chdir(tmp_path)
    nl = NeurolangPDL()
    nl.execute_datalog_program('.import("lib/base.dl")')
    assert "base" in nl.symbol_table


def test_import_a_module_of_a_package():
    """``package:path`` names a module shipped in a Python package."""
    nl = NeurolangPDL()
    result = nl.execute_datalog_program(
        '.import("neurolang.tests.datalog_modules:sides.dl")\n'
        'ans(x) :- opposite("left", x)'
    )
    assert _rows(result) == {("right",)}


def test_a_module_of_probabilistic_facts(tmp_path):
    module = _write(
        tmp_path / "p.dl", "0.5 :: p(1)\n0.2 :: p(2)\nq(x) :- p(x)\n"
    )
    nl = NeurolangPDL()
    nl.import_datalog_module(module)
    result = nl.execute_datalog_program("ans(x, PROB) :- q(x)")
    assert _rows(result) == {(1, 0.5), (2, 0.2)}


def test_a_module_is_imported_once(tmp_path):
    """Importing a module again, from the program or from another module,
    does nothing."""
    module = _write(tmp_path / "m.dl", "n(1)\ntwice(x) :- n(x)\n")
    _write(tmp_path / "other.dl", '.import("m.dl")\n')
    nl = NeurolangPDL()
    nl.execute_datalog_program(f'.import("{module}")')
    nl.execute_datalog_program(
        f'.import("{tmp_path}/./m.dl")\n.import("{tmp_path}/other.dl")'
    )
    nl.import_datalog_module(module)
    assert len(nl.symbol_table["twice"].formulas) == 1
    assert len(nl.symbol_table["n"].value) == 1


def test_a_module_imported_by_a_module_is_imported_once(tmp_path):
    _write(tmp_path / "m.dl", "n(1)\ntwice(x) :- n(x)\n")
    other = _write(tmp_path / "other.dl", '.import("m.dl")\n')
    nl = NeurolangPDL()
    nl.import_datalog_module(other)
    nl.import_datalog_module(tmp_path / "m.dl")
    assert len(nl.symbol_table["twice"].formulas) == 1
    assert set(nl.program_ir.imported_modules()) == {
        str((tmp_path / name).resolve()) for name in ("m.dl", "other.dl")
    }


def test_a_module_is_imported_again_after_its_scope(tmp_path):
    """Leaving the scope a module was imported in forgets the module with
    its rules: it can be imported again."""
    module = _write(tmp_path / "m.dl", "n(1)\n")
    nl = NeurolangPDL()
    with nl.scope:
        nl.import_datalog_module(module)
    assert "n" not in nl.symbol_table
    nl.import_datalog_module(module)
    assert "n" in nl.symbol_table


def test_an_import_cycle_is_an_error(tmp_path):
    a = _write(tmp_path / "a.dl", '.import("b.dl")\na(1)\n')
    _write(tmp_path / "b.dl", '.import("a.dl")\nb(1)\n')
    nl = NeurolangPDL()
    with pytest.raises(InvalidCommandExpression, match="cycle") as error:
        nl.import_datalog_module(a)
    assert "a.dl -> " in str(error.value)
    assert "b.dl -> " in str(error.value)


def test_a_module_with_a_query_is_refused(tmp_path):
    module = _write(tmp_path / "q.dl", "r(1)\nans(x) :- r(x)\n")
    nl = NeurolangPDL()
    with pytest.raises(InvalidCommandExpression, match="query") as error:
        nl.import_datalog_module(module)
    assert str(module) in str(error.value)
    assert "r" not in nl.symbol_table


def test_a_parse_error_in_a_module_names_its_file_and_line(tmp_path):
    module = _write(tmp_path / "bad.dl", "r(1)\n\nq(x) :- r(x\n")
    nl = NeurolangPDL()
    with pytest.raises(InvalidCommandExpression) as error:
        nl.execute_datalog_program(f'.import("{module}")')
    assert f"{module}, line 3" in str(error.value)
    assert getattr(error.value, "line", None) is None


def test_an_error_in_a_module_names_its_file(tmp_path):
    _write(tmp_path / "bad.dl", "r(1)\n.nope(1)\n")
    outer = _write(tmp_path / "outer.dl", '.import("bad.dl")\n')
    nl = NeurolangPDL()
    with pytest.raises(InvalidCommandExpression) as error:
        nl.import_datalog_module(outer)
    message = str(error.value)
    assert f"{tmp_path / 'bad.dl'}" in message
    assert "not supported" in message
    assert message.count("Datalog module") == 1


def test_a_missing_module_is_an_error(tmp_path):
    nl = NeurolangPDL()
    with pytest.raises(InvalidCommandExpression, match="No Datalog module"):
        nl.import_datalog_module(tmp_path / "missing.dl")
    with pytest.raises(InvalidCommandExpression, match="No Datalog module"):
        nl.import_datalog_module("neurolang.tests.datalog_modules:no.dl")
    with pytest.raises(InvalidCommandExpression, match="package"):
        nl.import_datalog_module("no_such_package_xyz:m.dl")


def test_import_takes_one_path():
    nl = NeurolangPDL()
    with pytest.raises(InvalidCommandExpression, match="path"):
        nl.execute_datalog_program('.import("a.dl", "b.dl")')
