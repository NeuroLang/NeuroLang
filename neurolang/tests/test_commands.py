from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ..frontend.probabilistic_frontend import NeurolangPDL

from ..datalog.basic_representation import DatalogProgram
from ..exceptions import InvalidCommandExpression, UnsupportedProgramError
from ..expression_walker import ExpressionBasicEvaluator
from ..expressions import Command, Constant, Symbol
from ..probabilistic.cplogic.program import CPLogicMixin
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
    LeftSulcus(name_, region) :- Destrieux(name_, region) & startswith("L S", name_)"""

    with nl.scope:
        nl.execute_datalog_program(query)
        res = nl.solve_all()

    assert len(res["LeftSulcus"]) == 31
    assert "LeftSulcus" not in nl.symbol_table
