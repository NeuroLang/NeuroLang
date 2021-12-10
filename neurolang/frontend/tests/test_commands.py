from unittest.mock import patch

import pandas as pd
import pytest

from ...datalog.basic_representation import DatalogProgram
from ...exceptions import InvalidCommandExpression, UnsupportedProgramError
from ...expression_walker import ExpressionBasicEvaluator
from ...expressions import Command, Constant, Symbol
from ...probabilistic.cplogic.program import CPLogicMixin
from ..commands import CommandsMixin


class Datalog(
    CommandsMixin, DatalogProgram, ExpressionBasicEvaluator, CPLogicMixin
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


@patch("neurolang.frontend.commands.pd.read_csv")
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
