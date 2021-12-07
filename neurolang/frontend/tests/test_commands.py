import pytest

from ...datalog.basic_representation import DatalogProgram
from ...exceptions import InvalidCommandExpression, UnsupportedProgramError
from ...expression_walker import ExpressionBasicEvaluator
from ...expressions import Command, Symbol
from ..commands import CommandsMixin


class Datalog(CommandsMixin, DatalogProgram, ExpressionBasicEvaluator):
    pass


def test_invalid_commands_raises_exceptions():
    cmd = Command("load_csv", ("https://somerandomurl.csv",), {})

    datalog = Datalog()
    with pytest.raises(InvalidCommandExpression):
        datalog.walk(cmd)

    cmd = Command("print", (Symbol("A"),), {})
    with pytest.raises(UnsupportedProgramError):
        datalog.walk(cmd)


def test_load_csv_command_adds_tuple_set():
    studies = Symbol("Studies")
    cmd = Command(
        "load_csv",
        (
            studies,
            "https://github.com/neuroquery/neuroquery_data/raw/master/data/data-neuroquery_version-1_metadata.tsv.gz",
        ),
        {"sep": "\t"},
    )

    datalog = Datalog()
    datalog.walk(cmd)

    assert studies in datalog.symbol_table
