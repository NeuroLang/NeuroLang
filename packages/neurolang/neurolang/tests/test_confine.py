"""Tests of the confinement of the statements of a session to its data."""
import pandas as pd
import pytest

from nl_expr.exceptions import InvalidCommandExpression
from neurolang.commands import DatalogModuleError
from neurolang.confine import (
    CONFINE_ENV,
    ConfinedError,
    check_read,
    check_write,
    is_url,
)
from neurolang.frontend import NeurolangDL


@pytest.fixture
def confined(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv(CONFINE_ENV, str(data))
    return data


def test_unconfined_statements_reach_what_the_process_can(tmp_path,
                                                          monkeypatch):
    monkeypatch.delenv(CONFINE_ENV, raising=False)
    check_read(tmp_path / "anything")
    check_read("https://example.org/table.csv")
    check_write(tmp_path / "out.csv")


def test_confined_statements_read_their_data_only(confined, tmp_path):
    check_read(confined / "table.csv")
    check_read(confined / "sub" / ".." / "table.csv")
    for path in (tmp_path / "secret.csv", confined / ".." / "secret.csv",
                 "/etc/passwd"):
        with pytest.raises(ConfinedError, match="reads no file"):
            check_read(path)
    with pytest.raises(ConfinedError, match="fetches no address"):
        check_read("http://169.254.169.254/latest/meta-data")
    with pytest.raises(ConfinedError, match="writes no result"):
        check_write(confined / "out.csv", "result")
    assert is_url("s3://bucket/key") and not is_url("C:/data/table.csv")


def test_the_commands_of_a_confined_program(confined, tmp_path):
    pd.DataFrame({"a": [1, 2]}).to_csv(confined / "ok.csv", index=False)
    pd.DataFrame({"a": [3]}).to_csv(tmp_path / "secret.csv", index=False)
    (tmp_path / "secret.dl").write_text("secret(1).\n")
    nl = NeurolangDL()
    nl.execute_datalog_program(f'.load_csv(ok, "{confined / "ok.csv"}")')
    for statement, error in (
        (f'.load_csv(s, "{tmp_path / "secret.csv"}")',
         InvalidCommandExpression),
        ('.load_csv(s, "https://example.org/x.csv")',
         InvalidCommandExpression),
        (f'.import("{tmp_path / "secret.dl"}")', DatalogModuleError),
        (f'.load_atlas(a, "{tmp_path / "secret.nii"}", "{tmp_path / "l.csv"}")',
         InvalidCommandExpression),
    ):
        with pytest.raises(error, match="reads no|fetches no"):
            nl.execute_datalog_program(statement)


def test_the_modules_an_engine_ships_are_imported(confined, tmp_path):
    shipped = tmp_path / "engine_modules"
    shipped.mkdir()
    (shipped / "vocabulary.dl").write_text("word(1).\n")
    nl = NeurolangDL()
    nl.program_ir.module_directories = (str(shipped),)
    nl.execute_datalog_program('.import("vocabulary.dl")')
    assert nl.execute_datalog_program("ans(x) :- word(x)") is not None
