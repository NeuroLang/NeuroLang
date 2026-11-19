"""Tests of the one copy of the tables engines built in one process share."""

import gc

import numpy as np
import pandas as pd
import pytest

from neurolang.frontend import NeurolangPDL
from neurolang_apps.utils import shared_tables

ROWS = [(i, i % 7, f"s{i % 100}") for i in range(shared_tables.MIN_ROWS)]


@pytest.fixture(autouse=True)
def empty_pool(monkeypatch):
    monkeypatch.setattr(shared_tables, "_POOL", {})


def engine(rows=ROWS, small=((1, 2),)):
    nl = NeurolangPDL()
    nl.add_tuple_set(rows, name="big")
    nl.add_tuple_set(list(small), name="small")
    return nl


def frame(nl, name="big"):
    return nl.symbol_table[name].value._container


def test_an_identical_table_of_another_engine_is_shared():
    first, second = engine(), engine()
    assert shared_tables.share(first) == 0
    assert shared_tables.share(second) > 0
    a, b = frame(first), frame(second)
    assert a is not b and a.equals(b)
    assert np.shares_memory(a.iloc[:, 0].to_numpy(), b.iloc[:, 0].to_numpy())
    assert shared_tables.share(second) == 0  # already shared
    # a query reads the same rows
    query = "ans(x, s) :- big(x, 3, s)"
    assert set(first.execute_datalog_program(query).to_unnamed()) == \
        set(second.execute_datalog_program(query).to_unnamed())


def test_other_rows_or_small_tables_are_not_shared():
    first = engine()
    other = engine(rows=ROWS[:-1] + [(-1, 0, "x")])
    shared_tables.share(first)
    assert shared_tables.share(other) == 0
    assert not np.shares_memory(
        frame(first).iloc[:, 0].to_numpy(), frame(other).iloc[:, 0].to_numpy()
    )
    assert frame(first, "small") is not frame(other, "small")


def test_writing_to_a_shared_table_leaves_the_other_as_it_was():
    first, second = engine(), engine()
    shared_tables.share(first)
    shared_tables.share(second)
    relation = second.symbol_table["big"].value
    relation._container.iloc[0, 0] = -5  # copy on write: its own now
    assert frame(first).iloc[0, 0] == 0


def test_a_table_mapped_from_a_file_is_left_to_the_system(tmp_path):
    import pyarrow as pa
    import pyarrow.ipc as ipc

    path = tmp_path / "t.arrow"
    table = pa.table({"a": np.arange(shared_tables.MIN_ROWS)})
    with ipc.new_file(str(path), table.schema) as writer:
        writer.write_table(table)
    mapped = ipc.open_file(pa.memory_map(str(path))).read_all().to_pandas(
        split_blocks=True)
    assert shared_tables._mapped(mapped)
    assert not shared_tables._mapped(pd.DataFrame({"a": [1]}))


def test_an_engine_dropped_leaves_the_pool():
    first = engine()
    shared_tables.share(first)
    del first
    gc.collect()
    second = engine()
    assert shared_tables.share(second) == 0
