"""Tests for QueryBuilderDatalog's public API."""

import pytest


def test_querybuilder_accepts_show_ra():
    """show_ra=True should be accepted by execute_datalog_program
    and threaded through the call chain without error at the
    signature level (the actual RA printing is implemented later)."""
    from neurolang.frontend import NeurolangPDL

    nl = NeurolangPDL()
    nl.add_tuple_set([(1,), (2,)], name="P")
    nl.execute_datalog_program("ans(x) :- P(x).", show_ra=True)
