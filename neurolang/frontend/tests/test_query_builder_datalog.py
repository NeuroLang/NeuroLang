"""Tests for QueryBuilderDatalog's public API."""


def test_querybuilder_accepts_show_ra():
    """
    Check that execute_datalog_program accepts show_ra=True.

    show_ra=True should be accepted by execute_datalog_program
    and threaded through the call chain without error at the
    signature level (the actual RA printing is implemented later).
    """
    from neurolang.frontend import NeurolangPDL

    nl = NeurolangPDL()
    nl.add_tuple_set([(1,), (2,)], name="P")
    nl.execute_datalog_program("ans(x) :- P(x).", show_ra=True)


def test_show_ra_deterministic_frontend_reports_wrong_arity_error():
    """
    Check that show-ra in the deterministic frontend reports arity errors.

    ``schaefer`` has one column but the rule uses two: evaluating the
    query raises, so printing its RA plan must raise too.
    """
    import pytest

    from neurolang.exceptions import WrongArgumentsInPredicateError
    from neurolang.frontend import NeurolangDL

    for show_ra in (False, True):
        nl = NeurolangDL()
        nl.add_tuple_set([("A",), ("B",)], name="schaefer")
        with pytest.raises(WrongArgumentsInPredicateError):
            nl.execute_datalog_program(
                "label(l) :- schaefer(l, r).\nans(l) :- label(l).",
                show_ra=show_ra,
            )


def test_query_with_two_constants_on_a_derived_predicate():
    """Magic sets seed ``magic_q2^bbf(5, 'a')``, a rule of two constants."""
    from neurolang.frontend import NeurolangDL

    nl = NeurolangDL()
    nl.add_tuple_set([(5, "a", 2), (5, "b", 3), (2, "a", 4)], name="e")
    nl.execute_datalog_program(
        "q(x, y, z) :- e(x, y, z)\nq2(x, y, z) :- q(x, y, z)"
    )
    with nl.scope as e:
        res = nl.query((e.z,), e.q2(5, "a", e.z))
    assert set(res) == {(2,)}
