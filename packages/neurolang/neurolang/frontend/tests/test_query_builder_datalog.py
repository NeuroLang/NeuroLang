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

    from nl_expr.exceptions import WrongArgumentsInPredicateError
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


def test_negation_within_an_existential_is_kept():
    """
    Check that ``exists(y; …, ~B(y))`` keeps the negation of ``B``.

    The parser used to drop the negated conjuncts of an existential, so
    the query below answered the rows of ``R`` without any ``S`` instead
    of those without an ``S`` outside ``U``.
    """
    from neurolang.frontend import NeurolangDL

    nl = NeurolangDL()
    nl.add_tuple_set([("a",), ("b",)], name="R")
    nl.add_tuple_set([("a", 1), ("b", 2)], name="S")
    nl.add_tuple_set([("a", 1)], name="U")
    res = nl.execute_datalog_program(
        "ans(x) :- R(x), ~exists(y; S(x, y), ~U(x, y))"
    )
    assert set(res.as_pandas_dataframe()["x"]) == {"a"}


def _near_engine(frontend):
    """
    An engine of 14371 studies, uniformly chosen, 158 of which are near
    a first focus and 50 others near a second, as in issue #1072.
    """
    nl = frontend()
    studies = [(s,) for s in range(14371)]
    if hasattr(nl, "add_uniform_probabilistic_choice_over_set"):
        nl.add_uniform_probabilistic_choice_over_set(studies, name="study")
    else:
        nl.add_tuple_set(studies, name="study")
    nl.add_tuple_set(
        [(s, "first") for s in range(158)]
        + [(s, "second") for s in range(158, 208)],
        name="focus",
    )
    return nl


def _near_query(focus, frontend):
    head = "ans(PROB)" if frontend.__name__ == "NeurolangPDL" else "ans(s)"
    return (
        f"near(s) :- focus(s, '{focus}').\n"
        f"{head} :- near(s), study(s)."
    )


def _near_count(result, frontend):
    """The studies near in *result*: its rows, or P times their number."""
    rows = result.as_pandas_dataframe()
    if frontend.__name__ == "NeurolangPDL":
        return round(rows.iloc[0, 0] * 14371)
    return len(rows)


def test_rules_of_a_second_query_are_added_with_a_warning():
    """
    A query program redefining a rule of an earlier query's adds to it:
    the result is their union (158 + 50 = 208 of 14371 studies, P =
    0.014474 for NeurolangPDL), with a warning; the scoped query gets
    the 50 studies of its own rule (brute force).
    """
    import warnings

    import pytest

    from neurolang.frontend import NeurolangDL, NeurolangPDL

    for frontend in (NeurolangDL, NeurolangPDL):
        nl = _near_engine(frontend)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            first = nl.execute_datalog_program(_near_query("first", frontend))
        assert _near_count(first, frontend) == 158
        with pytest.warns(UserWarning, match="near already defined"):
            second = nl.execute_datalog_program(
                _near_query("second", frontend)
            )
        assert _near_count(second, frontend) == 208

        nl = _near_engine(frontend)
        for focus, expected in (("first", 158), ("second", 50)):
            with warnings.catch_warnings(), nl.scope:
                warnings.simplefilter("error")
                result = nl.execute_datalog_program(
                    _near_query(focus, frontend)
                )
            assert _near_count(result, frontend) == expected


def test_rules_added_without_a_query_do_not_warn():
    """
    A rules-only program extends a predicate without warnings, and so
    does a query program defining a new one with several rules.
    """
    import warnings

    from neurolang.frontend import NeurolangDL

    nl = _near_engine(NeurolangDL)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        nl.execute_datalog_program("near(s) :- focus(s, 'first').")
        nl.execute_datalog_program("near(s) :- focus(s, 'second').")
        result = nl.execute_datalog_program(
            "far(s) :- near(s), focus(s, 'first').\n"
            "far(s) :- near(s), focus(s, 'second').\n"
            "ans(s) :- far(s), study(s)."
        )
    assert len(result) == 208
