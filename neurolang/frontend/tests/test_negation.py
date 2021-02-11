import io
import itertools
from typing import AbstractSet, Callable, Tuple

import numpy as np
import pandas as pd
import pytest

from ...exceptions import (
    NegativeFormulaNotNamedRelationException,
    NegativeFormulaNotSafeRangeException,
    UnsupportedProgramError,
    UnsupportedQueryError,
    UnsupportedSolverError,
)
from ...probabilistic.exceptions import (
    ForbiddenConditionalQueryNonConjunctive,
    UnsupportedProbabilisticQueryError,
)
from ...regions import SphericalVolume
from ...utils.relational_algebra_set import (
    NamedRelationalAlgebraFrozenSet,
    RelationalAlgebraFrozenSet,
)
from ..probabilistic_frontend import (
    NeurolangPDL,
    lifted_solve_marg_query,
    lifted_solve_succ_query,
)


def assert_almost_equal(set_a, set_b):
    for tupl_a in set_a:
        assert any(
            all(
                np.isclose(term_a, term_b)
                if isinstance(term_a, (float, np.float32, np.float64))
                else term_a == term_b
                for term_a, term_b in zip(tupl_a, tupl_b)
            )
            for tupl_b in set_b
        )


def test_postprob_conjunct_with_wlq_result():
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [("a",), ("b",)],
        name="Network",
    )
    nl.add_tuple_set(
        [
            ("a", "s1"),
            ("b", "s1"),
            ("b", "s2"),
            ("c", "s2"),
            ("c", "s3"),
        ],
        name="NetworkReported",
    )
    with nl.environment as e:
        e.SegregationQuery[e.n, e.s] = (
            e.NetworkReported(e.n, e.s)
            & ~e.NetworkReported(e.n2, e.s)
            & e.Network(e.n2)
            & (e.n != e.n2)
        )
        result = nl.query((e.n, e.s), e.SegregationQuery(e.n, e.s))
    expected = {
        ("c", "s3"),
    }
    assert_almost_equal(result, expected)
