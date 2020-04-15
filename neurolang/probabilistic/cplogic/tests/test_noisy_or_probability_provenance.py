import collections
import typing

import pandas as pd

from ....expressions import Constant
from ....relational_algebra import (
    ColumnStr,
    NamedRelationalAlgebraFrozenSet,
    Projection,
)
from ....relational_algebra_provenance import ProvenanceAlgebraSet
from ..noisy_or_probability_provenance import (
    NoisyORProbabilityProvenanceSolver,
    noisy_or_aggregation,
)


def test_noisy_or_aggregation():
    probs = pd.Series([0.8, 0.2, 0.3, 0.0])
    assert noisy_or_aggregation(probs) == 0.888


def test_noisy_or_projection():
    prov_col = "foo"
    columns = (prov_col, "bar", "baz")
    iterable = [
        (0.2, "a", "x"),
        (0.5, "b", "y"),
        (0.1, "a", "z"),
    ]
    relation = Constant[typing.AbstractSet](
        NamedRelationalAlgebraFrozenSet(iterable=iterable, columns=columns)
    )
    prov_relation = ProvenanceAlgebraSet(relation, prov_col)
    projection = Projection(prov_relation, (Constant(ColumnStr("bar")),))
    solver = NoisyORProbabilityProvenanceSolver()
    result = solver.walk(projection)
    expected_tuples = [(1 - 0.8 * 0.9, "a"), (0.5, "b")]
    itertuple = collections.namedtuple("tuple", result.value.columns)
    expected_namedtuples = set(map(itertuple._make, expected_tuples))
    result_namedtuples = set(result.value)
    assert expected_namedtuples == result_namedtuples
