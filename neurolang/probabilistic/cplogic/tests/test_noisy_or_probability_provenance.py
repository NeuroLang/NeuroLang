import collections

from ....relational_algebra import (
    NamedRelationalAlgebraFrozenSet,
    NaturalJoin,
    Projection,
    str2columnstr,
)
from ....relational_algebra_provenance import ProvenanceAlgebraSet
from ..noisy_or_probability_provenance import (
    NoisyORProbabilityProvenanceSolver,
)


def make_prov_set(iterable, columns):
    return ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(columns, iterable),
        str2columnstr(columns[0]),
    )


def test_simple_noisy_or_projection():
    prov_col = "foo"
    columns = (prov_col, "bar", "baz")
    iterable = [
        (0.2, "a", "x"),
        (0.5, "b", "y"),
        (0.1, "a", "z"),
    ]
    prov_set = make_prov_set(iterable, columns)
    projection = Projection(prov_set, (str2columnstr("bar"),))
    solver = NoisyORProbabilityProvenanceSolver()
    result = solver.walk(projection)
    expected_tuples = [(1 - 0.8 * 0.9, "a"), (0.5, "b")]
    itertuple = collections.namedtuple("tuple", result.value.columns)
    expected_namedtuples = set(map(itertuple._make, expected_tuples))
    result_namedtuples = set(result.value)
    assert expected_namedtuples == result_namedtuples


def test_noisy_or_projection_and_naturaljoin():
    """
    A(x) <- ∃y Q(x, y)
    B(x) <- ∃y Z(x, y)
    C(x) <- A(x), B(x)

    R_Q =   | _p_ | x  | y  |   R_Z =   | _p_ | x  | y  |
            | 0.2 | x1 | y1 |           | 0.6 | x1 | y3 |
            | 0.1 | x1 | y2 |           | 0.9 | x3 | y3 |
            | 0.9 | x2 | y2 |

    R_A =   | _p_  | x  |       R_B =   | _p_  | x  |
            | 0.28 | x1 |               | 0.6  | x1 |
            | 0.9  | x2 |               | 0.9  | x3 |

    R_C =   | _p_   | x  |
            | 0.168 | x1 |
    """
    r_Q = make_prov_set(
        [(0.2, "x1", "y1"), (0.1, "x1", "y2"), (0.9, "x2", "y2")],
        ("_p_", "x", "y"),
    )
    r_Z = make_prov_set(
        [(0.6, "x1", "y3"), (0.9, "x3", "y3")], ("_p_", "x", "y"),
    )
    r_A = Projection(r_Q, (str2columnstr("x"),))
    r_B = Projection(r_Z, (str2columnstr("x"),))
    r_C = NaturalJoin(r_A, r_B)
    result = NoisyORProbabilityProvenanceSolver().walk(r_A)
    expected = make_prov_set(
        [(1 - 0.8 * 0.9, "x1"), (0.9, "x2")], ("_p_", "x")
    )
    assert result == expected
    result = NoisyORProbabilityProvenanceSolver().walk(r_B)
    expected = make_prov_set([(0.6, "x1"), (0.9, "x3")], ("_p_", "x"))
    assert result == expected
    result = NoisyORProbabilityProvenanceSolver().walk(r_C)
    expected = make_prov_set(
        [((1 - (1 - 0.2) * (1 - 0.1)) * 0.6, "x1")], ("_p_", "x"),
    )
    assert result == expected
