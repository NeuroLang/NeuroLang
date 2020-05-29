import collections

from ....relational_algebra import (
    NaturalJoin,
    Projection,
    str2columnstr_constant,
)
from .. import testing
from ..noisy_or_probability_provenance import (
    NoisyORProbabilityProvenanceSolver,
)


def test_simple_noisy_or_projection():
    prov_col = "foo"
    columns = (prov_col, "bar", "baz")
    iterable = [
        (0.2, "a", "x"),
        (0.5, "b", "y"),
        (0.1, "a", "z"),
    ]
    prov_set = testing.make_prov_set(iterable, columns)
    projection = Projection(prov_set, (str2columnstr_constant("bar"),))
    solver = NoisyORProbabilityProvenanceSolver()
    result = solver.walk(projection)
    expected_tuples = [(1 - 0.8 * 0.9, "a"), (0.5, "b")]
    itertuple = collections.namedtuple("tuple", result.value.columns)
    assert all(itertuple._make(nt) in result.value for nt in expected_tuples)


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
    r_Q = testing.make_prov_set(
        [(0.2, "x1", "y1"), (0.1, "x1", "y2"), (0.9, "x2", "y2")],
        ("_p_", "x", "y"),
    )
    r_Z = testing.make_prov_set(
        [(0.6, "x1", "y3"), (0.9, "x3", "y3")], ("_p_", "x", "y"),
    )
    r_A = Projection(r_Q, (str2columnstr_constant("x"),))
    r_B = Projection(r_Z, (str2columnstr_constant("x"),))
    r_C = NaturalJoin(r_A, r_B)
    result = NoisyORProbabilityProvenanceSolver().walk(r_A)
    expected = testing.make_prov_set(
        [(1 - 0.8 * 0.9, "x1"), (0.9, "x2")], ("_p_", "x")
    )
    assert result == expected
    result = NoisyORProbabilityProvenanceSolver().walk(r_B)
    expected = testing.make_prov_set([(0.6, "x1"), (0.9, "x3")], ("_p_", "x"))
    assert result == expected
    result = NoisyORProbabilityProvenanceSolver().walk(r_C)
    expected = testing.make_prov_set(
        [((1 - (1 - 0.2) * (1 - 0.1)) * 0.6, "x1")], ("_p_", "x"),
    )
    assert result == expected
