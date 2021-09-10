import numpy as np

from ..probabilistic_frontend import NeurolangPDL


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


def test_deterministic_segregation_query():
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [("a",), ("b",), ("c",)],
        name="Network",
    )
    nl.add_tuple_set(
        [("s1",), ("s2",), ("s3",)],
        name="Study",
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
        e.SegregationQuery[e.n, e.s] = e.NetworkReported(
            e.n, e.s
        ) & ~nl.exists(e.n2, e.NetworkReported(e.n2, e.s) & (e.n2 != e.n))
        result = nl.query((e.n, e.s), e.SegregationQuery(e.n, e.s))
    expected = {
        ("c", "s3"),
    }
    assert_almost_equal(result, expected)
