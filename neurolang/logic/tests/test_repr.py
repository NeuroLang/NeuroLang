from .. import Conjunction, Disjunction


def test_conjunction_repr_ellipsis_pattern():
    pattern = Conjunction(...)
    repr(pattern)


def test_disjunction_repr_ellipsis_pattern():
    pattern = Disjunction(...)
    repr(pattern)
