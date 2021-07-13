from .. import Conjunction


def test_conjunction_repr_ellipsis_pattern():
    pattern = Conjunction(...)
    repr(pattern)
