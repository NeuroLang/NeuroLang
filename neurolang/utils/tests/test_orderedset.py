from .. import OrderedSet


def test_ordered_set():
    a = [5, 4, 3, 2, 3, 1]
    os = OrderedSet(a)

    assert all(a_ in os for a_ in a)
    assert len(os) == len(a) - 1
    assert all(o_ == a_ for o_, a_ in zip(os, [5, 4, 3, 2, 1]))
    assert all(os[i] == a[i] for i in range(4))
