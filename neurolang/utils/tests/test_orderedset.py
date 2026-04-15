from .. import OrderedSet


def test_ordered_set():
    a = [5, 4, 3, 2, 3, 1]
    os = OrderedSet(a)

    assert all(a_ in os for a_ in a)
    assert len(os) == len(a) - 1
    assert all(o_ == a_ for o_, a_ in zip(os, [5, 4, 3, 2, 1]))
    assert all(os[i] == a[i] for i in range(4))

    b = os.copy()

    assert os == b
    assert os is not b
    b.remove(4)
    assert os != b
    assert len(os) == len(b) + 1
    assert os.issuperset(b)
    assert b.issubset(os)

    b.replace(1, 0)
    assert len(b) == len(os) - 1
    assert not b.issubset(os)
