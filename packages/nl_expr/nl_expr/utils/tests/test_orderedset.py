from ...utils.orderedset import OrderedSet


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


def test_ordered_set_does_not_compare_its_elements_one_by_one():
    """Dropping the repeated elements takes their hashes, not a scan of
    the list of those before them, which compares expressions one by one."""
    comparisons = []

    class Element:
        def __init__(self, value):
            self.value = value

        def __hash__(self):
            return hash(self.value)

        def __eq__(self, other):
            comparisons.append(self.value)
            return self.value == other.value

    values = list(range(300)) + list(range(300))
    os = OrderedSet(Element(v) for v in values)

    assert [e.value for e in os] == list(range(300))
    assert len(os) == 300
    assert len(comparisons) <= len(values)
