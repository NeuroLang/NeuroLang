from .. import RelationalAlgebraSet


def test_relational_algebra_set_semantics_empty():
    ras = RelationalAlgebraSet()

    assert len(ras) == 0
    assert list(iter(ras)) == []

    ras.add((0, 1))
    assert (0, 1) in ras
    assert len(ras) == 1


def test_relational_algebra_set_semantics():
    a = [5, 4, 3, 2, 3, 1]
    ras = RelationalAlgebraSet(a)

    assert len(ras) == len(a) - 1
    ras.discard(5)
    assert 5 not in ras
    assert len(ras) == len(a) - 2
    ras.add(10)
    assert len(ras) == len(a) - 1
    assert 10 in ras
    assert all(a_ in ras for a_ in a if a_ != 5)


def test_relational_algebra_ra_projection():
    a = [(i % 2, i, i * 2) for i in range(5)]
    ras = RelationalAlgebraSet(a)

    ras_0 = ras.projection(0)
    assert (0,) in ras_0 and (1,) in ras_0
    assert len(ras_0) == 2

    ras_0 = ras.projection(0, 2)
    assert all((i % 2, i * 2) for i in range(5))


def test_relational_algebra_ra_selection():
    a = [(i % 2, i, i * 2) for i in range(5)]

    ras = RelationalAlgebraSet(a)

    ras_0 = ras.selection({0: 1})
    a_sel = set((i % 2, i, i * 2) for i in range(5) if i % 2 == 1)
    assert ras_0 == a_sel

    ras_0 = ras.selection({0: 1, 1: 2})
    a_sel = set(
        (i % 2, i, i * 2) for i in range(5)
        if i % 2 == 1 and i == 2
    )
    assert ras_0 == a_sel


def test_relational_algebra_ra_natural_join():
    a = [(i, i * 2) for i in range(5)]
    b = [(i * 2, i * 3) for i in range(5)]
    c = [(i, i * 2, i * 2, i * 3) for i in range(5)]

    ras_a = RelationalAlgebraSet(a)
    ras_b = RelationalAlgebraSet(b)
    ras_c = RelationalAlgebraSet(c)

    res = ras_a.natural_join(ras_b, [(1, 0)])
    assert res == ras_c
