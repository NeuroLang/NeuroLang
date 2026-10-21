from numpy import random
from ..interval_algebra import (
    converse, meets, before, starts, during,
    finishes, equals, overlaps, negate,
    get_intervals_relations
)
from ..regions import Region
from copy import deepcopy


def app(x, z, first, second, elements):
    elements.remove(x)
    elements.remove(z)
    for y in elements:
        if first(x, y) and second(y, z):
            return True
    return False


def composition(relations, domain, convert=False):
    if convert:
        relations = [[converse(f), converse(g)] for [f, g] in relations]
    return [
        lambda x, z, f=pair_of_fs[0], g=pair_of_fs[1]:
        app(x, z, f, g, deepcopy(domain)) for pair_of_fs in relations
    ]


def apply_composition(
    relations,
    parameters,
    negate_whole_expression=False,
    negations=None,
    conversion=False
):

    if conversion:
        parameters = list(reversed(parameters))
    res = True
    for i in range(len(relations)):
        result = relations[i](parameters[i][0], parameters[i][1])
        if negations:
            if negations[i] != (not result):
                res = False
                break
        else:
            if not result:
                res = False
                break

    return negate_whole_expression != res


def test_ia_relations_functions():
    intervals = [
        tuple([1, 2]),
        tuple([5, 7]),
        tuple([1, 5]),
        tuple([4, 6]),
        tuple([2, 4]),
        tuple([6, 7]),
        tuple([2, 4])
    ]

    assert before(intervals[0], intervals[1])
    assert meets(intervals[0], intervals[4])
    assert starts(intervals[0], intervals[2])
    assert during(intervals[4], intervals[2])
    assert overlaps(intervals[3], intervals[1])
    assert finishes(intervals[5], intervals[1])
    assert equals(intervals[4], intervals[6])

    assert not equals(intervals[1], intervals[0])
    assert not during(intervals[1], intervals[2])
    assert not overlaps(intervals[0], intervals[2])
    assert not starts(intervals[3], intervals[4])


def test_compositions():
    elems = [tuple([1, 2]), tuple([4, 6]), tuple([8, 10])]

    rel = composition([[before, before]], elems)
    assert apply_composition(rel, [[tuple([1, 2]), tuple([8, 10])]])

    rel = composition([[before, before]], elems)
    assert not apply_composition(rel, [[tuple([4, 6]), tuple([8, 10])]])

    elems.append(tuple([1, 5]))
    rel = composition([[starts, before]], elems)
    assert apply_composition(rel, [[tuple([1, 2]), tuple([8, 10])]])

    elems.append(tuple([1, 2]))
    rel = composition([[equals, starts]], elems)
    assert apply_composition(rel, [[tuple([1, 2]), tuple([1, 5])]])

    elems.append(tuple([2, 5]))
    rel = composition([[meets, overlaps]], elems)
    assert apply_composition(rel, [[tuple([1, 2]), tuple([4, 6])]])

    # multiple compositions
    elems.append(tuple([1, 2]))
    elems.append(tuple([1, 2]))
    rel = composition([[equals, equals], [equals, equals]], elems)
    assert apply_composition(
        rel, [[tuple([1, 2]), tuple([1, 2])], [tuple([1, 2]),
                                               tuple([1, 2])]]
    )

    elems.append(tuple([5, 8]))
    elems.append(tuple([0, 1]))
    rel = composition([[before, overlaps], [overlaps, overlaps]], elems)
    assert apply_composition(
        rel, [[tuple([0, 1]), tuple([4, 6])], [tuple([2, 5]),
                                               tuple([5, 8])]]
    )


def test_calculus_axioms():
    elems = [tuple(random.randint(1, 100, size=2)) for _ in range(10)]

    # Huntington's axiom
    r, s = random.choice([
        before, overlaps, during, meets, starts, finishes, equals
    ], 2)
    i, j = random.choice(range(len(elems)), 2, replace=False)
    assert not (not r(elems[i], elems[j]) or not s(elems[i], elems[j])) or (
        not ((not r(elems[i], elems[j])) or s(elems[i], elems[j]))
    ) == r(elems[i], elems[j])

    # identity
    i, j = random.choice(range(len(elems)), 2)
    elems.append(elems[j])
    rel = composition([[meets, equals]], elems)
    assert apply_composition(rel, [[elems[i], elems[j]]
                                   ]) == meets(elems[i], elems[j])

    i, j = random.choice(range(len(elems)), 2)
    elems.append(elems[i])
    rel = composition([[equals, meets]], elems)
    assert apply_composition(rel, [[elems[i], elems[j]]
                                   ]) == meets(elems[i], elems[j])

    # involution
    for op in [before, overlaps, during, meets, starts, finishes, equals]:
        i, j = random.choice(range(len(elems)), 2, replace=False)
        converse(converse(op))(elems[i], elems[j]) == op(elems[i], elems[j])

    # associativity
    r, s, t = random.choice([
        before, overlaps, during, meets, starts, finishes, equals
    ], 3)
    c1 = composition([[r, s]], elems)
    c2 = composition([[s, t]], elems)
    c = composition([[r, s], [s, t]], elems)
    i, j, k, l = random.choice(range(len(elems)), 4, replace=False)
    t1, t2, t3, t4 = elems[i], elems[j], elems[k], elems[l]
    assert (
        apply_composition(c1, [[t1, t2]]) and
        apply_composition(c2, [[t3, t4]])
    ) == apply_composition(c, [[t1, t2], [t3, t4]])

    # distributivity
    r, s, t = random.choice([
        before, overlaps, during, meets, starts, finishes, equals
    ], 3)
    c1 = composition([[r, t]], elems)
    c2 = composition([[s, t]], elems)
    c = composition([[random.choice([s, r]), t]], elems)
    i, j = random.choice(range(len(elems)), 2, replace=False)
    app = apply_composition(c, [[elems[i], elems[j]]])
    assert (apply_composition(c1, [[
        elems[i], elems[j]
    ]]) == app) or (apply_composition(c2, [[elems[i], elems[j]]]) == app)

    # inv-distrib
    r, s = random.choice([
        before, overlaps, during, meets, starts, finishes, equals
    ], 2)
    i, j, k, l = random.choice(range(len(elems)), 4, replace=False)
    [i, j, k,
     l] = [(q, p) for (p, q) in [elems[i], elems[j], elems[k], elems[l]]]
    assert any([r(i, j), s(k,
                           l)]) == any([converse(r)(j, i),
                                        converse(s)(l, k)])

    # inv-involutive-distr
    s, t = random.choice([
        before, overlaps, during, meets, starts, finishes, equals
    ], 2)
    c = composition([[s, t]], elems)
    inv_c = composition([[converse(t), converse(s)]], elems)
    i, j = random.choice(range(len(elems)), 2, replace=False)
    assert apply_composition(c, [[
        elems[i], elems[j]
    ]], conversion=True) == apply_composition(inv_c, [[elems[j], elems[i]]])

    # Tarski/ de Morgan
    r, s = random.choice([
        before, overlaps, during, meets, starts, finishes, equals
    ], 2)
    i, j = random.choice(range(len(elems)), 2, replace=False)
    c = composition([[converse(r), negate(r)]], elems)
    c2 = composition([[r, s]], elems)
    assert (
        apply_composition(
            c, [[elems[i], elems[j]]],
            negate_whole_expression=False,
            negations=[False, True]
        ) and apply_composition(
            c2, [[elems[i], elems[j]]], negate_whole_expression=True
        ) and (not s)
    ) == (not s)


def test_get_interval_relations_of_regions():
    r1 = Region((1, 1, 1), (2, 2, 2))
    r2 = Region((5, 5, 5), (8, 8, 8))
    assert get_intervals_relations(
        r1.bounding_box.limits, r2.bounding_box.limits
    ) == tuple(['b', 'b', 'b'])

    r1 = Region((1, 1, 1), (10, 10, 10))
    assert get_intervals_relations(
        r1.bounding_box.limits, r2.bounding_box.limits
    ) == tuple(['di', 'di', 'di'])

    r1 = Region((1, 1, 1), (6, 6, 6))
    assert get_intervals_relations(
        r1.bounding_box.limits, r2.bounding_box.limits
    ) == tuple(['o', 'o', 'o'])

    r2 = Region((1, 1, 1), (2, 2, 2))
    assert get_intervals_relations(
        r1.bounding_box.limits, r2.bounding_box.limits
    ) == tuple(['si', 'si', 'si'])

    r2 = Region((1, 1, 1), (6, 6, 6))
    assert get_intervals_relations(
        r1.bounding_box.limits, r2.bounding_box.limits
    ) == tuple(['e', 'e', 'e'])

    r1 = Region((5, 5, 5), (8, 8, 8))
    r2 = Region((8, 7, 12), (10, 8, 14))
    assert get_intervals_relations(
        r1.bounding_box.limits, r2.bounding_box.limits
    ) == tuple(['m', 'fi', 'b'])
    assert get_intervals_relations(
        r2.bounding_box.limits, r1.bounding_box.limits
    ) == tuple(['mi', 'f', 'bi'])

    r1 = Region((5, 5, 5), (8, 8, 8))
    r2 = Region((3, 3, 7), (6, 6, 9))
    assert get_intervals_relations(
        r1.bounding_box.limits, r2.bounding_box.limits
    ) == tuple(['oi', 'oi', 'o'])
    assert get_intervals_relations(
        r2.bounding_box.limits, r1.bounding_box.limits
    ) == tuple(['o', 'o', 'oi'])
