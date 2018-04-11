from numpy import random
from ..interval_algebra import *
from copy import deepcopy


def foo(x, z, first, second, elems):
    elems.remove(x)
    elems.remove(z)
    for y in elems:
        if first(x, y) and second(y, z):
            return True
    return False


def composition(comp, elems, conv=False):
    if conv:
        comp = [[converse(f),converse(g)] for [f, g] in comp]
    return [lambda x, z, f=functions[0], g=functions[1]: foo(x, z, f, g, deepcopy(elems))  for functions in comp]


def apply_composition(foo, pars, negate=False, neg=None, conversion=False):
    # for i in range(len(foo)):
    #     if not foo[i](pars[i][0], pars[i][1]):
    #         return False
    # return True
    if conversion:
        # foo = [converse(f) for f in reversed(foo)]
        pars = list(reversed(pars))
    res = True
    for i in range(len(foo)):
        result = foo[i](pars[i][0], pars[i][1])
        if not neg:
            if not result:
                res = False
                break
        else:
            if neg[i] != (not result):
                res = False
                break

    #valid = all([neg[i] != foo[i](pars[i][0], pars[i][1]) for i in range(len(foo))])

    return negate != res


def test_ia_relations_functions():
    intervals = [tuple([1, 2]), tuple([5, 7]), tuple([1, 5]), tuple([4, 6]), tuple([2, 4]), tuple([6, 7]), tuple([2, 4])]

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


def test_IA_axioms():
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

    #multiple compositions
    elems.append(tuple([1, 2]))
    elems.append(tuple([1, 2]))
    rel = composition([[equals, equals], [equals, equals]], elems)
    assert apply_composition(rel, [[tuple([1, 2]), tuple([1, 2])], [tuple([1, 2]), tuple([1, 2])]])

    elems.append(tuple([5, 8]))
    elems.append(tuple([0, 1]))
    rel = composition([[before, overlaps], [overlaps, overlaps]], elems)
    assert apply_composition(rel, [[tuple([0, 1]), tuple([4, 6])], [tuple([2, 5]), tuple([5, 8])]])


    #Huntington's axiom
    r, s = random.choice([before, overlaps, during, meets, starts, finishes, equals], 2)
    i, j = random.choice(range(len(elems)), 2)
    assert not (not r(elems[i], elems[j]) or not s(elems[i], elems[j])) or (not (
    (not r(elems[i], elems[j])) or s(elems[i], elems[j]))) == r(elems[i], elems[j])


    #identity
    rel = composition([[meets, equals]], elems)
    all([apply_composition(rel, [[tuple([1, 2]), x]]) == meets(tuple([1, 2]), x) for x in elems])

    rel = composition([[equals, meets]], elems)
    all([apply_composition(rel, [[x, tuple([1, 2])]]) == meets(x, tuple([1, 2])) for x in elems])


    #involution
    for op in [before, overlaps, during, meets, starts, finishes, equals]:
        i, j = random.choice(range(len(elems)), 2)
        converse(converse(op))(elems[i], elems[j]) == op(elems[i], elems[j])


    #associativity
    elems.append(tuple([-1, 0]))
    c1 = composition([[before, meets]], elems)
    c2 = composition([[meets, before]], elems)
    c = composition([[before, meets], [meets, before]], elems)
    assert apply_composition(c1, [[tuple([-1, 0]), tuple([2, 5])]]) and apply_composition(c2, [[tuple([1, 2]), tuple([8, 10])]])  \
                                                        == apply_composition(c, [[tuple([-1, 0]), tuple([2, 5])], [tuple([1, 2]), tuple([8, 10])]])

    #composition
    r, s, t = random.choice([before, overlaps, during, meets, starts, finishes, equals], 3)
    c1 = composition([[r, t]], elems)
    c2 = composition([[s, t]], elems)
    c = composition([[random.choice([s, r]), t]], elems)
    i, j = random.choice(range(len(elems)), 2, replace=False)

    assert (apply_composition(c1, [[elems[i], elems[j]]]) or apply_composition(c2, [[elems[i], elems[j]]]) ) == apply_composition(c, [[elems[i], elems[j]]])


    # inv-distrib
    r, s = random.choice([before, overlaps, during, meets, starts, finishes, equals], 2)
    i, j, k, l = random.choice(range(len(elems)), 4)
    [i, j, k, l] = [(q, p) for (p, q) in [elems[i], elems[j], elems[k], elems[l]]]
    assert any([r(i, j), s(k, l)]) == converse(r)(i,j) or converse(s)(k, l)


    # inv-involutive-distr
    s, t = random.choice([before, overlaps, during, meets, starts, finishes, equals], 2)
    c = composition([[s, t]], elems)
    inv_c = composition([[converse(t), converse(s)]], elems)
    i, j = random.choice(range(len(elems)), 2, replace=False)
    assert apply_composition(c, [[elems[i], elems[j]]], conversion=True) == apply_composition(inv_c, [[elems[j], elems[i]]])

    # Tarski/ de Morgan
    r, s = random.choice([before, overlaps, during, meets, starts, finishes, equals], 2)
    i, j = random.choice(range(len(elems)), 2, replace=False)
    c = composition([[converse(r), negate(r)]], elems)
    c2 = composition([[r, s]], elems)
    assert (apply_composition(c, [[elems[i], elems[j]]], negate=False, neg=[False, True]) and apply_composition(c2, [[elems[i], elems[j]]], negate=True) and (not s)) == (not s)
