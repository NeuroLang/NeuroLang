from ...expressions import Constant, Symbol
from .. import Fact
from ..instance import (FrozenMapInstance, FrozenSetInstance, MapInstance,
                        SetInstance)

S_ = Symbol
C_ = Constant
F_ = Fact

Q = Symbol('Q')
P = Symbol('P')


def test_frozenset_instance_contains_facts():
    elements = {Q: ({(C_(2), ), (C_(3), )})}
    instance = FrozenSetInstance(elements)
    assert Q(C_(2)) in instance
    assert Q(C_(3)) in instance
    assert Q(C_(4)) not in instance
    assert hash(instance) is not None


def test_set_instance_contains_facts():
    elements = {Q: ({(C_(2), ), (C_(3), )})}
    instance = SetInstance(elements)
    assert Q(C_(2)) in instance
    assert Q(C_(3)) in instance
    assert Q(C_(4)) not in instance
    assert len(set(instance) & {Q(C_(2)), Q(C_(3))}) == 2


def test_frozen_map_instance_contains_facts():
    elements = {Q: ({(C_(2), ), (C_(3), )})}
    instance = FrozenMapInstance(elements)
    assert Q in instance
    assert instance[Q].value == elements[Q]
    assert hash(instance) is not None


def test_map_instance_contains_facts():
    elements = {Q: ({(C_(2), ), (C_(3), )})}
    instance = MapInstance(elements)
    assert Q in instance
    assert instance[Q].value == elements[Q]


def test_copy():
    elements = {Q: ({(C_(2), ), (C_(3), )})}
    instance = SetInstance(elements)
    instance2 = instance.copy()

    assert instance == instance2


def test_build_instance_from_instance():
    elements = {Q: ({(C_(2), ), (C_(3), )})}
    instance = SetInstance(elements)
    instance2 = SetInstance(instance)

    assert instance == instance2
    assert instance.elements is not instance2.elements


def test_construct_instance_from_factset():
    factset = {
        Q(C_(1)),
        Q(C_(2)),
        Q(C_(3)),
    }
    instance = SetInstance(factset)
    assert len(instance) == 3
    assert all(f in instance for f in factset)


def test_map_instance_iterators():
    elements = {
        Q: frozenset({(C_(2), ), (C_(3), )}),
    }
    map_instance = FrozenMapInstance(elements)

    values = list(map_instance.values())
    assert len(values) == 1
    assert len(values[0].value - elements[Q]) == 0


def test_convert_instance_to_set():
    elements = {
        Q: frozenset({(C_(2), ), (C_(3), )}),
    }
    set_instance = FrozenSetInstance(elements)
    map_instance = set_instance.as_map()
    assert set_instance.elements is map_instance.elements


def test_convert_instance_to_map():
    elements = {
        Q: frozenset({(C_(2), ), (C_(3), )}),
    }
    map_instance = MapInstance(elements)
    set_instance = map_instance.as_set()
    assert set_instance.elements is map_instance.elements


def test_instance_union():
    elements1 = {
        Q: frozenset({(C_(2), ), (C_(3), )}),
    }
    elements2 = {
        Q: frozenset({(C_(4), ), (C_(3), )}),
        P: frozenset({(C_(3), )}),
    }
    instance = SetInstance(elements1) | SetInstance(elements2)
    assert Q(C_(2)) in instance
    assert Q(C_(3)) in instance
    assert Q(C_(4)) in instance
    assert P(C_(3)) in instance


def test_instance_difference():
    elements1 = {
        Q: frozenset({(C_(2), ), (C_(3), )}),
        P: frozenset({(C_(3), )})
    }
    elements2 = {
        Q: frozenset({(C_(4), )}),
        P: frozenset({(C_(3), )}),
    }
    instance = SetInstance(elements2) - SetInstance(elements1)
    assert Q(C_(4)) in instance
    assert len(instance) == 1


def test_instance_intersection():
    elements1 = {
        Q: frozenset({(C_(2), ), (C_(3), )}),
        P: frozenset({(C_(3), )})
    }
    elements2 = {
        Q: frozenset({(C_(3), )}),
        P: frozenset({(C_(3), )}),
    }
    instance = SetInstance(elements2) & SetInstance(elements1)
    assert Q(C_(3)) in instance
    assert P(C_(3)) in instance
    assert len(instance) == 2


def test_mutable_set_instance():
    elements1 = {
        Q: frozenset({(C_(2), ), (C_(3), )}),
    }
    instance = SetInstance(elements1)
    instance.discard(Q(C_(2)))

    assert len(instance) == 1
    assert Q(C_(3)) in instance

    instance.add(Q(C_(2)))
    instance.add(P(C_(3)))

    assert len(instance) == 3
    assert Q(C_(2)) in instance
    assert Q(C_(3)) in instance
    assert P(C_(3)) in instance


def test_instance_union_intersection_diff_update():
    elements1 = {
        Q: frozenset({(C_(2), ), (C_(3), )}),
    }
    elements2 = {
        Q: frozenset({(C_(4), ), (C_(3), )}),
        P: frozenset({(C_(3), )}),
    }
    instance = SetInstance(elements1)
    instance |= SetInstance(elements2)
    assert len(instance) == 4
    assert Q(C_(2)) in instance
    assert Q(C_(3)) in instance
    assert Q(C_(4)) in instance
    assert P(C_(3)) in instance

    instance = SetInstance(elements1)
    instance &= SetInstance(elements2)
    assert len(instance) == 1
    assert Q(C_(3)) in instance

    instance = SetInstance(elements1)
    instance -= SetInstance(elements2)
    assert len(instance) == 1
    assert Q(C_(2)) in instance


def test_instance_difference_update():
    elements1 = {
        Q: frozenset({(C_(2), ), (C_(3), )}),
        P: frozenset({(C_(3), )})
    }
    elements2 = {
        Q: frozenset({(C_(4), )}),
        P: frozenset({(C_(3), )}),
    }
    instance = SetInstance(elements2)
    instance -= SetInstance(elements1)
    assert Q(C_(4)) in instance
    assert len(instance) == 1


def test_instance_intersection_update():
    elements1 = {
        Q: frozenset({(C_(2), ), (C_(3), )}),
        P: frozenset({(C_(3), )})
    }
    elements2 = {
        Q: frozenset({(C_(3), )}),
        P: frozenset({(C_(3), )}),
    }
    instance = SetInstance(elements2)
    instance &= SetInstance(elements1)
    assert Q(C_(3)) in instance
    assert P(C_(3)) in instance
    assert len(instance) == 2
