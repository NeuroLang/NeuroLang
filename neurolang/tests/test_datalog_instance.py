from ..datalog import Fact
from ..datalog.instance import (
    FrozenMapInstance, FrozenSetInstance, MapInstance, SetInstance
)
from ..expressions import Constant, Symbol

S_ = Symbol
C_ = Constant
F_ = Fact

Q = Symbol('Q')
P = Symbol('P')


def test_set_instance_contains_facts():
    elements = {Q: ({(C_(2), ), (C_(3), )})}
    instance = FrozenSetInstance(elements)
    assert Q(C_(2)) in instance
    assert Q(C_(3)) in instance
    assert Q(C_(4)) not in instance
    assert hash(instance) is not None
    assert set(iter(instance)) == {Q(C_(2)), Q(C_(3))}


def test_map_instance_contains_facts():
    elements = {Q: ({(C_(2), ), (C_(3), )})}
    instance = FrozenMapInstance(elements)
    assert Q in instance
    assert instance[Q].value == elements[Q]
    assert hash(instance) is not None


def test_construct_instance_from_factset():
    factset = {
        Q(C_(1)),
        Q(C_(2)),
        Q(C_(3)),
    }
    instance = SetInstance(factset)
    assert instance.elements == {
        Q: frozenset({(C_(1), ), (C_(2), ), (C_(3), )})
    }


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


def test_instance_union_update():
    elements1 = {
        Q: frozenset({(C_(2), ), (C_(3), )}),
    }
    elements2 = {
        Q: frozenset({(C_(4), ), (C_(3), )}),
        P: frozenset({(C_(3), )}),
    }
    instance = SetInstance(elements1)
    instance |= SetInstance(elements2)
    assert Q(C_(2)) in instance
    assert Q(C_(3)) in instance
    assert Q(C_(4)) in instance
    assert P(C_(3)) in instance


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
