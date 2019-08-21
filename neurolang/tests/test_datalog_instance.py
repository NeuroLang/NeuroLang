from ..expressions import Constant, Symbol
from ..solver_datalog_naive import Fact
from ..datalog.instance import SetInstance, MapInstance

S_ = Symbol
C_ = Constant
F_ = Fact

Q = Symbol('Q')
P = Symbol('P')


def test_set_instance_contains_facts():
    elements = {Q: frozenset({(C_(2), ), (C_(3), )})}
    instance = SetInstance(elements)
    assert F_(Q(C_(2))) in instance
    assert F_(Q(C_(3))) in instance
    assert F_(Q(C_(4))) not in instance
    assert hash(instance) is not None


def test_map_instance_contains_facts():
    elements = {Q: frozenset({(C_(2), ), (C_(3), )})}
    instance = MapInstance(elements)
    assert Q in instance
    assert hash(instance) is not None


def test_construct_instance_from_factset():
    factset = {
        F_(Q(C_(1))),
        F_(Q(C_(2))),
        F_(Q(C_(3))),
    }
    instance = SetInstance(factset)
    assert instance.elements == {
        Q: frozenset({(C_(1), ), (C_(2), ), (C_(3), )})
    }


def test_convert_instance_to_set():
    elements = {
        Q: frozenset({(C_(2), ), (C_(3), )}),
    }
    set_instance = SetInstance(elements)
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
    assert F_(Q(C_(2))) in instance
    assert F_(Q(C_(3))) in instance
    assert F_(Q(C_(4))) in instance
    assert F_(P(C_(3))) in instance


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
    assert F_(Q(C_(4))) in instance
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
    assert F_(Q(C_(3))) in instance
    assert F_(P(C_(3))) in instance
    assert len(instance) == 2
