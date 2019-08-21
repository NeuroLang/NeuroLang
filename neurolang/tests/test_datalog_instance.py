from ..expressions import Constant, Symbol
from ..solver_datalog_naive import Fact
from ..datalog.instance import SetInstance, MapInstance

S_ = Symbol
C_ = Constant
F_ = Fact

Q = Symbol('Q')


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
    elements = {Q: frozenset({(C_(2), ), (C_(3), )})}
    set_instance = SetInstance(elements)
    map_instance = set_instance.as_map()
    assert set_instance.elements is map_instance.elements


def test_convert_instance_to_map():
    elements = {Q: frozenset({(C_(2), ), (C_(3), )})}
    map_instance = MapInstance(elements)
    set_instance = map_instance.as_set()
    assert set_instance.elements is map_instance.elements
