from ..expressions import Constant, Symbol
from ..solver_datalog_naive import Fact
from ..datalog.instance import SetInstance, MapInstance

S_ = Symbol
C_ = Constant
F_ = Fact

Q = Symbol('Q')
P = Symbol('P')
a = C_(2)
b = C_(3)
c = C_(4)
d = C_(5)


def test_construct_empty_instance():
    SetInstance()
    MapInstance()


def test_set_instance_contains_facts():
    elements = {Q: frozenset({(a, ), (b, )})}
    instance = SetInstance(elements)
    assert F_(Q(a)) in instance
    assert F_(Q(b)) in instance
    assert F_(Q(c)) not in instance
    assert hash(instance) is not None


def test_map_instance_contains_facts():
    elements = {Q: frozenset({(a, ), (b, )})}
    instance = MapInstance(elements)
    assert Q in instance
    assert hash(instance) is not None


def test_construct_instance_from_factset():
    factset = {
        F_(Q(C_(1))),
        F_(Q(a)),
        F_(Q(b)),
    }
    instance = SetInstance(factset)
    assert instance.elements == {Q: frozenset({(C_(1), ), (a, ), (b, )})}


def test_convert_instance_to_set():
    elements = {
        Q: frozenset({(a, ), (b, )}),
    }
    set_instance = SetInstance(elements)
    map_instance = set_instance.as_map()
    assert set_instance.elements is map_instance.elements


def test_convert_instance_to_map():
    elements = {
        Q: frozenset({(a, ), (b, )}),
    }
    map_instance = MapInstance(elements)
    set_instance = map_instance.as_set()
    assert set_instance.elements is map_instance.elements


def test_instance_union():
    elements1 = {
        Q: frozenset({(a, ), (b, )}),
    }
    elements2 = {
        Q: frozenset({(c, ), (b, )}),
        P: frozenset({(b, )}),
    }
    instance = SetInstance(elements1) | SetInstance(elements2)
    assert F_(Q(a)) in instance
    assert F_(Q(b)) in instance
    assert F_(Q(c)) in instance
    assert F_(P(b)) in instance


def test_instance_difference():
    elements1 = {Q: frozenset({(a, ), (b, )}), P: frozenset({(b, )})}
    elements2 = {
        Q: frozenset({(c, )}),
        P: frozenset({(b, )}),
    }
    instance = SetInstance(elements2) - SetInstance(elements1)
    assert F_(Q(c)) in instance
    assert len(instance) == 1


def test_instance_intersection():
    elements1 = {Q: frozenset({(a, ), (b, )}), P: frozenset({(b, )})}
    elements2 = {
        Q: frozenset({(b, )}),
        P: frozenset({(b, )}),
    }
    instance = SetInstance(elements2) & SetInstance(elements1)
    assert F_(Q(b)) in instance
    assert F_(P(b)) in instance
    assert len(instance) == 2


def test_union_many_instances():
    instance_1 = SetInstance({
        Q: frozenset({(a, ), (b, )}),
    })
    instance_2 = SetInstance({
        Q: frozenset({(b, ), (c, )}),
    })
    instance_3 = SetInstance({
        Q: frozenset({(d, )}),
    })
    instances = [instance_1, instance_2, instance_3]
    instance = SetInstance.union(*instances)
    assert len(instance) == 4
    for x in (a, b, c, d):
        assert F_(Q(x)) in instance
