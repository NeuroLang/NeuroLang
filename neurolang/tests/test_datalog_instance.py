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
