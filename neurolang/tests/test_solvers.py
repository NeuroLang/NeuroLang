import typing

from .. import solver
from .. import expressions


def test_type_properties():
    class NewSolver(solver.GenericSolver):
        type = float
        type_name = 'float'

    n = NewSolver()
    assert n.type == float
    assert n.type_name == 'float'
    assert n.plural_type_name == 'floats'


def test_symbol_table():
    st = dict(a=1)

    gs = solver.GenericSolver()

    assert gs.symbol_table == {}

    gs.set_symbol_table(st)

    assert gs.symbol_table == st


def test_predicate_match():
    class NewSolver(solver.GenericSolver[solver.T]):
        def predicate_test(self, a: solver.T) -> bool:
            return True

    ir = expressions.Predicate(
            expressions.Symbol('test'),
            (expressions.Constant[float](0.),)
        )

    ns = NewSolver[float]()
    assert ns.walk(ir).type == bool


def test_predicate_symbol_table():
    class NewSolver(solver.GenericSolver[solver.T]):
        type = float

        def predicate_test(self, a: float) -> bool:
            return expressions.Symbol[bool]('b')

    ns = NewSolver[float]()

    def ff(x: float) -> bool:
        return x % 2 == 0
    ft = typing.Callable[[float], bool]

    sym = expressions.Symbol[ft]('a')
    f = expressions.Constant[ft](ff)
    ns.symbol_table[sym] = f

    symc = expressions.Symbol[float]('c')

    pt = expressions.Predicate(sym, (expressions.Constant(2.),))
    pf = expressions.Predicate(sym, (expressions.Constant(1.),))
    ps = expressions.Predicate(sym, (symc,))

    assert ns.walk(pt).value
    assert not ns.walk(pf).value
    assert ns.walk(ps).value


def test_numeric_operations_solver():
    s = solver.NumericOperationsSolver[int]()

    e = (
        expressions.Symbol[int]('a') -
        expressions.Symbol[int]('b')
    )

    assert e.type == expressions.ToBeInferred
    assert s.walk(e).type == int
