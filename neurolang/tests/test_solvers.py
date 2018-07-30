import typing
import operator as op

from .. import solver
from .. import expressions


class ReturnSymbolConstantApplication(solver.PatternWalker):
    @solver.add_match(solver.Constant)
    def constant(self, expression):
        return expression

    @solver.add_match(solver.Symbol)
    def symbol(self, expression):
        return expression

    @solver.add_match(solver.FunctionApplication)
    def fa(self, expression):
        new_f = self.walk(expression.functor)
        new_args = tuple(
            self.walk(a)
            for a in expression.args
        )

        if (
            new_f is expression.functor and
            all(a is b for a, b in zip(expression.args, new_args))
        ):
            return expression
        else:
            return self.walk(expressions.FunctionApplication(new_f, new_args))


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


def test_boolean_operations_solver_or():
    s = solver.BooleanOperationsSolver()

    or_ = (
        expressions.Constant(True) |
        expressions.Symbol[bool]('b')
    )

    r = s.walk(or_)
    assert isinstance(r, expressions.Constant)
    assert r.value

    or_ = (
        expressions.Symbol[bool]('b') |
        expressions.Constant(True)
    )

    r = s.walk(or_)
    assert isinstance(r, expressions.Constant)
    assert r.value


def test_boolean_operations_solver_and():
    s = solver.BooleanOperationsSolver()

    and_ = (
        expressions.Constant(False) &
        expressions.Symbol[bool]('b')
    )

    r = s.walk(and_)
    assert isinstance(r, expressions.Constant)
    assert not r.value

    and_ = (
        expressions.Symbol[bool]('b') &
        expressions.Constant(False)
    )

    r = s.walk(and_)
    assert isinstance(r, expressions.Constant)
    assert not r.value


def test_boolean_operations_rewrite_cast():
    class Dummy(
        solver.BooleanRewriteSolver,
        ReturnSymbolConstantApplication
    ):
        pass

    s = Dummy()
    a = expressions.Symbol[bool]('a')
    b = expressions.Symbol[bool]('b')

    or_ = a | b

    assert or_.type == expressions.ToBeInferred
    assert s.walk(or_).type == bool

    and_ = a & b

    assert and_.type == expressions.ToBeInferred
    assert s.walk(and_).type == bool


def test_boolean_operations_rewrite_constant_left():
    class Dummy(
        solver.BooleanRewriteSolver, ReturnSymbolConstantApplication
    ):
        pass

    s = Dummy()
    a = expressions.Symbol[bool]('a')

    original = a | expressions.Constant(True)
    rewritten = s.walk(original)
    assert rewritten.functor.value is original.functor.value
    assert rewritten.args[0] is original.args[1]
    assert rewritten.args[1] is original.args[0]


def test_boolean_operations_rewrite_nested_constant():
    class Dummy(
        solver.BooleanRewriteSolver, ReturnSymbolConstantApplication
    ):
        pass

    s = Dummy()
    a = expressions.Symbol[bool]('a')
    b = expressions.Symbol[bool]('b')

    or_ = (
        a |
        (expressions.Constant(True) | b)
    )

    rewritten_or = s.walk(or_)
    assert rewritten_or.args[0] is or_.args[1].args[0]
    assert rewritten_or.args[1].args[0] is a
    assert rewritten_or.args[1].args[1] is b
    assert rewritten_or.functor.value is or_.functor.value
    assert rewritten_or.args[1].functor.value is or_.args[1].functor.value

    original = a | b | a | expressions.Constant(True)
    t_ = s.walk(original)
    assert isinstance(t_.args[0], expressions.Constant)
    assert t_.args[0].value
    assert t_.args[1].args[0].args[0] is a
    assert t_.args[1].args[0].args[1] is b
    assert t_.args[1].args[1] is a


def test_boolean_operations_rewrite_inversion():
    class Dummy(
        solver.BooleanRewriteSolver, ReturnSymbolConstantApplication
    ):
        pass

    s = Dummy()
    a = expressions.Symbol[bool]('a')
    b = expressions.Symbol[bool]('b')

    t_ = ~(a | b)
    t_r = s.walk(t_)
    assert t_r.functor.value is op.and_
    assert t_r.args[0].functor.value is op.invert
    assert t_r.args[1].functor.value is op.invert
    assert t_r.args[0].args[0] is a
    assert t_r.args[1].args[0] is b
