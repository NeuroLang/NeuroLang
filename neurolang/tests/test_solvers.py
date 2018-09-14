import typing
import operator as op

from .. import solver
from .. import expressions

C_ = expressions.Constant
S_ = expressions.Symbol
F_ = expressions.FunctionApplication


class ReturnSymbolConstantApplication(solver.PatternWalker):
    @solver.add_match(expressions.Constant)
    def constant(self, expression):
        return expression

    @solver.add_match(expressions.Symbol)
    def symbol(self, expression):
        return expression

    @solver.add_match(expressions.FunctionApplication)
    def fa(self, expression):
        new_f = self.walk(expression.functor)
        new_args = tuple(self.walk(a) for a in expression.args)

        if (
            new_f is expression.functor and
            all(a is b for a, b in zip(expression.args, new_args))
        ):
            return expression
        else:
            return self.walk(F_(new_f, new_args))


class DummySolver(
    solver.BooleanRewriteSolver, ReturnSymbolConstantApplication
):
    pass


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
        def function_test(self, a: solver.T) -> bool:
            return True

    ir = expressions.FunctionApplication(S_('test'), (C_[float](0.), ))

    ns = NewSolver[float]()
    assert ns.walk(ir).type == bool


def test_predicate_symbol_table():
    class NewSolver(solver.GenericSolver[solver.T]):
        def function_test(self, a: float) -> bool:
            return S_[bool]('b')

    ns = NewSolver[float]()

    def ff(x: float) -> bool:
        return x % 2 == 0

    ft = typing.Callable[[float], bool]

    sym = S_[ft]('a')
    f = C_[ft](ff)
    ns.symbol_table[sym] = f

    symc = S_[float]('c')

    pt = expressions.FunctionApplication(sym, (C_(2.), ))
    pf = expressions.FunctionApplication(sym, (C_(1.), ))
    ps = expressions.FunctionApplication(sym, (symc, ))

    assert ns.walk(pt).value
    assert not ns.walk(pf).value
    assert isinstance(ns.walk(ps), expressions.NonConstant)


def test_numeric_operations_solver():
    class MatchAll(solver.PatternWalker):
        @solver.add_match(...)
        def expression(self, expression):
            return expression

    class TheSolver(solver.NumericOperationsSolver[int], MatchAll):
        pass

    s = TheSolver()

    e = S_[int]('a') - S_[int]('b')

    assert e.type == expressions.Unknown
    assert s.walk(e).type == int


def test_boolean_operations_solver_or():
    s = solver.BooleanOperationsSolver()

    or_ = C_(True) | S_[bool]('b')

    r = s.walk(or_)
    assert isinstance(r, expressions.Constant)
    assert r.value

    or_ = S_[bool]('b') | C_(True)

    r = s.walk(or_)
    assert isinstance(r, expressions.Constant)
    assert r.value


def test_boolean_operations_solver_and():
    s = solver.BooleanOperationsSolver()

    and_ = C_(False) & S_[bool]('b')

    r = s.walk(and_)
    assert isinstance(r, expressions.Constant)
    assert not r.value

    and_ = S_[bool]('b') & C_(False)

    r = s.walk(and_)
    assert isinstance(r, expressions.Constant)
    assert not r.value


def test_boolean_operations_rewrite_cast():
    s = DummySolver()
    a = S_[bool]('a')
    b = S_[bool]('b')

    or_ = a | b

    assert or_.type == expressions.Unknown
    assert s.walk(or_).type == bool

    and_ = a & b

    assert and_.type == expressions.Unknown
    assert s.walk(and_).type == bool


def test_boolean_operations_rewrite_constant_left():
    s = DummySolver()
    a = S_[bool]('a')

    original = a | C_(True)
    rewritten = s.walk(original)
    assert rewritten.functor.value is original.functor.value
    assert rewritten.args[0] is original.args[1]
    assert rewritten.args[1] is original.args[0]


def test_boolean_operations_rewrite_nested_constant():
    s = DummySolver()
    a = S_[bool]('a')
    b = S_[bool]('b')

    or_ = (a | (C_(True) | b))

    rewritten_or = s.walk(or_)
    assert rewritten_or.args[0] is or_.args[1].args[0]
    assert rewritten_or.args[1].args[0] is a
    assert rewritten_or.args[1].args[1] is b
    assert rewritten_or.functor.value is or_.functor.value
    assert rewritten_or.args[1].functor.value is or_.args[1].functor.value

    original = a | b | a | C_(True)
    t_ = s.walk(original)
    assert isinstance(t_.args[0], expressions.Constant)
    assert t_.args[0].value is True
    assert t_.args[1].args[0] is a
    assert t_.args[1].args[1].args[0] is a
    assert t_.args[1].args[1].args[1] is b


def test_boolean_operations_rewrite_inversion():
    s = DummySolver()
    a = S_[bool]('a')
    b = S_[bool]('b')

    t_ = ~(a | b)
    t_r = s.walk(t_)
    assert t_r.functor.value is op.and_
    assert t_r.args[0].functor.value is op.invert
    assert t_r.args[1].functor.value is op.invert
    assert t_r.args[0].args[0] is a
    assert t_r.args[1].args[0] is b


def test_partial_binary_evaluation():
    class ExpressionWalkHistorySolver(DummySolver):
        def __init__(self, *args, **kwargs):
            self.walked = []
            super().__init__(*args, **kwargs)

        def walk(self, expression):
            self.walked.append(expression)
            return super().walk(expression)

        def assert_walked_before(self, e1, e2):
            # make sure e1 was walked
            assert any(e is e1 for e in self.walked)
            # make sure e2 was walked
            assert any(e is e2 for e in self.walked)
            # get postion of e1 in walk history
            e1_idx = next(i for i, e in enumerate(self.walked) if e is e1)
            # get position of e2 in walk history
            e2_idx = next(i for i, e in enumerate(self.walked) if e is e2)
            # make sure e1 was walked before e2
            assert e1_idx < e2_idx

    s = ExpressionWalkHistorySolver()

    a = S_[bool]('a')
    b = S_[bool]('b')
    c = S_[bool]('c')
    d = S_[bool]('d')

    exp = (~(a | b)) & (~(c | d))
    s.walk(exp)
    s.assert_walked_before(exp.args[0], exp.args[1])

    s = ExpressionWalkHistorySolver()
    exp = a & (a | b)
    s.walk(exp)
    s.assert_walked_before(exp.args[0], exp.args[1])

    s = ExpressionWalkHistorySolver()
    exp = a & (~(b | c))
    s.walk(exp)
    s.assert_walked_before(exp.args[0], exp.args[1])


def test_boolean_operations_rewrite_inversion_in_conjunction():
    s = DummySolver()
    a = S_[bool]('a')
    b = S_[bool]('b')
    c = S_[bool]('c')
    d = S_[bool]('d')

    e = a & (~(b | c) & d)
    we = s.walk(e)
    assert we.functor.value is op.and_
    assert we.args[0] is a
    assert we.args[1].functor.value is op.and_
    assert we.args[1].args[0] is d
    assert we.args[1].args[1].functor.value is op.and_
    assert we.args[1].args[1].args[0].functor.value is op.invert
    assert we.args[1].args[1].args[1].functor.value is op.invert
    assert we.args[1].args[1].args[0].args[0] is b
    assert we.args[1].args[1].args[1].args[0] is c


def test_boolean_operations_rewrite_conj_composition_order():
    s = DummySolver()
    a = S_[bool]('a')
    b = S_[bool]('b')
    d = S_[bool]('d')
    e = S_[bool]('e')
    f = S_('f')

    exp = (b | (d & e)) & a
    assert exp.functor.value is op.and_
    assert exp.args[1] is a
    we = s.walk(exp)
    assert we.args[0] is a

    exp = (b | (d & e)) & f(e)
    assert exp.functor.value is op.and_
    assert exp.args[1].args[0] is e
    we = s.walk(exp)
    assert we.args[0].args[0] is e


def test_boolean_operations_conjunction_distribution():
    s = solver.FirstOrderLogicSolver()
    a, b, c = S_('a'), S_('b'), S_('c')
    e = (a & b) & c
    we = s.walk(e)

    assert we.args[0] is a
    assert we.args[1].args[0] is b
    assert we.args[1].args[1] is c
    assert we.functor is e.functor
    assert we.args[1].functor is e.args[0].functor
