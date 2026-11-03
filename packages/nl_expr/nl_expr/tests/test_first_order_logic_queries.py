import typing

from .. import solver
from .. import expressions
from .. import logic
from ..typed_symbol_table import TypedSymbolTable

C_ = expressions.Constant
S_ = expressions.Symbol


def test_simple_symbol_query():
    ds = solver.FirstOrderLogicSolver(TypedSymbolTable())

    for s in range(5):
        sym = S_[int](str(s))
        ds.symbol_table[sym] = C_[int](s)

    def gt(a: int, b: int) -> bool:
        return a > b

    ds.symbol_table[S_('gt')] = C_(gt)

    x = S_[int]('x')
    query = expressions.Query[typing.AbstractSet[int]](
        x, ds.symbol_table['gt'](ds.symbol_table['3'], x)
    )

    res = ds.walk(query)
    assert isinstance(res, expressions.Constant[query.type])
    assert expressions.type_validation_value(
        res.value, typing.AbstractSet[int]
    )
    assert set(['0', '1', '2']) == res.value


def test_multiple_symbol_query():
    ds = solver.FirstOrderLogicSolver(TypedSymbolTable())

    for s in range(5):
        sym = S_[int](str(s))
        ds.symbol_table[sym] = C_[int](s)

    def gt(a: int, b: int) -> bool:
        return a > b

    ds.symbol_table[S_('gt')] = C_(gt)

    x = S_[int]('x')
    y = S_[int]('y')

    query = expressions.Query[typing.AbstractSet[typing.Tuple[int, int]]](
        C_((x, y)), (
            ds.symbol_table['gt'](ds.symbol_table['3'], x) &
            ds.symbol_table['gt'](x, y)
        )
    )

    res = ds.walk(query)
    assert isinstance(res, expressions.Constant[query.type])
    assert expressions.type_validation_value(
        res.value, typing.AbstractSet[typing.Tuple[int, int]]
    )
    assert set((str(x), str(y))
               for x in range(5)
               for y in range(5)
               if 3 > x and x > y) == res.value


def test_tuple_symbol_query():
    ds = solver.FirstOrderLogicSolver(TypedSymbolTable())

    for s in range(5):
        sym = S_[typing.Tuple[int, str]](str(s))
        ds.symbol_table[sym] = C_[typing.Tuple[int, str]](
            (C_[int](s), C_[str]('a' * s))
        )

    def gt(a: int, b: int) -> bool:
        return a > b

    ds.symbol_table[S_('gt')] = C_(gt)

    x = S_[typing.Tuple[int, str]]('x')

    with expressions.expressions_behave_as_objects():
        query = expressions.Query[typing.AbstractSet[typing.Tuple[int, str]]](
            x, ds.symbol_table['gt'](x[C_(0)], C_(2))
        )

    res = ds.walk(query)
    assert isinstance(res, expressions.Constant[query.type])
    assert expressions.type_validation_value(
        res.value, typing.AbstractSet[typing.Tuple[int, str]]
    )
    assert set(str(x) for x in range(5) if x > 2) == res.value


def test_existential_predicate():
    ds = solver.FirstOrderLogicSolver(TypedSymbolTable())

    for s in range(5):
        sym = S_[int](str(s))
        ds.symbol_table[sym] = C_[int](s)

    def gt(a: int, b: int) -> bool:
        return a > b

    ds.symbol_table[S_('gt')] = C_(gt)

    x = S_[int]('x')

    expression = logic.ExistentialPredicate(
        x, ds.symbol_table['gt'](x, C_(2))
    )
    res = ds.walk(expression)
    assert isinstance(res, expressions.Constant)
    assert res.type is bool
    assert res.value

    expression = logic.ExistentialPredicate(
        x, ds.symbol_table['gt'](x, C_(10))
    )
    res = ds.walk(expression)
    assert isinstance(res, expressions.Constant)
    assert res.type is bool
    assert not res.value


def test_existential_predicate_trivial():
    ds = solver.FirstOrderLogicSolver(TypedSymbolTable())

    for s in range(5):
        sym = S_[int](str(s))
        ds.symbol_table[sym] = C_[int](s)

    def gt(a: int, b: int) -> bool:
        return a > b

    ds.symbol_table[S_('gt')] = C_(gt)

    x = S_[int]('x')

    expression = logic.ExistentialPredicate(
        x,
        C_(True) | ds.symbol_table['gt'](x, C_(2))
    )
    res = ds.walk(expression)
    assert isinstance(res, expressions.Constant)
    assert res.type is bool
    assert res.value

    expression = logic.ExistentialPredicate(
        x,
        C_(False) & ds.symbol_table['gt'](x, C_(2))
    )
    res = ds.walk(expression)
    assert isinstance(res, expressions.Constant)
    assert res.type is bool
    assert not res.value


def test_existential_predicate_not_solved():
    ds = solver.FirstOrderLogicSolver(TypedSymbolTable())

    for s in range(5):
        sym = S_[int](str(s))
        ds.symbol_table[sym] = C_[int](s)

    def gt(a: int, b: int) -> bool:
        return a > b

    ds.symbol_table[S_('gt')] = C_(gt)

    x = S_[int]('x')
    y = S_[int]('y')

    expression = logic.ExistentialPredicate(
        x, ds.symbol_table['gt'](x, C_(2)) & ds.symbol_table['gt'](y, C_(2))
    )
    res = ds.walk(expression)
    assert isinstance(res, logic.ExistentialPredicate)
    assert res.head == expression.head
    assert res.body == res.body


def test_universal_predicate():
    ds = solver.FirstOrderLogicSolver(TypedSymbolTable())

    for s in range(5):
        sym = S_[int](str(s))
        ds.symbol_table[sym] = C_[int](s)

    def le(a: int, b: int) -> bool:
        return a <= b

    ds.symbol_table[S_('le')] = C_(le)

    x = S_[int]('x')

    expression = logic.UniversalPredicate(
        x, ds.symbol_table['le'](x, C_(2))
    )
    res = ds.walk(expression)
    assert isinstance(res, expressions.Constant)
    assert res.type is bool
    assert not res.value

    expression = logic.UniversalPredicate(
        x, ds.symbol_table['le'](x, C_(10))
    )
    res = ds.walk(expression)
    assert isinstance(res, expressions.Constant)
    assert res.type is bool
    assert res.value


def test_universal_predicate_trivial():
    ds = solver.FirstOrderLogicSolver(TypedSymbolTable())

    for s in range(5):
        sym = S_[int](str(s))
        ds.symbol_table[sym] = C_[int](s)

    def le(a: int, b: int) -> bool:
        return a <= b

    ds.symbol_table[S_('le')] = C_(le)

    x = S_[int]('x')

    expression = logic.UniversalPredicate(
        x,
        C_(False) & ds.symbol_table['le'](x, C_(2))
    )
    res = ds.walk(expression)
    assert isinstance(res, expressions.Constant)
    assert res.type is bool
    assert not res.value

    expression = logic.UniversalPredicate(
        x,
        C_(True) | ds.symbol_table['le'](x, C_(2))
    )
    res = ds.walk(expression)
    assert isinstance(res, expressions.Constant)
    assert res.type is bool
    assert res.value


def test_universal_predicate_not_solved():
    ds = solver.FirstOrderLogicSolver(TypedSymbolTable())

    for s in range(5):
        sym = S_[int](str(s))
        ds.symbol_table[sym] = C_[int](s)

    def le(a: int, b: int) -> bool:
        return a <= b

    ds.symbol_table[S_('le')] = C_(le)

    x = S_[int]('x')
    y = S_[int]('y')

    expression = logic.UniversalPredicate(
        x, ds.symbol_table['le'](x, C_(2)) & ds.symbol_table['le'](y, C_(2))
    )
    res = ds.walk(expression)
    assert isinstance(res, logic.UniversalPredicate)
    assert res.head == expression.head
    assert res.body == res.body
