from .. import neurolang as nl
from ..symbols_and_types import is_subtype, Symbol, Constant
from typing import AbstractSet, Callable, Container
from uuid import uuid1
import operator as op
from functools import wraps

__all__ = ['QueryBuilder']


class QueryBuilderExpression:
    def __init__(self, query_builder, expression):
        self.query_builder = query_builder
        self.expression = expression

    def do(self, result_symbol_name=None):
        return self.query_builder.execute_expression(
            self.expression,
            result_symbol_name=result_symbol_name
        )

    def __call__(self, *args, **kwargs):
        new_args = [
            a.expression if isinstance(a, QueryBuilderExpression)
            else Constant(a)
            for a in args
        ]

        new_kwargs = {
            k: (
                v.expression if isinstance(v, QueryBuilderExpression)
                else Constant(v)
            )
            for k, v in kwargs.items()
        }
        new_expression = self.expression(*new_args, **new_kwargs)
        return QueryBuilderOperation(
                self.query_builder, new_expression, self, args)

    def __repr__(self):
        if isinstance(self.expression, Constant):
            return repr(self.expression.value)
        elif isinstance(self.expression, Symbol):
            return f'{self.expression.name}'
        else:
            return object.__repr__(self)


binary_opeations = (
    op.add, op.sub, op.mul
)


def op_bind(op):
    @wraps(op)
    def f(self, *args):
        new_args = [
            arg.expression if isinstance(arg, QueryBuilderExpression)
            else Constant(arg)
            for arg in args
        ]
        new_expression = op(self.expression, *new_args)
        return QueryBuilderOperation(
            self.query_builder, new_expression, op,
            (self,) + args, infix=len(args) > 0
        )

    return f


def rop_bind(op):
    @wraps(op)
    def f(self, value):
        original_value = value
        if isinstance(value, QueryBuilderExpression):
            value = value.expression
        else:
            value = Constant(value)

        return QueryBuilderOperation(
            self.query_builder, op(self.expression, value),
            op, (self, original_value), infix=True
        )

    return f


for operator_name in dir(op):
    operator = getattr(op, operator_name)
    if operator_name.startswith('_'):
        continue

    name = f'__{operator_name}__'
    if name.endswith('___'):
        name = name[:-1]

    if not hasattr(QueryBuilderExpression, name):
        setattr(QueryBuilderExpression, name, op_bind(operator))


for operator in [
    op.add, op.sub, op.mul, op.matmul, op.truediv, op.floordiv,
    op.mod,  # op.divmod,
    op.pow, op.lshift, op.rshift, op.and_, op.xor,
    op.or_
]:
    name = f'__r{operator.__name__}__'
    if name.endswith('___'):
        name = name[:-1]

    setattr(QueryBuilderExpression, name, rop_bind(operator))


class QueryBuilderOperation(QueryBuilderExpression):
    def __init__(
        self, query_builder, expression,
        operator, arguments, infix=False
    ):
        self.query_builder = query_builder
        self.expression = expression
        self.operator = operator
        self.arguments = arguments
        self.infix = infix

    def __repr__(self):
        if isinstance(self.operator, QueryBuilderSymbol):
            op_repr = self.operator.symbol_name
        elif isinstance(self.operator, QueryBuilderOperation):
            op_repr = '({})'.format(repr(self.operator))
        else:
            op_repr = repr(self.operator)

        arguments_repr = []
        for a in self.arguments:
            if isinstance(a, QueryBuilderOperation):
                arguments_repr.append(
                    '( {} )'.format(repr(a))
                )
            elif isinstance(a, QueryBuilderSymbol):
                arguments_repr.append(a.symbol_name)
            else:
                arguments_repr.append(repr(a))

        if self.infix:
            return ' {} '.format(op_repr).join(arguments_repr)
        else:
            return '{}({})'.format(
                op_repr,
                ', '.join(arguments_repr)
            )


class QueryBuilderSymbol(QueryBuilderExpression):
    def __init__(self, query_builder, symbol_name):
        self.symbol_name = symbol_name
        self.query_builder = query_builder

    def __repr__(self):
        symbol = self.symbol
        if isinstance(symbol, Symbol):
            return(f'{self.symbol_name}: {symbol.type}')
        elif isinstance(symbol, Constant):
            if is_subtype(symbol.type, AbstractSet):
                contained = []
                all_symbols = (
                    self.query_builder.solver.symbol_table.symbols_by_type(
                        symbol.type.__args__[0]
                    )
                )
                for s in symbol.value:
                    if isinstance(s, Constant):
                        for k, v in all_symbols.items():
                            if isinstance(v, Constant) and s is v.value:
                                contained.append(k.name)
                                break
                    if isinstance(s, Symbol):
                        contained.append(s.name)
                return (f'{self.symbol_name}: {symbol.type} = {contained}')
            else:
                return (f'{self.symbol_name}: {symbol.type} = {symbol.value}')
        else:
            raise ValueError('...')

    @property
    def symbol(self):
        return self.query_builder.solver.symbol_table[self.symbol_name]

    @property
    def expression(self):
        return self.symbol

    @property
    def value(self):
        if isinstance(self.symbol, Constant):
            return self.symbol.value
        else:
            raise ValueError("This result type has no value")


class QueryBuilderQuery(QueryBuilderExpression):
    def __init__(self, query_builder, expression, symbol, predicate):
        self.query_builder = query_builder
        self.expression = expression
        self.symbol = symbol
        self.predicate = predicate

    def __repr__(self):
        return u'{{{s} | \u2203{s}: {p}}}'.format(
            s=repr(self.symbol),
            p=repr(self.predicate)
        )


class QueryBuilder:
    def __init__(self, solver):
        self.solver = solver
        self.set_type = AbstractSet[self.solver.type]
        self.type = self.solver.type

    def get_symbol(self, symbol_name):
        if symbol_name not in self.solver.symbol_table:
            raise ValueError('')
        return QueryBuilderSymbol(self, symbol_name)

    def __getitem__(self, symbol_name):
        return self.get_symbol(symbol_name)

    def __contains__(self, symbol):
        return symbol in self.solver.symbol_table

    @property
    def region_names(self):
        return [
            s.name for s in
            self.solver.symbol_table.symbols_by_type(
                self.type
            )
        ]

    @property
    def region_set_names(self):
        return [
            s.name for s in
            self.solver.symbol_table.symbols_by_type(
                self.set_type
            )
        ]

    @property
    def functions(self):
        return [
            s.name for s in self.solver.symbol_table
            if is_subtype(s.type, Callable)
        ]

    def define_predicate(self, predicate_name, symbol):
        if isinstance(symbol, str):
            symbol = self.get_symbol(symbol)
        if isinstance(symbol, QueryBuilderSymbol):
            symbol = symbol.symbol

        functor = self.solver.symbol_table[predicate_name]

        predicate = nl.Predicate[self.set_type](functor, (symbol,))

        return predicate

    def define_function_application(self, function_name, symbol_name):
        if isinstance(symbol_name, str):

            symbol = nl.Symbol[self.set_type](symbol_name)
        elif isinstance(symbol_name, QueryBuilderSymbol):
            symbol = symbol_name.symbol

        fa = nl.FunctionApplication[self.set_type](
            nl.Symbol[Callable[[self.set_type], self.set_type]](function_name),
            (symbol,)
        )
        return fa

    def execute_expression(self, expression, result_symbol_name=None):
        if result_symbol_name is None:
            result_symbol_name = str(uuid1())

        result = self.solver.walk(expression)
        self.solver.symbol_table[nl.Symbol[result.type](
            result_symbol_name
        )] = result
        return QueryBuilderSymbol(self, result_symbol_name)

    def solve_query(self, query, result_symbol_name=None):

        if isinstance(query, QueryBuilderExpression):
            query = query.expression

        if not isinstance(query, nl.Query):
            if result_symbol_name is None:
                result_symbol_name = str(uuid1())

            query = nl.Query[self.set_type](
                nl.Symbol[self.set_type](result_symbol_name),
                query
            )
        else:
            if result_symbol_name is not None:
                raise ValueError(
                    "Query result symbol name "
                    "already defined in query expression"
                )
            result_symbol_name = query.symbol.name

        self.solver.walk(query)
        return QueryBuilderSymbol(self, result_symbol_name)

    def neurosynth_term_to_region_set(self, term, result_symbol_name=None):

        if result_symbol_name is None:
            result_symbol_name = str(uuid1())

        predicate = nl.Predicate[str](
            nl.Symbol[Callable[[str], self.set_type]]('neurosynth_term'),
            (nl.Constant[str](term),)
        )

        query = nl.Query[self.set_type](
            nl.Symbol[self.set_type](result_symbol_name),
            predicate
        )
        query_res = self.solver.walk(query)
        for r in query_res.value.value:
            self.add_region(r)

        return QueryBuilderSymbol(self, result_symbol_name)

    def new_region_symbol(self, symbol_name=None):
        if symbol_name is None:
            symbol_name = str(uuid1())
        return QueryBuilderExpression(
            self,
            nl.Symbol[self.type](symbol_name)
        )

    def query(self, symbol, predicate):
        return QueryBuilderQuery(
            self,
            nl.Query[symbol.expression.type](
                symbol.expression,
                predicate.expression
            ),
            symbol, predicate
        )

    def add_symbol(self, value, result_symbol_name=None):
        if result_symbol_name is None:
            result_symbol_name = str(uuid1())

        if isinstance(value, QueryBuilderExpression):
            value = value.expression
        else:
            value = nl.Constant(value)

        symbol = nl.Symbol[self.set_type](result_symbol_name)
        self.solver.symbol_table[symbol] = value

        return QueryBuilderSymbol(self, result_symbol_name)

    def add_region(self, region, result_symbol_name=None):
        if not isinstance(region, self.type):
            raise ValueError(f"region must be instance of {self.type}")

        if result_symbol_name is None:
            result_symbol_name = str(uuid1())

        symbol = nl.Symbol[self.set_type](result_symbol_name)
        self.solver.symbol_table[symbol] = nl.Constant[self.type](region)

        return QueryBuilderSymbol(self, result_symbol_name)

    def add_region_set(self, region_set, result_symbol_name=None):
        if not isinstance(region_set, Container):
            raise ValueError(f"region must be instance of {self.set_type}")

        for region in region_set:
            if result_symbol_name is None:
                result_symbol_name = str(uuid1())

            symbol = nl.Symbol[self.set_type](result_symbol_name)
            self.solver.symbol_table[symbol] = nl.Constant[self.type](region)

        return QueryBuilderSymbol(self, result_symbol_name)

    @property
    def symbols(self):
        return QuerySymbolsProxy(self)


class QuerySymbolsProxy:
    def __init__(self, query_builder):
        self._query_builder = query_builder
        for k, v in self._query_builder.solver.included_predicates.items():
            self._query_builder.solver.symbol_table[Symbol[v.type](k)] = v

    def __getattr__(self, attr):
        try:
            return self._query_builder.get_symbol(attr)
        except ValueError as e:
            raise AttributeError()

    def __getitem__(self, attr):
        return self._query_builder.get_symbol(attr)

    def __setitem__(self, key, value):
        return self._query_builder.add_symbol(value, result_symbol_name=key)

    def __contains__(self, symbol):
        return symbol in self._query_builder.solver.symbol_table

    def __len__(self):
        return len(self._query_builder.solver.symbol_table)

    def __dir__(self):
        init = object.__dir__(self)
        init += [
            symbol.name
            for symbol in self._query_builder.solver.symbol_table
        ]
        return init

    def __repr__(self):
        init = [
            symbol.name
            for symbol in self._query_builder.solver.symbol_table
        ]

        return f'QuerySymbolsProxy with symbols {init}'
