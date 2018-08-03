from typing import AbstractSet, Callable, Container
from uuid import uuid1
from .. import neurolang as nl
from ..symbols_and_types import is_subtype

from .query_resolution_expressions import (
    Expression, Symbol,
    Query, Exists
)

__all__ = ['QueryBuilder']


class QueryBuilder:
    def __init__(self, solver):
        self.solver = solver
        self.set_type = AbstractSet[self.solver.type]
        self.type = self.solver.type

        for k, v in self.solver.included_predicates.items():
            self.solver.symbol_table[nl.Symbol[v.type](k)] = v

        for k, v in self.solver.included_functions.items():
            self.solver.symbol_table[nl.Symbol[v.type](k)] = v

    def get_symbol(self, symbol_name):
        if symbol_name not in self.solver.symbol_table:
            raise ValueError('')
        return Symbol(self, symbol_name)

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
        if isinstance(symbol, Symbol):
            symbol = symbol.symbol

        functor = self.solver.symbol_table[predicate_name]

        predicate = nl.Predicate[self.set_type](functor, (symbol,))

        return predicate

    def define_function_application(self, function_name, symbol_name):
        if isinstance(symbol_name, str):

            symbol = nl.Symbol[self.set_type](symbol_name)
        elif isinstance(symbol_name, Symbol):
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
        return Symbol(self, result_symbol_name)

    def solve_query(self, query, result_symbol_name=None):

        if isinstance(query, Expression):
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
        return Symbol(self, result_symbol_name)

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

        return Symbol(self, result_symbol_name)

    def new_region_symbol(self, symbol_name=None):
        if symbol_name is None:
            symbol_name = str(uuid1())
        return Expression(
            self,
            nl.Symbol[self.type](symbol_name)
        )

    def query(self, symbol, predicate):
        return Query(
            self,
            nl.Query[symbol.expression.type](
                symbol.expression,
                predicate.expression
            ),
            symbol, predicate
        )

    def exists(self, symbol, predicate):
        return Exists(
            self,
            nl.ExistentialPredicate[symbol.expression.type](
                symbol.expression,
                predicate.expression
            ),
            symbol, predicate
        )

    def add_symbol(self, value, result_symbol_name=None):
        if result_symbol_name is None:
            result_symbol_name = str(uuid1())

        if isinstance(value, Expression):
            value = value.expression
        else:
            value = nl.Constant(value)

        symbol = nl.Symbol[self.set_type](result_symbol_name)
        self.solver.symbol_table[symbol] = value

        return Symbol(self, result_symbol_name)

    def add_region(self, region, result_symbol_name=None):
        if not isinstance(region, self.type):
            raise ValueError(f"region must be instance of {self.type}")

        if result_symbol_name is None:
            result_symbol_name = str(uuid1())

        symbol = nl.Symbol[self.type](result_symbol_name)
        self.solver.symbol_table[symbol] = nl.Constant[self.type](region)

        return Symbol(self, result_symbol_name)

    def add_region_set(self, region_set, result_symbol_name=None):
        if not isinstance(region_set, Container):
            raise ValueError(f"region must be instance of {self.set_type}")

        for region in region_set:
            if result_symbol_name is None:
                result_symbol_name = str(uuid1())

            symbol = nl.Symbol[self.set_type](result_symbol_name)
            self.solver.symbol_table[symbol] = nl.Constant[self.type](region)

        return Symbol(self, result_symbol_name)

    @property
    def symbols(self):
        return QuerySymbolsProxy(self)


class QuerySymbolsProxy:
    def __init__(self, query_builder):
        self._query_builder = query_builder

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
