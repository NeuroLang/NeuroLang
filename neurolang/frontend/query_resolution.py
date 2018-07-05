from .. import neurolang as nl
from ..symbols_and_types import is_subtype, Symbol, Constant
from typing import AbstractSet, Callable, Container
from uuid import uuid1

__all__ = ['QueryBuilder']


class QueryBuilderSymbol:
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
                all_symbols = self.query_builder.solver.symbol_table.symbols_by_type(
                    symbol.type
                )
                for s in symbol.value:
                    for k, v in all_symbols.items():
                        if len(v.value) == 1 and s in v.value:
                            contained.append(k.name)
                if len(contained) == 1 and contained[0] == self.symbol_name:
                    return (f'{self.symbol_name}: {symbol.type} = {symbol.value}')
                else:
                    return (f'{self.symbol_name}: {symbol.type} = {contained}')
            else:
                return (f'{self.symbol_name}: {symbol.type} = {symbol.value}')
        else:
            raise ValueError('...')

    @property
    def symbol(self):
        return self.query_builder.solver.symbol_table[self.symbol_name]

    @property
    def value(self):
        if isinstance(self.symbol, Constant):
            return self.symbol.value
        else:
            raise ValueError("This result type has no value")


class QueryBuilder:
    def __init__(self, solver):
        self.solver = solver
        self.set_type = AbstractSet[self.solver.type]
        self.type = self.solver.type

    def get_symbol(self, symbol_name):
        if symbol_name not in self.solver.symbol_table:
            raise ValueError('')
        return QueryBuilderSymbol(self, symbol_name)

    @property
    def region_names(self):
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
        self.solver.symbol_table[nl.Symbol[result.type](result_symbol_name)] = result
        return QueryBuilderSymbol(self, result_symbol_name)

    def solve_query(self, query, result_symbol_name=None):

        if isinstance(query, QueryBuilderSymbol):
            query = query.symbol

        if not isinstance(query, nl.Query):
            if result_symbol_name is None:
                result_symbol_name = str(uuid1())

            query = nl.Query[self.set_type](nl.Symbol[self.set_type](result_symbol_name), query)
        else:
            if result_symbol_name is not None:
                raise ValueError("Query result symbol name already defined in query expression")
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

        query = nl.Query[self.set_type](nl.Symbol[self.set_type](result_symbol_name), predicate)
        query_res = self.solver.walk(query)
        for r in query_res.value.value:
            self.add_region(r)

        return QueryBuilderSymbol(self, result_symbol_name)

    def add_region(self, region, result_symbol_name=None):
        if not isinstance(region, self.type):
            raise ValueError(f"region must be instance of {self.type}")

        if result_symbol_name is None:
            result_symbol_name = str(uuid1())

        symbol = nl.Symbol[self.set_type](result_symbol_name)
        self.solver.symbol_table[symbol] = nl.Constant[self.set_type](frozenset((region,)))

        return QueryBuilderSymbol(self, result_symbol_name)

    def add_region_set(self, region, result_symbol_name=None):
        if not isinstance(region, Container):
            raise ValueError(f"region must be instance of {self.set_type}")

        if result_symbol_name is None:
            result_symbol_name = str(uuid1())

        symbol = nl.Symbol[self.set_type](result_symbol_name)
        self.solver.symbol_table[symbol] = nl.Constant[self.set_type](frozenset(region))

        return QueryBuilderSymbol(self, result_symbol_name)