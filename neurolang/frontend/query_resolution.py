from typing import AbstractSet, Callable, Container, Tuple
from uuid import uuid1

import numpy as np

from .. import neurolang as nl
from ..symbols_and_types import is_subtype
from ..region_solver_ds import Region
from ..regions import ExplicitVBR

from .query_resolution_expressions import (
    Expression, Symbol,
    Query, Exists, All
)

__all__ = ['QueryBuilder']


class QueryBuilder:
    def __init__(self, solver):
        self.solver = solver
        self.set_type = AbstractSet[self.solver.type]

        for k, v in self.solver.included_functions.items():
            self.solver.symbol_table[nl.Symbol[v.type](k)] = v

        for k, v in self.solver.included_functions.items():
            self.solver.symbol_table[nl.Symbol[v.type](k)] = v

    def get_symbol(self, symbol_name):
        if isinstance(symbol_name, Expression):
            symbol_name = symbol_name.expression.name
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
                Region
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

    @property
    def types(self):
        return self.solver.symbol_table.types

    def new_symbol(self, type, name=None):
        if isinstance(type, (tuple, list)):
            type = tuple(type)
            type = Tuple[type]

        if name is None:
            name = str(uuid1())
        return Expression(
            self,
            nl.Symbol[type](name)
        )

    def new_region_symbol(self, name=None):
        return self.new_symbol(Region, name=name)

    def add_tuple_set(self, iterable, types, name=None):
        if not isinstance(types, tuple) or len(types) == 1:
            if isinstance(types, tuple) and len(types) == 1:
                types = types[0]
                iterable = (e[0] for e in iterable)

            set_type = AbstractSet[types]
        else:
            types = tuple(types)
            set_type = AbstractSet[Tuple[types]]

        element_type = set_type.__args__[0]
        new_set = []
        for e in iterable:
            s = self.new_symbol(element_type).expression
            c = nl.Constant[element_type](e)
            self.solver.symbol_table[s] = c
            new_set.append(s)

        constant = nl.Constant[set_type](frozenset(new_set))

        symbol = self.new_symbol(set_type, name=name)
        self.solver.symbol_table[symbol.expression] = constant

        return symbol

    def query(self, symbol, predicate):
        return Query(
            self,
            nl.Query[AbstractSet[symbol.expression.type]](
                symbol.expression,
                predicate.expression
            ),
            symbol, predicate
        )

    def exists(self, symbol, predicate):
        return Exists(
            self,
            nl.ExistentialPredicate[bool](
                symbol.expression,
                predicate.expression
            ),
            symbol, predicate
        )

    def all(self, symbol, predicate):
        return All(
            self,
            nl.UniversalPredicate[bool](
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
        if not isinstance(region, Region):
            raise ValueError(f"region must be instance of {Region}")

        if result_symbol_name is None:
            result_symbol_name = str(uuid1())

        symbol = nl.Symbol[Region](result_symbol_name)
        self.solver.symbol_table[symbol] = nl.Constant[Region](region)

        return Symbol(self, result_symbol_name)

    def add_region_set(self, region_set, result_symbol_name=None):
        if not isinstance(region_set, Container):
            raise ValueError(f"region must be instance of {self.set_type}")

        for region in region_set:
            if result_symbol_name is None:
                result_symbol_name = str(uuid1())

            symbol = nl.Symbol[self.set_type](result_symbol_name)
            self.solver.symbol_table[symbol] = nl.Constant[Region](region)

        return Symbol(self, result_symbol_name)

    def create_region(self, spatial_image, label=1):
        region = ExplicitVBR(
            np.transpose((spatial_image.get_data() == label).nonzero()),
            spatial_image.affine, img_dim=spatial_image.shape
        )

        return region

    def add_atlas_set(self, name, atlas_labels, spatial_image):
        atlas_set = set()
        for label_number, label_name in atlas_labels:
            region = self.create_region(spatial_image, label=label_number)
            if len(region.voxels) == 0:
                continue
            symbol = nl.Symbol[Region](label_name)
            self.solver.symbol_table[symbol] = nl.Constant[Region](region)
            atlas_set.add(
                nl.Constant[Tuple[str, Region]](
                    (nl.Constant[str](label_name), symbol)
                )
            )
        atlas_set = nl.Constant[AbstractSet[Tuple[str, Region]]](
            frozenset(atlas_set)
        )
        atlas_symbol = nl.Symbol[atlas_set.type](name)
        self.solver.symbol_table[atlas_symbol] = atlas_set
        return self[atlas_symbol]

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
