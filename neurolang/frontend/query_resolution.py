import numpy as np
from uuid import uuid1
from typing import AbstractSet, Callable, Tuple
from neurolang.frontend.neurosynth_utils import NeuroSynthHandler
from .query_resolution_expressions import (
    Expression, Symbol,
    Query, Exists, All
)
from .. import neurolang as nl
from ..region_solver import Region
from ..regions import (
    ExplicitVBR,
    take_principal_regions
)
from ..expressions import is_leq_informative

__all__ = ['QueryBuilder']


class QueryBuilderBase(object):
    def __init__(self, solver, logic_programming=False):
        self.solver = solver
        self.set_type = AbstractSet[self.solver.type]
        self.logic_programming = logic_programming

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
        if isinstance(symbol_name, Symbol):
            symbol_name = symbol_name.symbol_name
        return self.get_symbol(symbol_name)

    def __contains__(self, symbol):
        return symbol in self.solver.symbol_table

    @property
    def types(self):
        return self.solver.symbol_table.types

    def new_symbol(self, type_, name=None):
        if isinstance(type_, (tuple, list)):
            type_ = tuple(type_)
            type_ = Tuple[type_]

        if name is None:
            name = str(uuid1())
        return Expression(
            self,
            nl.Symbol[type_](name)
        )

    @property
    def functions(self):
        return [
            s.name for s in self.solver.symbol_table
            if is_leq_informative(s.type, Callable)
        ]

    def add_symbol(self, value, name=None):
        if name is None:
            name = str(uuid1())

        if isinstance(value, Expression):
            value = value.expression
        else:
            value = nl.Constant(value)

        symbol = nl.Symbol[value.type](name)
        self.solver.symbol_table[symbol] = value

        return Symbol(self, name)

    def add_tuple_set(self, iterable, types, name=None):
        if not isinstance(types, tuple) or len(types) == 1:
            if isinstance(types, tuple) and len(types) == 1:
                types = types[0]
                iterable = (e[0] for e in iterable)

            set_type = AbstractSet[types]
        else:
            types = tuple(types)
            set_type = AbstractSet[Tuple[types]]

        constant = self._add_tuple_set_elements(iterable, set_type)
        if name is None:
            name = str(uuid1())

        symbol = nl.Symbol[set_type](name)
        self.solver.symbol_table[symbol] = constant

        return Symbol(self, name)

    def _add_tuple_set_elements(self, iterable, set_type):
        element_type = set_type.__args__[0]
        new_set = []
        for e in iterable:
            if not(isinstance(e, Symbol)):
                s = nl.Symbol[element_type](str(uuid1()))
                if is_leq_informative(element_type, Tuple):
                    c = nl.Constant[element_type](
                        tuple(nl.Constant(ee) for ee in e)
                    )
                else:
                    c = nl.Constant[element_type](e)
                self.solver.symbol_table[s] = c
            else:
                s = e.neurolang_symbol
            new_set.append(s)

        return nl.Constant[set_type](frozenset(new_set))


class RegionMixin(object):
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

    def new_region_symbol(self, name=None):
        return self.new_symbol(Region, name=name)

    def add_region(self, region, result_symbol_name=None):
        if not isinstance(region, self.solver.type):
            raise ValueError(
                f"type mismatch between region and solver type:"
                f" {self.solver.type}"
            )

        return self.add_symbol(region, result_symbol_name)

    def add_region_set(self, region_set, name=None):
        return self.add_tuple_set(region_set, Region, name=name)

    @staticmethod
    def create_region(spatial_image, label=1):
        region = ExplicitVBR(
            np.transpose((spatial_image.get_data() == label).nonzero()),
            spatial_image.affine, image_dim=spatial_image.shape
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
            self.solver.symbol_table[self.new_symbol(str).expression] = (
                nl.Constant[str](label_name)
            )

            tuple_symbol = self.new_symbol(Tuple[str, Region]).expression
            self.solver.symbol_table[tuple_symbol] = (
                nl.Constant[Tuple[str, Region]](
                    (nl.Constant[str](label_name), symbol)
                )
            )
            atlas_set.add(tuple_symbol)
        atlas_set = nl.Constant[AbstractSet[Tuple[str, Region]]](
            frozenset(atlas_set)
        )
        atlas_symbol = nl.Symbol[atlas_set.type](name)
        self.solver.symbol_table[atlas_symbol] = atlas_set
        return self[atlas_symbol]


class QueryBuilder(RegionMixin, QueryBuilderBase):
    def __init__(self, solver, logic_programming=False):
        super().__init__(
            solver, logic_programming=logic_programming
        )

        self.neurosynth_db = NeuroSynthHandler()

    def execute_expression(self, expression, result_symbol_name=None):
        if result_symbol_name is None:
            result_symbol_name = str(uuid1())

        result = self.solver.walk(expression)
        self.solver.symbol_table[nl.Symbol[result.type](
            result_symbol_name
        )] = result
        return Symbol(self, result_symbol_name)

    def query(self, head, predicate):

        if isinstance(head, tuple):
            symbols = ()
            for e in head:
                symbols += (e.expression,)
            head = nl.Constant(symbols)
        else:
            head = head.expression
        return Query(
            self,
            nl.Query[AbstractSet[head.type]](
                head,
                predicate.expression
            ),
            head, predicate
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

    def load_neurosynth_term_region(
        self, term: str, n_components=None, result_symbol_name=None
    ):
        if not result_symbol_name:
            result_symbol_name = str(uuid1())
        region_set = self.neurosynth_db.ns_region_set_from_term(term)
        if n_components:
            region_set = take_principal_regions(region_set, n_components)

        return self.add_tuple_set(region_set, ExplicitVBR, result_symbol_name)

    @property
    def symbols(self):
        return QuerySymbolsProxy(self)


class QuerySymbolsProxy(object):
    def __init__(self, query_builder):
        self._query_builder = query_builder

    def __getattr__(self, attr):
        try:
            return self._query_builder.get_symbol(attr)
        except ValueError:
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
