from contextlib import contextmanager
from typing import AbstractSet, Callable, Tuple, List
from uuid import uuid1

import numpy as np

from .. import expressions as exp
from .. import logic
from ..region_solver import Region
from ..regions import (ExplicitVBR, ImplicitVBR, SphericalVolume,
                       take_principal_regions)
from ..type_system import Unknown, is_leq_informative
from .neurosynth_utils import NeuroSynthHandler, StudyID, TfIDf
from .query_resolution_expressions import (All, Exists, Expression, Query,
                                           Symbol)

__all__ = ['QueryBuilderFirstOrder']


class QueryBuilderBase:
    def __init__(self, solver, logic_programming=False):
        self.solver = solver
        self.set_type = AbstractSet[self.solver.type]
        self.logic_programming = logic_programming

        for k, v in self.solver.included_functions.items():
            self.symbol_table[exp.Symbol[v.type](k)] = v

        for k, v in self.solver.included_functions.items():
            self.symbol_table[exp.Symbol[v.type](k)] = v

        self._symbols_proxy = QuerySymbolsProxy(self)

    def get_symbol(self, symbol_name):
        if isinstance(symbol_name, Expression):
            symbol_name = symbol_name.expression.name
        if symbol_name not in self.symbol_table:
            raise ValueError(f'Symbol {symbol_name} not defined')
        return Symbol(self, symbol_name)

    def __getitem__(self, symbol_name):
        if isinstance(symbol_name, Symbol):
            symbol_name = symbol_name.symbol_name
        return self.get_symbol(symbol_name)

    def __contains__(self, symbol):
        return symbol in self.symbol_table

    @property
    def types(self):
        return self.symbol_table.types

    @property
    def symbol_table(self):
        return self.solver.symbol_table

    @property
    def symbols(self):
        return self._symbols_proxy

    @property
    @contextmanager
    def environment(self):
        old_dynamic_mode = self._symbols_proxy._dynamic_mode
        self._symbols_proxy._dynamic_mode = True
        try:
            yield self._symbols_proxy
        finally:
            self._symbols_proxy._dynamic_mode = old_dynamic_mode

    @property
    @contextmanager
    def scope(self):
        old_dynamic_mode = self._symbols_proxy._dynamic_mode
        self._symbols_proxy._dynamic_mode = True
        self.solver.push_scope()
        try:
            yield self._symbols_proxy
        finally:
            self.solver.pop_scope()
            self._symbols_proxy._dynamic_mode = old_dynamic_mode

    def new_symbol(self, type_=Unknown, name=None):
        if isinstance(type_, (tuple, list)):
            type_ = tuple(type_)
            type_ = Tuple[type_]

        if name is None:
            name = str(uuid1())
        return Expression(
            self,
            exp.Symbol[type_](name)
        )

    @property
    def functions(self):
        return [
            s.name for s in self.symbol_table
            if is_leq_informative(s.type, Callable)
        ]

    def add_symbol(self, value, name=None):
        name = self._obtain_symbol_name(name, value)

        if isinstance(value, Expression):
            value = value.expression
        elif isinstance(value, exp.Constant):
            pass
        else:
            value = exp.Constant(value)

        symbol = exp.Symbol[value.type](name)
        self.symbol_table[symbol] = value

        return Symbol(self, name)

    def _obtain_symbol_name(self, name, value):
        if name is not None:
            return name

        if not hasattr(value, '__qualname__'):
            return str(uuid1())

        if '.' in value.__qualname__:
            ix = value.__qualname__.rindex('.')
            name = value.__qualname__[ix + 1:]
        else:
            name = value.__qualname__

        return name

    def del_symbol(self, name):
        del self.symbol_table[name]

    def add_tuple_set(self, iterable, type_=Unknown, name=None):
        if not isinstance(type_, tuple) or len(type_) == 1:
            if isinstance(type_, tuple) and len(type_) == 1:
                type_ = type_[0]
                iterable = (e[0] for e in iterable)

            set_type = AbstractSet[type_]
        else:
            type_ = tuple(type_)
            set_type = AbstractSet[Tuple[type_]]

        constant = self._add_tuple_set_elements(iterable, set_type)
        if name is None:
            name = str(uuid1())

        symbol = exp.Symbol[set_type](name)
        self.symbol_table[symbol] = constant

        return Symbol(self, name)

    def _add_tuple_set_elements(self, iterable, set_type):
        element_type = set_type.__args__[0]
        new_set = []
        for e in iterable:
            if not(isinstance(e, Symbol)):
                s, c = self._create_symbol_and_get_constant(e, element_type)
                self.symbol_table[s] = c
            else:
                s = e.neurolang_symbol
            new_set.append(s)

        return exp.Constant[set_type](self.solver.new_set(new_set))

    @staticmethod
    def _create_symbol_and_get_constant(element, element_type):
        symbol = exp.Symbol[element_type].fresh()
        if isinstance(element, exp.Constant):
            constant = element.cast(element_type)
        elif is_leq_informative(element_type, Tuple):
            constant = exp.Constant[element_type](
                tuple(exp.Constant(ee) for ee in element)
            )
        else:
            constant = exp.Constant[element_type](element)
        return symbol, constant


class RegionMixin:
    @property
    def region_names(self):
        return [
            s.name for s in
            self.symbol_table.symbols_by_type(
                Region
            )
        ]

    @property
    def region_set_names(self):
        return [
            s.name for s in
            self.symbol_table.symbols_by_type(
                self.set_type
            )
        ]

    def new_region_symbol(self, name=None):
        return self.new_symbol(type_=Region, name=name)

    def add_region(self, region, name=None):
        if not isinstance(region, self.solver.type):
            raise ValueError(
                f"type mismatch between region and solver type:"
                f" {self.solver.type}"
            )

        return self.add_symbol(region, name)

    def add_region_set(self, region_set, name=None):
        return self.add_tuple_set(region_set, name=name, type_=Region)

    @staticmethod
    def create_region(spatial_image, label=1, prebuild_tree=False):
        voxels = np.transpose(
            (
                np.asanyarray(spatial_image.dataobj) == label
            ).nonzero()
        )
        if len(voxels) == 0:
            return None
        region = ExplicitVBR(
            voxels,
            spatial_image.affine, image_dim=spatial_image.shape,
            prebuild_tree=prebuild_tree
        )
        return region

    def add_atlas_set(self, name, atlas_labels, spatial_image):
        atlas_set = set()
        for label_number, label_name in atlas_labels:
            region = self.create_region(spatial_image, label=label_number)
            if region is None:
                continue
            symbol = exp.Symbol[Region](label_name)
            self.symbol_table[symbol] = exp.Constant[Region](region)
            self.symbol_table[self.new_symbol(type_=str).expression] = (
                exp.Constant[str](label_name)
            )

            tuple_symbol = self.new_symbol(type_=Tuple[str, Region]).expression
            self.symbol_table[tuple_symbol] = (
                exp.Constant[Tuple[str, Region]](
                    (exp.Constant[str](label_name), symbol)
                )
            )
            atlas_set.add(tuple_symbol)
        atlas_set = exp.Constant[AbstractSet[Tuple[str, Region]]](
            frozenset(atlas_set)
        )
        atlas_symbol = exp.Symbol[atlas_set.type](name)
        self.symbol_table[atlas_symbol] = atlas_set
        return self[atlas_symbol]

    def sphere(self, center, radius, name=None):
        sr = SphericalVolume(center, radius)
        symbol = self.add_region(sr, name)
        return symbol

    def make_implicit_regions_explicit(self, affine, dim):
        for region_symbol_name in self.region_names:
            region_symbol = self.get_symbol(region_symbol_name)
            region = region_symbol.value
            if isinstance(region, ImplicitVBR):
                self.add_region(
                    region.to_explicit_vbr(affine, dim), region_symbol_name
                )


class NeuroSynthMixin:
    def load_neurosynth_term_regions(
        self, term: str, n_components=None, name=None, q=0.01
    ):
        if not hasattr(self, 'neurosynth_db'):
            self.neurosynth_db = NeuroSynthHandler()
        if not name:
            name = str(uuid1())
        region_set = self.neurosynth_db.ns_region_set_from_term(term, q)
        if n_components:
            region_set = take_principal_regions(region_set, n_components)

        region_set = ((t,) for t in region_set)
        return self.add_tuple_set(
            region_set,
            type_=Tuple[ExplicitVBR],
            name=name
        )

    def load_neurosynth_term_study_ids(
        self, term: str, name: str = None, frequency_threshold: float = 0.05
    ):
        if not hasattr(self, 'neurosynth_db'):
            self.neurosynth_db = NeuroSynthHandler()
        if not name:
            name = str(uuid1())
        study_set = self.neurosynth_db.ns_study_id_set_from_term(
            term, frequency_threshold
        )
        return self.add_tuple_set(study_set, type_=Tuple[StudyID], name=name)

    def load_neurosynth_study_tfidf_feature_for_terms(
        self, terms: List[str], name: str = None,
    ):
        if not hasattr(self, 'neurosynth_db'):
            self.neurosynth_db = NeuroSynthHandler()
        if not name:
            name = str(uuid1())
        result_set = self.neurosynth_db.ns_study_tfidf_feature_for_terms(terms)
        return self.add_tuple_set(
            result_set, type_=Tuple[StudyID, str, TfIDf], name=name
        )


class QueryBuilderFirstOrder(RegionMixin, NeuroSynthMixin, QueryBuilderBase):
    def __init__(self, solver, logic_programming=False):
        super().__init__(
            solver, logic_programming=logic_programming
        )

    def execute_expression(self, expression, name=None):
        if name is None:
            name = str(uuid1())

        result = self.solver.walk(expression)
        self.symbol_table[exp.Symbol[result.type](
            name
        )] = result
        return Symbol(self, name)

    def query(self, head, predicate):

        if isinstance(head, tuple):
            symbols = ()
            for e in head:
                symbols += (e.expression,)
            head = exp.Constant(symbols)
        else:
            head = head.expression
        return Query(
            self,
            exp.Query[AbstractSet[head.type]](
                head,
                predicate.expression
            ),
            head, predicate
        )

    def exists(self, symbol, predicate):
        return Exists(
            self,
            logic.ExistentialPredicate[bool](
                symbol.expression,
                predicate.expression
            ),
            symbol, predicate
        )

    def all(self, symbol, predicate):
        return All(
            self,
            logic.UniversalPredicate[bool](
                symbol.expression,
                predicate.expression
            ),
            symbol, predicate
        )


class QuerySymbolsProxy:
    def __init__(self, query_builder):
        self._dynamic_mode = False
        self._query_builder = query_builder

    def __getattr__(self, name):
        if name in self.__getattribute__('_query_builder'):
            return self._query_builder.get_symbol(name)

        try:
            return super().__getattribute__(name)
        except AttributeError:
            if self._dynamic_mode:
                return self._query_builder.new_symbol(
                    Unknown, name=name
                )
            else:
                raise

    def __setattr__(self, name, value):
        if name == '_dynamic_mode':
            return super().__setattr__(name, value)
        elif self._dynamic_mode:
            return self._query_builder.add_symbol(value, name=name)
        else:
            return super().__setattr__(name, value)

    def __delattr__(self, name):
        if self._dynamic_mode and name:
            self._query_builder.del_symbol(name)
        else:
            super().__delattr__(name)

    def __getitem__(self, attr):
        return self._query_builder.get_symbol(attr)

    def __setitem__(self, key, value):
        return self._query_builder.add_symbol(value, name=key)

    def __contains__(self, symbol):
        return symbol in self._query_builder.symbol_table

    def __iter__(self):
        return iter(
            sorted(set(
                s.name for s in
                self._query_builder.symbol_table
            ))
        )

    def __len__(self):
        return len(self._query_builder.symbol_table)

    def __dir__(self):
        init = object.__dir__(self)
        init += [
            symbol.name
            for symbol in self._query_builder.symbol_table
        ]
        return init

    def __repr__(self):
        init = [
            symbol.name
            for symbol in self._query_builder.symbol_table
        ]

        return f'QuerySymbolsProxy with symbols {init}'
