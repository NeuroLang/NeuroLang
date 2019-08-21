from typing import AbstractSet, Callable, Tuple
from uuid import uuid1

import numpy as np

from .. import neurolang as nl
from .. import solver_datalog_naive as sdb
from ..datalog_chase import build_chase_solution
from ..expressions import is_leq_informative, Unknown
from ..region_solver import Region
from ..regions import (
   ExplicitVBR, take_principal_regions, SphericalVolume, ImplicitVBR
)
from .neurosynth_utils import NeuroSynthHandler
from .query_resolution_expressions import (All, Exists, Expression, Fact,
                                           Implication, Query, Symbol)

__all__ = ['QueryBuilderFirstOrder']


class QueryBuilderBase(object):
    def __init__(self, solver, logic_programming=False):
        self.solver = solver
        self.set_type = AbstractSet[self.solver.type]
        self.logic_programming = logic_programming

        for k, v in self.solver.included_functions.items():
            self.symbol_table[nl.Symbol[v.type](k)] = v

        for k, v in self.solver.included_functions.items():
            self.symbol_table[nl.Symbol[v.type](k)] = v

    def get_symbol(self, symbol_name):
        if isinstance(symbol_name, Expression):
            symbol_name = symbol_name.expression.name
        if symbol_name not in self.symbol_table:
            raise ValueError('')
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
        return QuerySymbolsProxy(self)

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
            s.name for s in self.symbol_table
            if is_leq_informative(s.type, Callable)
        ]

    def add_symbol(self, value, name=None):
        if name is None:
            name = str(uuid1())

        if isinstance(value, Expression):
            value = value.expression
        elif isinstance(value, nl.Constant):
            pass
        else:
            value = nl.Constant(value)

        symbol = nl.Symbol[value.type](name)
        self.symbol_table[symbol] = value

        return Symbol(self, name)

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

        symbol = nl.Symbol[set_type](name)
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

        return nl.Constant[set_type](self.solver.new_set(new_set))

    @staticmethod
    def _create_symbol_and_get_constant(element, element_type):
        symbol = nl.Symbol[element_type](str(uuid1()))
        if isinstance(element, nl.Constant):
            constant = element.cast(element_type)
        elif is_leq_informative(element_type, Tuple):
            constant = nl.Constant[element_type](
                tuple(nl.Constant(ee) for ee in element)
            )
        else:
            constant = nl.Constant[element_type](element)
        return symbol, constant


class RegionMixin(object):
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
        return self.new_symbol(Region, name=name)

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
        voxels = np.transpose((spatial_image.get_data() == label).nonzero())
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
            symbol = nl.Symbol[Region](label_name)
            self.symbol_table[symbol] = nl.Constant[Region](region)
            self.symbol_table[self.new_symbol(str).expression] = (
                nl.Constant[str](label_name)
            )

            tuple_symbol = self.new_symbol(Tuple[str, Region]).expression
            self.symbol_table[tuple_symbol] = (
                nl.Constant[Tuple[str, Region]](
                    (nl.Constant[str](label_name), symbol)
                )
            )
            atlas_set.add(tuple_symbol)
        atlas_set = nl.Constant[AbstractSet[Tuple[str, Region]]](
            frozenset(atlas_set)
        )
        atlas_symbol = nl.Symbol[atlas_set.type](name)
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


class NeuroSynthMixin(object):
    def load_neurosynth_term_regions(
        self, term: str, n_components=None, name=None
    ):
        if not hasattr(self, 'neurosynth_db'):
            self.neurosynth_db = NeuroSynthHandler()

        if not name:
            name = str(uuid1())
        region_set = self.neurosynth_db.ns_region_set_from_term(term)
        if n_components:
            region_set = take_principal_regions(region_set, n_components)

        region_set = ((t,) for t in region_set)
        return self.add_tuple_set(
            region_set,
            type_=Tuple[ExplicitVBR],
            name=name
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
        self.symbol_table[nl.Symbol[result.type](
            name
        )] = result
        return Symbol(self, name)

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


class QueryBuilderDatalog(RegionMixin, NeuroSynthMixin, QueryBuilderBase):
    def __init__(self, solver):
        super().__init__(
            solver, logic_programming=True
        )

        self.current_program = []

    def new_symbol(self, type_=sdb.Unknown, name=None):
        if isinstance(type_, (tuple, list)):
            type_ = tuple(type_)
            type_ = Tuple[type_]

        if name is None:
            name = str(uuid1())
        return Expression(
            self,
            nl.Symbol[type_](name)
        )

    def assign(self, consequent, antecedent):
        if (
            isinstance(antecedent.expression, nl.Constant) and
            antecedent.expression.value is True
        ):
            expression = sdb.Fact(consequent.expression)
            self.current_program.append(
                Fact(self, expression, consequent)
            )
        else:
            expression = sdb.Implication(
                consequent.expression,
                antecedent.expression
            )
            self.current_program.append(
                Implication(self, expression, consequent, antecedent)
            )
        self.solver.walk(self.current_program[-1].expression)

    def query(self, head, predicate):
        self.solver.symbol_table = self.symbol_table.create_scope()
        functor_orig = head.expression.functor
        new_head = self.new_symbol()(*head.arguments)
        functor = new_head.expression.functor
        self.assign(new_head, predicate)
        solution = build_chase_solution(self.solver)
        solution_set = solution.get(functor.name, nl.Constant(set()))
        out_symbol = nl.Symbol[solution_set.type](functor_orig.name)
        self.current_program = self.current_program[:-1]
        self.solver.symbol_table = self.symbol_table.enclosing_scope
        self.add_tuple_set(
            solution_set.value, name=functor_orig.name
        )
        return Symbol(self, out_symbol.name)

    def reset_program(self):
        self.symbol_table.clear()
        self.current_program = []

    def add_tuple_set(self, iterable, type_=Unknown, name=None):
        if name is None:
            name = str(uuid1())

        symbol = nl.Symbol[type_](name)
        self.solver.add_extensional_predicate_from_tuples(
            symbol, iterable, type_=type_
        )

        return Symbol(self, name)


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
        return self._query_builder.add_symbol(value, name=key)

    def __contains__(self, symbol):
        return symbol in self._query_builder.symbol_table

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
