from .regions_and_symbols import *
from .. import neurolang as nl
from ..regions import Region
from typing import AbstractSet, Callable
from uuid import uuid1

__all__ = ['QueryBuilder']


class QueryBuilder:

    def __init__(self, solver):
        self.solver = solver

    def query_from_relation_and_region(self, relation, region, store_symbol_name=None):
        query = self.define_query(relation, region, 'query')
        obtained = self.solve_query(query, result_symbol_name=store_symbol_name)
        return obtained

    def define_predicate(self, functor, symbol_name):
        set_type = AbstractSet[self.solver.type]
        predicate = nl.Predicate[set_type](
            nl.Symbol[Callable[[set_type], set_type]](functor),
            (nl.Symbol[set_type](symbol_name),)
        )
        return predicate

    def define_query(self, relation, symbol_name, query_name):
        set_type = AbstractSet[self.solver.type]
        predicate = nl.Predicate[set_type](
            nl.Symbol[Callable[[set_type], set_type]](relation),
            (nl.Symbol[set_type](symbol_name),)
        )
        query = nl.Query[set_type](nl.Symbol[set_type](query_name), predicate)
        return query

    def solve_query(self, query, result_symbol_name=None):
        if result_symbol_name is None:
            result_symbol_name = str(uuid1())
        if not isinstance(query, nl.Query):
            set_type = AbstractSet[self.solver.type]
            query = nl.Query[set_type](nl.Symbol[set_type](result_symbol_name), query)
        self.solver.walk(query)
        result = self.solver.symbol_table[query.symbol.name].value
        obtained = symbol_names_of_region_set(self.solver, result)
        if result_symbol_name is not None:
            self.solver.symbol_table[nl.Symbol[self.solver.type](result_symbol_name)] = nl.Constant[AbstractSet[self.solver.type]](result)
        return obtained

    def query_from_plane(self, relation, plane_dict, store_into=None):
        self.solver.symbol_table[nl.Symbol[dict]('elem')] = nl.Constant[dict](plane_dict)
        predicate = nl.Predicate[dict](
            nl.Symbol[Callable[[dict], AbstractSet[Region]]](relation),
            (nl.Symbol[dict]('elem'),)
        )

        query = nl.Query[AbstractSet[Region]](nl.Symbol[dict]('q'), predicate)
        self.solver.walk(query)
        result = self.solver.symbol_table['q'].value
        self.solver.symbol_table[nl.Symbol[self.solver.type](store_into)] = nl.Constant[AbstractSet[Region]](result)
        return result

    def neurosynth_term_to_region_set(self, term):
        self.solver.symbol_table[nl.Symbol[str]('term')] = nl.Constant[str](term)
        predicate = nl.Predicate[str](
            nl.Symbol[Callable[[str], self.solver.type]]('neurosynth_term'),
            (nl.Symbol[str]('term'),)
        )

        query = nl.Query[str](nl.Symbol[str]('p'), predicate)
        self.solver.walk(query)
        return self.solver.symbol_table['p'].value
