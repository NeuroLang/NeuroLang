from .regions_and_symbols import *
from .. import neurolang as nl
from ..regions import Region
from typing import AbstractSet, Callable

__all__ = ['query_relation_region', 'define_query', 'solve_query',
           'query_from_plane', 'neurosynth_term_to_region_set']


def query_relation_region(solver, relation, region, store_symbol_name=None):
    set_type = AbstractSet[solver.type]
    query = define_query(set_type, relation, region, 'query')
    obtained = solve_query(solver, query, result_symbol_name=store_symbol_name)
    return obtained


def define_query(set_type, functor, symbol_name, query_name):
    predicate = nl.Predicate[set_type](
        nl.Symbol[Callable[[set_type], set_type]](functor),
        (nl.Symbol[set_type](symbol_name),)
    )
    query = nl.Query[set_type](nl.Symbol[set_type](query_name), predicate)
    return query


def solve_query(solver, query, result_symbol_name=None):
    solver.walk(query)
    result = solver.symbol_table[query.symbol.name].value
    obtained = symbol_names_of_region_set(solver, result)
    if result_symbol_name is not None:
        solver.symbol_table[nl.Symbol[solver.type](result_symbol_name)] = nl.Constant[AbstractSet[solver.type]](result)
    return obtained


def query_from_plane(solver, relation, plane_dict, store_into=None):
    solver.symbol_table[nl.Symbol[dict]('elem')] = nl.Constant[dict](plane_dict)
    predicate = nl.Predicate[dict](
        nl.Symbol[Callable[[dict], AbstractSet[Region]]](relation),
        (nl.Symbol[dict]('elem'),)
    )

    query = nl.Query[AbstractSet[Region]](nl.Symbol[dict]('q'), predicate)
    solver.walk(query)
    result = solver.symbol_table['q'].value
    solver.symbol_table[nl.Symbol[solver.type](store_into)] = nl.Constant[AbstractSet[Region]](result)
    return result


def neurosynth_term_to_region_set(solver, term):

    solver.symbol_table[nl.Symbol[str]('term')] = nl.Constant[str](term)
    predicate = nl.Predicate[str](
        nl.Symbol[Callable[[str], solver.type]]('neurosynth_term'),
        (nl.Symbol[str]('term'),)
    )

    query = nl.Query[str](nl.Symbol[str]('p'), predicate)
    solver.walk(query)
    return solver.symbol_table['p'].value
