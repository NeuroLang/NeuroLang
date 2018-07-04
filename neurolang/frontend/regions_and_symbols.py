from ..regions import region_union
from typing import AbstractSet

__all__ = ['symbol_names_of_region_set', 'region_from_symbol_name']


def symbol_names_of_region_set(solver, regions):
    '''by convention explicit defined regions names are define in uppercase
     while result and functions in lowercase'''
    res = []
    for k, v in solver.symbol_table.symbols_by_type(AbstractSet[solver.type]).items():
        if not k.name.isupper():
            continue
        for region in v.value:
            if region in regions:
                res.append(k.name)
                if len(res) == len(regions): return res
    return res


def region_from_symbol_name(solver, label):
    set = solver.symbol_table[label].value
    first = next(iter(set))
    if len(set) == 1:
        return first
    else:
        return region_union(set, first._affine_matrix)
