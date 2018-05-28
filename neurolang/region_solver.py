from .RCD_relations import direction_matrix, is_in_direction, inverse_direction
from .regions import *
from .solver import SetBasedSolver
from .utils.data_manipulation import *
import typing
import operator
from . import neurolang as nl


__all__ = ['RegionsSetSolver', 'define_query', 'get_singleton_element_from_frozenset']

#todo: this goto utils
def define_query(set_type, functor, symbol_name, query_name):
    predicate = nl.Predicate[set_type](
        nl.Symbol[typing.Callable[[set_type], set_type]](functor),
        (nl.Symbol[set_type](symbol_name),)
    )
    query = nl.Query[set_type](nl.Symbol[set_type](query_name), predicate)
    return query

def get_singleton_element_from_frozenset(fs):
    return next(iter(fs))

class RegionsSetSolver(SetBasedSolver):
    type = Region
    type_name = 'Region'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pred_type = typing.Callable[
            [typing.AbstractSet[self.type], ],
            typing.AbstractSet[self.type]
        ]

        for key, value in {'superior_of': 'S', 'inferior_of': 'I', 'anterior_of': 'A',
                           'posterior_of': 'P', 'overlapping': 'O',
                           'left_of': 'L', 'aligned': 'C', 'right_of': 'R'}.items():
            setattr(self, key, self.define_dir_based_fun(value))
            self.symbol_table[
                nl.Symbol[pred_type](key)
            ] = nl.Constant[pred_type](self.__getattribute__(key))

            setattr(self, key, self.define_inv_dir_based_fun(value))
            self.symbol_table[
                nl.Symbol[pred_type]('converse ' + key)
            ] = nl.Constant[pred_type](self.__getattribute__(key))

        self.symbol_table[nl.Symbol[pred_type]('universal')] = nl.Constant[pred_type](self.symbols_of_type())

    #todo replace 'Region' type for self.type

    def symbols_of_type(self) -> typing.AbstractSet[Region]:
        def f(elem: typing.AbstractSet[Region]) -> typing.AbstractSet[Region]:
            res = frozenset()
            for elem in self.symbol_table.symbols_by_type(typing.AbstractSet[Region]).values():
                res = res.union(elem.value)
            return res
        return f

    def define_dir_based_fun(self, direction) -> typing.AbstractSet[Region]:
        def f(reference_region: typing.AbstractSet[Region]) -> typing.AbstractSet[Region]:
            return self.direction(direction, reference_region)
        return f

    def define_inv_dir_based_fun(self, direction) -> typing.AbstractSet[Region]:
        def f(reference_region: typing.AbstractSet[Region]) -> typing.AbstractSet[Region]:
            return self.direction(inverse_direction(direction), reference_region)
        return f

    # add a match for the predicate "singleton" with a region as parameter
    # that will produce a set with just that region as a result
    @nl.add_match(nl.Predicate(nl.Symbol('singleton'), (nl.Constant[typing.Tuple[int, int, int, int]],)))
    def singleton(self, expression):
        a = expression.args[0].value
        r = Region((a[0].value, a[1].value), (a[2].value, a[3].value))
        res = nl.Constant[typing.AbstractSet[self.type]](
            frozenset((r,))
        )
        return res

    def direction(self, direction, reference_region: typing.AbstractSet[Region]) -> typing.AbstractSet[Region]:
        result, visited = set(), set()
        for symbol in self.symbol_table.symbols_by_type(typing.AbstractSet[Region]).values():
            regions_set = symbol.value - reference_region
            if not regions_set.issubset(visited):
                visited.update(regions_set)
                for region in regions_set:
                    for elem in reference_region:
                        mat = direction_matrix(region, elem)
                        if not is_in_direction(mat, direction):
                            break
                    else:
                        result.update((region,))

        return self.walk(frozenset(result))

    @nl.add_match(nl.FunctionApplication(nl.Constant(operator.invert), (nl.Constant[typing.AbstractSet],)))
    def rewrite_finite_domain_inversion(self, expression):
        set_constant = expression.args[0]
        set_type, set_value = nl.get_type_and_value(set_constant)
        all_regions = frozenset(
            (
                v.value for v in
                self.symbol_table.symbols_by_type(
                    set_type.__args__[0]
                ).values()
            )
        )
        for v in self.symbol_table.symbols_by_type(set_type).values():
            all_regions = all_regions.union(v.value)

        result = all_regions - set_value
        return self.walk(nl.Constant[set_type](result))

    def load_regions_to_solver(self, parc_im):
        labels = parc_im.get_data()
        label_regions_map = parse_region_label_map(parc_im)
        for region_name, region_key in label_regions_map.items():
            voxel_coordinates = np.transpose((labels == region_key).nonzero())
            region = ExplicitVBR(voxel_coordinates, parc_im.affine)
            self.symbol_table[nl.Symbol[self.type](region_name)] = nl.Constant[typing.AbstractSet[self.type]](frozenset([region]))

    def name_of_regions(self, regions: typing.AbstractSet[Region]):
        '''by convention regions names are define in uppercase while result and functions in lowercase'''
        res = []
        for k, v in self.symbol_table.symbols_by_type(typing.AbstractSet[self.type]).items():
            if not k.name.isupper():
                continue
            for region in v.value:
                if region in regions:
                    res.append(k.name)
                    if len(res) == len(regions): return res
        return res

    def query_and_store_result(self, query, result_symbol_name):
        self.walk(query)
        result = self.symbol_table[query.symbol.name].value
        self.symbol_table[nl.Symbol[self.type](result_symbol_name)] = nl.Constant[typing.AbstractSet[self.type]](result)
        obtained = self.name_of_regions(result)
        return obtained, result_symbol_name

    def solve_query(self, query):
        self.walk(query)
        result = self.symbol_table[query.symbol.name].value
        obtained = self.name_of_regions(result)
        return obtained

    def run_query(self, relation, region, store_into=None):
        set_type = typing.AbstractSet[self.type]
        query = define_query(set_type, relation, region, 'query')
        obtained = self.solve_query(query)
        if store_into:
            obtained, symbol = self.query_and_store_result(query, store_into)
        return obtained
