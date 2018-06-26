from .CD_relations import cardinal_relation, inverse_directions
from .regions import *
from .solver import SetBasedSolver
from .utils.data_manipulation import *
import typing
import operator
import os
import numpy as np
from neurosynth import Dataset, meta
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

    def __init__(self, *args, overlap_iter=None, **kwargs):
        super().__init__(*args, **kwargs)
        pred_type = typing.Callable[
            [typing.AbstractSet[self.type], ],
            typing.AbstractSet[self.type]
        ]

        self.stop_refinement_at = overlap_iter if overlap_iter is not None else None

        for key, value in {'inferior_of': 'I', 'superior_of': 'S',
                           'posterior_of': 'P', 'anterior_of': 'A',
                           'overlapping': 'O', 'left_of': 'L', 'right_of': 'R'}.items():
            setattr(self, key, self.define_dir_based_fun(value))
            self.symbol_table[
                nl.Symbol[pred_type](key)
            ] = nl.Constant[pred_type](self.__getattribute__(key))

            setattr(self, key, self.define_inv_dir_based_fun(value))
            self.symbol_table[
                nl.Symbol[pred_type]('converse ' + key)
            ] = nl.Constant[pred_type](self.__getattribute__(key))

        self.symbol_table[nl.Symbol[pred_type]('universal')] = nl.Constant[pred_type](self.symbols_of_type())

        dict_to_region = typing.Callable[[typing.DefaultDict, ], typing.AbstractSet[Region]]
        for key, value in {'superior_from_plane': 1, 'inferior_from_plane': -1}.items():
            setattr(self, key, self.region_from_plane(value))
            self.symbol_table[
                nl.Symbol[dict_to_region](key)
            ] = nl.Constant[dict_to_region](self.__getattribute__(key))

    def region_from_plane(self, bb_direction) -> typing.AbstractSet[Region]:
        def f(elem: typing.DefaultDict) -> typing.AbstractSet[Region]:
            elem['direction'] = bb_direction
            return frozenset([PlanarVolume(**elem)])
        return f

    def symbols_of_type(self) -> typing.AbstractSet[Region]:
        def f(elem: typing.AbstractSet[Region]) -> typing.AbstractSet[Region]:
            res = frozenset()
            for elem in self.symbol_table.symbols_by_type(typing.AbstractSet[Region]).values():
                res = res.union(elem.value)
            return res
        return f

    def define_dir_based_fun(self, direction) -> typing.AbstractSet[Region]:
        def f(reference_region: typing.AbstractSet[Region]) -> typing.AbstractSet[Region]:
            return self.direction(direction, reference_region, refinement_resolution=self.stop_refinement_at)
        return f

    def define_inv_dir_based_fun(self, direction) -> typing.AbstractSet[Region]:
        def f(reference_region: typing.AbstractSet[Region]) -> typing.AbstractSet[Region]:
            return self.direction([inverse_directions[d] for d in direction], reference_region, refinement_resolution=self.stop_refinement_at)
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

    def direction(self, direction, reference_regions: typing.AbstractSet[Region], refinement_resolution=None) -> typing.AbstractSet[Region]:
        result, visited = set(), set()
        for symbol_in_table in self.symbol_table.symbols_by_type(typing.AbstractSet[self.type]).values():
            regions_set = symbol_in_table.value - reference_regions
            if not regions_set.issubset(visited):
                visited.update(regions_set)
                for region in regions_set:
                    for ref in reference_regions:
                        if not cardinal_relation(region, ref, direction, refine_overlapping=True, stop_at=refinement_resolution):
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

    def load_term_defined_regions(self, term, k=None):

        file_dir = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(file_dir, 'utils/neurosynth')
        file = path + '/dataset.pkl'
        if not os.path.isfile(file):
            dataset = generate_neurosynth_dataset(path)
        else:
            dataset = Dataset.load(file)
        studies_ids = dataset.get_studies(features=term, frequency_threshold=0.05)
        ma = meta.MetaAnalysis(dataset, studies_ids, q=0.01, prior=0.5)
        data = ma.images['pAgF_z_FDR_0.01']
        affine = dataset.masker.get_header().get_sform()
        voxels = dataset.masker.unmask(data)
        regions_set = generate_connected_regions_set(voxels, affine, k)
        self.symbol_table[nl.Symbol[self.type](term.upper())] = \
            nl.Constant[typing.AbstractSet[self.type]](frozenset(regions_set))
        return regions_set

    def load_parcellation_regions_to_solver(self, parc_im, k=None):
        labels = parc_im.get_data()
        label_regions_map = parse_region_label_map(parc_im, k)
        for region_name, region_key in label_regions_map.items():
            voxel_coordinates = np.transpose((labels == region_key).nonzero())
            region = ExplicitVBR(voxel_coordinates, parc_im.affine)
            frozenset([region])
            self.symbol_table[nl.Symbol[self.type](region_name)] = \
                nl.Constant[typing.AbstractSet[self.type]](frozenset([region]))

    def regions_symbol_names_from_set(self, regions: typing.AbstractSet[Region]):
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

    def query_and_store_symbol(self, query, symbol_name):
        self.walk(query)
        result = self.symbol_table[query.symbol.name].value
        self.symbol_table[nl.Symbol[self.type](symbol_name)] = nl.Constant[typing.AbstractSet[self.type]](result)
        obtained = self.regions_symbol_names_from_set(result)
        return obtained, symbol_name

    def solve_query(self, query):
        self.walk(query)
        result = self.symbol_table[query.symbol.name].value
        obtained = self.regions_symbol_names_from_set(result)
        return obtained

    def query_relation_region(self, relation, region, store_into=None):
        set_type = typing.AbstractSet[self.type]
        query = define_query(set_type, relation, region, 'query')
        obtained = self.solve_query(query)
        if store_into:
            obtained, symbol = self.query_and_store_symbol(query, store_into)
        return obtained

    def query_from_plane(self, relation, plane_dict, store_into=None):

        self.symbol_table[nl.Symbol[dict]('elem')] = nl.Constant[dict](plane_dict)
        p1 = nl.Predicate[dict](
            nl.Symbol[typing.Callable[[dict], typing.AbstractSet[Region]]](relation),
            (nl.Symbol[dict]('elem'),)
        )

        query = nl.Query[typing.AbstractSet[Region]](nl.Symbol[dict]('a'), p1)
        self.walk(query)
        result = self.symbol_table[query.symbol.name].value
        self.symbol_table[nl.Symbol[self.type](store_into)] = nl.Constant[typing.AbstractSet[Region]](result)
        return result
