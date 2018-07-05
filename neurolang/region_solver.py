from .CD_relations import cardinal_relation, inverse_directions
from .regions import *
from .solver import SetBasedSolver
from .utils.data_manipulation import *
import typing
import operator
import os
import re
from neurosynth import Dataset, meta
from . import neurolang as nl


__all__ = ['RegionsSetSolver']


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
            setattr(self, key, self._define_dir_based_fun(value))
            self.symbol_table[
                nl.Symbol[pred_type](key)
            ] = nl.Constant[pred_type](self.__getattribute__(key))

            setattr(self, key, self._define_inv_dir_based_fun(value))
            self.symbol_table[
                nl.Symbol[pred_type]('converse ' + key)
            ] = nl.Constant[pred_type](self.__getattribute__(key))

        self.symbol_table[nl.Symbol[pred_type]('universal')] = nl.Constant[pred_type](self.symbols_of_type())

        str_to_region_set_type = typing.Callable[[typing.Text, ], typing.AbstractSet[Region]]

        self.symbol_table[nl.Symbol[str_to_region_set_type]('neurosynth_term')] = nl.Constant[str_to_region_set_type](
            self._neurosynth_term_rois())

        self.symbol_table[nl.Symbol[str_to_region_set_type]('regexp')] = nl.Constant[str_to_region_set_type](
            self._region_set_from_regexp())

    def _define_dir_based_fun(self, direction) -> typing.AbstractSet[Region]:
        def f(reference_region: typing.AbstractSet[Region]) -> typing.AbstractSet[Region]:
            return self.direction(direction, reference_region, refinement_resolution=self.stop_refinement_at)

        return f

    def _define_inv_dir_based_fun(self, direction) -> typing.AbstractSet[Region]:
        def f(reference_region: typing.AbstractSet[Region]) -> typing.AbstractSet[Region]:
            return self.direction([inverse_directions[d] for d in direction], reference_region,
                                  refinement_resolution=self.stop_refinement_at)

        return f

    def _region_from_plane(self, bb_direction) -> typing.AbstractSet[Region]:
        def f(plane_attributes: typing.DefaultDict) -> typing.AbstractSet[Region]:
            plane_attributes['direction'] = bb_direction
            return frozenset([PlanarVolume(**plane_attributes)])
        return f

    def symbols_of_type(self) -> typing.AbstractSet[Region]:
        def f(elem: typing.AbstractSet[Region]) -> typing.AbstractSet[Region]:
            res = frozenset()
            for elem in self.symbol_table.symbols_by_type(typing.AbstractSet[Region]).values():
                res = res.union(elem.value)
            return res
        return f

    def _neurosynth_term_rois(self) -> typing.AbstractSet[Region]:
        def f(elem: typing.Text) -> typing.AbstractSet[Region]:

            file_dir = os.path.abspath(os.path.dirname(__file__))
            path = os.path.join(file_dir, 'utils/neurosynth')
            file = os.path.join(path, 'dataset.pkl')
            if not os.path.isfile(file):
                dataset = fetch_neurosynth_dataset(path)
            else:
                dataset = Dataset.load(file)

            studies_ids = dataset.get_studies(features=elem, frequency_threshold=0.05)
            ma = meta.MetaAnalysis(dataset, studies_ids, q=0.01, prior=0.5)
            data = ma.images['pAgF_z_FDR_0.01']
            affine = dataset.masker.get_header().get_sform()
            dim = dataset.masker.dims
            masked_data = dataset.masker.unmask(data)
            regions_set = frozenset(region_set_from_masked_data(masked_data, affine, dim))

            return regions_set
        return f

    def _region_set_from_regexp(self) -> typing.AbstractSet[Region]:
        def f(regexp: typing.Text) -> typing.AbstractSet[Region]:
            regions = set()
            match = False
            for k, v in self.symbol_table.symbols_by_type(typing.AbstractSet[self.type]).items():
                if re.search(regexp, k.name):
                    match = True
                    iterator = iter(v.value)
                    first = next(iterator)
                    affine = first._affine_matrix
                    regions.add(first)
                    for region in iterator:
                        regions.add(region)

            result = frozenset([region_union(regions, affine)]) if match else frozenset([])
            return result
        return f

    def direction(self, direction, reference_regions: typing.AbstractSet[Region], refinement_resolution=None) -> typing.AbstractSet[Region]:
        result, visited = set(), set()
        for symbol_in_table in self.symbol_table.symbols_by_type(typing.AbstractSet[self.type]).values():
            resolved_symbol = self.walk(symbol_in_table)
            if isinstance(resolved_symbol, nl.Constant):
                regions_set = resolved_symbol.value - reference_regions
                if not regions_set.issubset(visited):
                    visited.update(regions_set)
                    for region in regions_set:
                        for ref in reference_regions:
                            if not cardinal_relation(region, ref, direction, refine_overlapping=True, stop_at=refinement_resolution):
                                break
                        else:
                            result.update((region,))

        return frozenset(result)

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
