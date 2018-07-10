import operator
import os
import re
import typing

try:
    from neurosynth import Dataset, meta
    __has_neurosynth__ = True
except ModuleNotFoundError:
    __has_neurosynth__ = False


from . import neurolang as nl
from .CD_relations import cardinal_relation, inverse_directions
from .regions import Region, region_union, region_set_from_masked_data
from .solver import SetBasedSolver
from .utils.data_manipulation import fetch_neurosynth_dataset


__all__ = ['RegionsSetSolver']


class RegionsSetSolver(SetBasedSolver[Region]):
    type = Region
    set_type = typing.AbstractSet[type]
    get_set_type = typing.Callable[[typing.Text], set_type]
    type_name = type.__name__

    def __init__(self, *args, overlap_iter=None, **kwargs):
        super().__init__(*args, **kwargs)
        pred_type = typing.Callable[[self.type, ], self.set_type]

        self.stop_refinement_at = overlap_iter

        for key, value in {'inferior_of': 'I', 'superior_of': 'S',
                           'posterior_of': 'P', 'anterior_of': 'A',
                           'overlapping': 'O', 'left_of': 'L',
                           'right_of': 'R'}.items():
            setattr(self, key, self._define_dir_based_fun(value))
            self.symbol_table[
                nl.Symbol[pred_type](key)
            ] = nl.Constant[pred_type](self.__getattribute__(key))

            setattr(self, key, self._define_inv_dir_based_fun(value))
            self.symbol_table[
                nl.Symbol[pred_type]('converse ' + key)
            ] = nl.Constant[pred_type](self.__getattribute__(key))

        self.symbol_table[
            nl.Symbol[typing.Callable[[], self.set_type]]('universal')
        ] = nl.Constant[typing.Callable[[], self.set_type]](self.all_regions)

        self.symbol_table[nl.Symbol[self.get_set_type]('neurosynth_term')] = \
            nl.Constant[self.get_set_type](self._neurosynth_term_regions())

        self.symbol_table[nl.Symbol[self.get_set_type]('regexp')] = \
            nl.Constant[self.get_set_type](self._region_set_from_regexp())

    def _define_dir_based_fun(self, direction) -> set_type:
        def f(reference_region: self.type) -> self.set_type:
            return self.direction(
                direction, reference_region,
                refinement_resolution=self.stop_refinement_at
            )

        return f

    def _define_inv_dir_based_fun(self, direction) -> set_type:
        def f(reference_region: self.type) -> self.set_type:
            return self.direction(
                [inverse_directions[d] for d in direction],
                reference_region,
                refinement_resolution=self.stop_refinement_at
            )

        return f

    def all_regions(self) -> set_type:
        res = frozenset()
        for elem in self.symbol_table.symbols_by_type(self.set_type).values():
            res = res.union(elem.value)
        return res

    def _neurosynth_term_regions(self) -> get_set_type:
        def f(elem: typing.Text) -> self.set_type:
            if not __has_neurosynth__:
                raise NotImplemented("Neurosynth not installed")

            file_dir = os.path.abspath(os.path.dirname(__file__))
            path = os.path.join(file_dir, 'utils/neurosynth')
            file = os.path.join(path, 'dataset.pkl')
            if not os.path.isfile(file):
                dataset = fetch_neurosynth_dataset(path)
            else:
                dataset = Dataset.load(file)

            studies_ids = dataset.get_studies(features=elem,
                                              frequency_threshold=0.05)
            ma = meta.MetaAnalysis(dataset, studies_ids, q=0.01, prior=0.5)
            data = ma.images['pAgF_z_FDR_0.01']
            affine = dataset.masker.get_header().get_sform()
            dim = dataset.masker.dims
            masked_data = dataset.masker.unmask(data)
            region_set = frozenset(
                region_set_from_masked_data(masked_data, affine, dim)
            )

            return region_set
        return f

    def _region_set_from_regexp(self) -> get_set_type:
        def f(regexp: typing.Text) -> self.set_type:
            regions = set()
            match = False
            symbols = self.symbol_table.symbols_by_type(self.set_type)
            for k, v in symbols.items():
                if re.search(regexp, k.name):
                    match = True
                    iterator = iter(v.value)
                    first = next(iterator)
                    affine = first.affine
                    regions.add(first)
                    for region in iterator:
                        regions.add(region)
            return frozenset([region_union(regions, affine)] if match else [])
        return f

    def direction(self, direction, reference_region: type,
                  refinement_resolution=None) -> set_type:
        regions = self.symbol_table.symbols_by_type(self.type).values()
        return frozenset([
            region.value for region in regions if (
                isinstance(region, nl.Constant) and
                region.value != reference_region and
                cardinal_relation(region.value, reference_region, direction,
                                  refine_overlapping=True,
                                  stop_at=refinement_resolution)
            )
        ])

    # add a match for the predicate "singleton" with a region as parameter
    # that will produce a set with just that region as a result
    @nl.add_match(nl.Predicate(nl.Symbol('singleton'), (nl.Constant[type],)))
    def singleton(self, expression):
        region = expression.args[0].value
        res = nl.Constant[self.set_type](
            frozenset((region,))
        )
        return res

    @nl.add_match(nl.FunctionApplication(
        nl.Constant(operator.invert), (nl.Constant[typing.AbstractSet],)
    ))
    def rewrite_finite_domain_inversion(self, expression):
        set_constant = expression.args[0]
        set_type, set_value = nl.get_type_and_value(set_constant)
        all_regions = frozenset([
            v.value for v in
            self.symbol_table.symbols_by_type(set_type.__args__[0]).values()
        ])
        for v in self.symbol_table.symbols_by_type(set_type).values():
            all_regions = all_regions.union(v.value)

        result = all_regions - set_value
        return self.walk(nl.Constant[set_type](result))
