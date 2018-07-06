from neurolang import frontend
from ...regions import Region


def test_add_regions_and_query():
    neurolang = frontend.RegionFrontend()

    inferior = Region((0, 0, 0), (1, 1, 1))
    central = Region((0, 0, 2), (1, 1, 3))
    superior = Region((0, 0, 4), (1, 1, 5))

    neurolang.add_region_set([inferior, superior])
    neurolang.add_region(central, result_symbol_name='CENTRAL')
    central_predicate = neurolang.define_predicate('superior_of', 'CENTRAL')
    result = neurolang.solve_query(central_predicate)
    assert result.value == frozenset([superior])


def test_neurosynth_query():
    neurolang = frontend.RegionFrontend()
    result = neurolang.neurosynth_term_to_region_set('emotion')
    assert len(result.value) != 0
