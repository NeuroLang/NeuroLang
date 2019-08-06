import numpy as np

from ..probabilistic.distributions import TableDistribution

dog_cat_table = {
    'cat': 0.2,
    'dog': 0.8,
}

beach_jellyfish_table = {
    'chironex_fleckeri': 0.01,
    'portuguese_man_o_war': 0.05,
    'pelagia_noctiluca': 0.1,
    'lion_mane': 0.14,
    'chrysaora_hysoscella': 0.2,
    'moon_jelly': 0.2,
    'aequorea_victoria': 0.3,
}

jellyfish_dangerosity = {
    'chironex_fleckeri': 1.0,
    'portuguese_man_o_war': 0.8,
    'pelagia_noctiluca': 0.4,
    'lion_mane': 0.4,
    'chrysaora_hysoscella': 0.1,
    'moon_jelly': 0.05,
    'aequorea_victoria': 0.0,
}


def test_table_distribution():
    distrib = TableDistribution(dog_cat_table)
    assert distrib.probability('cat') == 0.2
    assert distrib.probability('dog') == 0.8
    assert distrib.support == frozenset({'cat', 'dog'})


def test_expectation_table_distribution():
    beach_jellyfish_dist = TableDistribution(beach_jellyfish_table)
    beach_dangerosity = beach_jellyfish_dist.expectation(
        lambda v: jellyfish_dangerosity[v]
    )
    expected_value = sum(
        jellyfish_dangerosity[k] * beach_jellyfish_table[k]
        for k in jellyfish_dangerosity.keys()
    )
    assert beach_dangerosity == expected_value


def test_conditional_table_distribution():
    observed_jellyfishes = {'chironex_fleckeri', 'moon_jelly'}
    condition = lambda value: value in observed_jellyfishes
    beach_jellyfish_dist = TableDistribution(beach_jellyfish_table)
    beach_jellyfish_dist = beach_jellyfish_dist.conditioned_on(condition)
    beach_dangerosity = beach_jellyfish_dist.expectation(
        lambda v: jellyfish_dangerosity[v]
    )
    assert np.isclose(beach_dangerosity, 0.09523809523809525)
