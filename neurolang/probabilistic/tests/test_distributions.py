from typing import Mapping

import numpy as np

from ...expressions import Constant
from ..distributions import TableDistribution

cat = Constant[str]('cat')
dog = Constant[str]('dog')
chironex_fleckeri = Constant[str]('chironex_fleckeri')
portuguese_man_o_war = Constant[str]('portuguese_man_o_war')
pelagia_noctiluca = Constant[str]('pelagia_noctiluca')
lion_mane = Constant[str]('lion_mane')
chrysaora_hysoscella = Constant[str]('chrysaora_hysoscella')
moon_jelly = Constant[str]('moon_jelly')
aequorea_victoria = Constant[str]('aequorea_victoria')
dog = Constant[str]('dog')

dog_cat_table = Constant[Mapping]({
    Constant('cat'): Constant[float](0.2),
    Constant('dog'): Constant[float](0.8),
})

beach_jellyfish_table = Constant[Mapping]({
    chironex_fleckeri:
    Constant[float](0.01),
    portuguese_man_o_war:
    Constant[float](0.05),
    pelagia_noctiluca:
    Constant[float](0.1),
    lion_mane:
    Constant[float](0.14),
    chrysaora_hysoscella:
    Constant[float](0.2),
    moon_jelly:
    Constant[float](0.2),
    aequorea_victoria:
    Constant[float](0.3),
})

jellyfish_dangerosity = Constant[Mapping]({
    chironex_fleckeri:
    Constant[float](1.0),
    portuguese_man_o_war:
    Constant[float](0.8),
    pelagia_noctiluca:
    Constant[float](0.4),
    lion_mane:
    Constant[float](0.4),
    chrysaora_hysoscella:
    Constant[float](0.1),
    moon_jelly:
    Constant[float](0.05),
    aequorea_victoria:
    Constant[float](0.0),
})


def test_table_distribution():
    distrib = TableDistribution(dog_cat_table)
    assert distrib.probability(cat) == Constant[float](0.2)
    assert distrib.probability(dog) == Constant[float](0.8)
    assert distrib.support == frozenset({cat, dog})


def test_expectation_table_istribution():
    beach_jellyfish_dist = TableDistribution(beach_jellyfish_table)
    beach_dangerosity = beach_jellyfish_dist.expectation(
        lambda v: jellyfish_dangerosity.value[v]
    )
    expected_value = Constant[float](sum(
        jellyfish_dangerosity.value[k].value *
        beach_jellyfish_table.value[k].value
        for k in jellyfish_dangerosity.value.keys()
    ))
    assert beach_dangerosity == expected_value


def test_conditional_table_distribution():
    observed_jellyfishes = {chironex_fleckeri, moon_jelly}
    condition = lambda value: value in observed_jellyfishes
    beach_jellyfish_dist = TableDistribution(beach_jellyfish_table)
    beach_jellyfish_dist = beach_jellyfish_dist.conditioned_on(condition)
    beach_dangerosity = beach_jellyfish_dist.expectation(
        lambda v: jellyfish_dangerosity.value[v]
    )
    assert np.isclose(beach_dangerosity.value, 0.09523809523809525)
