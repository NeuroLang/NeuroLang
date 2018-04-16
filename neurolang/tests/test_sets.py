from .. import solver
from .. import neurolang as nl

from typing import AbstractSet


def test_union_monoid_ir():
    elements = [frozenset((a, )) for a in (1, 2)]
    null_element = frozenset()
    symbols = {
        'element{}'.format(i + 1): nl.Constant[AbstractSet[int]](e)
        for i, e in enumerate(elements)
    }
    symbols['null'] = nl.Constant[AbstractSet[int]](null_element)

    class NLI(
        solver.SetBasedSolver,
        nl.NeuroLangIntermediateRepresentationCompiler
    ):
        pass

    nli = NLI(
        symbols=symbols, type_name_map={'dummy': AbstractSet[int]}
    )

    e1 = symbols['element1']
    e2 = symbols['element2']
    e1_union_e2 = e1 | e2

    res = nli.walk(e1_union_e2)
    assert res.type == e1.type
    assert 1 in res and 2 in res
    assert len(res.value) == 2


def test_intersection_monoid_ir():
    elements = [frozenset(a) for a in ({1}, {1, 2})]
    null_element = frozenset()
    symbols = {
        'element{}'.format(i + 1): nl.Constant[AbstractSet[int]](e)
        for i, e in enumerate(elements)
    }
    symbols['null'] = nl.Constant[AbstractSet[int]](null_element)

    class NLI(
        solver.SetBasedSolver,
        nl.NeuroLangIntermediateRepresentationCompiler
    ):
        pass

    nli = NLI(
        symbols=symbols, type_name_map={'dummy': AbstractSet[int]}
    )

    e1 = symbols['element1']
    e2 = symbols['element2']
    e1_intersection_e2 = e1 & e2

    res = nli.walk(e1_intersection_e2)
    assert res.type == e1.type
    assert 1 in res and len(res.value) == 1
