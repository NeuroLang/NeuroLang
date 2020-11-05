from typing import AbstractSet

from .. import expressions, solver
from ..neurolang_compiler import NeuroLangIntermediateRepresentationCompiler

C_ = expressions.Constant


def test_union_monoid_ir():
    elements = [frozenset((C_(a), )) for a in (1, 2)]
    null_element = frozenset()
    symbols = {
        'element{}'.format(i + 1): C_[AbstractSet[int]](e)
        for i, e in enumerate(elements)
    }
    symbols['null'] = C_[AbstractSet[int]](null_element)

    nli = NeuroLangIntermediateRepresentationCompiler(
        solver.FirstOrderLogicSolver(), symbols=symbols
    )

    e1 = symbols['element1']
    e2 = symbols['element2']
    e1_union_e2 = e1 | e2

    res = nli.compile(e1_union_e2)
    assert res.type == e1.type
    assert 1 in res.value and 2 in res.value
    assert len(res.value) == 2


def test_intersection_monoid_ir():
    elements = [frozenset(C_(a_) for a_ in a) for a in ({1}, {1, 2})]
    null_element = frozenset()
    symbols = {
        'element{}'.format(i + 1): C_[AbstractSet[int]](e)
        for i, e in enumerate(elements)
    }
    symbols['null'] = C_[AbstractSet[int]](null_element)

    nli = NeuroLangIntermediateRepresentationCompiler(
        solver.FirstOrderLogicSolver(), symbols=symbols
    )

    e1 = symbols['element1']
    e2 = symbols['element2']
    e1_intersection_e2 = e1 & e2

    res = nli.compile(e1_intersection_e2)
    assert res.type == e1.type
    assert 1 in res.value and len(res.value) == 1
