from ..region_solver import RegionsSetSolver, get_singleton_element_from_frozenset
from ..symbols_and_types import TypedSymbolTable
from .. import neurolang as nl
from typing import AbstractSet, Callable
from ..regions import *
import os
import numpy as np
import nibabel as nib

# todo: refa this awful tests

# subject = '100206'
# path = '../../data/%s/T1w/aparc.a2009s+aseg.nii.gz' % subject
# data_from_file = os.path.isfile(path)
#
# if data_from_file:
#     region_solver = RegionsSetSolver(TypedSymbolTable())
#     parc_data = nib.load(path)
#     region_solver.load_regions_to_solver(parc_data)


def test_relation_superior_of():
    region_set_type = AbstractSet[Region]
    solver = RegionsSetSolver(TypedSymbolTable())

    inferior = Region((0, 0, 0), (1, 1, 1))
    central = Region((0, 0, 2), (1, 1, 3))
    superior = Region((0, 0, 4), (1, 1, 5))

    all_elements = frozenset([inferior, central, superior])
    elem = frozenset([central])

    solver.symbol_table[nl.Symbol[region_set_type]('db')] = nl.Constant[region_set_type](all_elements)
    solver.symbol_table[nl.Symbol[region_set_type]('elem')] = nl.Constant[region_set_type](elem)

    superior_relation = 'superior_of'
    predicate = nl.Predicate[region_set_type](
            nl.Symbol[Callable[[region_set_type], region_set_type]](superior_relation),
            (nl.Symbol[region_set_type]('elem'),)
        )

    query = nl.Query[region_set_type](nl.Symbol[region_set_type]('p1'), predicate)
    solver.walk(query)

    assert solver.symbol_table['p1'].value == frozenset([superior])


def test_superior_u_inferior():
    solver = RegionsSetSolver(TypedSymbolTable())

    db_elems = frozenset([Region((0, 0, 5), (1, 1, 6)), Region((0, 0, -10), (1, 1, -8))])
    elem = frozenset([Region((0, 0, 0), (1, 1, 1))])

    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('db')] = nl.Constant[AbstractSet[Region]](db_elems)
    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('e1')] = nl.Constant[AbstractSet[Region]](elem)

    check_union_commutativity(AbstractSet[Region], solver, 'superior_of', 'inferior_of', 'e1')


def check_union_commutativity(set_type, solver, relation1, relation2, element):
    p1 = nl.Predicate[set_type](
        nl.Symbol[Callable[[set_type], set_type]](relation1),
        (nl.Symbol[set_type](element),)
    )

    p2 = nl.Predicate[set_type](
        nl.Symbol[Callable[[set_type], set_type]](relation2),
        (nl.Symbol[set_type](element),)
    )

    query_a = nl.Query[set_type](nl.Symbol[set_type]('a'), (p1 | p2))
    query_b = nl.Query[set_type](nl.Symbol[set_type]('b'), (p2 | p1))
    solver.walk(query_a)
    solver.walk(query_b)
    assert solver.symbol_table['a'] == solver.symbol_table['b']


def test_union_associativity():
    solver = RegionsSetSolver(TypedSymbolTable())

    db_elems = frozenset([Region((0, 0, 5), (1, 1, 6)), Region((0, 0, -10), (1, 1, -8))])
    elem = frozenset([Region((0, 0, 0), (1, 1, 1))])

    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('db')] = nl.Constant[AbstractSet[Region]](db_elems)
    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('elem')] = nl.Constant[AbstractSet[Region]](elem)

    check_union_associativity(AbstractSet[Region], solver, 'superior_of', 'inferior_of', 'posterior_of', 'elem')


def check_union_associativity(type, solver, relation1, relation2, relation3, element):
    p1 = nl.Predicate[type](
        nl.Symbol[Callable[[type], type]](relation1),
        (nl.Symbol[type](element),)
    )

    p2 = nl.Predicate[type](
        nl.Symbol[Callable[[type], type]](relation2),
        (nl.Symbol[type](element),)
    )

    p3 = nl.Predicate[type](
        nl.Symbol[Callable[[type], type]](relation3),
        (nl.Symbol[type](element),)
    )

    query_a = nl.Query[type](nl.Symbol[type]('a'), p1 | (p2 | p3))
    query_b = nl.Query[type](nl.Symbol[type]('b'), (p1 | p2) | p3)

    solver.walk(query_a)
    solver.walk(query_b)

    assert solver.symbol_table['a'] == solver.symbol_table['b']


def test_huntington_axiom():
    solver = RegionsSetSolver(TypedSymbolTable())

    db_elems = frozenset([
        Region((0, 0, 5), (1, 1, 6)), Region((0, 0, -10), (1, 1, -8))
    ])
    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('db')] = nl.Constant[AbstractSet[Region]](db_elems)

    elem = frozenset([Region((0, 0, 0), (1, 1, 1))])

    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('e')] = nl.Constant[AbstractSet[Region]](elem)
    check_huntington(AbstractSet[Region], solver, 'superior_of', 'inferior_of', 'e')


def check_huntington(type, solver, relation1, relation2, element):
    p1 = nl.Predicate[type](
        nl.Symbol[Callable[[type], type]](relation1),
        (nl.Symbol[type](element),)
    )

    p2 = nl.Predicate[type](
        nl.Symbol[Callable[[type], type]](relation2),
        (nl.Symbol[type](element),)
    )

    query_a = nl.Query[type](nl.Symbol[type]('a'), p1)
    solver.walk(query_a)

    query_b = nl.Query[type](nl.Symbol[type]('b'), ~(~p1 | ~p2) | ~(~p1 | p2))
    solver.walk(query_b)
    assert solver.symbol_table['a'] == solver.symbol_table['b']


def test_composition():
    solver = RegionsSetSolver(TypedSymbolTable())

    superior = Region((0, 0, 5), (1, 1, 6))
    central = Region((0, 0, 0), (1, 1, 1))
    lat1 = Region((0, 2, 0), (1, 3, 1))
    lat2 = Region((0, 6, 0), (1, 7, 1))
    lat3 = Region((0, 10, 0), (1, 11, 1))

    db_elements = frozenset([central, superior, lat1, lat2, lat3])
    set_type = AbstractSet[Region]

    solver.symbol_table[nl.Symbol[set_type]('db')] = nl.Constant[set_type](db_elements)
    solver.symbol_table[nl.Symbol[set_type]('c')] = nl.Constant[set_type](frozenset([superior]))

    res = do_composition_of_relations_from_region(set_type, solver, 'c', 'anterior_of', 'inferior_of')

    assert res == frozenset([lat1, lat2, lat3])


def do_composition_of_relations_from_region(set_type, solver, elem, relation1, relation2):

    bs = do_query_of_relation(set_type, solver, elem, relation2)
    if bs.value == frozenset():
        return bs.value

    result = frozenset()
    for b_element in bs.value:
        solver.symbol_table[nl.Symbol[set_type]('b_element')] = nl.Constant[AbstractSet[Region]](frozenset([b_element]))
        res = do_query_of_relation(set_type, solver, 'b_element', relation1)
        result = result.union(res.value)
    return result


def do_query_of_relation(set_type, solver, elem, relation):

    predicate = nl.Predicate[set_type](
            nl.Symbol[Callable[[set_type], set_type]](relation),
            (nl.Symbol[set_type](elem),)
        )

    query = nl.Query[set_type](nl.Symbol[set_type]('p'), predicate)
    solver.walk(query)

    return solver.symbol_table['p']


def test_composition_distributivity():
    solver = RegionsSetSolver(TypedSymbolTable())

    superior = Region((0, 0, 5), (1, 1, 6))
    central = Region((0, 0, 0), (1, 1, 1))
    lat0 = Region((0, -3, 0), (1, -2, 1))
    lat1 = Region((0, 2, 0), (1, 3, 1))
    lat2 = Region((0, 6, 0), (1, 7, 1))
    lat3 = Region((0, 10, 0), (1, 11, 1))

    db_elements = frozenset([central, superior, lat0, lat1, lat2, lat3])
    set_type = AbstractSet[Region]

    solver.symbol_table[nl.Symbol[set_type]('db')] = nl.Constant[set_type](db_elements)
    solver.symbol_table[nl.Symbol[set_type]('d')] = nl.Constant[set_type](frozenset([superior]))

    res = check_distributivity(set_type, solver, 'd', 'posterior_of', 'anterior_of', 'inferior_of')

    c1 = do_composition_of_relations_from_region(set_type, solver, 'd', 'posterior_of', 'inferior_of')
    c2 = do_composition_of_relations_from_region(set_type, solver, 'd', 'anterior_of', 'inferior_of')
    assert res == c1.union(c2)


def check_distributivity(set_type, solver, elem, rel1, rel2, rel3):

    res = do_query_of_relation(set_type, solver, elem, rel3)
    obtained = frozenset()

    #iterate over the result of a query b in B and apply R&S b to obtain the left side element (a)
    for elements in res.value:

        solver.symbol_table[nl.Symbol[set_type]('pred_base_element')] = nl.Constant[set_type](frozenset([elements]))
        p1 = nl.Predicate[set_type](
            nl.Symbol[Callable[[set_type], set_type]](rel1),
            (nl.Symbol[set_type]('pred_base_element'),)
        )

        p2 = nl.Predicate[set_type](
            nl.Symbol[Callable[[set_type], set_type]](rel2),
            (nl.Symbol[set_type]('pred_base_element'),)
        )

        query_union = nl.Query[set_type](nl.Symbol[set_type]('union'), (p1 | p2))
        solver.walk(query_union)
        obtained = obtained.union(solver.symbol_table['union'].value)

    return obtained


def check_distributivity_composition_of_relations_from_region(set_type, solver, elem, rel1, rel2, rel3):

    cs = do_query_of_relation(set_type, solver, elem, rel3)
    if cs.value == frozenset():
        return cs.value

    bs = frozenset()
    for c_elements in cs.value:
        solver.symbol_table[nl.Symbol[set_type]('c')] = [c_elements]
        b = do_query_of_relation(set_type, solver, 'c', rel2)
        bs.union(b)

    if bs.value == frozenset():
        return bs.value

    result = frozenset()
    for b_elements in bs:
        solver.symbol_table[nl.Symbol[set_type]('b')] = [b_elements]
        res = do_query_of_relation(set_type, solver, 'b', rel1)
        result.union(res)
    return result


def test_involution():
    class SBS(RegionsSetSolver):
        type = Region

    solver = SBS(TypedSymbolTable())

    inferior = Region((0, 0, -10), (1, 1, -8))
    central = Region((0, 0, 0), (1, 1, 1))
    superior = Region((0, 0, 5), (1, 1, 6))

    db_elems = frozenset([central, superior, inferior])
    elem = frozenset([central])

    set_type = AbstractSet[Region]

    solver.symbol_table[nl.Symbol[set_type]('db')] = nl.Constant[set_type](db_elems)
    solver.symbol_table[nl.Symbol[set_type]('element')] = nl.Constant[set_type](elem)

    north_relation = 'superior_of'
    p1 = nl.Predicate[set_type](
        nl.Symbol[Callable[[set_type], set_type]](north_relation),
        (nl.Symbol[set_type]('x'),)
    )

    exists = nl.ExistentialPredicate[set_type](
        nl.Symbol[set_type]('x'), p1
    )

    query = nl.Query[set_type](nl.Symbol[set_type]('existential_query'), exists)
    solver.walk(query)
    res = solver.symbol_table['existential_query'].value - elem

    p_conv = nl.Predicate[set_type](
        nl.Symbol[Callable[[set_type], set_type]]('converse ' + north_relation),
        (nl.Symbol[set_type]('element'),)
    )
    query_conv = nl.Query[set_type](nl.Symbol[set_type]('converse_query'), p_conv)
    solver.walk(query_conv)

    assert res == solver.symbol_table['converse_query'].value


def test_converse_distributivity():

    class SBS(RegionsSetSolver):
        type = Region

    solver = SBS(TypedSymbolTable())

    db_elems = frozenset([Region((0, 0, 5), (1, 1, 6)), Region((0, 0, -10), (1, 1, -8))])
    elem = frozenset([Region((0, 0, 0), (1, 1, 1))])

    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('db')] = nl.Constant[AbstractSet[Region]](db_elems)
    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('e1')] = nl.Constant[AbstractSet[Region]](elem)

    set_type = AbstractSet[Region]

    p1 = nl.Predicate[set_type](
        nl.Symbol[Callable[[set_type], set_type]]('superior_of'),
        (nl.Symbol[set_type]('x'),)
    )

    p2 = nl.Predicate[set_type](
        nl.Symbol[Callable[[set_type], set_type]]('inferior_of'),
        (nl.Symbol[set_type]('x'),)
    )

    exists = nl.ExistentialPredicate[set_type](
        nl.Symbol[set_type]('x'), p1 | p2
    )
    query_a = nl.Query[set_type](nl.Symbol[set_type]('a'), exists)
    solver.walk(query_a)
    res = solver.symbol_table['a'].value - elem

    p1 = nl.Predicate[set_type](
        nl.Symbol[Callable[[set_type], set_type]]('superior_of'),
        (nl.Symbol[set_type]('x'),)
    )

    p2 = nl.Predicate[set_type](
        nl.Symbol[Callable[[set_type], set_type]]('inferior_of'),
        (nl.Symbol[set_type]('x'),)
    )

    exists_p1 = nl.ExistentialPredicate[set_type](
        nl.Symbol[set_type]('x'), p1
    )
    exists_p2 = nl.ExistentialPredicate[set_type](
        nl.Symbol[set_type]('x'), p2
    )

    query_p1 = nl.Query[set_type](nl.Symbol[set_type]('p1'), exists_p1)
    query_p2 = nl.Query[set_type](nl.Symbol[set_type]('p2'), exists_p2)
    solver.walk(query_p1)
    solver.walk(query_p2)
    union_result = (solver.symbol_table['p1'].value | solver.symbol_table['p2'].value) - elem
    assert union_result == res


def test_universal_relation_return_all_elements():

    solver = RegionsSetSolver(TypedSymbolTable())
    superior = Region((0, 0, 5), (1, 1, 6))
    inferior = Region((0, 0, -10), (1, 1, -8))
    central = Region((0, 0, 0), (1, 1, 1))
    db_elems = frozenset([
        superior, inferior
    ])
    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('db')] = nl.Constant[AbstractSet[Region]](db_elems)

    elem = frozenset([central])
    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('e')] = nl.Constant[AbstractSet[Region]](elem)

    type = AbstractSet[Region]
    p1 = nl.Predicate[type](
        nl.Symbol[Callable[[type], type]]('universal'),
        (nl.Symbol[type]('e'),)
    )

    query_a = nl.Query[AbstractSet[Region]](nl.Symbol[AbstractSet[Region]]('a'), p1)
    solver.walk(query_a)
    assert solver.symbol_table['a'].value == frozenset([superior, inferior, central])
    #todo: review


def test_composition_identity():
    class SBS(RegionsSetSolver):
        type = Region

        def predicate_universal(self, reference_elem_in_set: AbstractSet[Region]) -> AbstractSet[Region]:
            res = frozenset()
            for elem in self.symbol_table.symbols_by_type(AbstractSet[Region]).values():
                res = res.union(elem.value)
            return res

    solver = SBS(TypedSymbolTable())

    db_elems = frozenset([Region((0, 0, 5), (1, 1, 6)), Region((0, 0, -10), (1, 1, -8))])
    elem = frozenset([Region((0, 0, 0), (1, 1, 1))])
    set_type = AbstractSet[Region]

    solver.symbol_table[nl.Symbol[set_type]('db')] = nl.Constant[set_type](db_elems)
    solver.symbol_table[nl.Symbol[set_type]('e')] = nl.Constant[set_type](elem)

    pred_type = Callable[
        [set_type],
        set_type
    ]
    solver.symbol_table[nl.Symbol[pred_type]('universal')] = nl.Constant[pred_type](solver.predicate_universal)
    solver.symbol_table[nl.Symbol[pred_type]('converse universal')] = nl.Constant[pred_type](
        solver.predicate_universal)

    id1 = relations_composition(AbstractSet[Region], solver, 'universal', 'superior_of')
    id2 = relations_composition(AbstractSet[Region], solver, 'converse superior_of', 'universal')
    assert id1 == id2


def relations_composition(set_type, solver, relation_1, relation_2):
    p1 = nl.Predicate[set_type](
        nl.Symbol(relation_1),
        (nl.Symbol[set_type]('x'),)
    )

    p2 = nl.Predicate[set_type](
        nl.Symbol('converse ' + relation_2),
        (nl.Symbol[set_type]('x'),)
    )

    exists = nl.ExistentialPredicate[set_type](
        nl.Symbol[set_type]('x'), p1 & p2
    )

    query_a = nl.Query[set_type](nl.Symbol[set_type]('result'), exists)
    solver.walk(query_a)
    return solver.symbol_table['result']


def test_relation_left_of_aligned_from_unit_box():
    region_set_type = AbstractSet[Region]
    solver = RegionsSetSolver(TypedSymbolTable())

    center = Region((0, 0, 0), (1, 1, 1))
    l1 = Region((-5, 0, 0), (-2, 1, 1))
    l2 = Region((-10, 0, 0), (-8, 1, 1))
    r1 = Region((3, 0, 0), (5, 1, 1))

    all_elements = frozenset([center, l1, l2, r1])
    elem = frozenset([center])

    solver.symbol_table[nl.Symbol[region_set_type]('db')] = nl.Constant[region_set_type](all_elements)
    solver.symbol_table[nl.Symbol[region_set_type]('elem')] = nl.Constant[region_set_type](elem)

    north_relation = 'left_of'
    predicate = nl.Predicate[region_set_type](
            nl.Symbol[Callable[[region_set_type], region_set_type]](north_relation),
            (nl.Symbol[region_set_type]('elem'),)
        )

    query = nl.Query[region_set_type](nl.Symbol[region_set_type]('p1'), predicate)
    solver.walk(query)

    assert solver.symbol_table['p1'].value == frozenset([l1, l2])


def test_relation_left_of_unaligned():
    region_set_type = AbstractSet[Region]
    solver = RegionsSetSolver(TypedSymbolTable())

    center = Region((0, 0, 0), (1, 1, 1))
    l1 = Region((-5, -2, -2), (-2, 0, 0))
    l2 = Region((-10, 5, 5), (-8, 8, 7))
    l3 = Region((-12, 9, 8), (-11, 10, 10))
    north = Region((0, 5, 5), (1, 7, 7))
    r_north = Region((10, 5, 5), (12, 7, 7))

    all_elements = frozenset([center, l1, l2, l3, north, r_north])
    elem = frozenset([center])

    solver.symbol_table[nl.Symbol[region_set_type]('db')] = nl.Constant[region_set_type](all_elements)
    solver.symbol_table[nl.Symbol[region_set_type]('elem')] = nl.Constant[region_set_type](elem)

    north_relation = 'left_of'
    predicate = nl.Predicate[region_set_type](
            nl.Symbol[Callable[[region_set_type], region_set_type]](north_relation),
            (nl.Symbol[region_set_type]('elem'),)
        )

    query = nl.Query[region_set_type](nl.Symbol[region_set_type]('p1'), predicate)
    solver.walk(query)

    assert solver.symbol_table['p1'].value == frozenset([l1, l2, l3])


def test_overlapped_hyperrect():
    region_set_type = AbstractSet[Region]
    solver = RegionsSetSolver(TypedSymbolTable())

    a0 = Region((1, -1, 1), (2, 0, 2))
    b0 = Region((0, -1, 0), (1, 0, 1))
    c0 = Region((0, -1, -1), (1, 0, 0))

    a1 = Region((0, 0, 1), (1, 1, 2))
    c1 = Region((0, 0, -1), (1, 1, 0))

    a2 = Region((0, 1, 1), (1, 2, 2))
    b2 = Region((0, 1, 0), (1, 2, 1))
    c2 = Region((0, 1, -1), (1, 2, 0))

    all_elements = frozenset([a0, b0, c0, a1, c1, a2, b2, c2])
    reference_set = frozenset([Region((0, 0, -0.5), (1, 1, 1.5))])

    solver.symbol_table[nl.Symbol[region_set_type]('db')] = nl.Constant[region_set_type](all_elements)
    solver.symbol_table[nl.Symbol[region_set_type]('a')] = nl.Constant[region_set_type](reference_set)

    north_relation = 'overlapping'
    predicate = nl.Predicate[region_set_type](
        nl.Symbol[Callable[[region_set_type], region_set_type]](north_relation),
        (nl.Symbol[region_set_type]('a'),)
    )

    query = nl.Query[region_set_type](nl.Symbol[region_set_type]('p'), predicate)
    solver.walk(query)

    assert solver.symbol_table['p'].value == frozenset([a1, c1])

    reference_set = frozenset([Region((10, 0, -0.5), (12, 1, 1.5))])

    solver.symbol_table[nl.Symbol[region_set_type]('db')] = nl.Constant[region_set_type](all_elements)
    solver.symbol_table[nl.Symbol[region_set_type]('a')] = nl.Constant[region_set_type](reference_set)

    north_relation = 'overlapping'
    predicate = nl.Predicate[region_set_type](
        nl.Symbol[Callable[[region_set_type], region_set_type]](north_relation),
        (nl.Symbol[region_set_type]('a'),)
    )

    query = nl.Query[region_set_type](nl.Symbol[region_set_type]('p'), predicate)
    solver.walk(query)

    assert solver.symbol_table['p'].value == frozenset()

    a0 = Region((5, -1, 0.5), (6, 1.5, 3))
    all_elements = frozenset([a0, b0, c0, a1, c1, a2, b2, c2])
    reference_set = frozenset([Region((5.5, 0, -0.5), (7, 1, 1.5))])

    solver.symbol_table[nl.Symbol[region_set_type]('db')] = nl.Constant[region_set_type](all_elements)
    solver.symbol_table[nl.Symbol[region_set_type]('a')] = nl.Constant[region_set_type](reference_set)

    north_relation = 'overlapping'
    predicate = nl.Predicate[region_set_type](
        nl.Symbol[Callable[[region_set_type], region_set_type]](north_relation),
        (nl.Symbol[region_set_type]('a'),)
    )

    query = nl.Query[region_set_type](nl.Symbol[region_set_type]('p'), predicate)
    solver.walk(query)

    assert solver.symbol_table['p'] == frozenset([a0])



def test_paper_composition_ex():
    set_type = AbstractSet[Region]
    solver = RegionsSetSolver(TypedSymbolTable())

    a = Region((0, 2, 2), (1, 7, 4))
    b = Region((0, 3, 4), (1, 5, 6))
    c = Region((0, 2, 4), (1, 4, 6))

    db_elements = frozenset([a, b, c])

    solver.symbol_table[nl.Symbol[set_type]('db')] = nl.Constant[set_type](db_elements)
    solver.symbol_table[nl.Symbol[set_type]('a')] = nl.Constant[set_type](frozenset([a]))
    res = do_composition_of_relations_from_region(set_type, solver, 'a', 'superior_of', 'superior_of')
    assert res == frozenset([])

    solver = RegionsSetSolver(TypedSymbolTable())
    a = Region((0, 2, 2), (1, 10, 4))
    b = Region((0, 3, 5), (1, 8, 7))
    c = Region((0, 4, 8), (1, 6, 10))

    db_elements = frozenset([a, b, c])

    solver.symbol_table[nl.Symbol[set_type]('db')] = nl.Constant[set_type](db_elements)
    solver.symbol_table[nl.Symbol[set_type]('a')] = nl.Constant[set_type](frozenset([a]))
    res = do_composition_of_relations_from_region(set_type, solver, 'a', 'superior_of', 'superior_of')
    assert res == frozenset([c])


def test_regions_names_from_table():
    region_set_type = AbstractSet[Region]
    solver = RegionsSetSolver(TypedSymbolTable())

    center = Region((0, 0, 0), (1, 1, 1))
    l1 = Region((-5, -2, -2), (-2, 0, 0))
    l2 = Region((-10, 5, 5), (-8, 8, 7))
    l3 = Region((-12, 9, 8), (-11, 10, 10))

    l1_elem = frozenset([l1])
    l2_elem = frozenset([l2])
    l3_elem = frozenset([l3])
    center_elem = frozenset([center])

    solver.symbol_table[nl.Symbol[region_set_type]('L1')] = nl.Constant[region_set_type](l1_elem)
    solver.symbol_table[nl.Symbol[region_set_type]('L2')] = nl.Constant[region_set_type](l2_elem)
    solver.symbol_table[nl.Symbol[region_set_type]('L3')] = nl.Constant[region_set_type](l3_elem)
    solver.symbol_table[nl.Symbol[region_set_type]('CENTRAL')] = nl.Constant[region_set_type](center_elem)
    search_for = frozenset([l1, center])
    res = solver.name_of_regions(search_for)
    assert res == ['L1', 'CENTRAL']


def test_do_query():

    region_set_type = AbstractSet[Region]
    solver = RegionsSetSolver(TypedSymbolTable())

    inferior = Region((0, 0, 0), (1, 1, 1))
    central = Region((0, 0, 2), (1, 1, 3))
    superior = Region((0, 0, 4), (1, 1, 5))

    all_elements = frozenset([inferior, central, superior])
    #todo function to load all symbols into solver
    solver.symbol_table[nl.Symbol[region_set_type]('db')] = nl.Constant[region_set_type](all_elements)
    solver.symbol_table[nl.Symbol[region_set_type]('CENTRAL')] = nl.Constant[region_set_type](frozenset([central]))
    obtained = solver.run_query_on_region('superior_of', 'CENTRAL')
    assert len(obtained) == 0

    solver.symbol_table[nl.Symbol[region_set_type]('BOTTOM')] = nl.Constant[region_set_type](frozenset([inferior]))
    solver.symbol_table[nl.Symbol[region_set_type]('TOP')] = nl.Constant[region_set_type](frozenset([superior]))
    obtained = solver.run_query_on_region('superior_of', 'CENTRAL')
    assert obtained == ['TOP']

    solver.run_query_on_region('superior_of', 'BOTTOM', store_into='not_bottom')
    assert solver.symbol_table['not_bottom'].value == frozenset([central, superior])


def test_regions_algebraic_op():

    bs_vox = np.load('brain-stem-voxels.npy').astype(float)
    affine = np.eye(4)
    brain_stem = ExplicitVBR(bs_vox, affine)
    assert np.array_equal(brain_stem._voxels, brain_stem.to_ijk(affine))

    affine = np.eye(4) * 2
    affine[-1] = 1
    brain_stem = ExplicitVBR(bs_vox, affine)
    assert np.array_equal(brain_stem._voxels, brain_stem.to_ijk(affine))

    affine = np.eye(4)
    affine[:, -1] = np.array([1, 1, 1, 1])
    brain_stem = ExplicitVBR(bs_vox, affine)
    assert np.array_equal(brain_stem._voxels, brain_stem.to_ijk(affine))

    affine = np.array([[-0.69999999, 0., 0., 90.], [0., 0.69999999, 0., -126.], [0., 0., 0.69999999, -72.], [0., 0., 0., 1.]]).round(2)
    brain_stem = ExplicitVBR(bs_vox, affine)
    assert np.array_equal(brain_stem._voxels, brain_stem.to_ijk(affine))
    union = region_union([brain_stem], affine)
    assert union.bounding_box == brain_stem.bounding_box

    center = brain_stem.bounding_box.ub - 5
    radius = 10
    sphere = SphericalVolume(center, radius)
    assert sphere.bounding_box.overlaps(brain_stem.bounding_box)
    intersect = region_intersection([brain_stem, sphere], affine)
    assert intersect is not None

    d1 = region_difference([brain_stem, sphere], affine)
    d2 = region_difference([sphere, brain_stem], affine)
    union = region_union([brain_stem, sphere], affine)
    intersect2 = region_difference([union, d1, d2], affine)
    assert intersect2 is not None
    assert intersect.bounding_box == intersect2.bounding_box


def test_planar_regions_from_query():
    solver = RegionsSetSolver(TypedSymbolTable())
    center = (1, 5, 6)
    vector = (1, 0, 0)
    solver.symbol_table[nl.Symbol[dict]('e')] = nl.Constant[dict]({'origin': center, 'vector': vector})

    p1 = nl.Predicate[dict](
        nl.Symbol[Callable[[dict], AbstractSet[Region]]]('superior_from_plane'),
        (nl.Symbol[dict]('e'),)
    )

    query_a = nl.Query[AbstractSet[Region]](nl.Symbol[dict]('a'), p1)
    solver.walk(query_a)

    region = get_singleton_element_from_frozenset(solver.symbol_table['a'].value)
    assert(np.all(region == PlanarVolume(center, vector)))

    p1 = nl.Predicate[dict](
        nl.Symbol[Callable[[dict], AbstractSet[Region]]]('inferior_from_plane'),
        (nl.Symbol[dict]('e'),)
    )

    query_a = nl.Query[AbstractSet[Region]](nl.Symbol[dict]('a'), p1)
    solver.walk(query_a)

    region = get_singleton_element_from_frozenset(solver.symbol_table['a'].value)
    assert (np.all(region == PlanarVolume(center, vector, direction=-1)))
