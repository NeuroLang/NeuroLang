from ..region_solver import RegionsSetSolver
from ..symbols_and_types import TypedSymbolTable
from .. import neurolang as nl
from typing import AbstractSet, Callable
from ..regions import Region


def test_relation_superior_of():
    region_set_type = AbstractSet[Region]
    solver = RegionsSetSolver(TypedSymbolTable())

    inferior = Region((0, 0), (1, 1))
    central = Region((0, 2), (1, 3))
    superior = Region((0, 4), (1, 5))

    all_elements = frozenset([inferior, central, superior])
    elem = frozenset([central])

    solver.symbol_table[nl.Symbol[region_set_type]('db')] = nl.Constant[region_set_type](all_elements)
    solver.symbol_table[nl.Symbol[region_set_type]('elem')] = nl.Constant[region_set_type](elem)

    north_relation = 'superior_of'
    predicate = nl.Predicate[region_set_type](
            nl.Symbol[Callable[[region_set_type], region_set_type]](north_relation),
            (nl.Symbol[region_set_type]('elem'),)
        )

    query = nl.Query[region_set_type](nl.Symbol[region_set_type]('p1'), predicate)
    solver.walk(query)

    assert solver.symbol_table['p1'].value == frozenset([superior])


def test_north_u_south():
    solver = RegionsSetSolver(TypedSymbolTable())

    db_elems = frozenset([Region((0, 5), (1, 6)), Region((0, -10), (1, -8))])
    elem = frozenset([Region((0, 0), (1, 1))])

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

    db_elems = frozenset([Region((0, 5), (1, 6)), Region((0, -10), (1, -8))])
    elem = frozenset([Region((0, 0), (1, 1))])

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
        Region((0, 5), (1, 6)), Region((0, -10), (1, -8))
    ])
    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('db')] = nl.Constant[AbstractSet[Region]](db_elems)

    elem = frozenset([Region((0, 0), (1, 1))])

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

    superior = Region((0, 5), (1, 6))
    central = Region((0, 0), (1, 1))
    lat1 = Region((2, 0), (3, 1))
    lat2 = Region((6, 0), (7, 1))
    lat3 = Region((10, 0), (11, 1))

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

    superior = Region((0, 5), (1, 6))
    central = Region((0, 0), (1, 1))
    lat0 = Region((-3, 0), (-2, 1))
    lat1 = Region((2, 0), (3, 1))
    lat2 = Region((6, 0), (7, 1))
    lat3 = Region((10, 0), (11, 1))

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

    inferior = Region((0, -10), (1, -8))
    central = Region((0, 0), (1, 1))
    superior = Region((0, 5), (1, 6))

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

    db_elems = frozenset([Region((0, 5), (1, 6)), Region((0, -10), (1, -8))])
    elem = frozenset([Region((0, 0), (1, 1))])

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
    superior = Region((0, 5), (1, 6))
    inferior = Region((0, -10), (1, -8))
    central = Region((0, 0), (1, 1))
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
'''todo: review'''


def test_composition_identity():
    class SBS(RegionsSetSolver):
        type = Region

        def predicate_universal(self, reference_elem_in_set: AbstractSet[Region]) -> AbstractSet[Region]:
            res = frozenset()
            for elem in self.symbol_table.symbols_by_type(AbstractSet[Region]).values():
                res = res.union(elem.value)
            return res

    solver = SBS(TypedSymbolTable())

    db_elems = frozenset([Region((0, 5), (1, 6)), Region((0, -10), (1, -8))])
    elem = frozenset([Region((0, 0), (1, 1))])
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


def test_overlapping_hyperrectangles():
    region_set_type = AbstractSet[Region]
    solver = RegionsSetSolver(TypedSymbolTable())

    a = Region((-10, -10), (5, 5))
    b = Region((0, 0), (15, 15))

    all_elements = frozenset([a, b])
    a = frozenset([a])
    b = frozenset([b])

    solver.symbol_table[nl.Symbol[region_set_type]('db')] = nl.Constant[region_set_type](all_elements)
    solver.symbol_table[nl.Symbol[region_set_type]('a')] = nl.Constant[region_set_type](a)
    solver.symbol_table[nl.Symbol[region_set_type]('b')] = nl.Constant[region_set_type](b)

    north_relation = 'overlapping'
    predicate = nl.Predicate[region_set_type](
        nl.Symbol[Callable[[region_set_type], region_set_type]](north_relation),
        (nl.Symbol[region_set_type]('a'),)
    )
    predicate_r = nl.Predicate[region_set_type](
        nl.Symbol[Callable[[region_set_type], region_set_type]](north_relation),
        (nl.Symbol[region_set_type]('b'),)
    )

    query = nl.Query[region_set_type](nl.Symbol[region_set_type]('p'), predicate)
    solver.walk(query)
    query = nl.Query[region_set_type](nl.Symbol[region_set_type]('pr'), predicate_r)
    solver.walk(query)

    assert solver.symbol_table['p'].value == b
    assert solver.symbol_table['pr'].value == a


def test_paper_composition_ex():
    set_type = AbstractSet[Region]
    solver = RegionsSetSolver(TypedSymbolTable())

    a = Region((2, 2), (7, 4))
    b = Region((3, 4), (5, 6))
    c = Region((2, 4), (4, 6))

    db_elements = frozenset([a, b, c])

    solver.symbol_table[nl.Symbol[set_type]('db')] = nl.Constant[set_type](db_elements)
    solver.symbol_table[nl.Symbol[set_type]('a')] = nl.Constant[set_type](frozenset([a]))
    res = do_composition_of_relations_from_region(set_type, solver, 'a', 'superior_of', 'superior_of')
    assert res == frozenset([])

    solver = RegionsSetSolver(TypedSymbolTable())
    a = Region((2, 2), (10, 4))
    b = Region((3, 5), (8, 7))
    c = Region((4, 8), (6, 10))

    db_elements = frozenset([a, b, c])

    solver.symbol_table[nl.Symbol[set_type]('db')] = nl.Constant[set_type](db_elements)
    solver.symbol_table[nl.Symbol[set_type]('a')] = nl.Constant[set_type](frozenset([a]))
    res = do_composition_of_relations_from_region(set_type, solver, 'a', 'superior_of', 'superior_of')
    assert res == frozenset([c])


def test_get_labels_from_region_results():
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

    solver.symbol_table[nl.Symbol[region_set_type]('l1')] = nl.Constant[region_set_type](l1_elem)
    solver.symbol_table[nl.Symbol[region_set_type]('l2')] = nl.Constant[region_set_type](l2_elem)
    solver.symbol_table[nl.Symbol[region_set_type]('l3')] = nl.Constant[region_set_type](l3_elem)
    solver.symbol_table[nl.Symbol[region_set_type]('central')] = nl.Constant[region_set_type](center_elem)
    search_for = frozenset([l1, center])
    res = solver.get_label_of_regions_by_limits(search_for)
    print(res)
