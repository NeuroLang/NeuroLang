from ..region_solver import RegionsSetSolver
from ..symbols_and_types import TypedSymbolTable
from .. import neurolang as nl
from typing import AbstractSet, Callable
from ..regions import Region, ExplicitVBR, take_principal_regions
from ..CD_relations import cardinal_relation
import nibabel as nib
import numpy as np
from numpy import random
import pytest


def do_query_of_regions_in_relation_to_region(
    solver, elem, relation, output_symbol_name='q'
):

    predicate = nl.FunctionApplication[solver.set_type](
        nl.Symbol[Callable[[solver.type], solver.set_type]](relation),
        (nl.Symbol[solver.type](elem), )
    )

    query = nl.Query[solver.set_type](
        nl.Symbol[solver.set_type](output_symbol_name), predicate
    )
    solver.walk(query)

    return solver.symbol_table[output_symbol_name]


def get_singleton_element_from_frozenset(fs):
    return next(iter(fs))


# todo: refa this awful tests
def test_relation_superior_of():
    solver = RegionsSetSolver(TypedSymbolTable())

    inferior = Region((0, 0, 0), (1, 1, 1))
    central = Region((0, 0, 2), (1, 1, 3))
    superior = Region((0, 0, 4), (1, 1, 5))

    solver.symbol_table[nl.Symbol[solver.type]('inf_region')
                        ] = nl.Constant[solver.type](inferior)
    solver.symbol_table[nl.Symbol[solver.type]('central_region')
                        ] = nl.Constant[solver.type](central)
    solver.symbol_table[nl.Symbol[solver.type]('sup_region')
                        ] = nl.Constant[solver.type](superior)

    obtained = do_query_of_regions_in_relation_to_region(
        solver, 'central_region', 'superior_of'
    ).value
    assert obtained == frozenset([superior])


def test_superior_u_inferior_relation():
    solver = RegionsSetSolver(TypedSymbolTable())

    inferior = Region((0, 0, -10), (1, 1, -8))
    central = Region((0, 0, 0), (1, 1, 1))
    superior = Region((0, 0, 5), (1, 1, 6))

    solver.symbol_table[nl.Symbol[solver.type]('inf_region')
                        ] = nl.Constant[solver.type](inferior)
    solver.symbol_table[nl.Symbol[solver.type]('central_region')
                        ] = nl.Constant[solver.type](central)
    solver.symbol_table[nl.Symbol[solver.type]('sup_region')
                        ] = nl.Constant[solver.type](superior)

    check_union_commutativity(
        solver, 'superior_of', 'inferior_of', 'central_region'
    )


def check_union_commutativity(solver, relation1, relation2, element):

    p1 = nl.FunctionApplication[solver.set_type](
        nl.Symbol[Callable[[solver.type], solver.set_type]](relation1),
        (nl.Symbol[solver.type](element), )
    )

    p2 = nl.FunctionApplication[solver.set_type](
        nl.Symbol[Callable[[solver.type], solver.set_type]](relation2),
        (nl.Symbol[solver.type](element), )
    )

    query_a = nl.Query[solver.set_type](
        nl.Symbol[solver.set_type]('a'), (p1 | p2)
    )
    query_b = nl.Query[solver.set_type](
        nl.Symbol[solver.set_type]('b'), (p2 | p1)
    )
    solver.walk(query_a)
    solver.walk(query_b)
    assert solver.symbol_table['a'] == solver.symbol_table['b']


def test_relations_union_associativity():
    solver = RegionsSetSolver(TypedSymbolTable())

    inferior = Region((0, 0, -10), (1, 1, -8))
    central = Region((0, 0, 0), (1, 1, 1))
    superior = Region((0, 0, 5), (1, 1, 6))

    solver.symbol_table[nl.Symbol[solver.type]('inf_region')
                        ] = nl.Constant[solver.type](inferior)
    solver.symbol_table[nl.Symbol[solver.type]('central_region')
                        ] = nl.Constant[solver.type](central)
    solver.symbol_table[nl.Symbol[solver.type]('sup_region')
                        ] = nl.Constant[solver.type](superior)

    check_union_associativity(
        solver, 'superior_of', 'inferior_of', 'posterior_of', 'central_region'
    )


def check_union_associativity(
    solver, relation1, relation2, relation3, element
):
    p1 = nl.FunctionApplication[solver.set_type](
        nl.Symbol[Callable[[solver.type], solver.set_type]](relation1),
        (nl.Symbol[solver.type](element), )
    )

    p2 = nl.FunctionApplication[solver.set_type](
        nl.Symbol[Callable[[solver.type], solver.set_type]](relation2),
        (nl.Symbol[solver.type](element), )
    )

    p3 = nl.FunctionApplication[solver.set_type](
        nl.Symbol[Callable[[solver.type], solver.set_type]](relation3),
        (nl.Symbol[solver.type](element), )
    )

    query_a = nl.Query[solver.set_type](
        nl.Symbol[solver.set_type]('a'), p1 | (p2 | p3)
    )
    query_b = nl.Query[solver.set_type](
        nl.Symbol[solver.set_type]('b'), (p1 | p2) | p3
    )

    solver.walk(query_a)
    solver.walk(query_b)

    assert solver.symbol_table['a'] == solver.symbol_table['b']


def test_huntington_axiom_satisfiability():
    solver = RegionsSetSolver(TypedSymbolTable())

    inferior = Region((0, 0, -10), (1, 1, -8))
    central = Region((0, 0, 0), (1, 1, 1))
    superior = Region((0, 0, 5), (1, 1, 6))

    solver.symbol_table[nl.Symbol[solver.type]('inf_region')
                        ] = nl.Constant[solver.type](inferior)
    solver.symbol_table[nl.Symbol[solver.type]('central_region')
                        ] = nl.Constant[solver.type](central)
    solver.symbol_table[nl.Symbol[solver.type]('sup_region')
                        ] = nl.Constant[solver.type](superior)

    check_huntington(solver, 'superior_of', 'inferior_of', 'central_region')


def check_huntington(solver, relation1, relation2, element):
    p1 = nl.FunctionApplication[solver.set_type](
        nl.Symbol[Callable[[solver.type], solver.set_type]](relation1),
        (nl.Symbol[solver.type](element), )
    )

    p2 = nl.FunctionApplication[solver.set_type](
        nl.Symbol[Callable[[solver.type], solver.set_type]](relation2),
        (nl.Symbol[solver.type](element), )
    )

    query_a = nl.Query[solver.set_type](nl.Symbol[solver.set_type]('a'), p1)
    solver.walk(query_a)

    query_b = nl.Query[solver.set_type](
        nl.Symbol[solver.set_type]('b'), ~(~p1 | ~p2) | ~(~p1 | p2)
    )
    solver.walk(query_b)
    assert solver.symbol_table['a'] == solver.symbol_table['b']


def test_composition_of_relations():
    solver = RegionsSetSolver(TypedSymbolTable())

    superior = Region((0, 0, 5), (1, 1, 6))
    central = Region((0, 0, 0), (1, 1, 1))
    lat1 = Region((0, 2, 0), (1, 3, 1))
    lat2 = Region((0, 6, 0), (1, 7, 1))
    lat3 = Region((0, 10, 0), (1, 11, 1))

    solver.symbol_table[nl.Symbol[solver.type]('sup_region')
                        ] = nl.Constant[solver.type](superior)
    solver.symbol_table[nl.Symbol[solver.type]('central_region')
                        ] = nl.Constant[solver.type](central)
    solver.symbol_table[nl.Symbol[solver.type]('l1')
                        ] = nl.Constant[solver.type](lat1)
    solver.symbol_table[nl.Symbol[solver.type]('l2')
                        ] = nl.Constant[solver.type](lat2)
    solver.symbol_table[nl.Symbol[solver.type]('l3')
                        ] = nl.Constant[solver.type](lat3)

    res = do_composition_of_relations(
        solver, 'sup_region', 'anterior_of', 'inferior_of'
    )

    assert res == frozenset([lat1, lat2, lat3])


def do_composition_of_relations(solver, region, relation1, relation2):

    bs = do_query_of_regions_in_relation_to_region(solver, region, relation2)
    if bs.value == frozenset():
        return bs.value

    result = frozenset()
    for b_element in bs.value:
        solver.symbol_table[nl.Symbol[solver.type]('b_element')
                            ] = nl.Constant[solver.type](b_element)
        res = do_query_of_regions_in_relation_to_region(
            solver, 'b_element', relation1
        )
        result = result.union(res.value)
    return result


def relations_composition(solver, relation1, relation2):
    p1 = nl.FunctionApplication[solver.set_type](
        nl.Symbol[Callable[[solver.type], solver.set_type]](relation1),
        (nl.Symbol[solver.type]('x'), )
    )

    p2 = nl.FunctionApplication[solver.set_type](
        nl.Symbol[Callable[[solver.type], solver.set_type]]
        ('converse' + relation2), (nl.Symbol[solver.type]('x'), )
    )

    exists = nl.ExistentialFunctionApplication[solver.set_type](
        nl.Symbol[solver.set_type]('x'), p1 & p2
    )

    query_a = nl.Query[solver.set_type](
        nl.Symbol[solver.set_type]('result'), exists
    )
    solver.walk(query_a)
    return solver.symbol_table['result']


def test_composition_distributivity():
    solver = RegionsSetSolver(TypedSymbolTable())

    superior = Region((0, 0, 5), (1, 1, 6))
    central = Region((0, 0, 0), (1, 1, 1))
    lat0 = Region((0, -3, 0), (1, -2, 1))
    lat1 = Region((0, 2, 0), (1, 3, 1))
    lat2 = Region((0, 6, 0), (1, 7, 1))
    lat3 = Region((0, 10, 0), (1, 11, 1))

    solver.symbol_table[nl.Symbol[solver.type]('l0')
                        ] = nl.Constant[solver.type](lat0)
    solver.symbol_table[nl.Symbol[solver.type]('l1')
                        ] = nl.Constant[solver.type](lat1)
    solver.symbol_table[nl.Symbol[solver.type]('l2')
                        ] = nl.Constant[solver.type](lat2)
    solver.symbol_table[nl.Symbol[solver.type]('l3')
                        ] = nl.Constant[solver.type](lat3)
    solver.symbol_table[nl.Symbol[solver.type]('central')
                        ] = nl.Constant[solver.type](central)
    solver.symbol_table[nl.Symbol[solver.type]('superior')
                        ] = nl.Constant[solver.type](superior)

    res = check_distributivity(
        solver, 'superior', 'posterior_of', 'anterior_of', 'inferior_of'
    )
    c1 = do_composition_of_relations(
        solver, 'superior', 'posterior_of', 'inferior_of'
    )
    c2 = do_composition_of_relations(
        solver, 'superior', 'anterior_of', 'inferior_of'
    )
    assert res == c1.union(c2)


def check_distributivity(solver, elem, rel1, rel2, rel3):

    res = do_query_of_regions_in_relation_to_region(solver, elem, rel3)
    obtained = frozenset()

    # iterate over the result of a query b in B and apply
    # R&S b to obtain the left side element (a)
    for elements in res.value:

        solver.symbol_table[nl.Symbol[solver.type]('pred_base_element')
                            ] = nl.Constant[solver.type](elements)
        p1 = nl.FunctionApplication[solver.set_type](
            nl.Symbol[Callable[[solver.type], solver.set_type]](rel1),
            (nl.Symbol[solver.type]('pred_base_element'), )
        )
        p2 = nl.FunctionApplication[solver.set_type](
            nl.Symbol[Callable[[solver.type], solver.set_type]](rel2),
            (nl.Symbol[solver.type]('pred_base_element'), )
        )

        query_union = nl.Query[solver.set_type](
            nl.Symbol[solver.set_type]('union'), (p1 | p2)
        )
        solver.walk(query_union)
        obtained = obtained.union(solver.symbol_table['union'].value)

    return obtained


def test_left_of():
    solver = RegionsSetSolver(TypedSymbolTable())

    center = Region((0, 0, 0), (1, 1, 1))
    l1 = Region((-5, -2, -2), (-2, 0, 0))
    l2 = Region((-10, 5, 5), (-8, 8, 7))
    l3 = Region((-12, 9, 8), (-11, 10, 10))
    north = Region((0, 5, 5), (1, 7, 7))
    r_north = Region((10, 5, 5), (12, 7, 7))

    solver.symbol_table[nl.Symbol[solver.type]('l1')
                        ] = nl.Constant[solver.type](l1)
    solver.symbol_table[nl.Symbol[solver.type]('l2')
                        ] = nl.Constant[solver.type](l2)
    solver.symbol_table[nl.Symbol[solver.type]('l3')
                        ] = nl.Constant[solver.type](l3)
    solver.symbol_table[nl.Symbol[solver.type]('center')
                        ] = nl.Constant[solver.type](center)
    solver.symbol_table[nl.Symbol[solver.type]('north')
                        ] = nl.Constant[solver.type](north)
    solver.symbol_table[nl.Symbol[solver.type]('r_north')
                        ] = nl.Constant[solver.type](r_north)

    obtained_value = do_query_of_regions_in_relation_to_region(
        solver, 'center', 'left_of'
    ).value
    assert obtained_value == frozenset([l1, l2, l3])


def test_overlapping_bounding_boxes():
    solver = RegionsSetSolver(TypedSymbolTable())

    a0 = Region((-1, 0, 1), (0, 1, 2))
    b0 = Region((-1, 0, 0), (0, 1, 1))
    c0 = Region((-1, 0, -1), (0, 1, 0))

    a1 = Region((0, 0, 1), (1, 1, 2))
    c1 = Region((0, 0, -1), (1, 1, 0))

    a2 = Region((0, 1, 1), (1, 2, 2))
    b2 = Region((0, 1, 0), (1, 2, 1))
    c2 = Region((0, 1, -1), (1, 2, 0))

    solver.symbol_table[nl.Symbol[solver.type]('a0')
                        ] = nl.Constant[solver.type](a0)
    solver.symbol_table[nl.Symbol[solver.type]('b0')
                        ] = nl.Constant[solver.type](b0)
    solver.symbol_table[nl.Symbol[solver.type]('c0')
                        ] = nl.Constant[solver.type](c0)

    solver.symbol_table[nl.Symbol[solver.type]('a1')
                        ] = nl.Constant[solver.type](a1)
    solver.symbol_table[nl.Symbol[solver.type]('c1')
                        ] = nl.Constant[solver.type](c1)

    solver.symbol_table[nl.Symbol[solver.type]('a2')
                        ] = nl.Constant[solver.type](a2)
    solver.symbol_table[nl.Symbol[solver.type]('b2')
                        ] = nl.Constant[solver.type](b2)
    solver.symbol_table[nl.Symbol[solver.type]('b3')
                        ] = nl.Constant[solver.type](c2)

    solver.symbol_table[nl.Symbol[solver.type]('reference')] = \
        nl.Constant[solver.type](Region((0, 0, -0.5), (1, 1, 1.5)))

    obtained_value = do_query_of_regions_in_relation_to_region(
        solver, 'reference', 'overlapping'
    ).value
    assert obtained_value == frozenset([a1, c1])

    solver.symbol_table[nl.Symbol[solver.type]('reference')] = \
        nl.Constant[solver.type](Region((10, 0, -0.5), (12, 1, 1.5)))

    obtained_value = do_query_of_regions_in_relation_to_region(
        solver, 'reference', 'overlapping'
    ).value
    assert obtained_value == frozenset()


def test_paper_composition_ex():
    solver = RegionsSetSolver(TypedSymbolTable())

    a = Region((0, 2, 2), (1, 7, 4))
    b = Region((0, 3, 4), (1, 5, 6))
    c = Region((0, 2, 4), (1, 4, 6))

    solver.symbol_table[nl.Symbol[solver.type]('a')
                        ] = nl.Constant[solver.type](a)
    solver.symbol_table[nl.Symbol[solver.type]('b')
                        ] = nl.Constant[solver.type](b)
    solver.symbol_table[nl.Symbol[solver.type]('c')
                        ] = nl.Constant[solver.type](c)

    res = do_composition_of_relations(
        solver, 'a', 'superior_of', 'superior_of'
    )
    assert res == frozenset([])

    solver = RegionsSetSolver(TypedSymbolTable())
    a = Region((0, 2, 2), (1, 10, 4))
    b = Region((0, 3, 5), (1, 8, 7))
    c = Region((0, 4, 8), (1, 6, 10))

    solver.symbol_table[nl.Symbol[solver.type]('a')
                        ] = nl.Constant[solver.type](a)
    solver.symbol_table[nl.Symbol[solver.type]('b')
                        ] = nl.Constant[solver.type](b)
    solver.symbol_table[nl.Symbol[solver.type]('c')
                        ] = nl.Constant[solver.type](c)

    solver.symbol_table[nl.Symbol[solver.type]('a')
                        ] = nl.Constant[solver.type](a)
    res = do_composition_of_relations(
        solver, 'a', 'superior_of', 'superior_of'
    )
    assert res == frozenset([c])


@pytest.mark.skip(reason="need to fix neurosynth-based test")
def test_term_defined_regions_creation():

    solver = RegionsSetSolver(TypedSymbolTable())
    solver.symbol_table[nl.Symbol[str]('term')] = nl.Constant[str]('emotion')
    obtained = do_query_of_regions_from_term(
        solver, 'term', 'neurosynth_term'
    ).value
    assert not obtained == frozenset([])


def do_query_of_regions_from_term(
    solver, elem, relation, output_symbol_name='q'
):

    predicate = nl.FunctionApplication[solver.set_type](
        nl.Symbol[Callable[[str], solver.set_type]](relation),
        (nl.Symbol[str](elem), )
    )

    query = nl.Query[solver.set_type](
        nl.Symbol[solver.set_type](output_symbol_name), predicate
    )
    solver.walk(query)

    return solver.symbol_table[output_symbol_name]


@pytest.mark.skip(reason="need to fix neurosynth-based test")
def test_term_defined_relative_position():

    solver = RegionsSetSolver(TypedSymbolTable())
    solver.symbol_table[nl.Symbol[str]('term')
                        ] = nl.Constant[str]('temporal lobe')
    temporal_lobe = do_query_of_regions_from_term(
        solver, 'term', 'neurosynth_term', output_symbol_name='TEMPORAL LOBE'
    )

    solver.symbol_table[nl.Symbol[solver.type]('temporal_region')] = (
        nl.Constant[solver.type](
            get_singleton_element_from_frozenset(temporal_lobe.value)
        )
    )

    superior_relation = 'anterior_of'
    predicate = nl.FunctionApplication[solver.set_type](
        nl.Symbol[Callable[[solver.set_type], solver.type]](superior_relation),
        (nl.Symbol[solver.type]('temporal_region'), )
    )

    anterior_region = ExplicitVBR(np.array([[50, 90, 50]]), np.eye(4))
    solver.symbol_table[nl.Symbol[solver.set_type]('anterior_region')
                        ] = nl.Constant[solver.type](anterior_region)

    query = nl.Query[solver.set_type](
        nl.Symbol[solver.set_type]('p2'), predicate
    )
    solver.walk(query)

    assert solver.symbol_table['p2'].value == frozenset([anterior_region])


@pytest.mark.skip(reason="need to fix neurosynth-based test")
def test_term_defined_solve_overlapping():

    solver = RegionsSetSolver(TypedSymbolTable())
    solver.symbol_table[nl.Symbol[str]('term')] = nl.Constant[str]('gambling')
    gambling = do_query_of_regions_from_term(
        solver, 'term', 'neurosynth_term', output_symbol_name='GAMBLING'
    )

    a_gambling_region = get_singleton_element_from_frozenset(
        take_principal_regions(gambling.value, 1)
    )
    solver.symbol_table[nl.Symbol[solver.type]('gambling')
                        ] = nl.Constant[solver.type](a_gambling_region)

    center = a_gambling_region.bounding_box.ub
    a_voxels = nib.affines.apply_affine(
        np.linalg.inv(a_gambling_region.affine),
        np.array([center - 1, center + 1])
    )

    solver.symbol_table[nl.Symbol[solver.type]('OVERLAPPING_RECTANGLE')
                        ] = nl.Constant[solver.type](
                            ExplicitVBR(a_voxels, a_gambling_region.affine)
                        )

    predicate = nl.FunctionApplication[solver.set_type](
        nl.Symbol[Callable[[solver.set_type], solver.type]]('overlapping'),
        (nl.Symbol[solver.type]('gambling'), )
    )

    query = nl.Query[solver.set_type](
        nl.Symbol[solver.set_type]('p1'), predicate
    )
    solver.walk(query)
    assert solver.symbol_table['p1'].value == set()

    assert cardinal_relation(
        a_gambling_region,
        ExplicitVBR(a_voxels, a_gambling_region.affine),
        'O',
        refine_overlapping=False
    )
    assert not cardinal_relation(
        a_gambling_region,
        ExplicitVBR(a_voxels, a_gambling_region.affine),
        'O',
        refine_overlapping=True
    )


def test_regexp_region_union():

    region_set_type = AbstractSet[Region]
    solver = RegionsSetSolver(TypedSymbolTable())
    vbr0 = ExplicitVBR(np.array([[0, 0, 0]]), np.eye(4))
    vbr1 = ExplicitVBR(np.array([[1, 1, 1]]), np.eye(4))
    vbr2 = ExplicitVBR(np.array([[2, 2, 2]]), np.eye(4))
    vbr_rand = ExplicitVBR(
        np.array([[
            random.randint(0, 1000),
            random.randint(0, 1000),
            random.randint(0, 1000)
        ]]), np.eye(4)
    )

    solver.symbol_table[nl.Symbol[region_set_type]('REGION_ZEROS')
                        ] = nl.Constant[region_set_type](frozenset([vbr0]))
    solver.symbol_table[nl.Symbol[region_set_type]('REGION_ONES')
                        ] = nl.Constant[region_set_type](frozenset([vbr1]))
    solver.symbol_table[nl.Symbol[region_set_type]('REGION_TWOS')
                        ] = nl.Constant[region_set_type](frozenset([vbr2]))
    solver.symbol_table[nl.Symbol[region_set_type]('RAND_REGIONS')
                        ] = nl.Constant[region_set_type](
                            frozenset([vbr_rand])
                        )

    solver.symbol_table[nl.Symbol[str]('term')] = nl.Constant[str]('^REG\w')
    obtained = do_query_of_regions_from_term(solver, 'term', 'regexp').value
    affine = get_singleton_element_from_frozenset(obtained).affine
    voxels = get_singleton_element_from_frozenset(obtained).voxels
    assert np.array_equal(affine, vbr0.affine)
    assert vbr1.voxels in voxels
    assert vbr2.voxels in voxels
    assert vbr_rand.voxels not in voxels

    solver.symbol_table[nl.Symbol[str]('term')] = nl.Constant[str]('REG\w')
    obtained = do_query_of_regions_from_term(solver, 'term', 'regexp').value

    voxels = get_singleton_element_from_frozenset(obtained).voxels
    assert vbr1.voxels in voxels
    assert vbr_rand.voxels in voxels

    solver.symbol_table[nl.Symbol[str]('term')] = nl.Constant[str]('[N|O]S$')
    obtained = do_query_of_regions_from_term(solver, 'term', 'regexp').value
    voxels = get_singleton_element_from_frozenset(obtained).voxels
    assert vbr0.voxels in voxels
    assert vbr_rand.voxels in voxels
    assert vbr1.voxels not in voxels
