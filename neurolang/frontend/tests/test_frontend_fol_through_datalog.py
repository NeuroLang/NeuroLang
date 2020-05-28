from .. import RegionFrontendFolThroughDatalog, RegionFrontend
from ...regions import ExplicitVBR, Region, SphericalVolume
from ..query_resolution_expressions import Symbol
from typing import AbstractSet, Tuple
from unittest.mock import patch
from unittest import skip

import numpy as np


def test_add_set():
    neurolang = RegionFrontendFolThroughDatalog()

    s = neurolang.add_tuple_set(range(10), int)
    res = neurolang[s]

    assert s.type is AbstractSet[Tuple[int]]
    assert res.type is AbstractSet[Tuple[int]]
    assert res.value == frozenset((i,) for i in range(10))

    v = frozenset(zip(("a", "b", "c"), range(3)))
    s = neurolang.add_tuple_set(v, (str, int))
    res = neurolang[s]

    assert s.type is AbstractSet[Tuple[str, int]]
    assert res.type is AbstractSet[Tuple[str, int]]
    assert res.value == v


def test_add_regions_and_query():
    neurolang = RegionFrontendFolThroughDatalog()

    inferior = Region((0, 0, 0), (1, 1, 1))
    superior = Region((0, 0, 4), (1, 1, 5))

    neurolang.add_region(inferior, name="inferior_region")
    neurolang.add_region(superior, name="superior_region")
    assert neurolang.symbols.inferior_region.value == inferior
    assert neurolang.symbols.superior_region.value == superior

    x = neurolang.new_region_symbol(name="x")
    query = neurolang.query(
        x, neurolang.symbols.superior_of(x, neurolang.symbols.inferior_region)
    )
    query_result = query.do(name="result_of_test_query")

    assert isinstance(query_result, Symbol)
    assert isinstance(query_result.value, AbstractSet)
    assert len(query_result.value) == 1
    assert (superior,) == next(iter(query_result.value))


def test_query_regions_from_region_set():
    neurolang = RegionFrontendFolThroughDatalog()

    central = ExplicitVBR(np.array([[0, 0, 5], [1, 1, 8]]), np.eye(4))
    neurolang.add_region(central, name="reference_region")

    i1 = ExplicitVBR(np.array([[0, 0, 2], [1, 1, 3]]), np.eye(4))
    i2 = ExplicitVBR(np.array([[0, 0, -1], [1, 1, 2]]), np.eye(4))
    i3 = ExplicitVBR(np.array([[0, 0, -10], [1, 1, -5]]), np.eye(4))
    regions = {i1, i2, i3}
    neurolang.add_tuple_set(regions, ExplicitVBR)

    x = neurolang.new_region_symbol(name="x")

    query = neurolang.query(
        x, neurolang.symbols.inferior_of(x, neurolang.symbols.reference_region)
    )
    query_result = query.do(name="result_of_test_query")

    assert len(query_result.value) == len(regions)
    assert query_result.value == {(i1,), (i2,), (i3,)}


def test_query_new_predicate():
    neurolang = RegionFrontendFolThroughDatalog()

    central = ExplicitVBR(np.array([[0, 0, 5], [1, 1, 8]]), np.eye(4))
    reference_symbol = neurolang.add_region(central, name="reference_region")

    inferior_posterior = ExplicitVBR(
        np.array([[0, -10, -10], [1, -5, -5]]), np.eye(4)
    )

    inferior_central = ExplicitVBR(
        np.array([[0, 0, -1], [1, 1, 2]]), np.eye(4)
    )
    inferior_anterior = ExplicitVBR(
        np.array([[0, 2, 2], [1, 5, 3]]), np.eye(4)
    )

    regions = {inferior_posterior, inferior_central, inferior_anterior}

    neurolang.add_tuple_set(regions, ExplicitVBR)

    def posterior_and_inferior(y, z):
        return neurolang.symbols.anatomical_posterior_of(
            y, z
        ) & neurolang.symbols.anatomical_inferior_of(y, z)

    x = neurolang.new_region_symbol(name="x")
    query = neurolang.query(x, posterior_and_inferior(x, reference_symbol))
    query_result = query.do(name="result_of_test_query")
    assert len(query_result.value) == 1
    assert next(iter(query_result.value)) == (inferior_posterior,)


@skip("Also fails for RegionFrontend because `len(query_result.value) != 1`")
def test_load_spherical_volume_first_order():
    neurolang = RegionFrontendFolThroughDatalog()

    inferior = ExplicitVBR(np.array([[0, 0, 0], [1, 1, 1]]), np.eye(4))

    neurolang.add_region(inferior, name="inferior_region")
    neurolang.sphere((0, 0, 0), 0.5, name="unit_sphere")
    assert neurolang.symbols["unit_sphere"].value == SphericalVolume(
        (0, 0, 0), 0.5
    )

    x = neurolang.new_region_symbol(name="x")
    query = neurolang.query(
        x, neurolang.symbols.overlapping(x, neurolang.symbols.unit_sphere)
    )
    query_result = query.do()
    assert len(query_result.value) == 1
    assert next(iter(query_result.value)) == (inferior,)

    neurolang.make_implicit_regions_explicit(np.eye(4), (500, 500, 500))
    query = neurolang.query(
        x, neurolang.symbols.overlapping(x, neurolang.symbols.unit_sphere)
    )
    query_result = query.do()
    sphere_constant = neurolang.symbols["unit_sphere"].value
    assert (
        isinstance(sphere_constant, ExplicitVBR)
        and np.array_equal(sphere_constant.affine, np.eye(4))
        and sphere_constant.image_dim == (500, 500, 500)
        and np.array_equal(sphere_constant.voxels, [[0, 0, 0]])
    )
    assert len(query_result.value) == 1
    assert next(iter(query_result.value)) == inferior


def test_multiple_symbols_query():
    neurolang = RegionFrontendFolThroughDatalog()
    r1 = ExplicitVBR(np.array([[0, 0, 5], [1, 1, 10]]), np.eye(4))
    r2 = ExplicitVBR(np.array([[0, 0, -10], [1, 1, -5]]), np.eye(4))
    neurolang.add_region(r1, name="r1")
    neurolang.add_region(r2, name="r2")

    central = ExplicitVBR(np.array([[0, 0, 1], [1, 1, 1]]), np.eye(4))
    neurolang.add_region(central, name="reference_region")

    x = neurolang.new_region_symbol(name="x")
    y = neurolang.new_region_symbol(name="y")
    pred = neurolang.symbols.superior_of(
        x, neurolang.symbols.reference_region
    ) & neurolang.symbols.inferior_of(y, neurolang.symbols.reference_region)

    res = neurolang.query((x, y), pred).do()
    assert res.value == frozenset({(r1, r2)})


def test_tuple_symbol_multiple_types_query():
    neurolang = RegionFrontendFolThroughDatalog()
    r1 = ExplicitVBR(np.array([[0, 0, 5], [1, 1, 10]]), np.eye(4))
    r2 = ExplicitVBR(np.array([[0, 0, -10], [1, 1, -5]]), np.eye(4))

    neurolang.add_region(r1, name="r1")
    neurolang.add_region(r2, name="r2")

    central = ExplicitVBR(np.array([[0, 0, 1], [1, 1, 1]]), np.eye(4))
    neurolang.add_region(central, name="reference_region")

    x = neurolang.new_region_symbol(name="x")
    z = neurolang.new_symbol(int, name="max_value")

    def norm_of_width(a: int, b: Region) -> bool:
        return bool(np.linalg.norm(b.width) < a)

    neurolang.add_tuple_set(range(10), int)

    neurolang.add_symbol(norm_of_width, "norm_of_width_gt")

    pred = neurolang.symbols.superior_of(
        x, neurolang.symbols.reference_region
    ) & neurolang.symbols.norm_of_width_gt(
        z, neurolang.symbols.reference_region
    )

    res = neurolang.query((x, z), pred).do()
    assert res.value != frozenset()


@skip("Can not evaluate expressions other than query")
def test_quantifier_expressions():

    neurolang = RegionFrontendFolThroughDatalog()

    i1 = ExplicitVBR(np.array([[0, 0, 2]]), np.eye(4))
    i2 = ExplicitVBR(np.array([[0, 0, 6]]), np.eye(4))
    i3 = ExplicitVBR(np.array([[0, 0, 10]]), np.eye(4))
    i4 = ExplicitVBR(np.array([[0, 0, 13], [0, 0, 15]]), np.eye(4))
    regions = {i1, i2, i3, i4}
    neurolang.add_tuple_set(regions, ExplicitVBR)

    central = ExplicitVBR(np.array([[0, 0, 15], [1, 1, 20]]), np.eye(4))
    neurolang.add_region(central, name="reference_region")

    x = neurolang.new_region_symbol(name="x")
    res = neurolang.all(
        x,
        ~neurolang.symbols.superior_of(x, neurolang.symbols.reference_region),
    )
    assert res.do().value

    res = neurolang.exists(
        x, neurolang.symbols.overlapping(x, neurolang.symbols.reference_region)
    )
    assert res.do().value


@patch(
    "neurolang.frontend.neurosynth_utils."
    "NeuroSynthHandler.ns_region_set_from_term"
)
def test_neurosynth_region(mock_ns_regions):
    mock_ns_regions.return_value = {
        ExplicitVBR(np.array([[1, 0, 0], [1, 1, 0]]), np.eye(4))
    }
    neurolang = RegionFrontendFolThroughDatalog()
    s = neurolang.load_neurosynth_term_regions(
        "gambling", 10, "gambling_regions"
    )
    res = neurolang[s]
    mock_ns_regions.assert_called()

    assert res.type is AbstractSet[Tuple[ExplicitVBR]]
    assert res.value == frozenset((t,) for t in mock_ns_regions.return_value)


def test_nested_quantifiers_in_query():
    nl = RegionFrontendFolThroughDatalog()

    A = nl.add_tuple_set(range(10), int, name="A")
    B = nl.add_tuple_set(range(20), int, name="B")
    L = nl.add_tuple_set("abcdefg", str, name="L")
    R = nl.add_tuple_set(
        {("a", 13), ("c", 5), ("e", 16), ("g", 2)}, Tuple[str, int], name="R"
    )

    x = nl.new_symbol(name="x", type_=int)
    y = nl.new_symbol(name="y", type_=int)
    z = nl.new_symbol(name="z", type_=str)

    def leq_f(x, y):
        return x < y

    leq = nl.add_symbol(leq_f, "leq")

    q = nl.query(x, nl.all(y, leq(y, x) | ~A(y)) & nl.exists(z, R(z, x)))
    res = q.do()
    assert res.value == frozenset({(13,), (16,)})


def test_isin():
    nl = RegionFrontendFolThroughDatalog()

    A = nl.add_tuple_set(range(10), int, name="A")
    B = nl.add_tuple_set(range(20), int, name="B")

    x = nl.new_symbol(name="x", type_=int)

    q = nl.query(x, nl.symbols.isin(x, A))
    res = q.do()
    assert res.value == frozenset(set((i,) for i in range(10)))


def test_isin_2():
    nl = RegionFrontendFolThroughDatalog()

    nl.add_tuple_set(range(10), int, name="D")

    is_even = nl.add_symbol(lambda x: x % 2 == 0, "is_even")

    x = nl.new_symbol(name="x", type_=int)
    B = nl.query(x, is_even(x)).do(name="B")

    R = nl.query(x, nl.symbols.isin(x, B)).do(name="R")
    expected = frozenset(set((i,) for i in range(10) if i % 2 == 0))
    assert R.value == expected


@skip("fails to import nilearn in the CI")
def test_compare_frontends():
    nl1 = _init(RegionFrontendFolThroughDatalog)
    nl2 = _init(RegionFrontend)

    def get1(result):
        return set(r[0] for r in result)

    def get2(result):
        return set(r.value for r in result)

    r1 = _query_1(nl1).do()
    r2 = _query_1(nl2).do()
    assert get1(r1) == get2(r2)

    r1 = _query_2(nl1).do()
    r2 = _query_2(nl2).do()
    assert get1(r1) == get2(r2)

    r1 = _query_3(nl1).do()
    r2 = _query_3(nl2).do()
    assert get1(r1) == get2(r2)

    r1 = _query_4(nl1, r1).do()
    r2 = _query_4(nl2, r2).do()
    assert get1(r1) == get2(r2)


def _init(frontend_class):
    from nilearn import datasets
    import nibabel as nib

    destrieux_dataset = datasets.fetch_atlas_destrieux_2009()
    destrieux_map = nib.load(destrieux_dataset["maps"])

    nl = frontend_class()
    for label_number, name in destrieux_dataset["labels"]:
        name = name.decode()
        if not name.startswith("L ") or not (
            "S_" in name or "Lat_Fis" in name or "Pole" in name
        ):
            continue

        # Create a region object
        region = nl.create_region(destrieux_map, label=label_number)

        # Fine tune the symbol name
        name = "L_" + name[2:].replace("-", "_")
        nl.add_region(region, name=name.lower())

    return nl


def _query_1(nl):
    x = nl.new_region_symbol("x")
    return nl.query(
        x, nl.symbols.anatomical_anterior_of(x, nl.symbols.l_s_central)
    )


def _query_2(nl):
    x = nl.new_region_symbol("x")
    return nl.query(
        x,
        nl.symbols.anatomical_anterior_of(x, nl.symbols.l_s_central)
        & nl.symbols.anatomical_superior_of(x, nl.symbols.l_s_temporal_sup),
    )


def _query_3(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    return nl.query(
        x,
        nl.symbols.anatomical_anterior_of(x, nl.symbols.l_s_central)
        & ~nl.exists(
            y,
            nl.symbols.anatomical_anterior_of(y, nl.symbols.l_s_central)
            & nl.symbols.anatomical_anterior_of(x, y),
        ),
    )


def _query_4(nl, temporal_lobe):
    x = nl.new_region_symbol("x")
    return nl.query(
        x,
        nl.symbols.isin(x, temporal_lobe)
        & ~nl.symbols.anatomical_inferior_of(x, nl.symbols.l_s_temporal_inf),
    )
