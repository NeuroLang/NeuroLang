from ..region_solver import RegionsSetSolver
from ..symbols_and_types import TypedSymbolTable
from .. import neurolang as nl
import typing
from typing import AbstractSet, Callable
from ..regions import Region


def inverse_direction(arg):
    dirs = ['north_of', 'east_of', 'overlapping', 'west_of', 'south_of']
    for i in range(len(arg)):
        arg[i] = dirs[len(dirs) - dirs.index(arg[i]) - 1]
    return arg


# def define_universal_rel_in_solver(solver, type):
#     def symbols_of_type():
#         res = frozenset()
#         for elem in solver.symbol_table.symbols_by_type(type).values():
#             res = res.union(elem.value)
#         return res
#     solver.symbol_table[nl.Symbol[type]('universal')] = nl.Constant[type](symbols_of_type)

def test_simple_relation_north_of():

    solver = RegionsSetSolver(TypedSymbolTable())

    inferior = Region((0, 0), (1, 1))
    central = Region((0, 2), (1, 3))
    superior = Region((0, 4), (1, 5))

    north_relation = 'north_of'

    all_elements = frozenset([inferior, central, superior])
    elem = frozenset([central])

    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('db')] = nl.Constant[typing.AbstractSet[Region]](all_elements)

    region_set_type = typing.AbstractSet[Region]

    def set_region_predicate(name='query_element', type=region_set_type, element=elem):
        solver.symbol_table[nl.Symbol[type](name)] = nl.Constant[type](element)

    def define_predicate(type=region_set_type, operation=north_relation, element='predicate_name'):
        return nl.Predicate[type](
            nl.Symbol[Callable[[type], type]](operation),
            (nl.Symbol[type](element),)
        )

    set_region_predicate(name='predicate_name', type=region_set_type, element=elem)
    pred = define_predicate(type=region_set_type, operation=north_relation, element='predicate_name')

    def solve_query(type=region_set_type, solver=solver, predicate=pred, target_name='query_result'):
        query = nl.Query[type](nl.Symbol[type](target_name), predicate)
        solver.walk(query)
        return solver.symbol_table[target_name].value

    expected = frozenset([superior])
    actual = solve_query(region_set_type, solver, pred)
    assert expected == actual

def test_north_U_south():
    # solver = RegionsSetSolver()
    # solver.set_symbol_table(TypedSymbolTable())

    solver = RegionsSetSolver(TypedSymbolTable())

    db_elems = frozenset([
        Region((0, 5), (1, 6)), Region((0, -10), (1, -8))
    ])
    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('db')] = nl.Constant[typing.AbstractSet[Region]](db_elems)


    elem = frozenset([
        Region((0, 0), (1, 1))
    ])
    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('e1')] = nl.Constant[typing.AbstractSet[Region]](elem)

    check_union_commutativity(AbstractSet[Region], solver, 'north_of', 'south_of', 'e1')

def check_union_commutativity(type, solver, relation1, relation2, element):
    p1 = nl.Predicate[type](
        nl.Symbol[Callable[[type], type]](relation1),
        (nl.Symbol[type](element),)
    )

    p2 = nl.Predicate[type](
        nl.Symbol[Callable[[type], type]](relation2),
        (nl.Symbol[type](element),)
    )

    query_a = nl.Query[type](nl.Symbol[type]('a'), p1 | p2)
    query_b = nl.Query[type](nl.Symbol[type]('b'), p2 | p1)

    solver.walk(query_a)
    solver.walk(query_b)

    assert solver.symbol_table['a'] == solver.symbol_table['b']


def test_union_asociativity():
    solver = RegionsSetSolver(TypedSymbolTable())

    db_elems = frozenset([
        Region((0, 5), (1, 6)), Region((0, -10), (1, -8))
    ])
    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('db')] = nl.Constant[AbstractSet[Region]](db_elems)

    elem = frozenset([Region((0, 0), (1, 1))])

    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('e')] = nl.Constant[AbstractSet[Region]](elem)
    check_union_associativity(AbstractSet[Region], solver, 'north_of', 'south_of', 'west_of', 'e')


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
    check_huntington(AbstractSet[Region], solver, 'north_of', 'south_of', 'e')


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

    query_b = nl.Query[type](nl.Symbol[type]('b'), ~(
        (~p1 | ~p2).cast(type)
    ).cast(type) | ~(~p1 | p2).cast(type))
    solver.walk(query_b)


    assert solver.symbol_table['a'] == solver.symbol_table['b']


def test_simple_converse():
    solver = RegionsSetSolver(TypedSymbolTable())

    db_elems = frozenset([
        Region((0, 5), (1, 6)), Region((0, -10), (1, -8))
    ])
    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('db')] = nl.Constant[AbstractSet[Region]](db_elems)

    elem = frozenset([Region((0, 0), (1, 1))])
    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('e')] = nl.Constant[AbstractSet[Region]](elem)

    type = AbstractSet[Region]

    p1 = nl.Predicate[type](
        nl.Symbol[Callable[[type], type]]('north_of'),
        (nl.Symbol[type]('e'),)
    )

    p2 = nl.Predicate[type](
        nl.Symbol[Callable[[type], type]]('inverse north_of'),
        (nl.Symbol[type]('e'),)
    )

    query_a = nl.Query[AbstractSet[Region]](nl.Symbol[AbstractSet[Region]]('a'), p1)
    solver.walk(query_a)
    assert solver.symbol_table['a'].value == frozenset({Region((0, 5), (1, 6))})

    query_b = nl.Query[AbstractSet[Region]](nl.Symbol[AbstractSet[Region]]('b'), p2)
    solver.walk(query_b)
    assert solver.symbol_table['b'].value == frozenset({Region((0, -10), (1, -8))})


def test_converse_involution():
    solver = RegionsSetSolver(TypedSymbolTable())

    db_elems = frozenset([
        Region((0, 5), (1, 6)), Region((0, -10), (1, -8))
    ])
    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('db')] = nl.Constant[AbstractSet[Region]](db_elems)

    elem = frozenset([Region((0, 0), (1, 1))])
    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('e')] = nl.Constant[AbstractSet[Region]](elem)

    involution = double_converse(AbstractSet[Region], solver, 'north_of', 'e')

    p1 = nl.Predicate[type](
        nl.Symbol[Callable[[type], type]]('north_of'),
        (nl.Symbol[type]('e'),)
    )

    query_a = nl.Query[AbstractSet[Region]](nl.Symbol[AbstractSet[Region]]('query_result'), p1)
    solver.walk(query_a)
    assert involution == solver.symbol_table['query_result'].value


def double_converse(type, solver, function, elem):

    p1 = nl.Predicate[type](
        nl.Symbol[Callable[[type], type]](function),
        (nl.Symbol[type](elem),)
    )

    query_a = nl.Query[type](nl.Symbol[type]('a'), p1)
    solver.walk(query_a)
    result = solver.symbol_table['a'].value
    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('f')] = nl.Constant[AbstractSet[Region]](result)
    p2 = nl.Predicate[type](
        nl.Symbol[Callable[[type], type]]('inverse ' + function),
        (nl.Symbol[type]('f'),)
    )

    query_b = nl.Query[type](nl.Symbol[type]('b'), p2)
    solver.walk(query_b)

    return result


# def test_converse_distributivity():
#
#     solver = RegionsSetSolver(TypedSymbolTable())
#
#     db_elems = frozenset([
#         Region((0, 5), (1, 6)), Region((0, -10), (1, -8))
#     ])
#     solver.symbol_table[nl.Symbol[AbstractSet[Region]]('db')] = nl.Constant[AbstractSet[Region]](db_elems)
#
#     elem = frozenset([Region((0, 0), (1, 1))])
#     solver.symbol_table[nl.Symbol[AbstractSet[Region]]('e')] = nl.Constant[AbstractSet[Region]](elem)
#
#     elem = frozenset([
#         Region((0, 0), (1, 1))
#     ])
#     solver.symbol_table[nl.Symbol[AbstractSet[Region]]('e1')] = nl.Constant[typing.AbstractSet[Region]](elem)
#
#     north_u_south = get_converse_of_union_elements(AbstractSet[Region], solver, 'north_of', 'south_of', 'e1')
#
#
#
# def get_converse_of_union_elements(type, solver, relation1, relation2, elem):
#
#     [relation1, relation2] = inverse_direction([relation1, relation2])
#     p1 = nl.Predicate[type](
#         nl.Symbol[Callable[[type], type]](relation1),
#         (nl.Symbol[type](elem),)
#     )
#
#     p2 = nl.Predicate[type](
#         nl.Symbol[Callable[[type], type]](relation2),
#         (nl.Symbol[type](elem),)
#     )
#
#     query_a = nl.Query[type](nl.Symbol[type]('a'), p1 | p2)
#     solver.walk(query_a)
#
#     return solver.symbol_table['a']
#

def test_universal_relation():

    solver = RegionsSetSolver(TypedSymbolTable())
    db_elems = frozenset([
        Region((0, 5), (1, 6)), Region((0, -10), (1, -8))
    ])
    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('db')] = nl.Constant[AbstractSet[Region]](db_elems)

    elem = frozenset([Region((0, 0), (1, 1))])
    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('e')] = nl.Constant[AbstractSet[Region]](elem)

    type = AbstractSet[Region]
    p1 = nl.Predicate[type](
        nl.Symbol[Callable[[type], type]]('universal'),
        (nl.Symbol[type]('e'),)
    )

    query_a = nl.Query[AbstractSet[Region]](nl.Symbol[AbstractSet[Region]]('a'), p1)
    solver.walk(query_a)
    print(solver.symbol_table['a'].value)



def test_basic_composition_relation():

    solver = RegionsSetSolver(TypedSymbolTable())

    db_elems = frozenset([
        Region((0, 5), (1, 6)), Region((0, -10), (1, -8))
    ])
    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('db')] = nl.Constant[typing.AbstractSet[Region]](db_elems)

    elem = frozenset([
        Region((0, 0), (1, 1))
    ])

    print(compose_relations(AbstractSet[Region], solver, 'north_of', 'south_of', elem))
    #assert compose_relations(AbstractSet[Region], solver, 'north_of', 'south_of', elem) == frozenset([Region((0, 0), (1, 1)), Region((0, -10), (1, -8)])


def compose_relations(type, solver, relation1, relation2, element):

    intermediate_results = frozenset()
    for elem in element:
        solver.symbol_table[nl.Symbol[type]('intermediate')] = nl.Constant[type](frozenset([elem]))
        p1 = nl.Predicate[type](
            nl.Symbol[Callable[[type], type]](relation1),
            (nl.Symbol[type]('intermediate'),)
        )

        query_a = nl.Query[type](nl.Symbol[type]('first_query'), p1)
        solver.walk(query_a)
        intermediate_results = intermediate_results.union(solver.symbol_table['first_query'].value)
    res = frozenset()
    for elem in intermediate_results:
        solver.symbol_table[nl.Symbol[AbstractSet[Region]]('intermediate')] = nl.Constant[typing.AbstractSet[Region]](frozenset([elem]))
        p2 = nl.Predicate[type](
            nl.Symbol[Callable[[type], type]](relation2),
            (nl.Symbol[type]('intermediate'),)
        )

        query_b = nl.Query[type](nl.Symbol[type]('second_query'), p2)

        solver.walk(query_b)
        res = res.union(solver.symbol_table['second_query'].value)
    return res

def test_compose_associativity():

    solver = RegionsSetSolver(TypedSymbolTable())

    inferior = Region((0, 0), (1, 1))
    central = Region((0, 2), (1, 3))
    superior = Region((0, 4), (1, 5))

    db_elems = frozenset([inferior, central, superior])

    north_relation, south_relation = 'north_of', 'south_of'
    type = AbstractSet[Region]

    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('db')] = nl.Constant[AbstractSet[Region]](db_elems)

    elem = frozenset([Region((0, 0), (1, 1))])
    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('e')] = nl.Constant[AbstractSet[Region]](elem)

    #first part
    intermediate_results = compose_relations(AbstractSet[Region], solver, north_relation, south_relation, elem)

    solver.symbol_table[nl.Symbol[type]('foo1')] = nl.Constant[type](intermediate_results)
    p2222 = nl.Predicate[type](
        nl.Symbol[Callable[[type], type]](north_relation),
        (nl.Symbol[type]('foo1'),)
    )

    query_b = nl.Query[type](nl.Symbol[type]('nuevo'), p2222)
    solver.walk(query_b)

    #TODO revisar este caso donde hay que iterar sobre todos los elementos para replicar el ciclo que se hace en la composicion
    first_result = frozenset()
    for elem in intermediate_results:
        solver.symbol_table[nl.Symbol[AbstractSet[Region]]('intermediate')] = nl.Constant[typing.AbstractSet[Region]](
            frozenset([elem]))
        p2 = nl.Predicate[type](
            nl.Symbol[Callable[[type], type]](north_relation),
            (nl.Symbol[type]('intermediate'),)
        )

        query_b = nl.Query[type](nl.Symbol[type]('b'), p2)

        solver.walk(query_b)
        first_result = first_result.union(solver.symbol_table['b'].value)


    #second part
    p1 = nl.Predicate[type](
        nl.Symbol[Callable[[type], type]](north_relation),
        (nl.Symbol[type]('e'),)
    )
    query_a = nl.Query[type](nl.Symbol[type]('a'), p1)
    solver.walk(query_a)
    intermediate = solver.symbol_table['a'].value
    second_result = compose_relations(AbstractSet[Region], solver, south_relation, north_relation, intermediate)

    assert first_result == second_result


def test_compose_distributibity():

    solver = RegionsSetSolver(TypedSymbolTable())

    db_elems = frozenset([
        Region((0, 5), (1, 6)), Region((0, 8), (1, 10)), Region((0, 20), (1, 21))
    ])
    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('db')] = nl.Constant[typing.AbstractSet[Region]](db_elems)

    elem = frozenset([
        Region((0, 0), (1, 1))
    ])
    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('element')] = nl.Constant[typing.AbstractSet[Region]](elem)

    rel1, rel2, rel3 = 'north_of', 'south_of', 'north_of'

    type = AbstractSet[Region]
    p1 = nl.Predicate[type](
        nl.Symbol[Callable[[type], type]](rel1),
        (nl.Symbol[type]('element'),)
    )

    p2 = nl.Predicate[type](
        nl.Symbol[Callable[[type], type]](rel2),
        (nl.Symbol[type]('element'),)
    )

    query_a = nl.Query[type](nl.Symbol[type]('a'), p1 | p2)
    solver.walk(query_a)

    intermediate_results = solver.symbol_table['a'].value
    first_result = frozenset()
    for elements in intermediate_results:
        solver.symbol_table[nl.Symbol[AbstractSet[Region]]('intermediate')] = nl.Constant[typing.AbstractSet[Region]](
            frozenset([elements]))
        p2 = nl.Predicate[type](
            nl.Symbol[Callable[[type], type]](rel3),
            (nl.Symbol[type]('intermediate'),)
        )

        query_b = nl.Query[type](nl.Symbol[type]('b'), p2)

        solver.walk(query_b)
        first_result = first_result.union(solver.symbol_table['b'].value)

    #print(first_result)


    #second part:
    u_one = compose_relations(AbstractSet[Region], solver, rel1, rel2, elem)
    u_two = compose_relations(AbstractSet[Region], solver, rel1, rel3, elem)

    second_results = u_one.union(u_two)
    print(second_results)
