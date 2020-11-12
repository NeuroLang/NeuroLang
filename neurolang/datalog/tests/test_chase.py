import operator as op
from itertools import product
from typing import AbstractSet, Callable, Tuple

from pytest import fixture, raises, skip

from ... import expression_walker as ew
from ... import expressions
from ..basic_representation import DatalogProgram
from ..chase import (ChaseGeneral, ChaseMGUMixin, ChaseNaive,
                     ChaseNamedRelationalAlgebraMixin, ChaseNode,
                     ChaseNonRecursive, ChaseRelationalAlgebraPlusCeriMixin,
                     ChaseSemiNaive, NeuroLangNonLinearProgramException,
                     NeuroLangProgramHasLoopsException)
from ..expressions import (Conjunction, Fact, Implication, TranslateToLogic,
                           Union)
from ..instance import MapInstance
from ..expression_processing import EQ

try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import suppress as nullcontext



C_ = expressions.Constant
S_ = expressions.Symbol
Imp_ = Implication
F_ = Fact
Eb_ = expressions.ExpressionBlock
Disj_ = Union


class DatalogTranslator(TranslateToLogic, ew.IdentityWalker):
    pass


DT = DatalogTranslator()


Q = S_('Q')
T = S_('T')
S = S_('S')
v = S_('v')
w = S_('w')
x = S_('x')
y = S_('y')
z = S_('z')
a = C_('a')
b = C_('b')
c = C_('c')
eq = C_[Callable[[expressions.Unknown, expressions.Unknown], bool]](op.eq)
gt = C_[Callable[[expressions.Unknown, expressions.Unknown], bool]](op.gt)
contains = C_[Callable[[expressions.Unknown, expressions.Unknown], bool]](
    op.contains
)


class Datalog(TranslateToLogic, DatalogProgram, ew.ExpressionBasicEvaluator):
    def function_gt(self, x: int, y: int) -> bool:
        return x > y


chase_configurations = [
    (step_class, cq_class)
    for step_class, cq_class in product(
        (
            ChaseNonRecursive,
            ChaseNaive,
            ChaseSemiNaive
        ),
        (
            ChaseMGUMixin,
            ChaseNamedRelationalAlgebraMixin,
            ChaseRelationalAlgebraPlusCeriMixin,
        )
    )
]


@fixture(
    params=chase_configurations,
    ids=[
        f'{strategy}-{step}'
        for strategy, step in chase_configurations
    ]
)
def chase_class(request):
    class C(request.param[0], request.param[1], ChaseGeneral):
        pass

    return C


def test_no_free_variable_case(chase_class):
    datalog_program = Eb_((
        F_(Q(a)),
        Imp_(T(b), Q(a)),
    ))

    dl = Datalog()
    dl.walk(datalog_program)

    instance_0 = MapInstance(dl.extensional_database())
    rule = datalog_program.expressions[-1]
    dc = chase_class(dl)
    instance_update = dc.chase_step(instance_0, rule)
    res = MapInstance({
        T: C_({C_(('b', ))}),
    })

    assert res == instance_update


def test_no_head_argument_case(chase_class):
    datalog_program = Eb_((
        F_(Q(a)),
        Imp_(T(), Q(x)),
    ))

    dl = Datalog()
    dl.walk(datalog_program)

    instance_0 = MapInstance(dl.extensional_database())
    rule = datalog_program.expressions[-1]
    dc = chase_class(dl)
    instance_update = dc.chase_step(instance_0, rule)
    res = MapInstance({
        T: C_({C_(tuple())}),
    })

    assert res == instance_update


def test_symmetric_elements(chase_class):
    datalog_program = Eb_((
        F_(Q(a, a)),
        F_(Q(b, c)),
        Imp_(T(x), Q(x, x)),
    ))

    dl = Datalog()
    dl.walk(datalog_program)

    instance_0 = MapInstance(dl.extensional_database())
    rule = datalog_program.expressions[-1]
    dc = chase_class(dl)
    instance_update = dc.chase_step(instance_0, rule)
    res = MapInstance({
        T: C_({C_((a,))}),
    })

    assert res == instance_update


def test_builtin_equality_only(chase_class):
    datalog_program = Eb_((Imp_(Q(x), eq(x, C_(5))), ))

    dl = Datalog()
    dl.walk(datalog_program)

    instance_0 = MapInstance(dl.extensional_database())

    rule = datalog_program.expressions[0]
    dc = chase_class(dl)
    instance_update = dc.chase_step(instance_0, rule)
    res = MapInstance({
        Q: C_({C_((5, ))}),
    })
    assert instance_update == res

    datalog_program = Eb_((Imp_(Q(x), eq(x, C_(5) + C_(7))), ))

    dl = Datalog()
    dl.walk(datalog_program)

    instance_0 = MapInstance(dl.extensional_database())

    rule = datalog_program.expressions[0]
    dc = chase_class(dl)
    instance_update = dc.chase_step(instance_0, rule)
    res = MapInstance({
        Q: C_({C_((12, ))}),
    })
    assert instance_update == res


def test_builtin_equality_only_sets(chase_class):
    const = C_(frozenset({5, 6}))
    datalog_program = Eb_((Imp_(Q(x), eq(x, const)),))

    dl = Datalog()
    dl.walk(datalog_program)

    instance_0 = MapInstance(dl.extensional_database())

    rule = datalog_program.expressions[0]
    dc = chase_class(dl)
    instance_update = dc.chase_step(instance_0, rule)
    res = MapInstance({
        Q: C_({(const.value,)}),
    })
    assert instance_update == res


def test_builtin_equality_only_sets_and_computations(chase_class):
    const = C_(frozenset({5, 6}))
    len_ = C_(len)
    datalog_program = Eb_((
        Fact(Q(const)),
        Imp_(T(x), eq(x, len_(y)) & Q(y))
    ))

    dl = Datalog()
    dl.walk(datalog_program)

    instance_0 = MapInstance(dl.extensional_database())
    assert instance_0[Q].type == AbstractSet[Tuple[AbstractSet[int]]]

    rule = dl.symbol_table[T].formulas[0]
    dc = chase_class(dl)
    instance_update = dc.chase_step(instance_0, rule)

    res = MapInstance({
        T: C_({(2,)}),
    })
    assert instance_update == res

    const1 = C_(frozenset({6, 8}))
    datalog_program = Eb_((
        Fact(Q(const)),
        Fact(Q(const1)),
        Imp_(T(y), contains(y, C_(5)) & Q(y))
    ))

    dl = Datalog()
    dl.walk(datalog_program)

    instance_0 = MapInstance(dl.extensional_database())

    rule = dl.symbol_table[T].formulas[0]
    dc = chase_class(dl)
    instance_update = dc.chase_step(instance_0, rule)

    res = MapInstance({
        T: C_({(frozenset({5, 6}),)}),
    })
    assert instance_update == res
    assert instance_update[T].type == AbstractSet[Tuple[AbstractSet[int]]]

    datalog_program = Eb_((
        Fact(Q(const)),
        Fact(Q(const1)),
        Fact(Q(C_(frozenset({0, 0})))),
        Imp_(T(x, y), Q(y) & contains(y, x) & eq(x, C_(6))),
        Imp_(S(x, y), Q(y) & eq(x, C_(6)) & contains(y, x)),
        Imp_(S(x, y), Q(y) & eq(x, C_(5)) & contains(y, x)),
    ))

    dl = Datalog()
    dl.walk(datalog_program)

    dc = chase_class(dl)
    instance_result = dc.build_chase_solution()

    res = MapInstance({
        T: C_({
            (C_(6), frozenset({5, 6}),),
            (C_(6), frozenset({6, 8}),)
        }),
        S: C_({
            (C_(5), frozenset({5, 6}),),
            (C_(6), frozenset({5, 6}),),
            (C_(6), frozenset({6, 8}),)
        }),
    })
    assert instance_result[T] == res[T]
    assert instance_result[S] == res[S]
    assert instance_result[T].type == AbstractSet[Tuple[int, AbstractSet[int]]]
    assert instance_result[S].type == AbstractSet[Tuple[int, AbstractSet[int]]]


def test_builtin_equality_simple_chase_solution(chase_class):
    datalog_program = Eb_((
        F_(Q(C_(5), C_(6))),
        F_(Q(C_(7), C_(8))),
        F_(S(C_(5))),
        Imp_(T(x), eq(C_(6), y) & Q(x, y)),
    ))

    dl = Datalog()
    dl.walk(datalog_program)

    dc = chase_class(dl)
    res = dc.build_chase_solution()
    assert res['T'] == res['S']


def test_builtin_equality_chase_solution(chase_class):
    datalog_program = Eb_((
        F_(Q(C_(5), C_(6))),
        F_(Q(C_(7), C_(8))),
        Imp_(T(x, z), eq(z, y) & Q(x, y)),
    ))

    dl = Datalog()
    dl.walk(datalog_program)

    dc = chase_class(dl)
    res = dc.build_chase_solution()
    assert res['Q'] == res['T']

    datalog_program = Eb_((
        F_(Q(C_(5), C_(6))),
        F_(Q(C_(7), C_(8))),
        Imp_(T(x, z), eq(z, y * C_(1)) & Q(x, y)),
    ))

    dl = Datalog()
    dl.walk(datalog_program)

    dc = chase_class(dl)
    res = dc.build_chase_solution()
    assert res['Q'] == res['T']


def test_chase_set_destroy(chase_class):
    consts = [
        C_(frozenset({5, 6})),
        C_(frozenset({5, 8})),
        C_(frozenset({15, 8})),
    ]

    datalog_program = Eb_(
        tuple(Fact(Q(c)) for c in consts) +
        (
            Imp_(T(x), contains(y, x) & Q(y)),
        )
    )

    dl = Datalog()
    dl.walk(datalog_program)

    instance_0 = MapInstance(dl.extensional_database())

    rule = dl.symbol_table['T'].formulas[0]
    dc = chase_class(dl)
    instance_update = dc.chase_step(instance_0, rule)

    res = MapInstance({
        T: C_({(5,), (6,), (8,), (15,)}),
    })
    assert instance_update == res


def test_chase_set_destroy_tuples(chase_class):
    if not issubclass(chase_class, ChaseNamedRelationalAlgebraMixin):
        skip(
            msg="Multiple column destroy only implemented for the RA chase"
        )

    consts = [
        C_(frozenset({(5, 6), (15, 8)})),
        C_(frozenset({(5, 8)})),
        C_(frozenset({(15, 8)})),
    ]

    datalog_program = Eb_(
        tuple(Fact(Q(c)) for c in consts) +
        (
            Imp_(T(x, y), contains(z, C_(((x, y)))) & Q(z)),
        )
    )

    dl = Datalog()
    dl.walk(datalog_program)

    instance_0 = MapInstance(dl.extensional_database())

    rule = dl.symbol_table['T'].formulas[0]
    dc = chase_class(dl)
    instance_update = dc.chase_step(instance_0, rule)

    res = MapInstance({
        T: C_({(5, 6), (5, 8), (15, 8)}),
    })
    assert instance_update == res


def test_builtin_equality_add_column(chase_class):
    datalog_program = Eb_((
        Imp_(Q(y), Conjunction((T(x), eq(y, C_(2) * x)))),
    ))

    dl = Datalog()
    dl.add_extensional_predicate_from_tuples(T, ((i,) for i in range(8)))
    dl.walk(datalog_program)

    dc = chase_class(dl)
    instance_out = dc.build_chase_solution()
    res = MapInstance({
        T: dl.extensional_database()[T],
        Q: C_(
            {C_((2 * i,)) for i in range(8)}
        ),
    })
    assert instance_out == res


def test_python_builtin_equaltiy_chase_step(chase_class):
    datalog_program = DT.walk(Eb_((
        F_(Q(C_(1), C_(2))),
        F_(Q(C_(2), C_(3))),
        Imp_(S(y),
             Q(x, z) & eq(z + C_(1), y)),
        Imp_(S(y),
             Q(x, z) & eq(y, z + C_(1))),
    )))

    dl = Datalog()
    dl.walk(datalog_program)

    instance_0 = MapInstance(dl.extensional_database())

    rule = DT.walk(datalog_program.formulas[-2])
    dc = chase_class(dl)
    instance_update = dc.chase_step(instance_0, rule)
    res = MapInstance({
        S: C_({C_((3, )), C_((4, ))}),
    })
    assert instance_update == res

    rule = datalog_program.formulas[-1]
    instance_update = dc.chase_step(instance_0, rule)
    assert instance_update == res


def test_python_builtin_chase_step(chase_class):
    datalog_program = DT.walk(Eb_((
        F_(Q(C_(1), C_(2))), F_(Q(C_(2), C_(3))), F_(Q(C_(8), C_(6))),
        Imp_(T(x, y), Q(x, z) & Q(z, y)),
        Imp_(S(x, y), Q(x, y) & gt(x, y))
    )))

    dl = Datalog()
    dl.walk(datalog_program)

    instance_0 = MapInstance(dl.extensional_database())

    rule = datalog_program.formulas[-1]
    dc = chase_class(dl)
    instance_update = dc.chase_step(instance_0, rule)
    assert instance_update == MapInstance({
        S: C_({C_((C_(8), C_(6)))}),
    })

    rule = datalog_program.formulas[-2]
    instance_update = dc.chase_step(instance_0, rule)
    assert instance_update == MapInstance({
        T: C_({C_((C_(1), C_(3)))}),
    })

    instance_1 = instance_0 | instance_update
    instance_update = dc.chase_step(instance_1, rule)
    assert len(instance_update) == 0


def test_python_nested_builtin_chase_step(chase_class):
    datalog_program = DT.walk(Eb_((
        F_(Q(C_(8), C_(15))),
        F_(Q(C_(8), C_(9))),
        Imp_(S(x, y),
             Q(x, y) & gt(x, y - C_(2))),
    )))

    dl = Datalog()
    dl.walk(datalog_program)

    instance_0 = MapInstance(dl.extensional_database())

    rule = datalog_program.formulas[-1]
    dc = chase_class(dl)
    instance_update = dc.chase_step(instance_0, rule)
    assert instance_update == MapInstance({
        S: C_({C_((C_(8), C_(9)))}),
    })


def test_non_recursive_predicate_chase_step(chase_class):
    gt = S_('gt')

    datalog_program = DT.walk(Eb_((
        F_(Q(C_(1), C_(2))),
        F_(Q(C_(2), C_(3))),
        F_(Q(C_(8), C_(6))),
        Imp_(T(x, y),
             Q(x, z) & Q(z, y)),
        Imp_(S(x, y),
             Q(x, y) & gt(x, y))
    )))

    dl = Datalog()
    dl.walk(datalog_program)

    instance_0 = MapInstance(dl.extensional_database())

    rule = datalog_program.formulas[-1]
    dc = chase_class(dl)
    instance_update = dc.chase_step(instance_0, rule)
    res = MapInstance({
        S: C_({C_((C_(8), C_(6)))}),
    })
    assert instance_update == res

    rule = datalog_program.formulas[-2]
    instance_update = dc.chase_step(instance_0, rule)
    assert instance_update == MapInstance({
        T: C_({C_((C_(1), C_(3)))}),
    })

    instance_1 = instance_0 | instance_update
    instance_update = dc.chase_step(instance_1, rule)
    assert len(instance_update) == 0


def test_python_multiple_builtins(chase_class):
    datalog_program = DT.walk(Eb_((
        F_(Q(C_(1), C_(2))),
        F_(Q(C_(2), C_(3))),
        Imp_(S(w),
             Q(x, z) & eq(z + C_(1), y) & eq(y, w)),
    )))

    dl = Datalog()
    dl.walk(datalog_program)

    instance_0 = MapInstance(dl.extensional_database())

    rule = datalog_program.formulas[-1]
    dc = chase_class(dl)
    instance_update = dc.chase_step(instance_0, rule)
    res = MapInstance({
        S: C_({C_((3, )), C_((4, ))}),
    })
    assert instance_update == res

    datalog_program = DT.walk(Eb_((
        F_(Q(C_(1), C_(2))),
        F_(Q(C_(2), C_(3))),
        Imp_(S(w),
             Q(x, z) & eq(y, w) & eq(z + C_(1), y)),
    )))

    dl = Datalog()
    dl.walk(datalog_program)

    instance_0 = MapInstance(dl.extensional_database())

    rule = datalog_program.formulas[-1]
    dc = chase_class(dl)
    instance_update = dc.chase_step(instance_0, rule)
    res = MapInstance({
        S: C_({C_((3, )), C_((4, ))}),
    })
    assert instance_update == res


def test_non_recursive_predicate_chase_tree(chase_class):
    datalog_program = DT.walk(Eb_((
        F_(Q(C_(1), C_(2))), F_(Q(C_(2),
                                  C_(3))), Imp_(T(x, y),
                                                Q(x, z) & Q(z, y))
    )))

    dl = Datalog()
    dl.walk(datalog_program)

    dc = chase_class(dl)
    res = dc.build_chase_tree()

    instance_update = MapInstance({T: C_({C_((C_(1), C_(3)))})})

    instance_1 = instance_update.copy()
    instance_1.update(dl.extensional_database())

    assert res.instance == MapInstance(dl.extensional_database())
    assert res.children == {
        datalog_program.formulas[-1]: ChaseNode(instance_1, dict())
    }


def test_recursive_predicate_chase_tree(chase_class):
    datalog_program = DT.walk(Eb_((
        F_(Q(C_(1), C_(2))), F_(Q(C_(2), C_(3))), Imp_(T(x, y), Q(x, y)),
        Imp_(T(x, y),
             Q(x, z) & T(z, y))
    )))

    dl = Datalog()
    dl.walk(datalog_program)

    dc = chase_class(dl)
    res = dc.build_chase_tree()

    instance_update = MapInstance({T: dl.extensional_database()[Q]})

    instance_1 = instance_update.copy()
    instance_1.update(dl.extensional_database())

    assert res.instance == MapInstance(dl.extensional_database())
    assert len(res.children) == 1
    first_child = res.children[datalog_program.formulas[-2]]
    assert first_child.instance == instance_1
    assert len(first_child.children) == 1
    second_child = first_child.children[datalog_program.formulas[-1]]

    instance_2 = MapInstance({
        Q: C_({
            C_((C_(1), C_(2))),
            C_((C_(2), C_(3))),
        }),
        T: C_({C_((C_(1), C_(2))),
               C_((C_(2), C_(3))),
               C_((C_(1), C_(3)))})
    })

    assert len(second_child.children) == 0
    assert second_child.instance == instance_2


def test_nonrecursive_predicate_chase_solution(chase_class, n=10):
    datalog_program = DT.walk(Eb_(
        tuple(F_(Q(C_(i), C_(i + 1)))
              for i in range(n)) + (Imp_(T(x, y),
                                         Q(x, z) & Q(z, y)), )
    ))

    dl = Datalog()
    dl.walk(datalog_program)

    dc = chase_class(dl)
    solution_instance = dc.build_chase_solution()

    final_instance = MapInstance({
        Q: C_({C_((C_(i), C_(i + 1)))
               for i in range(n)}),
        T: C_({C_((C_(i), C_(i + 2)))
               for i in range(n - 1)})
    })

    assert solution_instance == final_instance


def test_nonrecursive_predicate_chase_solution_constant(chase_class, n=10):
    datalog_program = Eb_(
        tuple(F_(Q(C_(i), C_(i + 1)))
              for i in range(n)) + (Imp_(T(y),
                                         Q(C_(1), z) & Q(z, y)), )
    )

    dl = Datalog()
    dl.walk(datalog_program)

    dc = chase_class(dl)
    solution_instance = dc.build_chase_solution()

    final_instance = MapInstance({
        Q: C_({C_((C_(i), C_(i + 1)))
               for i in range(n)}),
        T: C_({C_((C_(i + 2), ))
               for i in (1, )})
    })

    assert solution_instance == final_instance


def test_recursive_predicate_chase_solution(chase_class):
    datalog_program = Eb_((
        F_(Q(C_(1), C_(2))),
        F_(Q(C_(2), C_(3))),
        Imp_(T(x, y), Q(x, y)),
        Imp_(T(x, y),
             Q(x, z) & T(z, y))
    ))

    dl = Datalog()
    dl.walk(datalog_program)

    dc = chase_class(dl)

    if issubclass(chase_class, ChaseNonRecursive):
        context = raises(NeuroLangProgramHasLoopsException)
    else:
        context = nullcontext()

    with context:
        solution_instance = dc.build_chase_solution()

        final_instance = MapInstance({
            Q: C_({
                C_((C_(1), C_(2))),
                C_((C_(2), C_(3))),
            }),
            T: C_({C_((C_(1), C_(2))),
                C_((C_(2), C_(3))),
                C_((C_(1), C_(3)))})
        })

        assert solution_instance == final_instance


def test_another_recursive_chase(chase_class):
    x = S_('X')
    y = S_('Y')
    z = S_('Z')
    anc = S_('anc')
    par = S_('par')
    q = S_('q')
    a = C_('a')
    b = C_('b')
    c = C_('c')
    d = C_('d')

    edb = Eb_([
        F_(par(a, b)),
        F_(par(b, c)),
        F_(par(c, d)),
    ])

    code = Eb_([
        Imp_(q(x), anc(a, x)),
        Imp_(anc(x, y), par(x, y)),
        Imp_(anc(x, y),
             anc(x, z) & par(z, y)),
    ])

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)

    if issubclass(chase_class, ChaseNonRecursive):
        context = raises(NeuroLangProgramHasLoopsException)
    else:
        context = nullcontext()

    with context:
        solution = chase_class(dl).build_chase_solution()
        assert solution['q'].value == {C_((e, )) for e in (b, c, d)}


def test_transitive_closure(chase_class):
    x = S_('X')
    y = S_('Y')
    z = S_('Z')
    edge = S_('edge')
    reaches = S_('reaches')

    dl = Datalog()
    dl.add_extensional_predicate_from_tuples(edge, {(1, 2), (2, 3), (3, 4)})

    code = Eb_([
        Imp_(reaches(x, y), edge(x, y)),
        Imp_(reaches(x, y), reaches(x, z) & reaches(z, y))
    ])

    dl.walk(code)

    if issubclass(chase_class, ChaseNonRecursive):
        context = raises(NeuroLangProgramHasLoopsException)
    elif issubclass(chase_class, ChaseSemiNaive):
        context = raises(NeuroLangNonLinearProgramException)
    else:
        context = nullcontext()

    with context:
        solution = chase_class(dl).build_chase_solution()
        assert solution[reaches].value == {
            C_((i, j))
            for i in range(1, 5)
            for j in range(i, 5)
            if i != j
        }


def test_nested_function_application(chase_class):
    x = S_("x")
    y = S_("y")
    z = S_("z")
    f = S_("f")
    g = S_("g")
    R = S_("R")
    Q = S_("Q")
    dl = Datalog()
    dl.symbol_table[f] = C_(lambda x, y: (x + y) // 2)
    dl.symbol_table[g] = C_(lambda x: x ** 2)
    dl.add_extensional_predicate_from_tuples(R, {(1,), (2,)})
    dl.walk(Implication(Q(z), Conjunction((R(x), R(y), EQ(z, f(g(x), g(y)))))))
    solution = chase_class(dl).build_chase_solution()
    assert solution[Q].value == {
        C_(((1 ** 2 + 1 ** 2) // 2,)),
        C_(((1 ** 2 + 2 ** 2) // 2,)),
        C_(((2 ** 2 + 1 ** 2) // 2,)),
        C_(((2 ** 2 + 2 ** 2) // 2,)),
    }
