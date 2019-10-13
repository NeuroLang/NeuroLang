import operator as op
from itertools import product
from typing import Callable

from pytest import fixture

from ... import expression_walker as ew
from ... import expressions
from ..basic_representation import DatalogProgram
from ..chase import (ChaseGeneral, ChaseMGUMixin, ChaseNaive,
                     ChaseNamedRelationalAlgebraMixin, ChaseNode,
                     ChaseRelationalAlgebraPlusCeriMixin, ChaseSemiNaive)
from ..expressions import Disjunction, Fact, Implication, TranslateToLogic
from ..instance import MapInstance

C_ = expressions.Constant
S_ = expressions.Symbol
Imp_ = Implication
F_ = Fact
Eb_ = expressions.ExpressionBlock
Disj_ = Disjunction


class DatalogTranslator(TranslateToLogic, ew.IdentityWalker):
    pass


DT = DatalogTranslator()


Q = S_('Q')
T = S_('T')
S = S_('S')
w = S_('w')
x = S_('x')
y = S_('y')
z = S_('z')
a = C_('a')
b = C_('b')
c = C_('c')
eq = C_[Callable[[expressions.Unknown, expressions.Unknown], bool]](op.eq)
gt = C_[Callable[[expressions.Unknown, expressions.Unknown], bool]](op.gt)


class Datalog(TranslateToLogic, DatalogProgram, ew.ExpressionBasicEvaluator):
    def function_gt(self, x: int, y: int) -> bool:
        return x > y


@fixture(params=[
    (step_class, cq_class)
    for step_class, cq_class in product(
        (
            ChaseNaive,
            #ChaseSemiNaive
        ),
        (
            ChaseMGUMixin,
            #ChaseNamedRelationalAlgebraMixin,
            #ChaseRelationalAlgebraPlusCeriMixin,
        )
    )
])
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
    instance_update = MapInstance(dc.chase_step(instance_0, rule))
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

    instance_1 = dc.merge_instances(instance_0, instance_update)
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
        F_(Q(C_(1), C_(2))), F_(Q(C_(2), C_(3))), F_(Q(C_(8), C_(6))),
        Imp_(T(x, y),
             Q(x, z) & Q(z, y)), Imp_(S(x, y),
                                      Q(x, y) & gt(x, y))
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

    instance_1 = dc.merge_instances(instance_0, instance_update)
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

    assert res.instance == dl.extensional_database()
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

    assert res.instance == dl.extensional_database()
    assert len(res.children) == 1
    first_child = res.children[datalog_program.formulas[-2]]
    assert first_child.instance == instance_1
    assert len(first_child.children) == 1
    second_child = first_child.children[datalog_program.formulas[-1]]

    instance_2 = {
        Q: C_({
            C_((C_(1), C_(2))),
            C_((C_(2), C_(3))),
        }),
        T: C_({C_((C_(1), C_(2))),
               C_((C_(2), C_(3))),
               C_((C_(1), C_(3)))})
    }

    assert len(second_child.children) == 0
    assert second_child.instance == instance_2


def test_nonrecursive_predicate_chase_solution(chase_class, N=10):
    datalog_program = DT.walk(Eb_(
        tuple(F_(Q(C_(i), C_(i + 1)))
              for i in range(N)) + (Imp_(T(x, y),
                                         Q(x, z) & Q(z, y)), )
    ))

    dl = Datalog()
    dl.walk(datalog_program)

    dc = chase_class(dl)
    solution_instance = dc.build_chase_solution()

    final_instance = {
        Q: C_({C_((C_(i), C_(i + 1)))
               for i in range(N)}),
        T: C_({C_((C_(i), C_(i + 2)))
               for i in range(N - 1)})
    }

    assert solution_instance == final_instance


def test_nonrecursive_predicate_chase_solution_constant(chase_class, N=10):
    datalog_program = Eb_(
        tuple(F_(Q(C_(i), C_(i + 1)))
              for i in range(N)) + (Imp_(T(y),
                                         Q(C_(1), z) & Q(z, y)), )
    )

    dl = Datalog()
    dl.walk(datalog_program)

    dc = chase_class(dl)
    solution_instance = dc.build_chase_solution()

    final_instance = {
        Q: C_({C_((C_(i), C_(i + 1)))
               for i in range(N)}),
        T: C_({C_((C_(i + 2), ))
               for i in (1, )})
    }

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
    solution_instance = dc.build_chase_solution()

    final_instance = {
        Q: C_({
            C_((C_(1), C_(2))),
            C_((C_(2), C_(3))),
        }),
        T: C_({C_((C_(1), C_(2))),
               C_((C_(2), C_(3))),
               C_((C_(1), C_(3)))})
    }

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

    solution = chase_class(dl).build_chase_solution()
    assert solution['q'].value == {C_((e, )) for e in (b, c, d)}
    