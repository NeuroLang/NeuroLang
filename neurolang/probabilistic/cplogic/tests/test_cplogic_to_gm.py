from ....datalog import Fact
from ....expressions import Constant, ExpressionBlock, Symbol
from ....logic import Implication
from .. import testing
from ..cplogic_to_gm import AndCPDFactory, BernoulliCPDFactory
from ..program import CPLogicProgram

P = Symbol("P")
Q = Symbol("Q")
x = Symbol("x")
a = Constant("a")
b = Constant("b")


def test_empty_program():
    code = ExpressionBlock(tuple())
    cpl_program = CPLogicProgram()
    cpl_program.walk(code)
    gm = testing.build_gm(cpl_program)
    assert len(gm.edges.value) == 0
    assert len(gm.cpd_factories.value) == 0


def test_simple_deterministic_program():
    code = ExpressionBlock((Fact(Q(a)), Fact(Q(b)), Implication(P(x), Q(x))))
    cpl_program = CPLogicProgram()
    cpl_program.walk(code)
    gm = testing.build_gm(cpl_program)
    assert set(gm.cpd_factories.value) == {Q, P}
    assert isinstance(gm.cpd_factories.value[Q], BernoulliCPDFactory)
    assert isinstance(gm.cpd_factories.value[P], AndCPDFactory)


def test_program_with_probfacts():
    code = ExpressionBlock((Implication(Q(x), P(x)),))
    probfacts_sets = {P: {(1.0, "a"), (0.5, "b"), (0.3, "c")}}
    cpl_program = CPLogicProgram()
    cpl_program.walk(code)
    for pred_symb, pfact_set in probfacts_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(pred_symb, pfact_set)
    gm = testing.build_gm(cpl_program)
    assert set(gm.edges.value) == {Q}
    assert gm.edges.value[Q] == {P}
    assert isinstance(gm.cpd_factories.value[Q], AndCPDFactory)
    assert isinstance(gm.cpd_factories.value[P], BernoulliCPDFactory)
    assert (
        testing.get_named_relation_tuples(gm.cpd_factories.value[P].relation)
        == probfacts_sets[P]
    )
