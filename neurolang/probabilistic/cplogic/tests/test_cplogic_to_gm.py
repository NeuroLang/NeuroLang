from ....datalog import Fact
from ....expressions import Constant, Symbol
from ....logic import Conjunction, Implication, Union
from .. import testing
from ..cplogic_to_gm import (
    AndPlateNode,
    BernoulliPlateNode,
    NaryChoicePlateNode,
    NaryChoiceResultPlateNode,
)
from ..program import CPLogicProgram

P = Symbol("P")
Q = Symbol("Q")
Z = Symbol("Z")
x = Symbol("x")
a = Constant("a")
b = Constant("b")


def test_empty_program():
    code = Union(tuple())
    cpl_program = CPLogicProgram()
    cpl_program.walk(code)
    gm = testing.build_gm(cpl_program)
    assert gm.edges == tuple()
    assert gm.nodes == tuple()


def test_simple_deterministic_program():
    code = Union((Fact(Q(a)), Fact(Q(b)), Implication(P(x), Q(x))))
    cpl_program = CPLogicProgram()
    cpl_program.walk(code)
    gm = testing.build_gm(cpl_program)
    assert len(gm.nodes) == 2
    Q_node = gm.get_node(Q)
    P_node = gm.get_node(P)
    assert isinstance(Q_node, BernoulliPlateNode)
    assert isinstance(P_node, AndPlateNode)
    assert len(Q_node.relation.value) == 2
    assert Q_node.relation.value.arity == 2
    assert len(P_node.relation.value) == 2
    assert P_node.relation.value.arity == 1


def test_program_with_probfacts():
    code = Union((Implication(Q(x), P(x)),))
    probfacts_sets = {P: {(1.0, "a"), (0.5, "b"), (0.3, "c")}}
    cpl_program = CPLogicProgram()
    cpl_program.walk(code)
    for pred_symb, pfact_set in probfacts_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(pred_symb, pfact_set)
    gm = testing.build_gm(cpl_program)
    assert len(gm.edges) == 1
    Q_node = gm.get_node(Q)
    P_node = gm.get_node(P)
    assert gm.get_parent_node_symbols(Q) == {P}
    assert isinstance(Q_node, AndPlateNode)
    assert isinstance(P_node, BernoulliPlateNode)
    assert (
        testing.get_named_relation_tuples(P_node.relation) == probfacts_sets[P]
    )


def test_program_with_probchoice():
    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_choice_from_tuples(
        P, {(0.6, "a"), (0.4, "b")}
    )
    gm = testing.build_gm(cpl_program)
    P_node = gm.get_node(P)
    assert isinstance(P_node, NaryChoiceResultPlateNode)
    assert len(gm.get_parent_node_symbols(P)) == 1
    choice_node_symb = next(iter(gm.get_parent_node_symbols(P)))
    choice_node = gm.get_node(choice_node_symb)
    assert isinstance(choice_node, NaryChoicePlateNode)


def test_program_with_probchoice_and_intensional_rule():
    cpl_program = CPLogicProgram()
    cpl_program.walk(Union((Implication(Q(x), Conjunction((P(x), Z(x)))),)))
    cpl_program.add_probabilistic_facts_from_tuples(
        Z, {(0.6, "a"), (1.0, "b")}
    )
    cpl_program.add_probabilistic_choice_from_tuples(
        P, {(0.6, "a"), (0.4, "b")}
    )
    gm = testing.build_gm(cpl_program)
    assert gm.get_parent_node_symbols(Q) == {P, Z}
    assert len(gm.get_parent_node_symbols(P)) == 1
    choice_node_symb = next(iter(gm.get_parent_node_symbols(P)))
    Z_node = gm.get_node(Z)
    P_node = gm.get_node(P)
    choice_node = gm.get_node(choice_node_symb)
    assert isinstance(Z_node, BernoulliPlateNode)
    assert isinstance(P_node, NaryChoiceResultPlateNode)
    assert isinstance(choice_node, NaryChoicePlateNode)
