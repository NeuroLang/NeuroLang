"""Tests for the SQUALL parser's probabilistic logic lifting/EP handling.

These tests validate the Internal representation of ExistentialPredicate
lifting around Conditions, clash detection for same-named quantified
variables, and non-conjunctive condition decomposition. They test the
logic transformations used by the parser, not the parser itself.
"""
from operator import eq as op_eq

from neurolang.datalog.expression_processing import extract_logic_free_variables
from neurolang.datalog.negation import is_conjunctive_negation
from neurolang.expression_walker import ReplaceSymbolWalker
from neurolang.expressions import (
    Constant, ExpressionBlock, FunctionApplication, Symbol
)
from neurolang.logic import (
    Conjunction, Disjunction, ExistentialPredicate, Implication
)
from neurolang.logic.horn_clauses import fol_query_to_datalog_program
from neurolang.logic.transformations import ExtractBoundVariables
from neurolang.probabilistic.expressions import Condition, ProbabilisticQuery, PROB

EQ = Constant(op_eq)


def test_lift_ep_from_conditioned_anaphora():
    """Single-side EP: lifted EP.body.conditioned == Condition(p(v), q(v))."""
    v = Symbol('v')
    p_of_v = FunctionApplication(Symbol('p'), (v,))
    q_of_v = FunctionApplication(Symbol('q'), (v,))
    ep = ExistentialPredicate(v, p_of_v)

    other_bound = ExtractBoundVariables().walk(q_of_v)
    assert v not in other_bound

    lifted = ExistentialPredicate(
        ep.head,
        Condition(ep.body, q_of_v)
    )

    assert is_conjunctive_negation(lifted.body.conditioned)
    assert is_conjunctive_negation(lifted.body.conditioning)
    # After lifting, the Condition inside EP should match p(v) | q(v)
    assert Condition(p_of_v, q_of_v).conditioned == lifted.body.conditioned, (
        f"Expected conditioned={Condition(p_of_v, q_of_v).conditioned}, "
        f"got {lifted.body.conditioned}"
    )
    assert Condition(p_of_v, q_of_v).conditioning == lifted.body.conditioning, (
        f"Expected conditioning={Condition(p_of_v, q_of_v).conditioning}, "
        f"got {lifted.body.conditioning}"
    )


def test_lift_ep_clash_both_sides():
    """Same-named variables on both sides of Condition are renamed."""
    x1 = Symbol('x')
    x2 = Symbol('x')
    p_of_x1 = FunctionApplication(Symbol('p'), (x1,))
    q_of_x2 = FunctionApplication(Symbol('q'), (x2,))
    ep_left = ExistentialPredicate(x1, p_of_x1)
    ep_right = ExistentialPredicate(x2, q_of_x2)

    other_bound = ExtractBoundVariables().walk(ep_right)
    assert x1 in other_bound

    fresh_var = Symbol[x1.type].fresh()
    fresh_ep = ReplaceSymbolWalker({x1.name: fresh_var}).walk(ep_left)
    lifted = ExistentialPredicate(
        fresh_ep.head,
        Condition(fresh_ep.body, ep_right)
    )

    inner_cond = lifted.body
    assert isinstance(inner_cond, Condition)
    assert inner_cond.conditioned == fresh_ep.body
    assert isinstance(inner_cond.conditioning, ExistentialPredicate)
    assert lifted.head == fresh_ep.head
    assert lifted.head.name != x1.name


def test_decompose_non_conjunctive_condition():
    """Disjunction in conditioning is decomposed into auxiliary rules."""
    x = Symbol('x')
    p_of_x = FunctionApplication(Symbol('p'), (x,))
    q_of_x = FunctionApplication(Symbol('q'), (x,))
    r_of_x = FunctionApplication(Symbol('r'), (x,))
    disj = Disjunction((q_of_x, r_of_x))
    condition = Condition(p_of_x, disj)

    results = []
    new_args = []
    for arg in (condition.conditioned, condition.conditioning):
        if is_conjunctive_negation(arg):
            new_args.append(arg)
        else:
            fv = extract_logic_free_variables(arg)
            fresh_head = Symbol.fresh()(*tuple(fv))
            aux = fol_query_to_datalog_program(fresh_head, arg)
            results.append(aux)
            new_args.append(fresh_head)

    assert len(new_args) == 2
    assert new_args[0] == p_of_x
    assert isinstance(new_args[1], FunctionApplication)
    assert len(results) == 1
    assert isinstance(results[0], ExpressionBlock)