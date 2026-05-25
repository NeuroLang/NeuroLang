"""Viterbi decoder: backpointer tracking for max-product semiring queries."""

from typing import AbstractSet

from ..expressions import Constant, Symbol
from ..logic import Conjunction, Implication, Union
from ..relational_algebra import (
    NamedRelationalAlgebraFrozenSet,
    str2columnstr_constant,
)
from ..relational_algebra_provenance import ProvenanceAlgebraSet
from . import dalvi_suciu_lift
from .semiring import MaxProductSemiring


def compute_backpointers(
    cpl_program,
    prev_symbol,
    transition_symbol,
):
    """Build and solve a backpointer query for a transition step.

    Walks *bp(z, y) :- prev(y), trans(y, z)* into the program then
    solves it with MaxProductSemiring, returning one row per result
    state with the argmax previous state.
    """
    z = Symbol("z")
    y = Symbol("y")

    bp_sym = Symbol.fresh()
    body = Conjunction((prev_symbol(y), transition_symbol(y, z)))
    rule = Implication(bp_sym(z, y), body)
    cpl_program.walk(Union((rule,)))

    ans_sym = Symbol("ans")
    query = Implication(ans_sym(z, y), bp_sym(z, y))
    raw = dalvi_suciu_lift.solve_succ_query(
        query, cpl_program, semiring=MaxProductSemiring()
    )

    best = {}
    for row in raw.relation.value:
        score = float(row[0])
        state = str(row[1])
        prev = str(row[2]) if len(row) > 2 else None
        if state not in best or score > best[state][0]:
            best[state] = (score, prev)

    data = [(score, state, prev) for state, (score, prev) in best.items()]
    return ProvenanceAlgebraSet(
        Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(("_p_", "z", "y"), data)
        ),
        str2columnstr_constant("_p_"),
    )


def decode_viterbi(
    cpl_program,
    variables,
    rules,
    query_symbols,
    transition_symbols=None,
):
    """Full Viterbi decode with backpointer tracking."""
    results = {}
    traces = {}
    backpointers = {}
    if transition_symbols is None:
        transition_symbols = {}

    x = Symbol("x")
    ans_sym = Symbol("ans")

    for step in sorted(rules.keys()):
        cpl_program.walk(Union((rules[step],)))

        qsym = query_symbols[step]
        query = Implication(ans_sym(x), qsym(x))
        results[step] = dalvi_suciu_lift.solve_succ_query(
            query, cpl_program, semiring=MaxProductSemiring()
        )

        if step in transition_symbols:
            ts = transition_symbols[step]
            bp_result = compute_backpointers(
                cpl_program, ts["prev"], ts["transition"]
            )
            traces[step] = list(bp_result.relation.value)
            bp_map = {}
            for row in bp_result.relation.value:
                score = float(row[0])
                state = str(row[1])
                prev_state = str(row[2]) if len(row) > 2 else None
                if state not in bp_map or score > bp_map[state][0]:
                    bp_map[state] = (score, prev_state)
            backpointers[step] = {k: v[1] for k, v in bp_map.items()}

    return results, traces, backpointers


def extract_backpointers(
    max_product_result,
    result_col=0,
    state_col=1,
    backpointer_col=2,
):
    """For each result state, find which existential gave the max score.
    Returns {state: argmax_value}.
    """
    backpointers = {}
    data = max_product_result.relation.value
    for row in data:
        score = row[result_col]
        state = row[state_col] if len(row) > state_col else None
        bp = row[backpointer_col] if len(row) > backpointer_col else None
        if state is not None and bp is not None:
            if state not in backpointers or score > backpointers[state][0]:
                backpointers[state] = (score, bp)
    return {k: v[1] for k, v in backpointers.items()}


def extract_traces(
    max_product_result,
    score_col=0,
    state_col=1,
    backpointer_col=2,
):
    """Extract all backpointer tuples [(score, state, backpointer), ...]."""
    traces = []
    data = max_product_result.relation.value
    for row in data:
        score = row[score_col] if len(row) > score_col else None
        state = row[state_col] if len(row) > state_col else None
        bp = row[backpointer_col] if len(row) > backpointer_col else None
        traces.append((score, state, bp))
    return traces


def trace_path(backpointers, final_state, steps):
    """Reconstruct the most likely state sequence from backpointer maps."""
    path = [final_state]
    for step in reversed(steps[1:]):
        bp = backpointers.get(step, {})
        prev = bp.get(path[-1])
        if prev is not None:
            path.append(prev)
        else:
            break
    return list(reversed(path))


__all__ = [
    "compute_backpointers",
    "decode_viterbi",
    "extract_backpointers",
    "extract_traces",
    "trace_path",
]
